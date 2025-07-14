import torch
from torch import nn
from tqdm.auto import tqdm
from copy import deepcopy

# 从项目中的其他文件中导入必要的类和函数
from graphs.base_graph import NodeType
from utils import get_merging_fn # 确保可以从 utils.py 导入此函数

class MergeHandler:
    """
    处理在模型图谱节点上应用的（非）合并变换。
    这个类作为应用变换的辅助工具，其逻辑是通用的，无需修改。
    """
    def __init__(self, graph, merge, unmerge):
        self.graph = graph
        # 存储不同类型模块层的（非）合并指令
        self.module_handlers = {
            'LayerNorm': self.handle_layernorm,
            'Linear': self.handle_linear,
            # 可以根据需要添加对其他特定模块类型的处理
        }
        self.merge = merge
        self.unmerge = unmerge

    def handle_layernorm(self, forward, node, module):
        """应用（非）合并操作到LayerNorm参数"""
        if forward:
            # 前向传播时，合并权重和偏置
            for parameter_name in ['weight', 'bias']:
                if hasattr(module, parameter_name) and getattr(module, parameter_name) is not None:
                    parameter = getattr(module, parameter_name)
                    parameter.data = self.merge @ parameter.data
            for succ in self.graph.succs(node):
                self.prop_forward(succ)
        else:
            # 反向传播时，不进行操作，让前一层处理
            if len(self.graph.preds(node)) > 0:
              self.prop_back(self.graph.preds(node)[0])


    def handle_linear(self, forward, node, module):
        """应用（非）合并操作到线性层参数"""
        if forward:  # unmerge
            module.weight.data = module.weight.data @ self.unmerge
        else:  # merge
            module.weight.data = self.merge @ module.weight.data
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data = self.merge @ module.bias.data

    def prop_back(self, node):
        """反向传播（非）合并度量"""
        if node in self.graph.merged:
            return
        
        info = self.graph.get_node_info(node)
        self.graph.merged.add(node)
        
        for succ in self.graph.succs(node):
            self.prop_forward(succ)
        
        if info['type'] == NodeType.MODULE:
            module = self.graph.get_module(info['layer'])
            handler = self.module_handlers.get(module.__class__.__name__)
            if handler:
                handler(False, node, module)
        elif info['type'] != NodeType.INPUT:
             for pred in self.graph.preds(node):
                self.prop_back(pred)
    
    def prop_forward(self, node):
        """正向传播（非）合并变换"""
        if node in self.graph.unmerged:
            return
        
        info = self.graph.get_node_info(node)
        self.graph.unmerged.add(node)
        
        if info['type'] == NodeType.MODULE:
            module = self.graph.get_module(info['layer'])
            handler = self.module_handlers.get(module.__class__.__name__)
            if handler:
                handler(True, node, module)
        elif info['type'] != NodeType.OUTPUT:
             for succ in self.graph.succs(node):
                self.prop_forward(succ)


class ModelMerge(nn.Module):
    """
    处理任意数量模型的合并操作。
    接收一个模型图谱列表（每个模型一个）。
    """
    def __init__(self, *graphs, device: torch.device):
        super().__init__()
        self.hooks = []
        self.init(graphs, device)

    def init(self, graphs, device: torch.device):
        """用新的图谱集初始化合并属性。"""
        for g in graphs:
            g.model.to(device).eval()
        self.graphs = graphs
        self.device = device
        self.merged_model = None
        # 为中间层计算添加钩子
        for graph in self.graphs:
            graph.add_hooks(device=device)

    def compute_metrics(self, dataloader, metric_classes):
        """
        计算模型间的成对对齐度量。
        此方法已被 `add_custom_merger_logic` 在主脚本中重写，以处理异构输入。
        这里的实现是一个占位符，展示了其基本逻辑。
        """
        raise NotImplementedError("此方法应由主脚本中的 `add_custom_merger_logic` 动态重写。")

    def compute_transformations(self, transform_fn, **kwargs):
        """
        使用对齐度量计算变换矩阵。
        """
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        start_time.record()

        self.merges = {}
        self.unmerges = {}
        
        # 获取所有需要计算度量的节点
        nodes = sorted([node for node in self.metrics.keys() if isinstance(node, int)])

        for node in tqdm(nodes, desc="正在计算变换矩阵"):
            metric = self.metrics[node]
            # fix_rate 控制合并后特征维度的大小
            fix_rate = kwargs.get("fix_rate", 0.5)
            # 这里的 (1-fix_rate) 类似于原始代码中的 r
            merge, unmerge = transform_fn(metric, model_dims=self.model_dims[node], **kwargs)
            
            # 乘以模型数量以进行归一化（在平均时）
            merge = merge * len(self.graphs)
            self.merges[node] = merge.chunk(len(self.graphs), dim=1)
            self.unmerges[node] = unmerge.chunk(len(self.graphs), dim=0)

        end_time.record()
        torch.cuda.synchronize()
        self.compute_transform_time = start_time.elapsed_time(end_time) / 1000
        print(f"计算变换矩阵耗时: {self.compute_transform_time:.2f} 秒")
        
        return self.merges, self.unmerges

    def apply_transformations(self):
        """
        将计算出的变换应用到每个模型的权重上。
        """
        print("正在应用变换到模型权重...")
        for node in tqdm(self.merges.keys(), desc="应用变换中"):
            merges = self.merges[node]
            unmerges = self.unmerges[node]
            for merge, unmerge, graph in zip(merges, unmerges, self.graphs):
                merger_handler = MergeHandler(graph, merge, unmerge)
                merger_handler.prop_back(node)

    def get_merged_state_dict(self):
        """
        通过线性插值（平均）合并后的模型权重，生成最终的状态字典。
        """
        print("正在平均化已对齐的权重...")
        # 使用第一个模型作为基础架构参考
        merged_dict = deepcopy(self.graphs[0].model.state_dict())
        
        for key in tqdm(merged_dict.keys(), desc="平均化权重"):
            # 只合并存在于所有模型中且形状相同的参数
            if all(key in g.model.state_dict() and 
                   merged_dict[key].shape == g.model.state_dict()[key].shape 
                   for g in self.graphs):
                
                # 累加所有模型的权重
                summed_weights = torch.stack([g.model.state_dict()[key] for g in self.graphs]).sum(dim=0)
                # 取平均
                merged_dict[key] = (summed_weights / len(self.graphs)).to(merged_dict[key].dtype)

        return merged_dict
              
    def transform(self, model, dataloader, transform_fn, metric_classes, **kwargs):
        """
        执行完整的合并流程：计算度量 -> 计算变换 -> 应用变换 -> 获取合并后的模型。
        """
        self.merged_model = model.to(self.device).eval()
        
        if not isinstance(metric_classes, dict):
            metric_classes = {x.name: x for x in metric_classes}
        
        # 1. 计算度量（此方法由主脚本中的 `add_custom_merger_logic` 重写）
        self.compute_metrics(dataloader, metric_classes=metric_classes)
        
        # 2. 计算变换
        self.compute_transformations(transform_fn, **kwargs)
        
        # 3. 应用变换
        self.apply_transformations()

    def clear_hooks(self):
        """从图谱中清除所有钩子。"""
        for g in self.graphs:
            g.clear_hooks()
        for hook in self.hooks:
            hook.remove()
        self.hooks = []