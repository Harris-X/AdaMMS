import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from functools import partial
from warnings import warn
from typing import List, Dict
import matplotlib.pyplot as plt
from .utils import add_colorbar

class CKA:
    def __init__(self,
                 model1: nn.Module,
                 model2: nn.Module,
                 model1_name: str = None,
                 model2_name: str = None,
                 model1_layers: List[str] = None, # 修改：直接接收层的名字列表
                 model2_layers: List[str] = None, # 修改：直接接收层的名字列表
                 device: str ='cpu'):
        """
        一个通用的、与模型架构无关的CKA实现。

        :param model1: (nn.Module) 神经网络 1
        :param model2: (nn.Module) 神经网络 2
        :param model1_name: (str) 模型1的名称
        :param model2_name: (str) 模型2的名称
        :param model1_layers: (List[str]) 需要从模型1中提取特征的层的名称列表
        :param model2_layers: (List[str]) 需要从模型2中提取特征的层的名称列表
        :param device: (str) 运行模型的设备
        """
        self.model1 = model1
        self.model2 = model2
        self.device = device
        self.model1_info = {}
        self.model2_info = {}

        self.model1_info['Name'] = model1_name or model1.__repr__().split('(')[0]
        self.model2_info['Name'] = model2_name or model2.__repr__().split('(')[0]

        if self.model1_info['Name'] == self.model2_info['Name']:
            warn(f"两个模型名称相同: {self.model2_info['Name']}。这可能会在解释结果时引起混淆。")

        # 直接使用传入的层列表
        self.model1_layers = model1_layers
        self.model2_layers = model2_layers
        
        self.model1_features = {}
        self.model2_features = {}

        self._insert_hooks()

        self.model1 = self.model1.to(self.device)
        self.model2 = self.model2.to(self.device)

        self.model1.eval()
        self.model2.eval()

    def _log_layer(self,
                   model: str,
                   name: str,
                   layer: nn.Module,
                   inp: torch.Tensor,
                   out: torch.Tensor):
        """Hook函数，用于记录指定层的输出特征"""
        # 对于Transformer，输出通常是元组 (hidden_states, ...)，我们只取 hidden_states
        if isinstance(out, tuple):
            feature = out[0]
        else:
            feature = out

        if model == "model1":
            self.model1_features[name] = feature
        elif model == "model2":
            self.model2_features[name] = feature
        else:
            raise RuntimeError("未知的模型名称 for _log_layer.")

    def _get_module_by_name(self, model, name):
        """通过字符串名称获取模块"""
        names = name.split('.')
        module = model
        for n in names:
            module = getattr(module, n)
        return module

    def _insert_hooks(self):
        """
        为self.model1_layers和self.model2_layers中指定的层注册前向钩子。
        """
        if self.model1_layers is None or self.model2_layers is None:
            raise ValueError("必须提供 model1_layers 和 model2_layers 列表。")

        # 为模型1注册钩子
        for name in self.model1_layers:
            try:
                layer = self._get_module_by_name(self.model1, name)
                layer.register_forward_hook(partial(self._log_layer, "model1", name))
            except AttributeError:
                warn(f"警告: 在模型1中未找到层 '{name}'。跳过。")


        # 为模型2注册钩子
        for name in self.model2_layers:
            try:
                layer = self._get_module_by_name(self.model2, name)
                layer.register_forward_hook(partial(self._log_layer, "model2", name))
            except AttributeError:
                warn(f"警告: 在模型2中未找到层 '{name}'。跳过。")
        
        print(f"已为模型1的 {len(self.model1_layers)} 个层注册钩子。")
        print(f"已为模型2的 {len(self.model2_layers)} 个层注册钩子。")


    def _HSIC(self, K, L):
        """
        计算HSIC度量的无偏估计。
        参考: https://arxiv.org/pdf/2010.15327.pdf Eq (3)
        """
        N = K.shape[0]
        if N < 4:
            # HSIC的无偏估计要求 N > 3
            return 0.0
            
        ones = torch.ones(N, 1, device=self.device)
        result = torch.trace(K @ L)
        result += ((ones.t() @ K @ ones @ ones.t() @ L @ ones) / ((N - 1) * (N - 2))).item()
        result -= ((ones.t() @ K @ L @ ones) * 2 / (N - 2)).item()
        return (1 / (N * (N - 3)) * result).item()

    def _center_gram(self, gram, unbiased=False):
        """中心化Gram矩阵"""
        if unbiased:
            # HSIC的无偏估计不需要中心化，因为其公式内部已经处理了
            return gram

        gram = gram.clone()
        mean = torch.mean(gram, dim=0, keepdim=True)
        gram = gram - mean
        gram = gram - torch.mean(gram, dim=1, keepdim=True)
        return gram

    def _cka(self, gram_x, gram_y, unbiased=False):
        """计算两个Gram矩阵之间的CKA相似度"""
        gram_x = self._center_gram(gram_x, unbiased)
        gram_y = self._center_gram(gram_y, unbiased)
        
        # CKA score is HSIC(K, L) / sqrt(HSIC(K, K) * HSIC(L, L))
        # 我们使用有偏估计来保持稳定性
        scaled_hsic = (gram_x.T @ gram_y).sum()
        norm_x = (gram_x.T @ gram_x).sum()
        norm_y = (gram_y.T @ gram_y).sum()
        
        # 加上一个很小的eps防止除以零
        eps = 1e-6
        return scaled_hsic / (torch.sqrt(norm_x * norm_y) + eps)


    def compare(self, dataloader1: DataLoader, dataloader2: DataLoader = None):
        """
        在给定的数据集上计算模型之间的特征相似度。
        """
        if dataloader2 is None:
            warn("模型2的Dataloader未提供。将为两个模型使用相同的Dataloader。")
            dataloader2 = dataloader1

        self.model1_info['Dataset'] = dataloader1.dataset.__class__.__name__
        self.model2_info['Dataset'] = dataloader2.dataset.__class__.__name__

        N = len(self.model1_layers)
        M = len(self.model2_layers)
        
        print(f"正在比较来自 '{self.model1_info['Name']}' 的 {N} 个层和来自 '{self.model2_info['Name']}' 的 {M} 个层。")

        self.similarity_matrix = torch.zeros(N, M, device=self.device)
        
        # 使用自定义的collate_fn来处理异构tokenizer
        def collate_for_compare(batch):
            # 假设batch是一个元组 (batch_for_model1, batch_for_model2)
            return batch[0], batch[1]

        # 这里的dataloader应该是一个能同时产出两个模型输入的迭代器
        num_batches = len(dataloader1)

        for (x1, *_), (x2, *_) in tqdm(zip(dataloader1, dataloader2), desc="| CKA | 比较特征", total=num_batches):
            
            self.model1_features.clear()
            self.model2_features.clear()
            
            with torch.no_grad():
                _ = self.model1(x1.to(self.device))
                _ = self.model2(x2.to(self.device))
            
            for i, name1 in enumerate(self.model1_layers):
                if name1 not in self.model1_features: continue
                X = self.model1_features[name1].flatten(1) # [Batch, Features]
                
                # 计算Gram矩阵 K = X @ X.T
                K = X @ X.T
                
                for j, name2 in enumerate(self.model2_layers):
                    if name2 not in self.model2_features: continue
                    Y = self.model2_features[name2].flatten(1) # [Batch, Features]

                    # 计算Gram矩阵 L = Y @ Y.T
                    L = Y @ Y.T
                    
                    # 累加每个批次的CKA值
                    self.similarity_matrix[i, j] += self._cka(K, L)
        
        # 对所有批次的结果取平均
        self.similarity_matrix /= num_batches

        assert not torch.isnan(self.similarity_matrix).any(), "CKA 计算结果中出现NAN"
        return self.similarity_matrix.cpu()


    def plot_results(self, save_path: str = None, title: str = None):
        fig, ax = plt.subplots()
        im = ax.imshow(self.similarity_matrix.cpu().numpy(), origin='lower', cmap='magma', vmin=0.0, vmax=1.0)
        ax.set_xlabel(f"Layers {self.model2_info['Name']}", fontsize=15)
        ax.set_ylabel(f"Layers {self.model1_info['Name']}", fontsize=15)

        if title is not None:
            ax.set_title(f"{title}", fontsize=18)
        else:
            ax.set_title(f"{self.model1_info['Name']} vs {self.model2_info['Name']}", fontsize=18)

        add_colorbar(im)
        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=300)

        plt.show()