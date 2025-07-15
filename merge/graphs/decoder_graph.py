from graphs.base_graph import BIGGraph, NodeType

class DecoderGraph(BIGGraph):
    def __init__(self, model, model_type='llama'):
        super().__init__(model)
        self.model_type = model_type
        # 根据模型类型确定层的名称
        if model_type == 'llama':
            self.layer_prefix = 'model.layers'
            self.attention_prefix = 'self_attn'
            self.q_proj, self.k_proj, self.v_proj, self.o_proj = 'q_proj', 'k_proj', 'v_proj', 'o_proj'
            self.mlp_prefix = 'mlp'
            self.up_proj, self.gate_proj, self.down_proj = 'up_proj', 'gate_proj', 'down_proj'
            self.input_layernorm = 'input_layernorm'
            self.post_attention_layernorm = 'post_attention_layernorm'
        elif model_type == 'qwen':
            self.layer_prefix = 'model.layers'
            self.attention_prefix = 'self_attn'
            # Qwen2使用一个大的线性层来同时完成Q, K, V的投影
            self.q_proj, self.k_proj, self.v_proj = 'q_proj', 'k_proj', 'v_proj'
            self.o_proj = 'o_proj'
            self.mlp_prefix = 'mlp'
            self.up_proj, self.gate_proj, self.down_proj = 'up_proj', 'gate_proj', 'down_proj'
            self.input_layernorm = 'input_layernorm'
            self.post_attention_layernorm = 'post_attention_layernorm'
        else:
            raise ValueError(f"Unsupported model type: {model_type}")


    def graphify(self):
        # 输入节点
        input_node = self.create_node(node_type=NodeType.INPUT)

        # Embedding 层
        emb_node = self.add_nodes_from_sequence('', ['model.embed_tokens'], input_node)

        # Transformer Blocks
        last_node = emb_node
        num_layers = self.model.config.num_hidden_layers
        for i in range(num_layers):
            layer_name = f'{self.layer_prefix}.{i}'
            
            # 残差连接的起点
            residual_node = last_node

            # Self-Attention 块
            attn_ln_node = self.add_nodes_from_sequence(layer_name, [self.input_layernorm, NodeType.PREFIX], last_node)
            
            # Q, K, V 投影
            q_node = self.add_nodes_from_sequence(f'{layer_name}.{self.attention_prefix}', [self.q_proj], attn_ln_node)
            k_node = self.add_nodes_from_sequence(f'{layer_name}.{self.attention_prefix}', [self.k_proj], attn_ln_node)
            v_node = self.add_nodes_from_sequence(f'{layer_name}.{self.attention_prefix}', [self.v_proj, NodeType.PREFIX], attn_ln_node) # 在V之后放置一个PREFIX

            # Attention输出
            attn_output_node = self.add_nodes_from_sequence(f'{layer_name}.{self.attention_prefix}', [self.o_proj], v_node)
            
            # 第一个残差连接
            sum_node_1 = self.create_node(node_type=NodeType.SUM)
            self.add_directed_edge(residual_node, sum_node_1)
            self.add_directed_edge(attn_output_node, sum_node_1)

            # MLP 块
            residual_node_2 = sum_node_1
            mlp_ln_node = self.add_nodes_from_sequence(layer_name, [self.post_attention_layernorm, NodeType.PREFIX], residual_node_2)
            
            gate_proj_node = self.add_nodes_from_sequence(f'{layer_name}.{self.mlp_prefix}', [self.gate_proj], mlp_ln_node)
            up_proj_node = self.add_nodes_from_sequence(f'{layer_name}.{self.mlp_prefix}', [self.up_proj], mlp_ln_node)

            # 通常在gate和up之后会有一个element-wise的乘法，这里简化为一个SUM节点
            act_node = self.create_node(node_type=NodeType.SUM)
            self.add_directed_edge(gate_proj_node, act_node)
            self.add_directed_edge(up_proj_node, act_node)

            down_proj_node = self.add_nodes_from_sequence(f'{layer_name}.{self.mlp_prefix}', [self.down_proj], act_node)

            # 第二个残差连接
            sum_node_2 = self.create_node(node_type=NodeType.SUM)
            self.add_directed_edge(residual_node_2, sum_node_2)
            self.add_directed_edge(down_proj_node, sum_node_2)

            last_node = sum_node_2


        # Final LayerNorm 和输出头
        final_ln_node = self.add_nodes_from_sequence('', ['model.norm', NodeType.PREFIX], last_node)
        output_head_node = self.add_nodes_from_sequence('', ['lm_head'], final_ln_node)

        # 输出节点
        output_node = self.create_node(node_type=NodeType.OUTPUT)
        self.add_directed_edge(output_head_node, output_node)
        
        return self