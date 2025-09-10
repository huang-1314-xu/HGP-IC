import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn
import pickle
import numpy as np


class SAGE1(nn.Module):  # 三层神经网络
    def __init__(self, in_feats, n_hidden, n_classes):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layer1 = dglnn.SAGEConv(in_feats, n_hidden, "mean")
        self.layer2 = dglnn.SAGEConv(n_hidden, n_hidden, "mean")
        self.layer3 = dglnn.SAGEConv(n_hidden, n_classes, "mean")
        self.dropout = nn.Dropout(0.5)

    # sg为子图
    def forward(self, g1, g2, g_h,g_h2):  # g_h2

        g1_ids = g1.ndata['_ID']
        g2_ids = g2.ndata['_ID']
        # print(g2_ids)

        ##将膨胀之后的子图的顺序和原子图相对应   原子图：3,7,5节点  膨胀图：2,5,4,7,6,3-》3,7,5,2,4,6
        # 将 _ID 转换为 CPU 上的 numpy 数组以进行处理
        g1_ids_np = g1_ids.detach().cpu().numpy()
        g2_ids_np = g2_ids.detach().cpu().numpy()
        # 建立从 g1_ids 到索引的映射
        id_to_idx_g1 = {id: idx for idx, id in enumerate(g1_ids_np)}
        # 为 g2 中的每个 _ID 查找对应的 g1 索引，如果未找到则为 -1
        idx1_list = [id_to_idx_g1.get(id2, -1) for id2 in g2_ids_np]
        idx1_tensor = torch.tensor(idx1_list).to(g2_ids.device)
        # 创建匹配和未匹配的掩码
        mask = idx1_tensor != -1
        not_mask = ~mask  ##未匹配的节点

        g2_mask_ids = g2.ndata['_ID'][mask]
        g2_not_mask_ids = g2.ndata['_ID'][not_mask]

        h = g2.ndata['feat']
        h = self.layer1(g2, h)  ###第一层GNN
        h[not_mask] = g_h.ndata['feat'][g2_not_mask_ids].detach()
        g_h.ndata['feat'][g2_mask_ids] = h[mask].detach()

        h = F.relu(h)
        h = self.dropout(h)
        # #第二层
        h = self.layer2(g2, h)  ###
        h[not_mask] = g_h2.ndata['feat'][g2_not_mask_ids].detach()
        g_h2.ndata['feat'][g2_mask_ids] = h[mask].detach()
        h = F.relu(h)
        h = self.dropout(h)
        # h = self.layer2(g2, h)
        #第三层
        h = self.layer3(g2, h)
        return h
###全图推理
    def inference(self, sg, x):
        h = x
        h = self.layer1(sg, h)
        h = F.relu(h)
        h = self.layer2(sg, h)
        h = F.relu(h)
        h = self.layer3(sg, h)
        return h

#####子图推理
    def inference1(self, g1, g2, g_h, g_h2):
        """推理函数,用于三层GNN模型的评估和预测
        Args:
            g1: 原始子图
            g2: 扩展后的子图(包含原始子图和它的邻居节点)
            g_h: 第一层GNN后的全局图隐藏特征
            g_h2: 第二层GNN后的全局图隐藏特征
        """
        # 获取原始子图和扩展子图的节点ID
        g1_ids = g1.ndata['_ID']  # 原始子图节点ID
        g2_ids = g2.ndata['_ID']  # 扩展子图节点ID
        # 构建原始子图节点ID到索引的映射
        id_to_idx_g1 = {id: idx for idx, id in enumerate(g1_ids.detach().cpu().numpy())}
        # 找出扩展子图中每个节点是否属于原始子图
        idx1_list = [id_to_idx_g1.get(id2, -1) for id2 in g2_ids.detach().cpu().numpy()]
        idx1_tensor = torch.tensor(idx1_list).to(g2_ids.device)
        # 创建节点掩码:区分原始节点和新增节点
        mask = idx1_tensor != -1  # True表示原始子图中的节点
        not_mask = ~mask  # True表示新增的邻居节点

        # 获取原始节点和新增节点的ID
        g2_mask_ids = g2.ndata['_ID'][mask]  # 原始子图节点的ID
        g2_not_mask_ids = g2.ndata['_ID'][not_mask]  # 新增邻居节点的ID

        # 第一层特征传播
        h = g2.ndata['feat']  # 获取节点特征
        h = self.layer1(g2, h)  # 第一层GNN传播

        # 第一层特征同步
        h[not_mask] = g_h.ndata['feat'][g2_not_mask_ids].detach()  # 新增节点使用全局图的特征
        g_h.ndata['feat'][g2_mask_ids] = h[mask].detach()  # 更新全局图中原始节点的特征
        # 第一层非线性变换
        h = F.relu(h)  # 激活函数
        # 第二层特征传播
        h = self.layer2(g2, h)  # 第二层GNN传播
        # 第二层特征同步
        h[not_mask] = g_h2.ndata['feat'][g2_not_mask_ids].detach()  # 使用g_h2
        g_h2.ndata['feat'][g2_mask_ids] = h[mask].detach()
        # 第二层非线性变换
        h = F.relu(h)
        # 第三层特征传播
        h = self.layer3(g2, h)  # 第三层GNN传播
        return h

class SAGE2_new(nn.Module):  ###两层
    def __init__(self, in_feats, n_hidden, n_classes):
        super().__init__()
        self.layer1 = dglnn.SAGEConv(in_feats, n_hidden, "mean")
        self.layer2 = dglnn.SAGEConv(n_hidden, n_classes, "mean")
        self.dropout = nn.Dropout(0.5)

    # sg为子图
    def forward(self, g1,g2,g_h):#g_h2

        g1_ids  = g1.ndata['_ID']
        g2_ids  = g2.ndata['_ID']
        # print(g2_ids)

        ##将膨胀之后的子图的顺序和原子图相对应   原子图：3,7,5节点  膨胀图：2,5,4,7,6,3-》3,7,5,2,4,6
        # 将 _ID 转换为 CPU 上的 numpy 数组以进行处理
        g1_ids_np  = g1_ids.detach().cpu().numpy()
        g2_ids_np  = g2_ids.detach().cpu().numpy()
        # 建立从 g1_ids 到索引的映射
        id_to_idx_g1 = {id: idx for idx, id in enumerate(g1_ids_np)}
        # 为 g2 中的每个 _ID 查找对应的 g1 索引，如果未找到则为 -1
        idx1_list = [id_to_idx_g1.get(id2, -1) for id2 in g2_ids_np]
        idx1_tensor = torch.tensor(idx1_list).to(g2_ids.device)
        # 创建匹配和未匹配的掩码
        mask = idx1_tensor != -1
        not_mask = ~mask##未匹配的节点

        g2_mask_ids = g2.ndata['_ID'][mask]
        g2_not_mask_ids = g2.ndata['_ID'][not_mask]

        h = g2.ndata['feat']
        h = self.layer1(g2, h)###第一层GNN
        h[not_mask] = g_h.ndata['feat'][g2_not_mask_ids].detach()
        g_h.ndata['feat'][g2_mask_ids] = h[mask].detach()

        h = F.relu(h)
        h = self.dropout(h)
        # #第二层
        # h = self.layer2(g2, h)  ###
        # h[not_mask] = g_h2.ndata['feat'][g2_not_mask_ids].detach()
        # g_h2.ndata['feat'][g2_mask_ids] = h[mask].detach()
        # h = F.relu(h)
        # h = self.dropout(h)
        h = self.layer2(g2, h)
        # h = self.layer3(g2, h)
        return h
###全图测试
    def inference(self, sg, x):
        h = x
        h = self.layer1(sg, h)
        h = F.relu(h)
        h = self.layer2(sg, h)
        return h
###子图测试
    def inference1(self, g1, g2, g_h):
        """推理函数,用于模型评估和预测
        Args:
            g1: 原始子图
            g2: 扩展后的子图(包含原始子图和它的邻居节点)
            g_h: 全局图的隐藏特征
        """
        # 获取原始子图和扩展子图的节点ID
        g1_ids = g1.ndata['_ID']  # 原始子图节点ID
        g2_ids = g2.ndata['_ID']  # 扩展子图节点ID

        # 构建原始子图节点ID到索引的映射
        id_to_idx_g1 = {id: idx for idx, id in enumerate(g1_ids.detach().cpu().numpy())}
        # 找出扩展子图中每个节点是否属于原始子图
        idx1_list = [id_to_idx_g1.get(id2, -1) for id2 in g2_ids.detach().cpu().numpy()]
        idx1_tensor = torch.tensor(idx1_list).to(g2_ids.device)

        # 创建节点掩码:区分原始节点和新增节点
        mask = idx1_tensor != -1  # True表示原始子图中的节点
        not_mask = ~mask  # True表示新增的邻居节点

        # 获取原始节点和新增节点的ID
        g2_mask_ids = g2.ndata['_ID'][mask]  # 原始子图节点的ID
        g2_not_mask_ids = g2.ndata['_ID'][not_mask]  # 新增邻居节点的ID

        # 第一层特征传播
        h = g2.ndata['feat']  # 获取节点特征
        h = self.layer1(g2, h)  # 第一层GNN传播

        # 特征同步:确保扩展图与全局图的特征一致性
        h[not_mask] = g_h.ndata['feat'][g2_not_mask_ids].detach()  # 新增节点使用全局图的特征
        g_h.ndata['feat'][g2_mask_ids] = h[mask].detach()  # 更新全局图中原始节点的特征

        # 非线性变换
        h = F.relu(h)  # 激活函数
        # 注意:推理时不使用dropout,因为需要完整的网络进行预测

        # 第二层特征传播
        h = self.layer2(g2, h)  # 第二层GNN传播

        return h


class SAGE2(nn.Module):  # 两层神经网络
    def __init__(self, in_feats, n_hidden, n_classes):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, "mean"))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, "mean"))
        self.dropout = nn.Dropout(0.5)
        # 通过门控机制控制特征加权
        self.gate = nn.Parameter(torch.tensor(0.5))  # 初始化门控值，可以通过训练进行调整

    def forward(self, sg, x, node_dict, replica_node_ids_set):
        node_ids = sg.ndata['_ID']
        node_ids_set = set(node_ids.tolist())  # 将节点 ID 转换为集合
        intersection_nodes = node_ids_set.intersection(replica_node_ids_set)  # 找到复制节点的交集

        h = x.to(next(self.parameters()).device)  # 将输入特征 x 移动到正确的设备

        # **第一层卷积：通过虚拟边进行消息传递并更新节点特征**
        for l, layer in enumerate(self.layers):
            h = layer(sg, h)  # 通过卷积层进行消息传递

            if l == 0:
                # **第一层输出**：获取第一层卷积后的节点特征
                layer_output = {node_id.item(): h[idx].clone().detach() for idx, node_id in enumerate(node_ids)}
                hist_dict = node_dict  # 历史特征字典

                # **通过虚拟边进行消息传递并聚合**
                for node_id in intersection_nodes:
                    if node_id in hist_dict:
                        # 获取历史特征，并进行归一化（防止数值不稳定）
                        norm_hist = hist_dict[node_id] / (torch.norm(hist_dict[node_id]) + 1e-6)
                        norm_feature = layer_output[node_id] / (torch.norm(layer_output[node_id]) + 1e-6)

                        # 使用门控机制对历史特征和当前特征进行加权
                        gate_value = torch.sigmoid(self.gate)  # 通过Sigmoid函数输出门控值（0到1之间）

                        # 使用门控值来加权特征
                        hist_dict[node_id] = gate_value * norm_hist + (1 - gate_value) * norm_feature

                # **将第一层输出的特征替换到字典中**
                for node_id in node_ids:
                    node_id_int = node_id.item()
                    node_dict[node_id_int] = layer_output[node_id_int]  # 更新字典中的节点特征

            # **第一层卷积后应用激活函数和 Dropout**
            if l != len(self.layers) - 1:  # 不是最后一层
                h = F.relu(h)  # ReLU 激活
                h = self.dropout(h)  # Dropout

        return h, node_dict



    # 推理函数不需要补偿机制
    def inference(self, sg, x):
        h = x
        for l, layer in enumerate(self.layers):
            h = layer(sg, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
        return h
