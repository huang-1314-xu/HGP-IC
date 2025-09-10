import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

import dgl
import dgl.nn as dglnn


class SAGE1(nn.Module):##三层神经网络
    def __init__(self, in_feats, n_hidden, n_classes):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, "mean"))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, "mean"))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, "mean"))
        self.dropout = nn.Dropout(0.5)

          
 

    # sg为子图
    def forward(self, sg, x):
        h = x
        for l, layer in enumerate(self.layers):
            h = layer(sg, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h

    def inference(self, sg, x):
        h = x
        for l, layer in enumerate(self.layers):
            h = layer(sg, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
        return h

class SAGE2(nn.Module): ###两层
    def __init__(self, in_feats, n_hidden, n_classes):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, "mean"))       
        self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, "mean"))
        self.dropout = nn.Dropout(0.5)
        # for layer in self.layers:
        #     init.xavier_uniform_(layer.weight)
          
 

    # sg为子图
    def forward(self, sg, x):
        h = x
        for l, layer in enumerate(self.layers):
            
            h = layer(sg, h)
                  
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h

    def inference(self, sg, x):
        h = x
        for l, layer in enumerate(self.layers):
            h = layer(sg, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
        return h


# class Graph_Conv(nn.Module):
#     def __init__(self, in_feats, n_hidden, n_classes):
#         super().__init__()
#         self.layers = nn.ModuleList()
#         self.layers.append(dglnn.GraphConv(in_feats, n_hidden))
#         self.layers.append(dglnn.GraphConv(n_hidden, n_hidden))
#         self.layers.append(dglnn.GraphConv(n_hidden, n_classes))
#         self.dropout = nn.Dropout(0.2)

#     # sg为子图
#     def forward(self, sg, x):
#         h = x
#         for l, layer in enumerate(self.layers):
#             h = layer(sg, h)
#             if l != len(self.layers) - 1:
#                 h = F.relu(h)
#                 h = self.dropout(h)
#         return h

#     def inference(self, sg, x):
#         h = x
#         for l, layer in enumerate(self.layers):
#             h = layer(sg, h)
#             if l != len(self.layers) - 1:
#                 h = F.relu(h)
#         return h

# 定义 GCN 模型
class Graph_Conv(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, dropout):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.GraphConv(in_feats, n_hidden))
        self.layers.append(dglnn.GraphConv(n_hidden, n_hidden))
        self.layers.append(dglnn.GraphConv(n_hidden, n_classes))

          ##两层 GCN
        # self.layers.append(dglnn.GraphConv(in_feats, n_hidden))  # 第一层
        # self.layers.append(dglnn.GraphConv(n_hidden, n_classes))  # 第二层
        self.dropout = nn.Dropout(dropout)

    # sg 为子图
    def forward(self, sg, x):
        h = x
        for l, layer in enumerate(self.layers):
            h = layer(sg, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h

    def inference(self, sg, x):
        sg = dgl.remove_self_loop(sg)  # 在这里添加自环
          # 为每个子图添加自环
        sg = dgl.add_self_loop(sg)  # 在这里添加自环
        h = x
        for l, layer in enumerate(self.layers):
            h = layer(sg, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
        return h
