import numpy as np
import torch
import torch.nn.functional as F
import torchmetrics.functional as MF
import dgl
import util.Sampler
import util.Sampler2
import util.Module
import util.Module4
from sklearn.metrics import f1_score
import csv
from ogb.nodeproppred import DglNodePropPredDataset
import random
import time
import pickle
import copy

##############################################     实验
# 参数
methods = ["LDG", "Fennel", "Metis", "DBH", "HDRF", "ne", "sne"]  ##点划分
method = methods[4]  ##"DBH", "HDRF","reverse_HDRF"
num_partitions = 100
num_epochs = 150
batch_size = 1
num_hidden = 128
lr = 0.001
weight_decay = 5e-4
dropout = 0.5

# --------------------

aveLOSS = []
aveTrain_acc = []
Val_acc = []
Test_acc = []
Val_f1 = []
Test_f1 = []
node_num = []
edge_num = []

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)


########定义一个统计函数
def calculate_counts(tensor, a):
    # 初始化计数变量
    count_0 = 0
    count_0_to_a = 0
    count_a_to_2a = 0
    count_2a_to_3a = 0
    count_3a_to_4a = 0
    count_4a_and_above = 0

    # 遍历 tensor 的每个元素
    for element in tensor:
        if element == 0:
            count_0 += 1
        elif element > 0 and element < a:
            count_0_to_a += 1
        elif element >= a and element < 2 * a:
            count_a_to_2a += 1
        elif element >= 2 * a and element < 3 * a:
            count_2a_to_3a += 1
        elif element >= 3 * a and element < 4 * a:
            count_3a_to_4a += 1
        elif element >= 4 * a:
            count_4a_and_above += 1

    # 计算长度
    length = len(tensor)

    # 计算比例并存储在列表中
    result = [a,
              count_0 / length,
              count_0_to_a / length,
              count_a_to_2a / length,
              count_2a_to_3a / length,
              count_3a_to_4a / length,
              count_4a_and_above / length
              ]

    return result


# new_list = []

# with open('/22085400417/22414/415/ama_idtxt/ama_id_id.txt', 'r') as file:
#     for line in file:
#         number = int(line.strip())
#         new_list.append(number)

dataset = dgl.data.AsNodePredDataset(DglNodePropPredDataset("ogbn-products", root="/23085411005/23085411005/hdrf_hy_pycharm/data/ogb-products"))
graph = dataset[0]
print("原图节点数:",graph.number_of_nodes())
##节点的出度
in_degrees = graph.in_degrees()
out_degrees = graph.out_degrees()
# 找出入度和出度都为0的节点（即孤立节点）
isolated_nodes = torch.where((in_degrees == 0) & (out_degrees == 0))[0]
if isolated_nodes.numel() > 0:
    # graph.remove_nodes(torch.tensor(isolated_nodes))
    graph.remove_nodes(isolated_nodes.clone().detach())
    print("处理后的节点数目：",graph.number_of_nodes())
else:
    print("原图未处理节点数目")

num_edges = graph.num_edges()
num_nodes = graph.num_nodes()

# rg = dgl.reorder_graph(graph, node_permute_algo='custom',
#                   permute_config={'nodes_perm': new_list})###连续

###dataset = dgl.data.RedditDataset(raw_dir="/22085400415/dataset/shujvji/red")
####dataset= dgl.data.YelpDataset("/22085400415/dataset/shujvji/yel")###Nodes: 716,847 Edges: 13,954,819 Number of classes: 100 (Multi-class)，反向边，自环
# graph = rg
###graph = dgl.remove_self_loop(graph)  ###去自环
# 获取边数和节点数
num_edges = graph.num_edges()
num_nodes = graph.num_nodes()
# print("节点数", num_nodes)
# print("边数", num_edges)

""""
np.random.seed(42)
train_ratio = 0.6  # 训练集占总节点数的比例
val_ratio = 0.2  # 验证集占总节点数的比例
test_ratio = 0.2  # 测试集占总节点数的比例

# 初始化布尔型列表
train_mask = np.zeros(num_nodes, dtype=bool)
val_mask = np.zeros(num_nodes, dtype=bool)
test_mask = np.zeros(num_nodes, dtype=bool)

# 从总节点中随机选择节点，并将对应的布尔值设为True
node_indices = np.arange(num_nodes)
np.random.shuffle(node_indices)

num_train = int(train_ratio * num_nodes)
num_val = int(val_ratio * num_nodes)
num_test = int(test_ratio * num_nodes)

train_indices = node_indices[:num_train]
val_indices = node_indices[num_train:num_train + num_val]
test_indices = node_indices[num_train + num_val:num_train + num_val + num_test]

train_mask[train_indices] = True
val_mask[val_indices] = True
test_mask[test_indices] = True

# 将NumPy数组转换为PyTorch张量
train_mask = torch.from_numpy(train_mask)
val_mask = torch.from_numpy(val_mask)
test_mask = torch.from_numpy(test_mask)

# 将划分结果存储到g.ndata中
graph.ndata["train_mask"] = train_mask
graph.ndata["val_mask"] = val_mask
graph.ndata["test_mask"] = test_mask
"""

###输入维度                   隐藏层           输出维度，也就是节点的类别
model = util.Module4.SAGE1(graph.ndata["feat"].shape[1], num_hidden, dataset.num_classes).cuda()
# model = compare.util.Module.Graph_Conv(graph.ndata["feat"].shape[1], num_hidden, dataset.num_classes).cuda()
opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

dataloader = dgl.dataloading.DataLoader(  #####边划分
    graph,
    torch.arange(num_partitions).to("cuda"),
    util.Sampler2.Edge_partition_sampler_id(
        graph,
        num_partitions,  ###huafen_blosk
        cache_path=f'/23085411005/23085411005/hdrf_hy_pycharm/vertex_partitions/hdrf/ogb-products/100_bian',
        prefetch_ndata=["feat", "label", "train_mask", "val_mask", "test_mask"],
    ),
    device="cuda",
    batch_size=batch_size,
    shuffle=True,
    drop_last=False,
    num_workers=0,
    use_uva=True,
)  ####训练验证测试用相同的划分方法

eval_dataloader = dgl.dataloading.DataLoader(  #####边划分
    graph,
    torch.arange(num_partitions).to("cuda"),
    util.Sampler2.Edge_partition_sampler_id(
        graph,
        num_partitions,  ####CoraGraphDataset   ##"DBH", "HDRF","reverse_HDRF"
        cache_path=f'/23085411005/23085411005/hdrf_hy_pycharm/vertex_partitions/hdrf/ogb-products/100_bian',
        prefetch_ndata=["feat", "label", "train_mask", "val_mask", "test_mask"],
        ###"train_mask", "val_mask", "test_mask"
    ),
    device="cuda",
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=0,
    use_uva=True,
)  ####训练验证测试用相同的划分方法

# durations存放每次训练的时间，后面计算平均值和标准差
durations = []

num_nodes = graph.number_of_nodes()  # 只计算一次节点数
node_ids = torch.arange(num_nodes, device=device)  # 只创建一次节点ID

# 第一层特征图
graph_hidden1 = copy.deepcopy(graph).to(device)
graph_hidden1.ndata['feat'] = torch.zeros(num_nodes, num_hidden, device=device)
graph_hidden1.ndata['_ID'] = node_ids

# 第二层特征图
graph_hidden2 = copy.deepcopy(graph).to(device)
graph_hidden2.ndata['feat'] = torch.zeros(num_nodes, num_hidden, device=device)
graph_hidden2.ndata['_ID'] = node_ids


for e in range(num_epochs):
    print(f"epoch: {e + 1}")
    # 训练部分
    t0 = time.time()
    model.train()
    i = 0
    Loss = []
    Acc = []
    batch10_avenode_alldegraees = []  ###所有epoch中每隔10batch计算节点平均的度
    batch10_avenode_indegraees = []
    batch10_avenode_outdegraees = []
    batch_node_alldegraees = []
    batch_node_indegraees = []
    batch_node_outdegraees = []

    for it, sg in enumerate(dataloader):  ###训练
        i += 1
        sg_nodes = sg[0].ndata['_ID']
        successors = graph.out_edges(sg_nodes)[1]
        predecessors = graph.in_edges(sg_nodes)[0]
        neighbors = torch.cat([successors, predecessors]).unique()
        expanded_nodes = torch.cat([sg_nodes, neighbors]).unique()
        expanded_subgraph = dgl.node_subgraph(graph, expanded_nodes)
        ##一阶邻居集合，张量
        neighbors_set = set(neighbors.tolist())
        sub_nodes_set = set(sg_nodes.tolist())
        new_neighbors_set = neighbors_set - sub_nodes_set
        new_neighbors = torch.tensor(list(new_neighbors_set), device=neighbors.device)

        # graph_obj = sg[0]  # 提取DGL图对象
        graph_obj = expanded_subgraph  # 提取DGL图对象

        # print(graph_obj)
        x = graph_obj.ndata["feat"]  # 提取节点特征
        y = graph_obj.ndata["label"]  # 提取节点标签
        m = graph_obj.ndata["train_mask"].bool().to(device)  # 将训练掩码转换为布尔类型
        sg_node_ids = graph_obj.ndata['_ID']
        ##加入的新节点不进行梯度计算
        mask = torch.isin(sg_node_ids, new_neighbors).to(device)
        m = m & ~mask

        # 获取模型的预测输出
        # print(torch.max(graph_hidden.ndata['feat']))
        y_hat = model(sg[0], graph_obj, graph_hidden1, graph_hidden2)
        # y_hat = model(graph_obj, x)

        # 计算交叉熵损失，使用掩码选择训练样本
        loss = F.cross_entropy(y_hat[m], y[m])

        # 清除之前的梯度
        opt.zero_grad()

        # 执行标准的后向传播，计算初始梯度
        loss.backward()

        # 使用优化器更新模型参数
        opt.step()

        # 记录损失值
        Loss.append(loss.item())

        # 记录当前批次的节点数量和边数量
        node_num.append(graph_obj.ndata["feat"].shape[0])  # 节点数量
        edge_num.append(graph_obj.num_edges())  # 边数量

        # 计算入度，并记录平均入度
        tensor1 = graph_obj.in_degrees()  # 入度
        inaverage = tensor1.float().mean()
        batch_node_indegraees.append(tensor1.tolist())  # 记录每个节点的入度
        inaverage = round(inaverage.item(), 2)  # 四舍五入，保留两位小数
        batch10_avenode_indegraees.append(inaverage)  # 记录平均入度

        # 计算出度，并记录平均出度
        tensor2 = graph_obj.out_degrees()  # 出度
        outaverage = tensor2.float().mean()
        batch_node_outdegraees.append(tensor2.tolist())  # 记录每个节点的出度
        outaverage = round(outaverage.item(), 2)  # 四舍五入，保留两位小数
        batch10_avenode_outdegraees.append(outaverage)  # 记录平均出度

        # 计算总度（入度 + 出度），并记录平均总度
        result = torch.add(tensor1, tensor2)  # 总度
        allaverage = result.float().mean()
        batch_node_alldegraees.append(result.tolist())  # 记录每个节点的总度
        allaverage = round(allaverage.item(), 2)  # 四舍五入，保留两位小数
        batch10_avenode_alldegraees.append(allaverage)  # 记录平均总度

        # 计算当前批次的分类准确率
        acc = MF.accuracy(y_hat[m], y[m], task="multiclass", num_classes=dataset.num_classes)
        Acc.append(acc.item())  # 记录精度值

        # 获取当前 CUDA 设备的最大内存使用量（MB）
        mem = torch.cuda.max_memory_allocated() / 1000000

        # 打印当前批次的损失、准确率和显存峰值
        print(f"Loss {loss.item():.4f} | Acc {acc.item():.4f} | Peak Mem {mem:.2f}MB")

    # print("every_epoch_loss", Loss)
    # print("every_epoch_traacc", Acc)
    aveLOSS.append(f"{sum(Loss) / i:.8f}")  ##每一轮最后一个loss
    aveTrain_acc.append(f"{sum(Acc) / i:.8f}")

    # tt - to为一轮训练的时间###一个epoch时间
    tt = time.time()
    print(f"time: {tt - t0:.2f}s")
    durations.append(tt - t0)
    # print("batch10_avenode_alldegraees", batch10_avenode_alldegraees)
    # print("batch10_avenode_indegraees", batch10_avenode_indegraees)
    # print("batch10_avenode_outdegraees", batch10_avenode_outdegraees)

    # print("batch_node_alldegraees", batch_node_alldegraees)
    # print("batch_node_indegraees", batch_node_indegraees)
    # print("batch_node_outdegraees", batch_node_outdegraees)

    ###################################
    # 评估部分
    model.eval()
    with torch.no_grad():
        device = next(model.parameters()).device

        # 初始化tensor来存储预测结果和标签
        y_hat_tensor = torch.zeros(num_nodes, dataset.num_classes, device=device)
        val_false_tensor = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        test_false_tensor = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        y_tensor = torch.empty(num_nodes, dtype=torch.long, device=device)

        for it, sg in enumerate(eval_dataloader):
            # 获取子图原始节点并移至正确设备
            sg_nodes = sg[0].ndata['_ID'].to(device)

            # 获取一阶邻居并确保在正确设备上
            successors = graph.out_edges(sg_nodes)[1].to(device)
            predecessors = graph.in_edges(sg_nodes)[0].to(device)
            neighbors = torch.cat([successors, predecessors]).unique()

            # 构建扩展子图
            expanded_nodes = torch.cat([sg_nodes, neighbors]).unique()
            expanded_subgraph = dgl.node_subgraph(graph, expanded_nodes)

            # 使用向量化操作替代集合操作，确保在GPU上
            new_neighbors_mask = ~torch.isin(neighbors, sg_nodes)
            new_neighbors = neighbors[new_neighbors_mask]

            # 获取节点信息并移至正确设备
            sg_node_ids = expanded_subgraph.ndata['_ID'].to(device)
            y = expanded_subgraph.ndata["label"].to(device)
            m_val = expanded_subgraph.ndata["val_mask"].bool().to(device)
            m_test = expanded_subgraph.ndata["test_mask"].bool().to(device)

            # 排除新邻居节点，确保在正确设备上
            mask = torch.isin(sg_node_ids, new_neighbors).to(device)
            m_val = m_val & ~mask
            m_test = m_test & ~mask

            # 获取模型预测
            y_hat = model.inference1(sg[0], expanded_subgraph, graph_hidden1, graph_hidden2)
            y_hat = F.softmax(y_hat, dim=1)

            # 使用向量化操作更新结果
            valid_indices = ~mask
            valid_node_ids = sg_node_ids[valid_indices]

            # 批量更新所有张量
            y_hat_tensor.index_add_(0, valid_node_ids, y_hat[valid_indices])
            val_false_tensor[valid_node_ids] = m_val[valid_indices]
            test_false_tensor[valid_node_ids] = m_test[valid_indices]
            y_tensor[valid_node_ids] = y[valid_indices]

        # 计算准确率
        val_acc = MF.accuracy(y_hat_tensor[val_false_tensor],
                              y_tensor[val_false_tensor],
                              task="multiclass",
                              num_classes=dataset.num_classes)
        test_acc = MF.accuracy(y_hat_tensor[test_false_tensor],
                               y_tensor[test_false_tensor],
                               task="multiclass",
                               num_classes=dataset.num_classes)

        print(f"Val Acc {val_acc.item():.4f} | Test Acc {test_acc.item():.4f}")
        Val_acc.append(val_acc.item())
        Test_acc.append(test_acc.item())

    model.train()
    model = model.to(device)
    # model.eval()  ###每个epoch评估一次    在sg里面评估测试
    # with torch.no_grad():
    #     model = model.cpu()
    #     x = graph.ndata["feat"]
    #     y = graph.ndata["label"]
    #     m_val = graph.ndata["val_mask"].bool()
    #     m_test = graph.ndata["test_mask"].bool()
    #     y_hat = model.inference(graph, x)
    #
    #     val_acc = MF.accuracy(y_hat[m_val], y[m_val], task="multiclass", num_classes=dataset.num_classes)
    #     test_acc = MF.accuracy(y_hat[m_test], y[m_test], task="multiclass", num_classes=dataset.num_classes)
    #     print(f"Val Acc {val_acc.item():.4f} | Test Acc {test_acc.item():.4f}")
    #     Val_acc.append(val_acc.item())
    #     Test_acc.append(test_acc.item())
    # model.train()
    # model = model.to(device)

print(f"Average time: {np.mean(durations):.2f}s, std: {np.std(durations):.2f}s")  ###打印每个epoch训练的平均时间和时间的标准差
print(f"Average node_num: {np.mean(node_num):.2f}")  ###打印平均节点数量   每个sg上的节点数
print(f"Average edge_num: {np.mean(edge_num):.2f}")  ###打印平均边数量   每个sg上的边数
# print("sg.node.num", node_num)
# print("sg.edge.num", edge_num)

###cora_results
with open(f"/23085411005/23085411005/hdrf_hy_pycharm/result/ogb_lmc/zitu_qiuhe/ogb_hdrf_mb_100_3_128_hot.csv", "w",
          newline="") as f:  ###使用 open() 函数创建一个 CSV 文件
    writer = csv.writer(f)  ###创建一个 CSV writer 对象，用于写入数据到 CSV 文件中
    writer.writerow(["aveloss", "avetrain_acc", "val_acc", "test_acc", "epoch_time"])  ###写入 CSV 文件的表头
    for i in range(len(aveLOSS)):  ###遍历损失值列表的长度
        writer.writerow(
            [aveLOSS[i], aveTrain_acc[i], Val_acc[i], Test_acc[i], durations[i]])  ###将损失值、训练集精度、验证集精度和测试集精度写入到 CSV 文件中
    writer.writerow([np.mean(node_num), np.mean(edge_num)])
    writer.writerow(
        ["block_node_edge_num", [np.mean(node_num), np.mean(edge_num)], "canshu", num_partitions, batch_size,
         num_epochs, num_hidden, lr, weight_decay, dropout])
