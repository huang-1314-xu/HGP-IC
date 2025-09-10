import numpy as np
import torch
import torch.nn.functional as F
import torchmetrics.functional as MF
import dgl
import util.Sampler
import util.Sampler2
import util.Module
import pickle
from sklearn.metrics import f1_score
import csv
from ogb.nodeproppred import DglNodePropPredDataset
import random
import time

##############################################     实验
# 参数
methods = ["LDG", "Fennel", "Metis", "DBH", "HDRF", "ne", "sne"]  ##点划分
method = methods[4]  ##"DBH", "HDRF","reverse_HDRF"
num_partitions = 76
num_epochs = 130
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


dataset = dgl.data.CoauthorCSDataset(raw_dir='/23085411005/hdrf_hy_pycharm/data/CoauthorCSDataset')
graph = dataset[0]
print("原图节点数:", graph.number_of_nodes())
##节点的出度
in_degrees = graph.in_degrees()
out_degrees = graph.out_degrees()
# 找出入度和出度都为0的节点（即孤立节点）
isolated_nodes = torch.where((in_degrees == 0) & (out_degrees == 0))[0]
if isolated_nodes.numel() > 0:
    # graph.remove_nodes(torch.tensor(isolated_nodes))
    graph.remove_nodes(isolated_nodes.clone().detach())
    print("处理后的节点数目：", graph.number_of_nodes())
else:
    print("原图未处理节点数目")
num_edges = graph.num_edges()
num_nodes = graph.num_nodes()
# print("节点数", num_nodes)
# print("边数", num_edges)

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

###输入维度                   隐藏层           输出维度，也就是节点的类别
model = util.Module.SAGE1(graph.ndata["feat"].shape[1], num_hidden, dataset.num_classes).cuda()
# model = compare.util.Module.Graph_Conv(graph.ndata["feat"].shape[1], num_hidden, dataset.num_classes).cuda()
opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

dataloader = dgl.dataloading.DataLoader(  #####边划分
    graph,
    torch.arange(num_partitions).to("cuda"),
    util.Sampler2.Edge_partition_sampler_id(
        graph,
        num_partitions,  ###huafen_blosk
        cache_path=f'/23085411005/hdrf_hy_pycharm/vertex_partitions/hdrf/coa/76_bian',
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
        num_partitions,  ###huafen_blosk
        cache_path=f'/23085411005/hdrf_hy_pycharm/vertex_partitions/hdrf/coa/76_bian',
        prefetch_ndata=["feat", "label", "train_mask", "val_mask", "test_mask"],
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
        graph_obj = sg[0]  # 提取DGL图对象
        x = graph_obj.ndata["feat"]  # 提取节点特征
        y = graph_obj.ndata["label"]  # 提取节点标签
        m = graph_obj.ndata["train_mask"].bool()  # 将训练掩码转换为布尔类型  ###.bool(): 将提取到的节点特征转换为布尔类型。这意味着只有为 True 的节点才会被选中，而 False 的节点将被忽略掉
        y_hat = model(graph_obj, x)
        ###loss = torch.nn.MultiLabelMarginLoss()(y_hat[m], y[m])  ###mulabel
        loss = F.cross_entropy(y_hat[m], y[m])  ###muclass

        opt.zero_grad()
        loss.backward()
        opt.step()
        Loss.append(loss.item())
        node_num.append(graph_obj.ndata["feat"].shape[0])  ###将sg上所有的节点数量添加到名为 node_num 的列表中
        edge_num.append(graph_obj.num_edges())  ###批次内的节点数，边数
        # ##if it % 10 == 0:
        # tensor1 = sg.in_degrees()  ###入度
        # inaverage = tensor1.float().mean()
        # # print("入度", calculate_counts(tensor1, int(inaverage)))
        # batch_node_indegraees.append(tensor1.tolist())
        # inaverage = round(inaverage.item(), 2)
        # batch10_avenode_indegraees.append(inaverage)
        #
        # tensor2 = sg.out_degrees()  ###出度
        # outaverage = tensor2.float().mean()
        # # print("出度", calculate_counts(tensor2, int(outaverage)))
        # batch_node_outdegraees.append(tensor2.tolist())
        # outaverage = round(outaverage.item(), 2)
        # batch10_avenode_outdegraees.append(outaverage)
        #
        # result = torch.add(tensor1, tensor2)  ###度
        # allaverage = result.float().mean()
        # # print("度", calculate_counts(result, int(allaverage)))
        # batch_node_alldegraees.append(result.tolist())
        # allaverage = round(allaverage.item(), 2)
        # batch10_avenode_alldegraees.append(allaverage)
        # if it % 20 == 0:
        # print(sg.ndata["feat"].shape[0])###节点数量
        acc = MF.accuracy(y_hat[m], y[m], task="multiclass",
                          num_classes=dataset.num_classes)  ###task="multiclass" 表示多类分类任务，num_classes=dataset.num_classes 表示类别的数量
        ##acc = MF.accuracy(y_hat[m], y[m], task="multilabel", num_labels=dataset.num_classes)
        Acc.append(acc.item())
        mem = torch.cuda.max_memory_allocated() / 1000000  ###获取 CUDA 设备当前的最大内存使用量，并将其转换为以 MB 为单位的值
        # Acc为训练集的精度

        print(f"Loss {loss.item():.4f} | Acc {acc.item():.4f} | Peak Mem {mem:.2f}MB")  ###打印训练损失值、训练集精度和最大内存使用量

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
    model.eval()
    with torch.no_grad():
        device = next(model.parameters()).device
        # 初始化tensor来存储累积的预测结果和标签
        y_hat_all = torch.zeros(num_nodes, dataset.num_classes).to(device)
        y_all = torch.zeros(num_nodes, dtype=torch.long).to(device)
        val_mask_all = torch.zeros(num_nodes, dtype=torch.bool).to(device)
        test_mask_all = torch.zeros(num_nodes, dtype=torch.bool).to(device)

        # 在每个子图上进行预测
        for sg in eval_dataloader:
            graph_obj = sg[0]  # 提取DGL图对象
            graph_obj = graph_obj.to(device)
            x = graph_obj.ndata["feat"]
            y = graph_obj.ndata["label"]
            m_val = graph_obj.ndata["val_mask"].bool()
            m_test = graph_obj.ndata["test_mask"].bool()
            sg_node_ids = graph_obj.ndata['_ID']  # 获取原图中的节点ID

            # 获取子图的预测结果
            y_hat = model.inference(graph_obj, x)
            y_hat = F.softmax(y_hat, dim=1)

            # 累积预测结果和标签
            for i, nid in enumerate(sg_node_ids):
                nid = nid.item()
                y_hat_all[nid] += y_hat[i]
                y_all[nid] = y[i]
                val_mask_all[nid] |= m_val[i]
                test_mask_all[nid] |= m_test[i]

        # 计算验证集和测试集的准确率
        val_acc = MF.accuracy(y_hat_all[val_mask_all],
                              y_all[val_mask_all],
                              task="multiclass",
                              num_classes=dataset.num_classes)

        test_acc = MF.accuracy(y_hat_all[test_mask_all],
                               y_all[test_mask_all],
                               task="multiclass",
                               num_classes=dataset.num_classes)

        print(f"Val Acc {val_acc.item():.4f} | Test Acc {test_acc.item():.4f}")
        Val_acc.append(val_acc.item())
        Test_acc.append(test_acc.item())

    model.train()
    model = model.to(device)

print(f"Average time: {np.mean(durations):.2f}s, std: {np.std(durations):.2f}s")  ###打印每个epoch训练的平均时间和时间的标准差
print(f"Average node_num: {np.mean(node_num):.2f}")  ###打印平均节点数量   每个sg上的节点数
print(f"Average edge_num: {np.mean(edge_num):.2f}")  ###打印平均边数量   每个sg上的边数
# print("sg.node.num", node_num)
# print("sg.edge.num", edge_num)

###cora_results
with open(f"/23085411005/hdrf_hy_pycharm/result/coa/zitu_qiuhe/coa_hdrf_76_3_128_hot.csv", "w",
          newline="") as f:  ###使用 open() 函数创建一个 CSV 文件
    writer = csv.writer(f)  ###创建一个 CSV writer 对象，用于写入数据到 CSV 文件中
    writer.writerow(["aveloss", "avetrain_acc", "val_acc", "test_acc", "epoch_time"])  ###写入 CSV 文件的表头
    for i in range(len(aveLOSS)):  ###遍历损失值列表的长度
        writer.writerow(
            [aveLOSS[i], aveTrain_acc[i], Val_acc[i], Test_acc[i], durations[i]])  ###将损失值、训练集精度、验证集精度和测试集精度写入到 CSV 文件中
    writer.writerow([np.mean(node_num), np.mean(edge_num), np.mean(durations)])
    writer.writerow(
        ["block_node_edge_num", [np.mean(node_num), np.mean(edge_num)], "canshu", num_partitions, batch_size,
         num_epochs, num_hidden, lr, weight_decay, dropout])
