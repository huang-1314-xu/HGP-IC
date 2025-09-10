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



methods = ["LDG", "Fennel", "Metis", "DBH", "HGP", "ne", "sne"]  
method = methods[4] 
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



def calculate_counts(tensor, a):
    count_0 = 0
    count_0_to_a = 0
    count_a_to_2a = 0
    count_2a_to_3a = 0
    count_3a_to_4a = 0
    count_4a_and_above = 0


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


    length = len(tensor)


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

in_degrees = graph.in_degrees()
out_degrees = graph.out_degrees()

isolated_nodes = torch.where((in_degrees == 0) & (out_degrees == 0))[0]
if isolated_nodes.numel() > 0:
    # graph.remove_nodes(torch.tensor(isolated_nodes))
    graph.remove_nodes(isolated_nodes.clone().detach())
    print("处理后的节点数目：", graph.number_of_nodes())
else:
    print("原图未处理节点数目")
num_edges = graph.num_edges()
num_nodes = graph.num_nodes()


np.random.seed(42)
train_ratio = 0.6  #
val_ratio = 0.2  
test_ratio = 0.2  


train_mask = np.zeros(num_nodes, dtype=bool)
val_mask = np.zeros(num_nodes, dtype=bool)
test_mask = np.zeros(num_nodes, dtype=bool)


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


train_mask = torch.from_numpy(train_mask)
val_mask = torch.from_numpy(val_mask)
test_mask = torch.from_numpy(test_mask)


graph.ndata["train_mask"] = train_mask
graph.ndata["val_mask"] = val_mask
graph.ndata["test_mask"] = test_mask


model = util.Module.SAGE1(graph.ndata["feat"].shape[1], num_hidden, dataset.num_classes).cuda()
# model = compare.util.Module.Graph_Conv(graph.ndata["feat"].shape[1], num_hidden, dataset.num_classes).cuda()
opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

dataloader = dgl.dataloading.DataLoader(  
    graph,
    torch.arange(num_partitions).to("cuda"),
    util.Sampler2.Edge_partition_sampler_id(
        graph,
        num_partitions,  
        cache_path=f'/23085411005/hdrf_hy_pycharm/vertex_partitions/hdrf/coa/76_bian',
        prefetch_ndata=["feat", "label", "train_mask", "val_mask", "test_mask"],
    ),
    device="cuda",
    batch_size=batch_size,
    shuffle=True,
    drop_last=False,
    num_workers=0,
    use_uva=True,
)  

eval_dataloader = dgl.dataloading.DataLoader(  
    graph,
    torch.arange(num_partitions).to("cuda"),
    util.Sampler2.Edge_partition_sampler_id(
        graph,
        num_partitions, 
        cache_path=f'/23085411005/hdrf_hy_pycharm/vertex_partitions/hdrf/coa/76_bian',
        prefetch_ndata=["feat", "label", "train_mask", "val_mask", "test_mask"],
    ),
    device="cuda",
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=0,
    use_uva=True,
)  


durations = []

for e in range(num_epochs):
    print(f"epoch: {e + 1}")

    t0 = time.time()
    model.train()
    i = 0
    Loss = []
    Acc = []
    batch10_avenode_alldegraees = [] 
    batch10_avenode_indegraees = []
    batch10_avenode_outdegraees = []
    batch_node_alldegraees = []
    batch_node_indegraees = []
    batch_node_outdegraees = []

    for it, sg in enumerate(dataloader):  
        i += 1
        graph_obj = sg[0]  
        x = graph_obj.ndata["feat"]  
        y = graph_obj.ndata["label"]  
        m = graph_obj.ndata["train_mask"].bool()  
        y_hat = model(graph_obj, x)

        loss = F.cross_entropy(y_hat[m], y[m]) 

        opt.zero_grad()
        loss.backward()
        opt.step()
        Loss.append(loss.item())
        node_num.append(graph_obj.ndata["feat"].shape[0])  
        edge_num.append(graph_obj.num_edges()) 
       
        acc = MF.accuracy(y_hat[m], y[m], task="multiclass",
                          num_classes=dataset.num_classes) 
        Acc.append(acc.item())
        mem = torch.cuda.max_memory_allocated() / 1000000 


        print(f"Loss {loss.item():.4f} | Acc {acc.item():.4f} | Peak Mem {mem:.2f}MB") 


    aveLOSS.append(f"{sum(Loss) / i:.8f}")  
    aveTrain_acc.append(f"{sum(Acc) / i:.8f}")

    tt = time.time()
    print(f"time: {tt - t0:.2f}s")
    durations.append(tt - t0)


 
    model.eval()
    with torch.no_grad():
        device = next(model.parameters()).device

        y_hat_all = torch.zeros(num_nodes, dataset.num_classes).to(device)
        y_all = torch.zeros(num_nodes, dtype=torch.long).to(device)
        val_mask_all = torch.zeros(num_nodes, dtype=torch.bool).to(device)
        test_mask_all = torch.zeros(num_nodes, dtype=torch.bool).to(device)


        for sg in eval_dataloader:
            graph_obj = sg[0] 
            graph_obj = graph_obj.to(device)
            x = graph_obj.ndata["feat"]
            y = graph_obj.ndata["label"]
            m_val = graph_obj.ndata["val_mask"].bool()
            m_test = graph_obj.ndata["test_mask"].bool()
            sg_node_ids = graph_obj.ndata['_ID']  

   
            y_hat = model.inference(graph_obj, x)
            y_hat = F.softmax(y_hat, dim=1)

            for i, nid in enumerate(sg_node_ids):
                nid = nid.item()
                y_hat_all[nid] += y_hat[i]
                y_all[nid] = y[i]
                val_mask_all[nid] |= m_val[i]
                test_mask_all[nid] |= m_test[i]

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

print(f"Average time: {np.mean(durations):.2f}s, std: {np.std(durations):.2f}s")  
print(f"Average node_num: {np.mean(node_num):.2f}")  
print(f"Average edge_num: {np.mean(edge_num):.2f}") 



with open(f"/23085411005/hdrf_hy_pycharm/result/coa/zitu_qiuhe/coa_hdrf_76_3_128_hot.csv", "w",
          newline="") as f:  
    writer = csv.writer(f) 
    writer.writerow(["aveloss", "avetrain_acc", "val_acc", "test_acc", "epoch_time"])  
    for i in range(len(aveLOSS)):  
        writer.writerow(
            [aveLOSS[i], aveTrain_acc[i], Val_acc[i], Test_acc[i], durations[i]])  
    writer.writerow([np.mean(node_num), np.mean(edge_num), np.mean(durations)])
    writer.writerow(
        ["block_node_edge_num", [np.mean(node_num), np.mean(edge_num)], "canshu", num_partitions, batch_size,
         num_epochs, num_hidden, lr, weight_decay, dropout])

