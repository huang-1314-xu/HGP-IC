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


methods = ["LDG", "Fennel", "Metis", "DBH", "HGP", "ne", "sne"] 
method = methods[4] 
num_partitions = 100
num_epochs = 150
batch_size = 1
num_hidden = 128
lr = 0.001
weight_decay = 5e-4
dropout = 0.5



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



dataset = dgl.data.AsNodePredDataset(DglNodePropPredDataset("ogbn-products", root="/23085411005/23085411005/hdrf_hy_pycharm/data/ogb-products"))
graph = dataset[0]
print("原图节点数:",graph.number_of_nodes())

in_degrees = graph.in_degrees()
out_degrees = graph.out_degrees()

isolated_nodes = torch.where((in_degrees == 0) & (out_degrees == 0))[0]
if isolated_nodes.numel() > 0:
    # graph.remove_nodes(torch.tensor(isolated_nodes))
    graph.remove_nodes(isolated_nodes.clone().detach())
    print("处理后的节点数目：",graph.number_of_nodes())
else:
    print("原图未处理节点数目")

num_edges = graph.num_edges()
num_nodes = graph.num_nodes()


num_edges = graph.num_edges()
num_nodes = graph.num_nodes()



np.random.seed(42)
train_ratio = 0.6 
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

model = util.Module4.SAGE1(graph.ndata["feat"].shape[1], num_hidden, dataset.num_classes).cuda()

opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

dataloader = dgl.dataloading.DataLoader( 
    graph,
    torch.arange(num_partitions).to("cuda"),
    util.Sampler2.Edge_partition_sampler_id(
        graph,
        num_partitions,  
        cache_path=f'/23085411005/23085411005/hdrf_hy_pycharm/vertex_partitions/hdrf/ogb-products/100_bian',
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
        cache_path=f'/23085411005/23085411005/hdrf_hy_pycharm/vertex_partitions/hdrf/ogb-products/100_bian',
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

num_nodes = graph.number_of_nodes() 
node_ids = torch.arange(num_nodes, device=device) 


graph_hidden1 = copy.deepcopy(graph).to(device)
graph_hidden1.ndata['feat'] = torch.zeros(num_nodes, num_hidden, device=device)
graph_hidden1.ndata['_ID'] = node_ids


graph_hidden2 = copy.deepcopy(graph).to(device)
graph_hidden2.ndata['feat'] = torch.zeros(num_nodes, num_hidden, device=device)
graph_hidden2.ndata['_ID'] = node_ids


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
        sg_nodes = sg[0].ndata['_ID']
        successors = graph.out_edges(sg_nodes)[1]
        predecessors = graph.in_edges(sg_nodes)[0]
        neighbors = torch.cat([successors, predecessors]).unique()
        expanded_nodes = torch.cat([sg_nodes, neighbors]).unique()
        expanded_subgraph = dgl.node_subgraph(graph, expanded_nodes)

        neighbors_set = set(neighbors.tolist())
        sub_nodes_set = set(sg_nodes.tolist())
        new_neighbors_set = neighbors_set - sub_nodes_set
        new_neighbors = torch.tensor(list(new_neighbors_set), device=neighbors.device)

        # graph_obj = sg[0]  
        graph_obj = expanded_subgraph 

        # print(graph_obj)
        x = graph_obj.ndata["feat"]  
        y = graph_obj.ndata["label"] 
        m = graph_obj.ndata["train_mask"].bool().to(device) 
        sg_node_ids = graph_obj.ndata['_ID']

        mask = torch.isin(sg_node_ids, new_neighbors).to(device)
        m = m & ~mask


       
        y_hat = model(sg[0], graph_obj, graph_hidden1, graph_hidden2)


        loss = F.cross_entropy(y_hat[m], y[m])


        opt.zero_grad()


        loss.backward()


        opt.step()


        Loss.append(loss.item())

        node_num.append(graph_obj.ndata["feat"].shape[0]) 
        edge_num.append(graph_obj.num_edges()) 

      
        tensor1 = graph_obj.in_degrees()  
        inaverage = tensor1.float().mean()
        batch_node_indegraees.append(tensor1.tolist()) 
        inaverage = round(inaverage.item(), 2)  
        batch10_avenode_indegraees.append(inaverage) 


        tensor2 = graph_obj.out_degrees()  
        outaverage = tensor2.float().mean()
        batch_node_outdegraees.append(tensor2.tolist()) 
        outaverage = round(outaverage.item(), 2)  
        batch10_avenode_outdegraees.append(outaverage)  

     
        result = torch.add(tensor1, tensor2)  
        allaverage = result.float().mean()
        batch_node_alldegraees.append(result.tolist()) 
        allaverage = round(allaverage.item(), 2)  
        batch10_avenode_alldegraees.append(allaverage)  


        acc = MF.accuracy(y_hat[m], y[m], task="multiclass", num_classes=dataset.num_classes)
        Acc.append(acc.item()) 


        mem = torch.cuda.max_memory_allocated() / 1000000


        print(f"Loss {loss.item():.4f} | Acc {acc.item():.4f} | Peak Mem {mem:.2f}MB")


    aveLOSS.append(f"{sum(Loss) / i:.8f}")  后一个loss
    aveTrain_acc.append(f"{sum(Acc) / i:.8f}")


    tt = time.time()
    print(f"time: {tt - t0:.2f}s")
    durations.append(tt - t0)



    model.eval()
    with torch.no_grad():
        device = next(model.parameters()).device

     
        y_hat_tensor = torch.zeros(num_nodes, dataset.num_classes, device=device)
        val_false_tensor = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        test_false_tensor = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        y_tensor = torch.empty(num_nodes, dtype=torch.long, device=device)

        for it, sg in enumerate(eval_dataloader):
    
            sg_nodes = sg[0].ndata['_ID'].to(device)

  上
            successors = graph.out_edges(sg_nodes)[1].to(device)
            predecessors = graph.in_edges(sg_nodes)[0].to(device)
            neighbors = torch.cat([successors, predecessors]).unique()

        
            expanded_nodes = torch.cat([sg_nodes, neighbors]).unique()
            expanded_subgraph = dgl.node_subgraph(graph, expanded_nodes)

  
            new_neighbors_mask = ~torch.isin(neighbors, sg_nodes)
            new_neighbors = neighbors[new_neighbors_mask]

  
            sg_node_ids = expanded_subgraph.ndata['_ID'].to(device)
            y = expanded_subgraph.ndata["label"].to(device)
            m_val = expanded_subgraph.ndata["val_mask"].bool().to(device)
            m_test = expanded_subgraph.ndata["test_mask"].bool().to(device)

            mask = torch.isin(sg_node_ids, new_neighbors).to(device)
            m_val = m_val & ~mask
            m_test = m_test & ~mask

  
            y_hat = model.inference1(sg[0], expanded_subgraph, graph_hidden1, graph_hidden2)
            y_hat = F.softmax(y_hat, dim=1)

  
            valid_indices = ~mask
            valid_node_ids = sg_node_ids[valid_indices]

      
            y_hat_tensor.index_add_(0, valid_node_ids, y_hat[valid_indices])
            val_false_tensor[valid_node_ids] = m_val[valid_indices]
            test_false_tensor[valid_node_ids] = m_test[valid_indices]
            y_tensor[valid_node_ids] = y[valid_indices]


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
   

print(f"Average time: {np.mean(durations):.2f}s, std: {np.std(durations):.2f}s")  
print(f"Average node_num: {np.mean(node_num):.2f}")  
print(f"Average edge_num: {np.mean(edge_num):.2f}")  



with open(f"/23085411005/23085411005/hdrf_hy_pycharm/result/ogb_lmc/zitu_qiuhe/ogb_hdrf_mb_100_3_128_hot.csv", "w",
          newline="") as f:  
    writer = csv.writer(f) 
    writer.writerow(["aveloss", "avetrain_acc", "val_acc", "test_acc", "epoch_time"]) 
    for i in range(len(aveLOSS)):  
        writer.writerow(
            [aveLOSS[i], aveTrain_acc[i], Val_acc[i], Test_acc[i], durations[i]]) 
    writer.writerow([np.mean(node_num), np.mean(edge_num)])
    writer.writerow(
        ["block_node_edge_num", [np.mean(node_num), np.mean(edge_num)], "canshu", num_partitions, batch_size,
         num_epochs, num_hidden, lr, weight_decay, dropout])
