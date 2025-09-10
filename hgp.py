import simu
import score
#import hashing,hdrfE,rr,rand,greedy
import hdrfE,rr,rand,greedy
import random
import sys

import hdrfEplus
import dgl
# from dgl.data import CoauthorCSDataset
from ogb.nodeproppred import DglNodePropPredDataset
import pickle 
import time
import torch

def openData(path):
    s = []
    with open("data/"+path,'r') as f:
        for line in f:
            hedge = eval(line)##每一行是一条边
            s.append(hedge)###边存到列表里
    return(s)


def vertexinset(file):
    v = set()
    with open("data/" + file,'r') as f:
        for line in f:
            hedge = eval(line)
            for vertex in hedge:###遍历每条边
                if vertex not in v:##把不重复的每个顶点存在set（）的集合里
                    v.add(vertex)
    return v

def output(s):
    with open("data.out",'a') as f:###存结果
        for i in s:
            f.write(str(i) + " ")
        f.write("\n")

def main(algo = hdrfE.chooseEdge,nbb=2):
    t0 = time.time()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    dataset = dgl.data.CoraGraphDataset(raw_dir='/23085411005/hdrf_hy_pycharm/data/CoraGraphDataset')
    graph = dataset[0].to(device) 
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
    # 确保图有边权重
    if 'weight' not in graph.edata:
        graph.edata['weight'] = torch.ones(graph.num_edges(), device=device)
    # print(2)
    src_ids, dst_ids = graph.edges()

    result = [(int(src), int(dst)) for src, dst in zip(src_ids, dst_ids)]
    stream =result
    # print(result)
    src_set = set()
    # print("测试点2")
    # item = 0
    for item in src_ids:
        # item += 1
        src_set.add(int(item))
        # print(item)
    vertexs=src_set

    #Set up
    random.seed(2)###应该是随机打乱边，保持每次接近
    # init = hdrfE.init###初始化所有顶点的热度值
    scoring = score.replicationFactor###下面直接引用了，多余了
    numberOfPartitions = nbb##设置分区数

    # print(1111)
    vpp = [set() for i in range(numberOfPartitions)] #Vertex per partitions   #####存的是每个分区的顶点

    random.shuffle(stream)##打乱顺序边
    ##vertexs = vertexinset(testFile)###文件所有顶点集合
    print("Vertex got. Total of: " + str(len(vertexs)) + "vertexs")###总共多少个点
    hdrfE.init(vertexs,numberOfPartitions,stream)###初始化所有顶点的热度值
    hotness = hdrfE.estimate_hotness_2(graph)

    res = simu.run(algo,numberOfPartitions,stream,vpp, graph)##algo是划分算法，分区数，数据流，分区节点，返回的是partitions = [[]for i in k]
    print("Simulation done, computing score:")
    rf = score.replicationFactor(res,vertexs,vpp)##复制因子
    print("Replication factor is: " + str(rf))
    # print(11111)
    tt = time.time()
    print(f"time: {tt - t0:.2f}s")
    return rf


s3=main( algo=hdrfE.chooseEdge, nbb=2)###nbb分区数nvi s
print(s3)

