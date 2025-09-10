###将高热度的顶点分配到合适的分区
import torch
import dgl
from tqdm import tqdm

ps = 1  ###
rho = 1.1  ####调整算法中与负载平衡相关的参数
maxsize = 0
minsize = 0
lamb = 1.01  ####调整不同部分在总得分中的权重
# partdeg = {}
hotness = {}
hotness2 = {}
s = 0




# 定义消息传递和聚合函数
def message_func(edges):
    # 只发送边的权重作为消息
    msg = edges.data['weight']
    return {'msg': msg}  # 返回边的权重

# 聚合函数，按照公式计算热度值
def reduce_func(nodes, graph, pbar):
    # 获取消息 msg，表示边的权重
    msg = nodes.mailbox['msg']  # 获取消息（边的权重）
    msg_sum = torch.zeros(msg.shape[0], device=msg.device)  # 使用 msg 的 device 获取设备信息

    # 对每个节点的入边信息进行遍历
    for i in range(msg.shape[0]):
        # 获取节点 i 的入度节点（前驱节点）
        src_ids = graph.predecessors(nodes.nodes()[i])  # 获取源节点（入边）
        # 初始化当前节点的热度值为 0
        hotness = 0
        # 遍历每个前驱节点并计算贡献
        for src in src_ids:
            # 获取前驱节点的出边
            out_edges = graph.out_edges(src, form='eid')
            # 计算前驱节点出边权重的总和
            out_weights_sum = torch.sum(graph.edata['weight'][out_edges])
            # # 避免除以0的情况，如果没有出边权重，则设为1
            # if out_weights_sum == 0:
            #     out_weights_sum = torch.tensor(1.0, device=msg.device)
            # 按公式计算热度贡献
            edge_weight = graph.edata['weight'][graph.edge_ids(src, nodes.nodes()[i])]
            hotness += edge_weight / out_weights_sum
        # 将热度累加到 msg_sum
        msg_sum[i] = hotness
        # 每处理完一个节点后更新一次进度条
        pbar.update(1)
    return {'hotness': msg_sum}




# 第二个热度值的消息函数
def message_func_2(edges):
    # 将 1 / 度数 作为消息发送
    return {'msg': 1.0 / edges.src['degree']}

# 第二个热度值的归约函数
def reduce_func_2(nodes):
    # 计算邻居节点度数倒数的和
    hotness_v = torch.sum(nodes.mailbox['msg'], dim=1)

    for i in range(hotness_v.shape[0]):
        node_id = nodes.nodes()[i].item()  # 获取节点的 ID

    return {'hotness2': hotness_v}

def estimate_hotness_2(graph):
    global hotness2  # 声明要使用全局的 hotness2 字典

    # 计算每个节点的度数
    graph.ndata['degree'] = graph.out_degrees()


    # 使用 tqdm 显示进度条
    with tqdm(total=graph.number_of_nodes(), desc="Calculating Node Hotness 2", unit="node") as pbar:
        # 使用消息传递机制计算热度2
        graph.update_all(message_func_2, reduce_func_2)

        # 将结果移回CPU
        hotness2_values = graph.ndata['hotness2'].cpu().tolist()

        # 将每个节点的热度值存储在全局 hotness2 字典中
        for i, hotness2_value in enumerate(hotness2_values):
            hotness2[i] = hotness2_value

        pbar.update(graph.number_of_nodes())

    return hotness2


# 计算某个顶点在给定边中的热度比值
def teta(vertex, hedge):
    d = 0.0  ### # 初始化变量 d，用于累加 hedge 中所有顶点的热度
    for v in hedge:
        d += hotness2.get(v, 0)  ####hotness2.get(v, 0) 表示从字典 hotness2 中获取键 v 对应的值。如果字典中存在键 v，则返回其对应的热度值；如果不存在键 v，则返回默认值 0。
    s = hotness2.get(vertex, 0) / d if d > 0 else 0
    return s  ####这个点vertex的热度值大的s值就大




def g(vertex, hedge, vpp, i):  ###vpp就是bocks[set(),  ..... k], i就是k。求的是vertex（边上一个顶点）在i的分区的奖励值
    if vertex in vpp[i]:
        return (1 + 1 - teta(vertex, hedge))
    return 0



def CgreedyBal(part):  ###part表示某个分区，求的是当前分区的惩罚值
    n = maxsize - len(part)
    d = eps + maxsize - minsize  ###_eps=1.1
    return (n / d)  ###通过剩余容量和分区不平衡来衡量


def CHDRFeREP(hedge, vpp, i):  ###求的是该边（边上两个观点）在i的分区上的奖励值
    s = 0.0
    for vertex in hedge:
        s += g(vertex, hedge, vpp, i)
    # print(s)
    return s


def CHDRFe(hedge, part, vpp, i):  ###求的是该边在i的分区上的最终得分
    part1 = CHDRFeREP(hedge, vpp, i)
    part2 = CgreedyBal(part)
    # print(" " + str(part1) + " " +str(part2))
    return part1 + lamb * part2  ####_lamb=5


# def init(vertexs,numberOfPartitions,stream):##初始化所有顶点的局部度，vertexs存的是所有顶点，应该是存的节点id，以id为健
#     for vertex in vertexs:
#         partdeg[vertex] = 0.0
def init(vertexs, numberOfPartitions, stream):  ####对 vertexs 中的每个顶点 vertex，将其热度初始化为 0.0。
    for vertex in vertexs:
        hotness[vertex] = 0.0
    # print("初始化")
    # print(hotness)


def chooseEdge(hedge, partitions, vpp, graph, _lamb=1.01, _eps=1, _rho=1.1):
    # return a the number of the partition where edge should go
    global lamb
    global eps
    global rho
    global maxsize
    global minsize
    lamb = _lamb
    eps = _eps
    rho = _rho
    buff = -1
    scoreBuff = -1

    # updatepartdeg(hedge, graph, hotness)###先更新要分配的边上两节点的局部的度，partitions = [[]for i in k]，应该存的是边
    for i, part in enumerate(partitions):
        scoreloc = CHDRFe(hedge, part, vpp, i)
        if scoreloc > scoreBuff:
            buff = i
            scoreBuff = scoreloc
        if len(vpp[i]) > maxsize:  ##因为每次只划分一条边，所以只用加一
            maxsize += 1
    for part in vpp:  ###vpp存的应该也是以边存的，遍历每个分区set的长度（边的个数）
        minsize = min(minsize, len(part))
    return buff
###将高热度值的边分配到合适的分区
