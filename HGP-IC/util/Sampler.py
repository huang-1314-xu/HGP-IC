from dgl import DGLError, set_node_lazy_features, set_edge_lazy_features
from dgl.dataloading import Sampler
import os
import pickle
import torch
import torch
import dgl
import numpy as np
import torch

class Universal_Sampler(Sampler):##继承自 Sampler 类
    def __init__(self, g, k, cache_path, partition_file_path=None, prefetch_ndata=None,
                 prefetch_edata=None, output_device=None):
        """
        :param g: DGLGraph
        :param k: 划分的区块数
        :param cache_path: 缓存文件路径
        :param partition_file_path: 区块划分文件路径                ###已划分好的结果数据
        :param partitioner: 图划分器
        :param prefetch_ndata: 预取节点数据
        :param prefetch_edata: 预取边数据
        :param output_device: 输出设备
        """
        super().__init__()###调用父类 Sampler 的初始化方法。super() 是用来获取父类的对象，然后调用其方法
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    self.partition_node_ids, self.partition_offset = pickle.load(f)
            except (EOFError, TypeError, ValueError):###如果存在，则尝试从缓存文件中加载 partition_node_ids 和 partition_offset。如果加载失败，会引发异常并提示缓存文件内容无效
                raise DGLError(
                    f'The contents in the cache file {cache_path} is invalid. '
                    f'Please remove the cache file {cache_path} or specify another path.')
        ###    if len(self.partition_offset) != k + 1:###检查 partition_offset 的长度是否等于 k+1，如果不匹配，则引发异常
                ###raise DGLError(
                 ###   f'Number of partitions in the cache does not match the value of k. '
                 ###   f'Please remove the cache file {cache_path} or specify another path.')
        else:###如果缓存文件不存在，则执行下面的代码块    使用提前准备的嵇的划分文件
            partition_offset = []###创建一个空的
            partition_node_ids = []
            partition_offset.append(0)###partition_offset 中添加一个初始值为0的元素
            # 使用已经划分完成的文件进行划分
            node_partition = {}###创建一个空的字典 node_partition ，用于保存读取来的划分结果
            # 读取划分文件
            with open(partition_file_path, 'r') as f:
                i = 0
                while True:
                    line = f.readline()###每一行都是一个整数，表示该索引所属的类别，就是classes
                    if not line:
                        break
                    classes = int(line)###open 函数打开 partition_file_path 文件，并逐行读取其内容。将每行转换为整数，存储在 classes 变量中
                    if classes not in node_partition:
                        node_partition[classes] = []###如果 classes 不在 node_partition 字典中，就在字典中创建一个以 classes 为键的空列表
                    node_partition[classes].append(i)###i代表的节点id，所有行对应着所有节点
                    i += 1
            # 生成划分文件
            for i in range(k):
                partition_node_ids += node_partition[i]###这里的i就是上述的classes，也是k
                partition_offset.append(partition_offset[-1] + len(partition_node_ids))

            partition_offset = torch.tensor(partition_offset)
            partition_node_ids = torch.tensor(partition_node_ids)

            # 保存划分文件
            with open(cache_path, 'wb') as f:
                pickle.dump((partition_node_ids, partition_offset), f)
            self.partition_offset = partition_offset
            self.partition_node_ids = partition_node_ids

        self.prefetch_ndata = prefetch_ndata or []
        self.prefetch_edata = prefetch_edata or []
        self.output_device = output_device


class Node_partition_sampler_wuhuan(Universal_Sampler):
    def __init__(self, g, k, cache_path, partition_file_path=None, prefetch_ndata=None,
                 prefetch_edata=None, output_device=None):
        super(Node_partition_sampler_wuhuan, self).__init__(g, k, cache_path, partition_file_path,
                                                     prefetch_ndata, prefetch_edata, output_device)

    def sample(self, g, partition_ids):
        """
        Parameters
        :param partition_ids: 区块的id列表   ###表示要抽取的区块序号
        :param g: DGLGraph
        :return: g.subgraph
        """
        node_ids = torch.cat([
            self.partition_node_ids[self.partition_offset[i]:self.partition_offset[i + 1]]
            for i in partition_ids], 0)
        sg = g.subgraph(node_ids, relabel_nodes=True, output_device=self.output_device)
        ###sg = dgl.add_self_loop(sg)    #带自环的数据集不用加，不带自环的加
        set_node_lazy_features(sg, self.prefetch_ndata)
        set_edge_lazy_features(sg, self.prefetch_edata)
        return sg

class Node_partition_sampler_youzihuan(Universal_Sampler):
    def __init__(self, g, k, cache_path, partition_file_path=None, prefetch_ndata=None,
                 prefetch_edata=None, output_device=None):
        super(Node_partition_sampler_youzihuan, self).__init__(g, k, cache_path, partition_file_path,
                                                     prefetch_ndata, prefetch_edata, output_device)

    def sample(self, g, partition_ids):
        """
        Parameters
        :param partition_ids: 区块的id列表   ###表示要抽取的区块序号
        :param g: DGLGraph
        :return: g.subgraph
        """
        node_ids = torch.cat([
            self.partition_node_ids[self.partition_offset[i]:self.partition_offset[i + 1]]
            for i in partition_ids], 0)
        sg = g.subgraph(node_ids, relabel_nodes=True, output_device=self.output_device)
        ###sg = dgl.add_self_loop(sg)    带自环的数据集不用加，不带自环的加
        set_node_lazy_features(sg, self.prefetch_ndata)
        set_edge_lazy_features(sg, self.prefetch_edata)
        return sg



def load_block_from_file(filename):
    with open(filename, 'rb') as file:
        block = pickle.load(file)
    return block

class hrdf_Sampler(Sampler):##继承自 Sampler 类
    def __init__(self, g, k, cache_path, partition_file_path=None, prefetch_ndata=None,
                 prefetch_edata=None, output_device=None):
        super().__init__()
        ####self.blocks = None   # 初始化 blocks 变量

        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    self.blocks = pickle.load(f)  # 加载缓存数据到 self.blocks   应该是这一步错了
            except (EOFError, TypeError, ValueError):
                raise DGLError(
                    f'The contents in the cache file {cache_path} is invalid. '
                    f'Please remove the cache file {cache_path} or specify another path.')
        else:
             pass
        self.prefetch_ndata = prefetch_ndata or []
        self.prefetch_edata = prefetch_edata or []
        self.output_device = output_device


class Edge_partition_sampler_new(hrdf_Sampler):###应该是成对的边   ，边划分数据集去掉自环，形成子图，再自环，无区别
    def __init__(self, g, k, cache_path, partition_file_path=None, prefetch_ndata=None,
                 prefetch_edata=None, output_device=None):
        super(Edge_partition_sampler_new, self).__init__(g, k, cache_path, partition_file_path,
                                                     prefetch_ndata, prefetch_edata, output_device)
        
    def sample(self, g, partition_ids):
        merged_set = set()
        ####blocks_list = []
        ####blocks_list = list(self.blocks)
        ##########print("print( partition_ids)",partition_ids)
      
        for i in partition_ids:#之前是正确的。9.5.23.45
             if i <= len(self.blocks) and self.blocks[i] is not None:  # 判断索引是否在范围内
                 merged_set = merged_set.union(self.blocks[i])
             else:
                 raise ValueError("over-long-blocks,kong-block.") # 处理索引越界的情况

        # for i in partition_ids:###qpt给的不一定对
        #     if i < len(self.blocks) and self.blocks[i] is not None:
        #         merged_set = merged_set.union(frozenset(tuple(map(tuple, self.blocks[i]))))
        #     else:
        #         raise ValueError("over-long-blocks,kong-block.")###到这
                
        src = []
        dst = []
        for my_tuple in merged_set:
            v1, v2 = my_tuple            
            src.append(int(v1))
            dst.append(int(v2))
        x, y = src,dst
        if len(x) != len(y):
            raise ValueError("The length of src and dst should be the same.")

        ###src_tensor = torch.tensor(src_ids, dtype=torch.int32).view(-1)
        ###dst_tensor = torch.tensor(dst_ids, dtype=torch.int32).view(-1)
        edge_id_list = []

        for i in range(len(x)):
            edge_id = g.edge_ids(int(x[i]), int(y[i]))
            edge_id_list.append(edge_id)
        
        ###print(edge_id)
        #edge_id = edge_id.tolist()
        sg = dgl.edge_subgraph(g, edge_id_list,relabel_nodes=True, output_device=self.output_device)
        ###########sg = dgl.add_self_loop(sg)    为了应对wiki特别所以去掉了
  
        ####sg= dgl.graph((x, y))  ###kg = dgl.graph((src, dst), num_nodes=n_ent)
        set_node_lazy_features(sg, self.prefetch_ndata)###将预取的节点特征和边特征设置到子图 sg 上
        set_edge_lazy_features(sg, self.prefetch_edata)
        return sg


def split_dataset(graph, seed, train_ratio, val_ratio, test_ratio):
    """将数据集划分为训练集、验证集和测试集，并将结果存储到图的ndata中。

    Parameters
    ----------
    graph : DGLGraph
        图。
    seed : int
        随机数种子。
    train_ratio : float
        训练集占总节点数的比例。
    val_ratio : float
        验证集占总节点数的比例。
    test_ratio : float
        测试集占总节点数的比例。

    Returns
    -------
    None
    """
    num_nodes = graph.num_nodes()
    np.random.seed(seed)

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
    val_indices = node_indices[num_train:num_train+num_val]
    test_indices = node_indices[num_train+num_val:num_train+num_val+num_test]

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
    return graph
