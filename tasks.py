import os
import numpy as np
import torch
from torch.utils.data import Dataset


class ECDataset(Dataset):

    def __init__(self, size=5, num_samples=1e6, seed=None, tasknum=10):
        # size is num_nodes, num_samples is train_size
        super(ECDataset, self).__init__()

        if seed is None:
            seed = np.random.randint(123456789)

        np.random.seed(seed)
        torch.manual_seed(seed)
        # self.dataset = torch.rand(num_samples, 2+2*tasknum, size)
        ec_server_bandwidth = 200*torch.rand(num_samples, size, 1) + 100
        ec_server_computing = 10*torch.rand(num_samples, size, 1) + 5
        ec_server_task = torch.ones(num_samples, size, 1)
        ec_server = torch.cat([ec_server_bandwidth, ec_server_computing, ec_server_task],dim=2)
        tasks_input = torch.normal(1, 0.3,(tasknum,1))
        tasks_theta = 50*torch.rand(tasknum,1)+10
        tasks = torch.cat([tasks_input, tasks_theta],dim=1)
        tasks = tasks.view([1,2*tasknum])
        tasks = tasks.expand(num_samples,size,2*tasknum)
        # inplace = True!!!!!!!!!!!!!
        ##self.dataset = torch.cat([ec_server, tasks],dim=2)
        self.dataset = tasks
        self.dataset = self.dataset.transpose(1,2)
        # rand为均匀分布，randn为正态分布
        # (a,b,c): a表示生成多少套数据，(b,c)表示b行c列，在这里就是说生成a个实例，每个实例b行c列，b就是静态元素数量，c是结点数量
        # 拼接得到的张量维度为(num_samples, size, 2+2*tasknum)
        # debug出现了问题，转置成(num_samples, 2+2*tasknum, size)
        ## self.dynamic = torch.zeros(num_samples, 1, size)
        self.dynamic = ec_server
        self.dynamic = self.dynamic.transpose(1, 2)
        self.num_nodes = size
        self.size = num_samples


    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # (static, dynamic, start_loc)
        return (self.dataset[idx], self.dynamic[idx], [])


def update_mask(mask, dynamic, chosen_idx):
    """Marks the visited city, so it can't be selected a second time."""
    mask.scatter_(1, chosen_idx.unsqueeze(1), 0)
    return mask

def update_fn(dynamic, ptr):
    ## ptr为200个index
    ## dynamic:[200,3,5]
    ## output: dynamic
    bandwidth = dynamic[:,0,:]*dynamic[:,2,:]
    computing = dynamic[:,1,:]*dynamic[:,2,:]
    update_dynamic = torch.zeros_like(dynamic)
    for i, element in enumerate(dynamic):
        update_dynamic[i] = element
    for index, element in enumerate(ptr):
        update_dynamic[index][2][element] = dynamic[index][2][element]+1
    update_dynamic[:,0,:] = bandwidth / dynamic[:,2,:]
    update_dynamic[:, 1, :] = computing / dynamic[:, 2, :]
    return update_dynamic



def reward(static, dynamic, indices, w1=1, w2=0):

    # unsqueeze: 增加一个维度
    idx = indices.unsqueeze(1)
    idx = idx.expand(dynamic.size(0), dynamic.size(1), idx.size(2))
    sequence = torch.gather(dynamic.data, 2, idx).permute(0, 2, 1)
    # 到这里，sequence就是还原出来的序列，[200,10,22]，包含了batch中200套数据的解序列的22个属性
    tasks = static[:,:,0]
    tasks = tasks.view([dynamic.size(0),idx.size(2),2])
    transition_time = tasks[:,:,0] / sequence[:,:,0]
    process_time = tasks[:,:,1] / sequence[:,:,1]
    delay = transition_time + process_time
    delay_sum = torch.sum(delay, dim=1)
    obj1 = delay_sum

    price_bandwidth = sequence[:,:,0] / torch.max(sequence[:,:,0])
    price_computing = sequence[:,:,1] / torch.max(sequence[:,:,1])
    price = price_bandwidth * 0.4 + price_computing * 0.6
    cost = torch.sum(price * process_time, dim=1)

    obj2 = cost

    obj = w1*obj1 + w2*obj2
    return obj, obj1, obj2
