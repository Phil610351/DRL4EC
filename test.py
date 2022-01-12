import torch
import tasks
from tasks import ECDataset, reward
from torch.utils.data import DataLoader
from model import DRL4EC
from main import StateCritic
import numpy as np
import os
import matplotlib.pyplot as plt
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# save_dir = "../ec_no_transfer/10"
# save_dir = "../ec_transfer-10ec-20task/10"
save_dir = "./ec_transfer-10ec-40task/10"
# param
update_fn = tasks.update_fn
STATIC_SIZE = 40  # (x, y)
DYNAMIC_SIZE = 3

# claim model
actor = DRL4EC(STATIC_SIZE,
               DYNAMIC_SIZE,
               128,
               20,
               update_fn,
               None,
               1,
               0.1).to(device)
critic = StateCritic(STATIC_SIZE, DYNAMIC_SIZE, 128).to(device)

Test_data = ECDataset(15, 500, 2600, 20)
Test_loader = DataLoader(Test_data, 100, False, num_workers=0)

# load 50 models
N=50
w = np.arange(N+1)/N
objs = np.zeros((N+1,2))
start  = time.time()
t1_all = 0
t2_all = 0
tours=[]

for i in range(0, N+1):
    t1 = time.time()
    ac = os.path.join(save_dir, "w_%2.2f_%2.2f" % (1-w[i], w[i]),"actor.pt")
    cri = os.path.join(save_dir, "w_%2.2f_%2.2f" % (1-w[i], w[i]),"critic.pt")
    actor.load_state_dict(torch.load(ac, device))
    critic.load_state_dict(torch.load(cri, device))
    t1_all = t1_all + time.time()-t1
    # calculate
    obj1s = []
    obj2s = []
    for batch_idx, batch in enumerate(Test_loader):
        static, dynamic, x0 = batch

        static = static.to(device)
        dynamic = dynamic.to(device)
        x0 = x0.to(device) if len(x0) > 0 else None

        with torch.no_grad():
            ec_server_indices, _ = actor(static, dynamic, x0)

        _, obj1, obj2 = reward(static, dynamic, ec_server_indices, 1-w[i], w[i])
        tours.append(ec_server_indices.cpu().numpy())
        obj1s.append(torch.mean(obj1.detach()).item())
        obj2s.append(torch.mean(obj2.detach()).item())

    objs[i,:] = [np.mean(obj1s), np.mean(obj2s)]

print("time_load_model:%2.4f"%t1_all)
print("time_predict_model:%2.4f"%t2_all)
print(time.time()-start)

plt.figure()
plt.plot(objs[:,0],objs[:,1],"ro")
plt.show()
print(objs)


