import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn
from torch.nn.modules import loss
import tqdm

import model
from data_preprocessing import load_tensor_lib, merge_tensors

agent = model.MouseCommsNet()
optimizer = optim.Adam(agent.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()

#data
x0, y0 = load_tensor_lib("nul", 10, 0)
x1, y1 = load_tensor_lib("hoz", 10, 1)
x2, y2 = load_tensor_lib("ver", 10, 2)
x3, y3 = load_tensor_lib("wav", 10, 3)
x4, y4 = load_tensor_lib("cir", 10, 4)

X, Y = merge_tensors((x0, x1, x2, x3, x4), (y0, y1, y2, y3, y4))


training_steps = 10000
agent.train()
for i in tqdm.tqdm(range(training_steps)):
    optimizer.zero_grad()
    output = agent(X)
    loss = criterion(output, Y)
    loss.backward()
    optimizer.step()
print(loss.item())

torch.save(agent.state_dict(), './model/agent-v0.2.pth')


