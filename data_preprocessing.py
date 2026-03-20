import torch
"""This module is for loading the velocity vectors in a proper way so the IA can learn"""


def load_tensor_lib(name:str, max_index:int, ind_one:int) -> tuple:
    """
    shape of returned value:
    t: [ [ [x], [y] ], [ [ [x], [y] ]
    i_o:
    """
    assert max_index >= 3, "there must be at least 3 elements for the training to work"
    t = torch.load("./tensors/{}{}".format(name, 0))            #loads the first tensor
    i_o = torch.Tensor([1 if ind_one==0 else 0, 1 if ind_one==1 else 0, 1 if ind_one==2 else 0, 1 if ind_one==3 else 0, 1 if ind_one==4 else 0])#loads second tensor

    t = torch.stack((t, torch.load("./tensors/{}{}".format(name, 1))), dim=0)           #initial stacking
    i_o = torch.stack((i_o, torch.Tensor(
        [1 if ind_one == 0 else 0, 1 if ind_one == 1 else 0, 1 if ind_one == 2 else 0, 1 if ind_one == 3 else 0,
         1 if ind_one == 4 else 0])))

    for i in range(2, max_index):
        t = torch.cat((t, torch.load("./tensors/{}{}".format(name, i)).unsqueeze(0)), dim=0)
        i_o = torch.cat((i_o, torch.Tensor([1 if ind_one==0 else 0, 1 if ind_one==1 else 0, 1 if ind_one==2 else 0, 1 if ind_one==3 else 0, 1 if ind_one==4 else 0]).unsqueeze(0)), dim=0)

    return t, i_o

def merge_tensors(xp:tuple, yp:tuple) -> tuple:
    a = torch.Tensor()
    for i in xp:
        a = torch.cat((a, i), dim=0)
    b = torch.Tensor()
    for i in yp:
        b = torch.cat((b, i), dim=-2)
    return a, b


