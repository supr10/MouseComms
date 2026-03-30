import torch
import pyautogui as pag
import time

def get_mouse_velocity(delay:float = 0.1, device:str="cpu", norm:float = 5.0) -> torch.Tensor:
    mx, my = pag.position()
    t1 = time.time()

    time.sleep(delay)

    nmx, nmy = pag.position()
    t2 = time.time()
    dt=t2-t1
    assert norm!=0.0, "norm parameter must not be 0.0"
    return torch.Tensor([(nmx - mx)/norm*dt, (nmy - my)/norm*dt], device=device)


def record(epochs:int=20, delay:float = 0.01, norm:float = 5.0, device:str="cpu") -> torch.Tensor:
    recording = torch.tensor([], device=device)
    for epoch in range(epochs):
        recording = torch.cat((recording, get_mouse_velocity(delay=delay, device=device, norm=norm)), dim=0)
    return recording


def emulate_mouse(tensor:torch.Tensor, delay:float = 0.1):
    for j in range(0, tensor.size(0), 2):
        x, y = pag.position()
        pag.moveTo(x+tensor[j], y+tensor[j+1], duration=delay)



# recording for training

if __name__== "__main__":
    for i in range(3, 0, -1):
        print(f"beginning in {i} seconds...")
        time.sleep(1)
    print("recording...")
    a=record(epochs = 40, delay = 0.05, norm=0.1)
    torch.save(a, "./tensors/nul9")


#testing different settings
"""if __name__== "__main__":
    for i in range(3, 0, -1):
        print(f"beginning in {i} seconds...")
        time.sleep(1)
    print("recording...")
    a=record(epochs=40, delay = 0.05, norm=0.1)
    print(a)x   
    for i in range(3, 0, -1):
        print(f"beginning in {i} seconds...")
        time.sleep(1)
    emulate_mouse(a, delay=0.05)"""