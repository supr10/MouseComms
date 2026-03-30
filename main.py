import torch
import mousefuncs
from model import MouseCommsNet
import debug_functions as df
import keyboard

debug = True

answers = ["no movement", "horizontal movement", "vertical movement", "waving", "circular movement"]

df.debug_message("loading model...", debug)
model = MouseCommsNet()

model.load_state_dict(torch.load('./model/agent-v0.2.pth'))

df.debug_message("done loading model", debug)
df.debug_message("ready", debug)

while not keyboard.is_pressed("esc"):
    if keyboard.read_key()=="=":
        df.debug_message("recording...", debug)
        a=mousefuncs.record(epochs = 40, delay = 0.05, norm=0.1)
        print(answers[model(a).argmax(dim=0)])