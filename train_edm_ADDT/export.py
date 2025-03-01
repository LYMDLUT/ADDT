import argparse
import torch
import os
import dnnlib
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, default=-1)
FLAGS = parser.parse_args()

input = FLAGS.input
print(input)
output = './'+os.path.splitext(os.path.basename(input))[0]+'.pth'
with dnnlib.util.open_url(input) as f:
    net = pickle.load(f)['ema'].to('cpu')
b = net.state_dict()
torch.save(b,output)