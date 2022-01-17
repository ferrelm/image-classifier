# Imports here
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import json
import numpy as np
import os, random
import time
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import argparse

import functions


parser = argparse.ArgumentParser(description='Use Neural Network')

parser.add_argument("image_dir", action="store", default="./flowers/test/6/image_07181.jpg", help="image dir")
parser.add_argument("checkpoint", action="store", default="./checkpoint.pth", help="checkpoint dir")
parser.add_argument("--top_k", action="store", dest="top_k", type=int, default=5, help="top k classes")
parser.add_argument("--category_names", action="store", dest="category_names", default="cat_to_name.json", help="category names")
parser.add_argument("--gpu", action="store_true", dest="gpu", default=False, help="processor")

args = parser.parse_args()

image_dir = args.image_dir
checkpoint = args.checkpoint
top_k = args.top_k
category_names = args.category_names
gpu = args.gpu

print(f"\nimage_dir = {image_dir}"
      f"\ncheckpoint = {checkpoint}"
      f"\ntop_k = {top_k}"
      f"\ncategory_names = {category_names}"
      f"\ngpu = {gpu}")


with open(category_names, 'r') as f:
    cat_to_name = json.load(f)

model, optimizer = functions.load_checkpoint(checkpoint)
print(model)

functions.sanity_check(model, image_dir, top_k, cat_to_name, gpu)