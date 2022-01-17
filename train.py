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


parser = argparse.ArgumentParser(description='Train Neural Network')

parser.add_argument("data_dir", action="store", default="./flowers/", help="data directory")
parser.add_argument("--save_dir", action="store", dest="save_dir", default="./checkpoint.pth", help="checkpoint dir")
parser.add_argument("--arch", action="store", dest="arch", default="vgg16", help="model")
parser.add_argument("--learning_rate", action="store", dest="learning_rate", type=float, default=0.001, help="learning rate")
parser.add_argument("--hidden_units", action="store", dest="hidden_units", type=int, default=1024, help="hidden units")
parser.add_argument("--epochs", action="store", dest="epochs", type=int, default=2, help="epochs")
parser.add_argument("--gpu", action="store_true", dest="gpu", default=False, help="processor")

args = parser.parse_args()

data_dir = args.data_dir
save_dir = args.save_dir
arch = args.arch
learning_rate = args.learning_rate
hidden_units= args.hidden_units
epochs = args.epochs
gpu = args.gpu

print(f"\ndata_dir = {data_dir}"
      f"\nsave_dir = {save_dir}"
      f"\narch = {arch}"
      f"\nlearning_rate = {learning_rate}"
      f"\nhidden_units = {hidden_units}"
      f"\nepochs = {epochs}"
      f"\ngpu = {gpu}")

     
dataset, loader = functions.load_data(data_dir)
model, classifier, criterion, optimizer = functions.build_model(arch, hidden_units, learning_rate, gpu)
model, train_loss, valid_loss, accuracy, time_elapsed = functions.train_model(loader, model, criterion, optimizer, epochs, gpu)
functions.test_model(loader, model, criterion, gpu)
functions.save_checkpoint(save_dir, dataset, model, arch, classifier, optimizer, hidden_units, learning_rate, epochs, train_loss, valid_loss, accuracy, time_elapsed)
