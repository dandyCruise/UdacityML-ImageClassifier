import argparse
import torch
import torchvision
import json
import torch.nn.functional as nFunc
import matplotlib.pyplot as plt
import numpy as np 

from torch import nn,optim
from PIL import Image
from torchvision import datasets, transforms, models
from collections import OrderedDict
from matplotlib.ticker import FormatStrFormatter
from utility import loadModel, predict

parse = argparse.ArgumentParser(description='Predict Image')

#check that the folder exists for this input name 
parse.add_argument('--image_loc'    ,action ='store',default = "flowers/train/1/image_06734.jpg")
parse.add_argument('--checkpoint'   ,action='store',dest='check_loc',default = 'checkpoint.pth')
parse.add_argument('--top_k'        ,action='store',dest ="top_k",type=int,default=1) 
parse.add_argument('--category_name',action='store',dest='category_name',default ='cat_to_name.json') 
parse.add_argument('--gpu'          ,action='store',dest='gpu', default='cuda',help='cpu used by default, to use GPU please specify --gpu "cuda"')

inputArg = parse.parse_args()

imagePath       = inputArg.image_loc
checkpoint      = inputArg.check_loc 
topK            = inputArg.top_k
device          = inputArg.gpu
categoryName    = inputArg.category_name

with open(categoryName, 'r') as f:
    cat_to_name = json.load(f)

model = loadModel(checkpoint,device)
probs, flowerList = predict(imagePath,model,topK,device)

flowerNames = [] 
for flower in flowerList:
    flowerNames.append(cat_to_name[flower])
for i in range(0,len(flowerNames)):
    print(flowerNames[i])
    print(probs[i])