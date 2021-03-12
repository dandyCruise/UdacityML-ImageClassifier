import argparse
import torch
import torchvision
import json
import torch.nn.functional as nFunc
import matplotlib.pyplot as plt
import numpy as np 
from torch import nn, optim
from PIL import Image
from torchvision import datasets, transforms, models
from collections import OrderedDict
from matplotlib.ticker import FormatStrFormatter
from utility import dataLoader, modelSetup, trainModel, saveModel

parse = argparse.ArgumentParser(description='Train Network') 

parse.add_argument('dataDir'  ,action='store'   ,default ="./flowers/" ,help='Please enter the path to training data')
parse.add_argument('--saveDir',action='store' ,dest ='saveDir' ,default='/home/workspace/ImageClassifier/checkpoint.pth')
parse.add_argument('--epochs' ,action='store' ,dest ='epochs' ,type=int ,default=1)
parse.add_argument('--lRate'  ,action='store' ,dest ='lRate' ,type=float ,default=0.01)
parse.add_argument('--gpu'    ,action='store' ,dest ='gpu' ,default='cpu')
parse.add_argument('--arch'   ,action='store' ,dest ='arch' ,default='vgg16')
parse.add_argument('--hLayers',action='store' ,dest ='hLayers' ,type=int ,default=4096)

inputArg = parse.parse_args()

dataDir      = inputArg.dataDir
saveDir      = inputArg.saveDir
lRate        = inputArg.lRate
hLayers      = inputArg.hLayers
epochs       = inputArg.epochs 
device       = inputArg.gpu
trainedModel = inputArg.arch

model        = getattr(models,trainedModel)(pretrained=True)
inputUnits   = model.classifier[0].in_features
criterion    = nn.NLLLoss() 
optimizer    = optim.Adam(model.classifier.parameters(),lRate)

trainLoader, testLoader, valLoader,trainData = dataLoader(dataDir) 
modelSetup(model,hLayers,lRate,inputUnits,trainData)

model = trainModel(model,criterion,optimizer,epochs,device,trainLoader,valLoader)
saveModel(saveDir,trainedModel,model,trainData,optimizer,epochs) 

