
import torch
import torchvision
import json
import torch.nn.functional as nFunc
import matplotlib.pyplot as plt
import numpy as np 
from torch import nn
from torch import optim
from PIL import Image
from torchvision import datasets, transforms, models
from collections import OrderedDict
from matplotlib.ticker import FormatStrFormatter


def dataLoader(data_dir): 
   
    trainDir = data_dir + '/train'
    valDir   = data_dir + '/valid'
    testDir  = data_dir + '/test'
    
    trainTransforms = transforms.Compose([transforms.RandomRotation(30),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    testTransforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
    
    valTransforms =  transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    trainData = datasets.ImageFolder(trainDir, transform = trainTransforms)
    valData   = datasets.ImageFolder(valDir,   transform = valTransforms) 
    testData  = datasets.ImageFolder(testDir,  transform = testTransforms) 
    
    trainLoader = torch.utils.data.DataLoader(trainData, batch_size=64, shuffle=True)
    valLoader   = torch.utils.data.DataLoader(valData,   batch_size=32, shuffle=True)
    testLoader  = torch.utils.data.DataLoader(testData,  batch_size=32, shuffle=True)
    
    return trainLoader, valLoader,testLoader,trainData
    
def modelSetup(model,hLayer,lRate,inputUnits,train_data):
    
    for param in model.parameters(): 
        param.requires_grad = False
    
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(inputUnits, hLayer)),
                              ('relu', nn.ReLU()),
                              ('dropout1', nn.Dropout(p=0.5)),
                              ('fc2', nn.Linear(hLayer, 102)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))

    model.classifier = classifier
    return model 
    
def trainModel(model,criterion,optimizer,epochs,device,trainLoader,valLoader):

    if device == 'cuda' and torch.cuda.is_available(): 
        model.to('cuda')
    else: 
        model.to('cpu')
        print("GPU not avaliable - using CPU")
    
    print_every = 40
    steps       = 0 

    for ep in range (epochs):
        running_loss = 0
    
        #Iterating through data to carry out training step
        for inputs, labels in trainLoader:
            steps += 1 
            inputs,labels = inputs.to(device),labels.to(device)
        
            #setting the gradients back to 0 
            optimizer.zero_grad()
        
            op = model.forward(inputs)
            loss = criterion(op,labels)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval() 
                with torch.no_grad(): 
                    valLoss = 0
                    valAccuracy = 0 
                    for inputs,labels in valLoader:
                        inputs,labels = inputs.to(device),labels.to(device)
                        outputs = model.forward(inputs)
                        valLoss += criterion(outputs,labels).item()
                        prob    = torch.exp(outputs)
                        equality = (labels.data == prob.max(dim=1)[1])
                        valAccuracy += equality.type(torch.FloatTensor).mean()
                     
                print("Epoch: {}/{}... ".format(ep+1, epochs),
                      "Loss: {:.4f}".format(running_loss/print_every),
                      "Validation Loss {:.4f}".format(valLoss/len(valLoader)),
                      "Accuracy: {:.4f}".format(valAccuracy/len(valLoader)))
                      
                running_loss = 0
                model.train()
                
    print("training model complete")        
    return model 
  

def saveModel(path,arch,model,trainData,optimizer,epochs):
        
    model.class_to_idx = trainData.class_to_idx

    checkpoint = {'architecture':arch,
                  'classifier'  :model.classifier,
                  'state_dict'  :model.state_dict(),
                  'opt_state'   :optimizer.state_dict,
                  'num_epochs'  :epochs,
                  'class_to_idx':model.class_to_idx}
    
    return torch.save(checkpoint,path)        
                        
def loadModel(filepath,device):

    if device == 'cuda' and torch.cuda.is_available(): 
        checkpoint = torch.load(filepath)
    else:
        checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)
    
    model= getattr(models,checkpoint['architecture'])(pretrained=True)
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model 
     
def processImage(image): 
        
    pil_img = Image.open(image)
    process_img = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])]) 
       
    pil_img = process_img(pil_img)
    
    return pil_img
    
def predict(imgPath,model,topk,device):
    
    model.eval()
    if device == 'cuda' and torch.cuda.is_available(): 
        model.to('cuda')
    else:
        model.to('cpu')
    
    image       = processImage(imgPath)
    image       = image.unsqueeze_(0)
    image       = image.float()
    idx         = model.class_to_idx
    flower_list = []
    
    with torch.no_grad():
        if device == 'cuda' and torch.cuda.is_available():  
            output = model.forward(image.cuda())
        else:
            output = model.forward(image)
    
    probability = nFunc.softmax(output.data,dim=1)
    probs,labs = probability.topk(topk)
  
    probs = probs.tolist()
    labs  = labs.tolist()   
    
    idx_to_class = {v: k for k, v in idx.items()}
    for vals in labs[0]:
        flower_list.append(idx_to_class[vals])

    return probs[0], flower_list
  
