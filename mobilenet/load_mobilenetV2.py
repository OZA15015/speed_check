import torch                                                                                                                            
from torch import nn
import torch.optim as optim   
import numpy as np
from tensorboardX import SummaryWriter
import argparse
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import sys
import time
import torch.backends.cudnn as cudnn

#sys.path.append('/home/oza/pre-experiment/speeding/test_dist/mobilenet')
from models import *
from collections import OrderedDict
from torchsummary import summary

# ToTensor：画像のグレースケール化（RGBの0~255を0~1の範囲に正規化）、Normalize：Z値化（RGBの平均と標準偏差を0.5で決め打ちして正規化）
#transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.49139968, 0.48215827, 0.44653124), (0.2023, 0.1994, 0.2010))])
model =  MobileNetV2().to("cuda") 
model = torch.nn.DataParallel(model)
cudnn.benchmark = True

# トレーニングデータをダウンロード)
transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.49139968, 0.48215827, 0.44653124), (0.2023, 0.1994, 0.2010)),
])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)                                                                    

transform_test = transforms.Compose([
    #transforms.RandomCrop(32, padding=4),
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.49139968, 0.48215827, 0.44653124), (0.2023, 0.1994, 0.2010)),
])

# テストデータをダウンロード
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

optimizer = optim.SGD(model.parameters(), lr=0.01,
                      momentum=0.9, weight_decay=4e-5)

'''
manual_seed = 1 
np.random.seed(manual_seed)
torch.manual_seed(manual_seed)
torch.cuda.manual_seed_all(manual_seed)
torch.backends.cudnn.benchmark = True
'''
#device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cuda'
def build_mobilenetV2():
    model = MobileNetV2().to(device)
    if device == 'cuda':  
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True
    return model

def test_accuracy(model):
    correct = 0
    total = 0
    i = 0
    time_sum = 0
    # 勾配を記憶せず（学習せずに）に計算を行う
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            #loss = criterion(outputs, targets)
 
            #test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            print(100*correct/total)
        '''
        for data in testloader:
            images, labels = data
            images = images.to('cuda')
            labels = labels.to('cuda')
            start_time = time.time()
            outputs = model(images)
            time_sum += time.time() - start_time
            print(time.time() - start_time)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            #print(100*correct/total)
            i += 1
        '''
    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
    #print("MnobileNetV2_AveTime: " + str(time_sum/i))

def main():
    #model = build_mobilenetV2()  
    checkpoint = torch.load("./checkpoint/ckpt.pth", map_location="cuda")['net'] 
    new_state_dict = OrderedDict()
    
    '''
    for k, v in checkpoint.items():
        print(k)
        if 'module' in k:
            k = k.replace('module.', '')
        new_state_dict[k] = v
    '''
    
    #model.load_state_dict(new_state_dict)
    model.load_state_dict(checkpoint)
    #start_time = time.time()
    model.eval()
    test_accuracy(model)
    #print("FBNetA: " + str(time.time() - start_time ))
    
    params = 0 
    for p in model.parameters():
        if p.requires_grad:
            params += p.numel()
    print(params)  # 12189
    
    summary(model, (3, 32, 32))

if __name__== "__main__":
    main()
