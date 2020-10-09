import torch                                                                                                                            
from torch import nn
import numpy as np
from tensorboardX import SummaryWriter
import argparse
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import sys
import time
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import statistics

sys.path.append('/home/oza/pre-experiment/speeding/speed_check')
from cifar10_models import *  
from collections import OrderedDict
from torchsummary import summary
import scipy.stats as stats

# ToTensor：画像のグレースケール化（RGBの0~255を0~1の範囲に正規化）、Normalize：Z値化（RGBの平均と標準偏差を0.5で決め打ちして正規化）
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.49139968, 0.48215827, 0.44653124), (0.2023, 0.1994, 0.2010))])
 
# トレーニングデータをダウンロード
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)                                                                    
 
# テストデータをダウンロード
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True, num_workers=16)
    
manual_seed = 1 
np.random.seed(manual_seed)
torch.manual_seed(manual_seed)
torch.cuda.manual_seed_all(manual_seed)
torch.backends.cudnn.benchmark = True
#device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cuda'


def test_accuracy(model):
    correct = 0
    total = 0
    i = 0
    time_sum = 0
    # 勾配を記憶せず（学習せずに）に計算を行う
    #with torch.no_grad():
    for data in testloader:
        images, labels = data
        images = images.to('cuda')
        labels = labels.to('cuda')
        start_time = time.time()
        outputs = model(images)
        time_sum += time.time() - start_time
        print(time.time() - start_time)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        i += 1
    
    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
    print("MnobileNetV2_AveTime: " + str(time_sum/i))

def main():
    #model = resnet18(pretrained=True, device=device)
    model, checkpoint = mobilenet_v2(pretrained=True, device=device)
    #model = vgg11_bn(pretrained=True, device=device)
    #model = densenet121(pretrained=True, device=device)  
    model.eval()
    key_list = list(checkpoint.keys())
    #print(key_list) 
    #quit()
    count = 0
    new_list = []
    
    for name in key_list:
        if checkpoint[name].ndim == 4:
            new_list.append(name)
            count += 1
        elif 'classifier.1.weight' in name:
            new_list.append(name)          
            count += 1
    '''
    for name in key_list:
        #print(name)
        #print(checkpoint[name].shape)
        
        if 'conv' in name and 'weight' in name:
            count += 1
            print(name)
            print(checkpoint[name].shape)
            new_list.append(name)
        elif 'classifier.1.weight' in name:
            count += 1
            print(name)
            print(checkpoint[name].ndim)
            new_list.append(name)
        elif 'features.0.0.weight' in name or 'features.0.1.weight' in name:
            count += 1
            print(name)
            print(checkpoint[name].shape)
            new_list.append(name)
        elif 'features.18.0.weight' in name:
            count += 1
            print(name)
            print(checkpoint[name].shape)
            new_list.append(name)
    '''

    print(new_list)
    print(count)
    #print(model)
    summary(model, (3, 32, 32))
    
    for name in new_list: #畳み込み層の重みを抽出
        fig = plt.figure()
        checkpoint[name] = torch.flatten(checkpoint[name])
        checkpoint[name] = checkpoint[name].to('cpu').detach().numpy().copy()
        print(checkpoint[name].shape)
        #quit()
        count = 0 

        for i in range(checkpoint[name].shape[0]):
            if checkpoint[name][i] > -0.03 and checkpoint[name][i] < 0.03:
                count += 1
    
        print("weight_percentage: " + str(100*count/checkpoint[name].shape[0]))
        plt.title(name, fontsize=10)
        plt.xlabel("Value of Weight", fontsize=8)
        plt.ylabel("Frequency", fontsize=8) 
        plt.hist(checkpoint[name], bins=30)
        print("weight_mean: " + str(np.mean(checkpoint[name])))
        #print(name)
        #fig.savefig("img_mobilenetv2_1/" + name + ".png")
        #plt.close()


if __name__== "__main__":
    main()
