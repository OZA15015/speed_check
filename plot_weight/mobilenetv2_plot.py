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

sys.path.append('/home/oza/pre-experiment/speeding/speed_check/mobilenet')
from models import *
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
    model = build_mobilenetV2()  
    model.eval()
    checkpoint = torch.load("/home/oza/pre-experiment/speeding/speed_check/mobilenet/checkpoint/ckpt.pth", map_location="cpu")['net'] 
    key_list = list(checkpoint.keys())
    print(key_list)
    count = 0
    new_list = []
    for name in key_list:
        if 'conv' in name:
            count += 1
            print(name)
            new_list.append(name)
        elif 'linear.weight' in name:
            count += 1
            print(name)
            new_list.append(name)
    print(count)
    quit()
    for name in new_list: #畳み込み層の重みを抽出
        fig = plt.figure()
        checkpoint[name] = torch.flatten(checkpoint[name])
        checkpoint[name] = checkpoint[name].to('cpu').detach().numpy().copy()
        count = 0 

        for i in range(checkpoint[name].shape[0]):
            if checkpoint[name][i] > -0.05 and checkpoint[name][i] < 0.05:
                count += 1
    
        print("weight_percentage: " + str(100*count/checkpoint[name].shape[0]))
        plt.title(name, fontsize=10)
        plt.xlabel("Value of Weight", fontsize=8)
        plt.ylabel("Frequency", fontsize=8) 
        plt.hist(checkpoint[name], bins=30)
        print("weight_mean: " + str(np.mean(checkpoint[name])))
        #print(name)
        fig.savefig("img_mobilenetv2_1/" + name + ".png")
        plt.close()


if __name__== "__main__":
    main()
