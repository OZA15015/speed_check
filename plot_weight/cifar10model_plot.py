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
import matplotlib.pyplot as plt

sys.path.append('/home/oza/pre-experiment/speeding/FBNet_load')
import fbnet_building_blocks.fbnet_builder as fbnet_builder
from collections import OrderedDict
from torchsummary import summary

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
device = "cuda"

def fbneta_cifar():
    model = fbnet_builder.get_model('first_test', cnt_classes=10).to(device)
    return model


def main():
    model = fbneta_cifar()
    #checkpoint = torch.load("/home/oza/pre-experiment/speeding/FBNet_load/architecture_functions/logs/saitest_FBA/best_model.pth", map_location=device)
    checkpoint = torch.load("/home/oza/pre-experiment/speeding/FBNet_load/architecture_functions/logs/first_test/best_model.pth", map_location=device) 
    #checkpoint = torch.load("/home/oza/pre-experiment/speeding/FBNet_load/architecture_functions/logs/second_test/best_model.pth", map_location=device) 
    #checkpoint = torch.load("/home/oza/pre-experiment/speeding/FBNet_load/architecture_functions/logs/third_test/best_model.pth", map_location=device) 
    #checkpoint = torch.load("/home/oza/pre-experiment/speeding/FBNet_load/architecture_functions/logs/fbnet_c/best_model.pth", map_location=device) 
    new_state_dict = OrderedDict()
    
    for k, v in checkpoint.items():
        #print(k)
        if 'module' in k:
            k = k.replace('module.', '')
        new_state_dict[k] = v
     
    #model.load_state_dict(checkpoint, strict=False)
    model.load_state_dict(new_state_dict)
    model.eval()
    key_list = list(checkpoint.keys())
    new_list = []
    count = 0
    for name in key_list:
        #print(checkpoint[name].shape)
    
        if checkpoint[name].ndim == 4:
            new_list.append(name)
            count += 1
            print(checkpoint[name].shape)
        elif 'module.last_stages.fc.weight' in name:
            new_list.append(name)
            count += 1
            print(checkpoint[name].shape)
    print("sum layers: " + str(count))
    #print(new_list) 
    summary(model, (3, 32, 32))
    #quit()
    sum = 0
    for name in new_list: #畳み込み層の重みを抽出
        fig = plt.figure()
        checkpoint[name] = torch.flatten(checkpoint[name])
        checkpoint[name] = checkpoint[name].to('cpu').detach().numpy().copy()
        print(checkpoint[name].shape)
        count = 0

        for i in range(checkpoint[name].shape[0]):
            if checkpoint[name][i] > -0.05 and checkpoint[name][i] < 0.05:
                count += 1

        print(name + "_percentage: " + str(100*count/checkpoint[name].shape[0]))
        sum += 100*count/checkpoint[name].shape[0]
        plt.title(name, fontsize=10)
        plt.xlabel("Value of Weight", fontsize=8)
        plt.ylabel("Frequency", fontsize=8)
        plt.hist(checkpoint[name], bins=30)
        print("weight_mean: " + str(np.mean(checkpoint[name])))
        #print(name)
        #fig.savefig("first_test/" + name + ".png")
        plt.close()

    print("sum_weight_percentage(-0.05 < weight < 0.05): " + str(sum / len(new_list)))
    
    '''
    params = 0 
    for p in model.parameters():
        if p.requires_grad:
            params += p.numel()
    print(params)  # 12189
    '''

if __name__== "__main__":
    main()
