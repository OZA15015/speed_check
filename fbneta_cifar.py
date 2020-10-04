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

sys.path.append('/home/oza/pre-experiment/speeding/FBNet')
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
    model = fbnet_builder.get_model('fbnet_a', cnt_classes=10).to(device)
    return model

def test_accuracy(model):
    correct = 0
    total = 0
    # 勾配を記憶せず（学習せずに）に計算を行う
    with torch.no_grad():
        i = 0
        time_sum = 0
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            start_time = time.time()
            outputs = model(images)
            time_sum += time.time() - start_time
            print(time.time() - start_time)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            i += 1

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
    print(100 * correct / total)
    print("FBNet Group: " + str(time_sum/i))

def main():
    model = fbneta_cifar()
    #checkpoint = torch.load("/home/oza/pre-experiment/FBNet/architecture_functions/logs/0830fbnet_a/best_model.pth", map_location=device) 
    checkpoint = torch.load("/home/oza/pre-experiment/speeding/FBNet/architecture_functions/logs/saitest_FBA/best_model.pth", map_location=device)
    #checkpoint = torch.load("/home/oza/pre-experiment/speeding/FBNet/architecture_functions/logs/first_test/best_model.pth", map_location=device) 
    #checkpoint = torch.load("/home/oza/pre-experiment/speeding/FBNet/architecture_functions/logs/second_test/best_model.pth", map_location=device) 
    #checkpoint = torch.load("/home/oza/pre-experiment/speeding/FBNet/architecture_functions/logs/third_test/best_model.pth", map_location=device) 
    new_state_dict = OrderedDict()
    
    for k, v in checkpoint.items():
        print(k)
        if 'module' in k:
            k = k.replace('module.', '')
        new_state_dict[k] = v
    
    #model.load_state_dict(checkpoint, strict=False)
    model.load_state_dict(new_state_dict)
    model.eval()
    test_accuracy(model)
    params = 0 
    for p in model.parameters():
        if p.requires_grad:
            params += p.numel()
    print(params)  # 12189

    summary(model, (3, 32, 32))

if __name__== "__main__":
    main()
