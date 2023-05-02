###################################3
# Visualize HearMap by sum
# Zheng, Zhedong, Liang Zheng, and Yi Yang. "A discriminatively learned cnn embedding for person reidentification." ACM Transactions on Multimedia Computing, Communications, and Applications (TOMM) 14, no. 1 (2018): 13.
###################################

import os
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np

import yaml
import argparse
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms
from PIL import Image
import cv2
parser = argparse.ArgumentParser(description='Training')

parser.add_argument('--data_dir',default='../Market/pytorch',type=str, help='./test_data')
parser.add_argument('--name', default='ft_ResNet50', type=str, help='save model path')
parser.add_argument('--batchsize', default=1, type=int, help='batchsize')
parser.add_argument('--which_epoch',default='59', type=str, help='0,1,2,3...or last')

opt = parser.parse_args()

# config_path = os.path.join('./model_test',opt.name,'opts.yaml')
# with open(config_path, 'r') as stream:
#         config = yaml.load(stream)
# opt.fp16 = config['fp16']
# opt.PCB = config['PCB']
# opt.use_dense = config['use_dense']
# opt.use_NAS = config['use_NAS']
# opt.stride = config['stride']
#
#
# if 'h' in config:
#     opt.h = config['h']
#     opt.w = config['w']

opt.nclasses = 318


def Normalize(data):
 m = data.mean()
 mx = data.max()
 mn = data.min()
 return (data-m)/(mx-mn)


def heatmap2d(img, arr, arr2):
    image_dir = os.path.split(img)
    save_dir=image_dir[0][:-10]+'heatmap'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    fig = plt.figure()
    ax0 = fig.add_subplot(121, title="Image")
    ax1 = fig.add_subplot(122, title="Heatmap")

    ax0.imshow(Image.open(img))
    heatmap = ax1.imshow(arr, cmap='viridis')

#
    img_cv = cv2.imread(img)
    # arr=Normalize(arr)
    arr2=np.mean(arr2, axis=0)
    arr2 = np.maximum(arr2, 0)
    print(arr2.shape)
    arr2 /= np.max(arr2)
    arr2 = cv2.resize(arr2, (img_cv.shape[1], img_cv.shape[0]))  # 将热力图的大小调整为与原始图像相同
    arr2 = np.uint8(255 * arr2)  # 将热力图转换为RGB格式
    arr2 = cv2.applyColorMap(arr2, cv2.COLORMAP_JET)  # 将热力图应用于原始图像
    superimposed_img = arr2 * 0.4 + img_cv  # 这里的0.4是热力图强度因子
    cv2.imwrite(save_dir+'/'+image_dir[1], superimposed_img)  # 将图像保存到硬盘
#
    fig.colorbar(heatmap)
    plt.show()
    fig.savefig('heatmap')


def load_network(network, name, opt):
    save_path = os.path.join('model_test', name, 'net_%s.pth' % opt.which_epoch)
    network.load_state_dict(torch.load(save_path))
    return network,opt.which_epoch

data_transforms = transforms.Compose([
        transforms.Resize((256,128), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

image_datasets = {x: datasets.ImageFolder(os.path.join(opt.data_dir,x) ,data_transforms) for x in ['train']}


dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                             shuffle=False, num_workers=1) for x in ['train']}


imgpath = image_datasets['train'].imgs
#

model, epoch = load_network(model_structure,opt.name, opt)
# model, epoch = load_network(opt.name, opt)
#
model.classifier.classifier = nn.Sequential()
model = model.eval().cuda()

ii=0
ss=len(dataloaders['train'])
for data in dataloaders['train']:
    img, label = data

# data = next(iter(dataloaders['train']))
# img, label = data
    with torch.no_grad():
        x = model.model.conv1(img.cuda())
        x = model.model.bn1(x)
        x = model.model.relu(x)
        x = model.model.maxpool(x)
        x = model.model.layer1(x)
        x = model.model.layer2(x)
        x = model.model.layer3(x)
        output = model.model.layer4(x)
   # print(output.shape)
    heatmap1 = output.squeeze().sum(dim=0).cpu().numpy()
    heatmap2 = output.squeeze().cpu().numpy()
   # print(heatmap1.shape)
    #print(heatmap2.shape)
    #test_array = np.arange(100 * 100).reshape(100, 100)
    # Result is saved tas `heatmap.png`
    heatmap2d(imgpath[ii][0],heatmap1,heatmap2)
    ii+=1
