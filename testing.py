import os, random
import conf as cfg
import torch
from torch import nn as nn
from matplotlib import pyplot as plt
import utils
import datasetup as dst
import model as m
import model2 as m2
import cv2


dev = torch.device('cpu')
testdata = cfg.paths['val']

def add_noise(img, sigma=1, mean=0):
    b, c, h, w = img.shape
    hc, wc = h//2, w//2
    noisypart = img[:, 0:1, hc-32:hc+32, wc-32:wc+32]
    filter = torch.randn_like(noisypart)*sigma + mean
    img[:, 0:1, hc-32:hc+32, wc-32:wc+32] = filter + noisypart
    return img


def copy_move(img, d=64):
    b, c, h, w = img.shape
    hc, wc = h//2, w//2
    copypart = img[:, :, hc-32:hc+32, wc-32:wc+32]
    img[:, :, hc-32+d:hc+32+d, wc-32+d:wc+32+d] = copypart
    return img


def splicing(img, splicimg):
    b, c, h, w = img.shape
    hc, wc = h//2, w//2
    b1, c1, h1, w1 = splicimg.shape
    h1c, w1c = h1//2, w1//2
    splicepart = splicimg[:, :, h1c-32:h1c+32, w1c-32:w1c+32]
    img[:, :, hc-32:hc+32, wc-32:wc+32] = splicepart
    return img



def coordinate(High, Width):
    xcoord = torch.ones(size=(High, Width), dtype=torch.float32)
    ycoord = torch.ones(size=(High, Width), dtype=torch.float32)
    for i in range(High):
        xcoord[i, :] = 2*(i/High) - 1
    for j in range(Width):
        ycoord[:, j] = 2*(j/Width) - 1
    
    coord = torch.cat((xcoord.unsqueeze(dim=0), ycoord.unsqueeze(dim=0)), dim=0)
    return coord


coordxy = coordinate(High=1080, Width=1920)

def cropimg(img, coord=False):
    h, w, c = img.shape
    hc, wc = h//2, w//2
    crop = img[hc-128:hc+128, wc-128:wc+128, 1:2]

    crop = torch.from_numpy(crop).float().permute(2, 0, 1)

    if coord:
        coordcrop = coordxy[:, hc-128:hc+128, wc-128:wc+128]
        crop = torch.cat((crop, coordcrop), dim=0)

    return crop.unsqueeze(dim=0)


  

def imgman(datapath, coord=False):

    imgfolders = os.listdir(datapath)
    imgfolders = cfg.rm_ds(imgfolders)
    imgfolder = random.sample(imgfolders, 1)
    imgfolderpath = os.path.join(datapath, imgfolder[0])
    imglist = os.listdir(imgfolderpath)
    imgs = random.sample(imglist, 2)
    img1 = cv2.imread(os.path.join(imgfolderpath, imgs[0]))/255
    img2 = cv2.imread(os.path.join(imgfolderpath, imgs[1]))/255
    
    crop1 = cropimg(img1, coord=coord)
    crop2 = cropimg(img2, coord=coord)

    crop1c = crop1.clone()
    crop1n = crop1.clone()
    crop1s = crop1.clone()
    
    crop2c = crop2.clone()
    crop2n =crop2.clone()
    crop2s = crop2.clone()

    
    firsttuple = [crop1, add_noise(crop1n), copy_move(crop1c), splicing(crop1s, crop2s)]
    secondtuple = [crop2, add_noise(crop2n), copy_move(crop2c), splicing(crop2s, crop1s)]

    return firsttuple, secondtuple


def visulize(list1img, list2img, kk):
    l = len(list1img)
    fig, axs = plt.subplots(nrows=2, ncols=l, figsize=(20, 8))
    for i in range(l):
        axs[0, i].imshow(list1img[i], cmap='gray')
        axs[0, i].axis('off')
        axs[1, i].imshow(list2img[i], cmap='gray')
        axs[1, i].axis('off')
    
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.savefig(os.path.join(cfg.paths['outpics'], f'{kk}.png'))
    plt.close('all')


def result(modelpath, coordinate=False):
    kt = utils.KeepTrack(path=modelpath)
    models = os.listdir(modelpath)
    models = cfg.rm_ds(models)
    inch=1
    kk=0
    imgset1 , imgset2 = imgman(datapath=testdata, coord=coordinate)
    for model in models:
        state = kt.load_ckp(fname=model)
        model_state_dict = state['model']
        if coordinate:
            inch=3
            net = m2.VideoPrint(inch=inch, depth=20)
        else:
            net = m.VideoPrint(inch=inch, depth=20)

        net.load_state_dict(model_state_dict)
        
            

        noiseset1 = []
        noiseset2 = []
        l = len(imgset1)
        net.eval()
        with torch.no_grad():
            for i in range(l):
                noise1, noise2 = net(imgset1[i], imgset2[i])
                noiseset1.append(noise1)
                noiseset2.append(noise2)

        list1 = [img.detach().squeeze().numpy() for img in noiseset1]
        list2 = [img.detach().squeeze().numpy() for img in noiseset2]

        visulize(list1img=list1, list2img=list2, kk=kk)
        kk+=1

    

    




def main():
    
    # mn = 'singlecamfingerprint_38.pt'
    modelpath = cfg.paths['model2']
    result(modelpath=modelpath, coordinate=True)
    
    # x = torch.zeros(size=(1,1, 3, 3))
    # y = x.clone()
    # x[0, 0, 0,0] = 10
    # print(x)
    # print(y)



if __name__ == '__main__':
    main()

