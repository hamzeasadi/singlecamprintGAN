import os, random
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import conf as cfg


def datatemp(datapath:str, H=64, W=64, href=1080, wref=1920):
    hstart = (href%64)//2
    wstart = 0
    numh = href//64
    numw = wref//64
    # (folderid, hi, wi)
    tmp = dict()
    folders = os.listdir(datapath)
    folders = cfg.rm_ds(folders)
    patch_cnt = 0
    for folder in folders:
        folderpath = os.path.join(datapath, folder)
        for i in range(numh):
            hi = hstart + i*H
            for j in range(numw):
                wi = wstart + j*W
                tmp[f'patch{patch_cnt}'] = (folderpath, hi, wi)
                patch_cnt+=1
    return tmp


def cropimg(img, hi, wi, H=64, W=64):
    return img[hi:hi+H, wi:wi+W, 1:2]/256


class NoisPrintData(Dataset):
    """
    doc
    """
    def __init__(self, datapath:str, batch_size=100) -> None:
        super().__init__()
        self.path = datapath
        self.bs = batch_size
        self.temp = datatemp(datapath=datapath, H=64, W=64, href=1080, wref=1920)
        self.patchids = list(self.temp.keys())

    def getpair(self, patchid):
        folderpath, h, w = self.temp[patchid]
        iframelist = os.listdir(folderpath)
        subiframes = random.sample(iframelist, 4)
        pair1 = []
        pair2 = []
        for i, iframe in enumerate(subiframes):
            iframepath = os.path.join(folderpath, iframe)
            img =cv2.imread(iframepath)
            crop = cropimg(img, hi=h, wi=w)
            if i<2:
                pair1.append(torch.from_numpy(crop).permute(2, 0, 1))
            else:
                pair2.append(torch.from_numpy(crop).permute(2, 0, 1))
            
        pp1 = torch.cat((pair1[0].unsqueeze(dim=0), pair1[1].unsqueeze(dim=0)), dim=0)
        pp2 = torch.cat((pair2[0].unsqueeze(dim=0), pair2[1].unsqueeze(dim=0)), dim=0)
        
        return pp1, pp2



    def __len__(self):
        return 1000

    def __getitem__(self, index):
        pairs = random.sample(self.patchids, self.bs//2)
        X1, X2 = self.getpair(pairs[0])
        for ip in range(1, self.bs//2):
            ppid = pairs[ip]
            x1, x2 = self.getpair(ppid)
            X1 = torch.cat((X1, x1), dim=0)
            X2 = torch.cat((X2, x2), dim=0)

        return X1.float(), X2.float()

       

def createdl():
    traindataset = NoisPrintData(datapath=cfg.paths['train'], batch_size=100)
    valdataset = NoisPrintData(datapath=cfg.paths['val'], batch_size=100)
    return DataLoader(traindataset, batch_size=1), DataLoader(valdataset, batch_size=1)

        





def main():
    print(2)
    trainl, vall = createdl()
    X1, X2 = next(iter(trainl))
    print(X1.shape, X2.shape)
    print(X1.squeeze(dim=0).shape)
  


if __name__ == '__main__':
    main()