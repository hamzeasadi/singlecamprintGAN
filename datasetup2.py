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

def cropimg(img, hi, wi, H=64, W=64):
    coordcrop = coordxy[:, hi:hi+H, wi:wi+W]
    imgc = torch.from_numpy(img[hi:hi+H, wi:wi+W, 1:2]/256).permute(2, 0, 1)

    return torch.cat((imgc, coordcrop), dim=0)


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
        print(len(self.patchids))

    def getpair(self, patchid):
        coords = coordinate(High=1080, Width=1920)
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
                # s1 = torch.cat((torch.from_numpy(crop).permute(2, 0, 1)))
                pair1.append(crop)
            else:
                pair2.append(crop)
            
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
    return DataLoader(traindataset, batch_size=1, pin_memory=True), DataLoader(valdataset, batch_size=1, pin_memory=True)

        





def main():
    print(2)
    # temp = datatemp(data)
    dataset = NoisPrintData(datapath=cfg.paths['train'], batch_size=100)
    X1, X2 = dataset[0]
    print(X1.shape)
    print(X2.shape)


if __name__ == '__main__':
    main()