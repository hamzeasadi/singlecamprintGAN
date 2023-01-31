import numpy as np
import os
import torch
from torch import nn
from torch.optim import Optimizer

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def losstemp(bs=4, g=2):
    indesx = torch.tensor(list(range(2*bs)))

    batch_train = dict()
    k = 0
    for i in range(2*bs):
        if i<bs:
            if i%2 == 0:
                sub1 = torch.cat((indesx[:i], indesx[i+1:bs+i], indesx[bs+i+2:]))
                lbl = (sub1 == i+1).nonzero(as_tuple=True)[0]
                batch_train[f's{k}'] = (i, i+1, lbl, sub1)
                k+=1

                sub2 = torch.cat((indesx[:i], indesx[i+2:bs+i+1], indesx[bs+i+2:]))
                lbl = (sub2 == bs+i).nonzero(as_tuple=True)[0]
                batch_train[f's{k}'] = (i, bs+i, lbl, sub2)
                k+=1

                sub3 = torch.cat((indesx[:i], indesx[i+2:bs+i], indesx[bs+i+1:]))
                lbl = (sub3 == bs+i+1).nonzero(as_tuple=True)[0]
                batch_train[f's{k}'] = (i, bs+i+1, lbl, sub3)
                k+=1

            else:
                sub1 = torch.cat((indesx[:i-1], indesx[i+1:bs+i], indesx[bs+i+1:]))
                lbl = (sub1 == bs+i-1).nonzero(as_tuple=True)[0]
                batch_train[f's{k}'] = (i, bs+i-1, lbl, sub1)
                k+=1

                sub2 = torch.cat((indesx[:i-1], indesx[i+1:bs+i-1], indesx[bs+i:]))
                lbl = (sub2 == bs+i).nonzero(as_tuple=True)[0]
                batch_train[f's{k}'] = (i, bs+i, lbl, sub2)
                k+=1

        else:
            if i%2 == 0:
                sub1 = torch.cat((indesx[:i-bs], indesx[i-bs+2:i], indesx[i+1:]))
                lbl = (sub1 == i+1).nonzero(as_tuple=True)[0]
                batch_train[f's{k}'] = (i, i+1, lbl, sub1)
                k+=1

    return batch_train


def calc_psd(x):
    # x = x.squeeze()
    dft = torch.fft.fft2(x)
    avgpsd =  torch.mean(torch.mul(dft, dft.conj()).real, dim=0)
    r = torch.mean(torch.log(avgpsd)) - torch.log(torch.mean(avgpsd))
    return r





class OneClassLoss(nn.Module):
    """
    doc
    """
    def __init__(self, batch_size, group_size, reg) -> None:
        super().__init__()
        self.bs = batch_size
        self.gs = group_size
        self.reg = reg
        self.temp = losstemp(bs=batch_size, g=group_size)
        self.crit = nn.CrossEntropyLoss()

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=0)

        row = x[self.temp['s10'][0]]
        rem = x[self.temp['s10'][-1]]
        labels = self.temp['s10'][-2].to(dev)

        logits = torch.square(torch.linalg.matrix_norm(torch.subtract(row, rem))).squeeze()
        for k, v in self.temp.items():
            row = x[self.temp[k][0]]
            rem = x[self.temp[k][-1]]
            logit = torch.square(torch.linalg.matrix_norm(torch.subtract(row, rem))).squeeze()
            logits = torch.vstack((logits, logit))
            labels = torch.cat((labels, self.temp[k][-2].to(dev)), dim=0)

        # print(logits.shape, labels.shape)
        l1 = self.crit(logits, labels)
        l2 = -self.reg*calc_psd(x.squeeze())
        return l1 + l2
        
       
class KeepTrack():
    def __init__(self, path) -> None:
        self.path = path
        self.state = dict(model="", opt="", epoch=1, trainloss=0.1, valloss=0.1)

    def save_ckp(self, model: nn.Module, opt: Optimizer, epoch, fname: str, trainloss=0.1, valloss=0.1):
        self.state['model'] = model.state_dict()
        self.state['opt'] = opt.state_dict()
        self.state['epoch'] = epoch
        self.state['trainloss'] = trainloss
        self.state['valloss'] = valloss
        save_path = os.path.join(self.path, fname)
        torch.save(obj=self.state, f=save_path)

    def load_ckp(self, fname):
        state = torch.load(os.path.join(self.path, fname), map_location=dev)
        return state









def main():
    x1 = torch.randn(size=(64, 1, 3, 3))
    x2 = torch.randn(size=(64, 1, 3, 3))

    x = torch.ones(size=(64, 1, 64, 64))
    x = torch.rand(size=(64, 1, 64, 64))
    y = torch.randn(size=(64, 1, 64, 64))
    xpsd = calc_psd(x)
    ypsd = calc_psd(y)
    print(xpsd)
    print(ypsd)

 









if __name__ == '__main__':
    main()