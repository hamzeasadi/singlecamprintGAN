import os
import conf as cfg
import torch
from torch import nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch import optim
import utils
import datasetup as dst
import model as m
import engine
import argparse
import testing as tst

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(prog='train.py', description='required flags and supplemtary parameters for training')
parser.add_argument('--train', action=argparse.BooleanOptionalAction)
parser.add_argument('--test', action=argparse.BooleanOptionalAction)
# parser.add_argument('--data', '-d', type=str, required=True, default='None')
parser.add_argument('--modelname', '-mn', type=str, required=True, default='None')
parser.add_argument('--epochs', '-e', type=int, required=False, metavar='epochs', default=1)
# parser.add_argument('--numcls', '-nc', type=int, required=True, metavar='numcls', default=10)

args = parser.parse_args()

def train(Net:nn.Module, optfunc:Optimizer, lossfunc:nn.Module, epochs, modelname):
    kt = utils.KeepTrack(path=cfg.paths['model'])
    traindata, valdata = dst.createdl()
    for epoch in range(epochs):
        trainloss = engine.train_setp(net=Net, data=traindata, opt=optfunc, criterion=lossfunc)
        valloss = engine.val_setp(net=Net, data=valdata, opt=optfunc, criterion=lossfunc)
        fname = f'{modelname}_{epoch}.pt'
        kt.save_ckp(model=Net, opt=optfunc, epoch=epoch, trainloss=trainloss, valloss=valloss, fname=fname)
        print(f"epoch={epoch}, trainloss={trainloss}, valloss={valloss}")










def main():
    
    model = m.VideoPrint(inch=1, depth=20)
    model.to(dev)
    optimizer = optim.Adam(params=model.parameters(), lr=3e-4)
    crt = utils.OneClassLoss(batch_size=100, group_size=2, reg=0.1)

    if args.train:
        train(Net=model, optfunc=optimizer, lossfunc=crt, epochs=args.epochs, modelname=args.modelname)

    if args.test:
        tst.result()
    




if __name__ == '__main__':
    main()