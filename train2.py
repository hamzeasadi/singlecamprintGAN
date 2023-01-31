import os
import conf as cfg
import torch
from torch import nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch import optim
import utils
import datasetup2 as dst
import model2 as m
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

def train(gen_net: nn.Module, disc_net: nn.Module, gen_opt: optim.Optimizer, disc_opt: optim.Optimizer, criterion: nn.Module, modelname: str, epochs):
    kt = utils.KeepTrack(path=cfg.paths['model'])
    for epoch in range(epochs):
        trainl, testl = dst.createdl()
        trainloss = engine.train_step(disc=disc_net, gen=gen_net, data=trainl, criterion=criterion, disc_opt=disc_opt, gen_opt=gen_opt)
        valloss, disc_loss = engine.val_step(disc=disc_net, gen=gen_net, data=testl, criterion=criterion, disc_opt=disc_opt, gen_opt=gen_opt)
  
        print(f"epoch={epoch}, trainloss={trainloss}, valloss={valloss}, disc_loss={disc_loss}")
        fname=f'{modelname}_{epoch}.pt'
        kt.save_ckp(model=gen_net, opt=gen_opt, epoch=epoch, trainloss=trainloss, valloss=valloss, fname=fname)










def main():
    
    mn = args.modelname
    GenNet = m.VideoPrint(inch=3, depth=20)
    GenNet.to(dev)
    discNet = m.Critic(channels_img=1, features_d=64)
    discNet.to(dev)
    crt = utils.OneClassLoss(batch_size=100, group_size=2, reg=0.1)
    gen_opt = optim.RMSprop(params=GenNet.parameters(), lr=3e-4)
    disc_opt = optim.RMSprop(params=discNet.parameters(), lr=3e-4)
    if args.train:
        train(gen_net=GenNet, disc_net=discNet, gen_opt=gen_opt, disc_opt=disc_opt, criterion=crt, modelname=mn, epochs=args.epochs)
    




if __name__ == '__main__':
    main()