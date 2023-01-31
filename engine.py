import os, random
import torch
from torch import nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
import conf as cfg
from torch.optim import Optimizer



dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_step(disc: nn.Module, gen: nn.Module, data: DataLoader, criterion: nn.Module, disc_opt: Optimizer, gen_opt: Optimizer):
    epoch_error = 0
    l = len(data)
    gen.train()
    disc.train()

    for i, (X1, X2) in enumerate(data):
        X1 = X1.to(dev).squeeze(dim=0)
        X2 = X2.to(dev).squeeze(dim=0)
        (_, _), (fake, real) = gen(X1, X2)
        
        discreal = real.detach().to(dev)
        discfake = fake.detach().to(dev)
        for _ in range(5):
            disc_real = disc(discreal).reshape(-1)
            disc_fake = disc(discfake).reshape(-1)
            loss_disc = -(torch.mean(disc_real) - torch.mean(disc_fake))
            disc_opt.zero_grad()
            loss_disc.backward(retain_graph=True)
            disc_opt.step()
            for p in disc.parameters():
                p.data.clamp_(-0.01, 0.01)

        # gen training
        disc_out = disc(fake).reshape(-1)
        # print(f"real engine={real.shape}, fake engine={fake.shape}")
        gen_loss1 = criterion(fake, real)
        gen_loss2 = -torch.mean(disc_out)
        gen_loss = gen_loss1 + gen_loss2
        gen_opt.zero_grad()
        gen_loss.backward()
        gen_opt.step()
        epoch_error += gen_loss.item()
        # print("p3")
        # break
    return epoch_error/l


def val_step(disc: nn.Module, gen: nn.Module, data: DataLoader, criterion: nn.Module, disc_opt: Optimizer, gen_opt: Optimizer):
    epoch_error = 0
    l = len(data)
    gen.eval()
    disc.eval()

    for i, (X1, X2) in enumerate(data):
        X1 = X1.to(dev).squeeze(dim=0)
        X2 = X2.to(dev).squeeze(dim=0)
        (_, _), (fake, real) = gen(X1, X2)
        discreal = real.detach().to(dev)

        disc_real = disc(discreal).reshape(-1)
        disc_fake = disc(fake).reshape(-1)
        loss_disc = -(torch.mean(disc_real) - torch.mean(disc_fake))
         

        # gen training
        gen_loss1 = criterion(fake, real)
        disc_out = disc(fake).reshape(-1)
        gen_loss2 = -torch.mean(disc_out)
        gen_loss = gen_loss1 + gen_loss2
 
        epoch_error += gen_loss.item()
        # break
    return epoch_error/l, loss_disc.item()




def main():
    pass




if __name__ == '__main__':
    main()