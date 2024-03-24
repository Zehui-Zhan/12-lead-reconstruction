import argparse
import os
import numpy as np

from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch
import glob
import matplotlib.pyplot as plt
import scipy.io
import random
from main_GAN import data_load
from evaluate import evaluate_,show_generated_lead,CC2
import json
from torch.nn import DataParallel
from GAN import Generator_gan,Discriminator_gan,gen_loss,discriminator_loss
from VAE_CNN import VAE,loss_fn
from LSTM import Generator_lstm,loss_function

def train_gan(generator, discriminator,epoch_begin,epoch,dataloader,dataloader_verify):
    # Configure data loader
    os.makedirs("./model/GAN1_group_gan")
    for epoch_i in range(epoch_begin, epoch,1):
        epoch_gloss = 0
        if epoch_i % 10 ==0:
            sim_cc = evaluate_(generator,device,dataloader_verify)
            record_dict['sim'].append(sim_cc.cpu().item())
        for i, data in enumerate(dataloader):
            # Adversarial ground truths
            ecg_lead1 = data[:,0,:]
            ecg_lead2 = data[:,1:12,:]
            # Configure input
            real_ecg_lead1 = ecg_lead1.unsqueeze(1)
            real_ecg_lead2 = ecg_lead2
            real_ecg_lead1 = real_ecg_lead1.to(device)
            real_ecg_lead2 = real_ecg_lead2.to(device)
            # train generator
            # Loss measures generator's ability to fool the discriminator
            g_ecg_lead = generator(real_ecg_lead1).detach()
            disc_real_output = discriminator(real_ecg_lead2)
            disc_generated_output = discriminator(g_ecg_lead)

            d_loss = discriminator_loss(disc_real_output, disc_generated_output)
            optimizer_D.zero_grad()
            d_loss.backward()
            optimizer_D.step()

            g_ecg_lead = generator(real_ecg_lead1)
            disc_generated_output = discriminator(g_ecg_lead)
            g_loss = gen_loss(disc_generated_output, gen_output=g_ecg_lead, target=real_ecg_lead2)
            optimizer_G.zero_grad()
            g_loss.backward()
            optimizer_G.step()

            epoch_gloss += g_loss.item()
            if i % 10==0:
                record_dict['g_loss_list'].append(g_loss.detach().cpu().item())
                record_dict['d_loss_list'].append(d_loss.detach().cpu().item())
                record_dict['d_fake_list'].append((torch.mean(disc_generated_output.detach().cpu())).item())
                record_dict['d_real_list'].append((torch.mean(disc_real_output.detach().cpu())).item())
                record_dict['mse'].append((torch.mean(torch.pow((real_ecg_lead2.cpu()-g_ecg_lead.detach().cpu()),2))).item())
                print(
                    "[Epoch %d/%d] [Batch %d/%d] d_real: %f, d_fake: %f, g_loss: %f"
                    % (epoch_i, epoch, i + 1, len(dataloader), torch.mean(disc_real_output),
                       torch.mean(disc_generated_output),
                       torch.mean(g_loss))
                )
        record_dict['epoch_gloss_list'].append(epoch_gloss)
        if epoch_i % 2 == 0:
            show_generated_lead(real_ecg_lead1,real_ecg_lead2,g_ecg_lead,0)
        if epoch_i % 20 == 0:
            with open('model/GAN1_group_gan/data.json', 'w') as f:
                json.dump(record_dict, f)
            torch.save(generator, "./model/GAN1_group_gan/Gparams_.pkl")
            torch.save(discriminator, "./model/GAN1_group_gan/Dparams_.pkl")
def train_VAE(device, dl, model,epoch):
    os.makedirs("./model/GAN1_group_vae")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.000001)
    for i in range(1,epoch,1):
        if i % 10 ==0:
            SIM = []
            for i, data in enumerate(dataloader_verify):
                ecg_lead1 = data[:, 0, :]
                ecg_lead2 = data[:, 1:12, :]
                # Configure input
                real_ecg_lead1 = ecg_lead1.unsqueeze(1).to(device)
                real_ecg_lead2 = ecg_lead2.to(device)
                g_ecg_lead, mean, logvar = generator(real_ecg_lead1)
                g_ecg_lead = g_ecg_lead.detach()
                for x in range(1, 12):
                    a = CC2(g_ecg_lead, real_ecg_lead2, lead=x)
                    a1 = sum(a) / len(a)
                    SIM.append(a1)
            sim = [elem for elem in SIM if elem < 1]
            sim_tensor = torch.tensor(sim)
            sim_average = torch.mean(sim_tensor)
            sim_average = sim_average.item()
            record_dict['sim'].append(sim_average)
        for idx, data in enumerate(dl):
            ecg_lead1 = data[:,0,:]
            ecg_lead2 = data[:,1:12,:]
            real_ecg_lead1 = ecg_lead1.unsqueeze(1)
            real_ecg_lead2 = ecg_lead2
            real_ecg_lead1 = real_ecg_lead1.to(device)
            real_ecg_lead2 = real_ecg_lead2.to(device)

            y_hat, mean, logvar = model(real_ecg_lead1)
            loss = loss_fn(real_ecg_lead2, y_hat, mean, logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if idx % 50 == 0:
                record_dict['g_loss_list'].append(loss.detach().cpu().item())
                print(
                    "[Epoch %d/%d] [Batch %d/%d] g_loss: %f"
                    % (i, epoch, idx, len(dataloader),torch.mean(loss)))
        if i % 2 == 0:
            show_generated_lead(real_ecg_lead1, real_ecg_lead2, y_hat, 0)
        if i % 20 == 0:
            with open('model/GAN1_group_vae/data.json', 'w') as f:
                json.dump(record_dict, f)
            torch.save(model, "./model/GAN1_group_vae/Gparams_.pkl")

def train_lstm(device, dl, model,epoch):
    os.makedirs("./model/GAN1_group_lstm")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.000001)
    for i in range(1,epoch,1):
        if i % 10 ==0:
            SIM = []
            for i, data in enumerate(dataloader_verify):
                ecg_lead1 = data[:, 0, :]
                ecg_lead2 = data[:, 1:12, :]
                # Configure input
                real_ecg_lead1 = ecg_lead1.unsqueeze(1).to(device)
                real_ecg_lead2 = ecg_lead2.to(device)
                g_ecg_lead = generator(real_ecg_lead1)
                g_ecg_lead = g_ecg_lead.detach()
                for x in range(1, 12):
                    a = CC2(g_ecg_lead, real_ecg_lead2, lead=x)
                    a1 = sum(a) / len(a)
                    SIM.append(a1)
            sim = [elem for elem in SIM if elem < 1]
            sim_tensor = torch.tensor(sim)
            sim_average = torch.mean(sim_tensor)
            sim_average = sim_average.item()
            record_dict['sim'].append(sim_average)
        for idx, data in enumerate(dl):
            ecg_lead1 = data[:,0,:]
            ecg_lead2 = data[:,1:12,:]
            real_ecg_lead1 = ecg_lead1.unsqueeze(1)
            real_ecg_lead2 = ecg_lead2
            real_ecg_lead1 = real_ecg_lead1.to(device)
            real_ecg_lead2 = real_ecg_lead2.to(device)

            y_hat= model(real_ecg_lead1)
            loss = loss_function(real_ecg_lead2, y_hat)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if idx % 50 == 0:
                record_dict['g_loss_list'].append(loss.detach().cpu().item())
                print(
                    "[Epoch %d/%d] [Batch %d/%d] g_loss: %f"
                    % (i, epoch, idx, len(dataloader),torch.mean(loss)))
        if i % 2 == 0:
            show_generated_lead(real_ecg_lead1, real_ecg_lead2, y_hat, 0)
        if i % 20 == 0:
            with open('model/GAN1_group_lstm/data.json', 'w') as f:
                json.dump(record_dict, f)
            torch.save(model, "./model/GAN1_group_lstm/Gparams_.pkl")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DCGAN')
    parser.add_argument('--train_epoch', type=int, default=300, help='epoch number of testing')
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--lr1", type=float, default=1e-4, help="adam: learning rate")
    parser.add_argument("--lr2", type=float, default=1e-4, help="adam: learning rate")
    parser.add_argument('--gpu_list', type=str, default='1,2,3', help='gpu index')
    parser.add_argument('--type', type=str, default='MLP', help='Review model')
    args = parser.parse_args()
    print(args)
    g_loss_list,d_loss_list,d_real_list,d_fake_list,mse,epoch_gloss_list,sim = [],[],[],[],[],[],[]
    keys = ['g_loss_list', 'd_loss_list', 'd_real_list', 'd_fake_list', 'mse', 'epoch_gloss_list', 'sim']
    record_dict = {
        'g_loss_list': g_loss_list,
        'd_loss_list': d_loss_list,
        'd_real_list': d_real_list,
        'd_fake_list': d_fake_list,
        'mse': mse,
        'epoch_gloss_list': epoch_gloss_list,
        'sim': sim
    }

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_list
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.type == 'GAN':
        generator = Generator_gan(in_channels=1, out_channels=11)
        discriminator = Discriminator_gan(in_channels=11)
        generator = generator.double()
        discriminator = discriminator.double()
        # Initialize weights
        generator = generator.to(device)
        discriminator = discriminator.to(device)
        generator = DataParallel(generator)
        discriminator = DataParallel(discriminator)
        # generator = torch.load('model/GAN1_group_cpsc2018individual_2.0/Gparams_.pkl')
        # discriminator = torch.load('model/GAN1_group_cpsc2018individual_2.0/Dparams_.pkl')
        # with open('model/GAN1_group_cpsc2018individual_2.0/data.json', 'r') as f:
        #     record_dict = json.load(f)

        train_dataset = data_load('/home/zhanzhehui_min/Engineering_project/lead_transformed/CPSC2018_path_traingan')
        verify_dataset = data_load('/home/zhanzhehui_min/Engineering_project/lead_transformed/CPSC2018_path_testgan')
        dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        dataloader_verify = DataLoader(verify_dataset, batch_size=args.batch_size, shuffle=True)
        optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0005, betas=(0, 0.9), weight_decay=1e-9)
        optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0, 0.9), weight_decay=1e-9)
    elif args.type =='VAE_CNN':
        generator = VAE()
        generator = generator.double()
        generator = generator.to(device)
        generator_vae = DataParallel(generator)
        train_dataset = data_load('/home/zhanzhehui_min/Engineering_project/lead_transformed/CPSC2018_path_train3.0')
        verify_dataset = data_load('/home/zhanzhehui_min/Engineering_project/lead_transformed/CPSC2018_path_test3.0')
        dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        dataloader_verify = DataLoader(verify_dataset, batch_size=args.batch_size, shuffle=True)
        train_VAE(device, dataloader, generator_vae, args.train_epoch)
    elif args.type == 'MLP':
        print('mistake')
