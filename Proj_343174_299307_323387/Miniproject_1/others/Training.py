# If you want to experiment with Madgrad and Adabelief please install them using the following lines and uncomment
# the corresponding lines in the code:
# ! pip install madgrad
# ! pip install adabelief-pytorch==0.2.0

import torch
import torch.nn as nn
import torch.nn.functional as F
from models import *
from losses import *
import numpy as np
import matplotlib.pyplot as plt
# from madgrad import MADGRAD
# from adabelief_pytorch import AdaBelief
import pickle

# Uncomment the model you desire to tune
model_name = "DnCNN"
# model_name = "DnCNN skip"
# model_name = "DnCNN noens"
# model_name = "Deep ResUnet"
# model_name = "UNet"
# model_name = "Shallow Unet"
# model_name = "Parallel Unet"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def compute_psnr(x, y, max_range=1.0):
    assert x.shape == y.shape and x.ndim == 4
    return 20 * torch.log10(torch.tensor(max_range)) - 10 * torch.log10(((x-y) ** 2).mean((1,2,3))).mean()

# Load dataset
if torch.cuda.is_available():
  noisy_imgs1, noisy_imgs2 = torch.load('train_data.pkl', map_location=torch.device("cuda"))
  noisy_val, clean_val = torch.load('val_data.pkl', map_location=torch.device("cuda"))
else:
  noisy_imgs1, noisy_imgs2 = torch.load('train_data.pkl')
  noisy_val, clean_val = torch.load('val_data.pkl')

# Map the values to [0,1] for larger networks to avoid nan values in the loss
if model_name in ["Deep ResUnet", "UNet", "Shallow Unet", "Parallel Unet"]:
    noisy_imgs1, noisy_imgs2 = noisy_imgs1.float() / 255.0, noisy_imgs2.float() / 255.0
    noisy_val, clean_val = noisy_val.float() / 255.0, clean_val.float() / 255.0

# Construct a valdiation set and a test set
noisy_val_validation, noisy_val_test = noisy_val[:400], noisy_val[400:]
clean_val_validation, clean_val_test = clean_val[:400], clean_val[400:]

# Define the self-ensamble function
def ensamble(model, inputs):
    output = torch.zeros(inputs.shape, dtype=torch.float).to("cuda")
    for i in range(4):
        output += model(inputs.rot90(i, [2,3]).float()).rot90(-i, [2,3])
    for i in range(4):
        noisy_val_filp = inputs.flip(3)
        output += model(noisy_val_filp.rot90(i, [2,3]).float()).rot90(-i, [2,3]).flip(3)
    return output/8


# Define the training function
def train_model(model, train_input, train_target, criterion, optimizer, mini_batch_size=4, epochs=500, normalize=False, self_ensamble=False):
    if normalize:
        mu, std = train_input.mean(), train_input.std()
        train_input.sub_(mu).div_(std)
    model.to(device)
    train_input, train_target = train_input.to(device), train_target.to(device)
    loss_history = []
    val_psnr = []
    for e in range(epochs):
        avg_loss = 0
        model.train()
        for b in range(0, train_input.size(0), mini_batch_size):
            output = model(train_input.narrow(0, b, mini_batch_size))
            # loss = criterion(output, train_target.narrow(0, b, mini_batch_size), epochs, e)
            loss = criterion(output, train_target.narrow(0, b, mini_batch_size))
            avg_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss_history.append((avg_loss / (train_input.size(0) / mini_batch_size)).cpu().detach().numpy())
        model.eval()
        with torch.no_grad():
            print(f"Loss at {e} is {avg_loss / (train_input.size(0) / mini_batch_size)}")
            if self_ensamble:
                output = ensamble(model, noisy_val)
            else:
                output = model(noisy_val.float())
            print(output.shape)
            print(f"PSNR: {compute_psnr(output, clean_val, max_range=255)}")
            val_psnr.append(compute_psnr(output, clean_val, max_range=255).cpu().detach().numpy())
    with torch.no_grad():
        if self_ensamble:
            output_test = ensamble(model, noisy_val_test)
        else:
            output_test = model(noisy_val_test.float())
        test_psnr = compute_psnr(output_test, clean_val_test, max_range=255)  #.clamp(min=0, max=255)

    return loss_history, val_psnr, test_psnr.cpu().detach().numpy()

# Define the training function for the case of the L0 loss
def train_model_l0(model, train_input, train_target, criterion, optimizer, mini_batch_size=4, epochs=500, normalize=False, self_ensamble=False):
    if normalize:
        mu, std = train_input.mean(), train_input.std()
        train_input.sub_(mu).div_(std)
    model.to(device)
    train_input, train_target = train_input.to(device), train_target.to(device)
    loss_history = []
    val_psnr = []
    for e in range(epochs):
        avg_loss = 0
        model.train()
        for b in range(0, train_input.size(0), mini_batch_size):
            output = model(train_input.narrow(0, b, mini_batch_size))
            loss = criterion(output, train_target.narrow(0, b, mini_batch_size), epochs, e)
            # loss = criterion(output, train_target.narrow(0, b, mini_batch_size))
            avg_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss_history.append((avg_loss / (train_input.size(0) / mini_batch_size)).cpu().detach().numpy())
        model.eval()
        with torch.no_grad():
            print(f"Loss at {e} is {avg_loss / (train_input.size(0) / mini_batch_size)}")
            if self_ensamble:
                output = ensamble(model, noisy_val)
            else:
                output = model(noisy_val.float())
            print(output.shape)
            print(f"PSNR: {compute_psnr(output, clean_val, max_range=255)}")
            val_psnr.append(compute_psnr(output, clean_val, max_range=255).cpu().detach().numpy())
    with torch.no_grad():
        if self_ensamble:
            output_test = ensamble(model, noisy_val_test)
        else:
            output_test = model(noisy_val_test.float())
        test_psnr = compute_psnr(output_test, clean_val_test, max_range=255)  #.clamp(min=0, max=255)

    return loss_history, val_psnr, test_psnr.cpu().detach().numpy()

# Initialize the model
model_DnCNN = DnCNN(activation='Relu', encoder_ch=64).to("cuda")
model_DnCNN_skip = DnCNN_skip(activation='Leaky_Relu', encoder_ch=64).to("cuda")
model_DnCNN_noens = DnCNN(activation='Relu', encoder_ch=32).to("cuda")
model_ParallelUNet = ParallelUNet(activation='PRelu', encoder_ch=64).to("cuda")
model_ResUnet_incr_ch = ResUnet_incr_ch(activation='Leaky_Relu').to("cuda")
model_UNet_lean = UNet_lean(activation='PRelu').to("cuda")
model_UNet_paper = UNet_paper(activation='Leaky_Relu').to("cuda")
# optimizer = MADGRAD(model.parameters(), lr=0.0005436922138131781, weight_decay=2.2202653045989276e-08, momentum=0.88)

# optmize the model, change it depending on the optimal hyperparameters for each model
optimizer = torch.optim.Adam(params= model_DnCNN.parameters(), lr=0.0031, betas=(0.875, 0.991), weight_decay=0.0000016800238576037557)
loss_history, val_psnr, test_psnr = train_model(model_DnCNN, noisy_imgs1.float(), noisy_imgs2.float(), torch.nn.MSELoss(), optimizer, mini_batch_size=25, epochs=30, self_ensamble=True)


# Uncomment the following to train the models and get average results after 5 independent runs of 30 epochs
# model = DnCNN(activation="Relu", encoder_ch=64)
# # optimizer = MADGRAD(model.parameters(), lr=0.0005436922138131781, weight_decay=2.2202653045989276e-08, momentum=0.88)
# optimizer = torch.optim.Adam(params= model.parameters(), lr=0.0031, betas=(0.875, 0.991), weight_decay=0.0000016800238576037557)
# loss_history, val_psnr, test_psnr = train_model(model, noisy_imgs1.float(), noisy_imgs2.float(), torch.nn.MSELoss(), optimizer, mini_batch_size=25, epochs=30, self_ensamble=True)
# losses = np.array([loss_history])
# psnrs = np.array([val_psnr])
# test_psnrs = [test_psnr]
# for i in range(4):
#     model = DnCNN(activation="Relu", encoder_ch=64)
#     # optimizer = MADGRAD(model.parameters(), lr=0.0005436922138131781, weight_decay=2.2202653045989276e-08, momentum=0.88)
#     optimizer = torch.optim.Adam(params= model.parameters(), lr=0.0031, betas=(0.875, 0.991), weight_decay=0.0000016800238576037557)
#     loss_history, val_psnr, test_psnr = train_model(model, noisy_imgs1.float(), noisy_imgs2.float(), torch.nn.MSELoss(), optimizer, mini_batch_size=25, epochs=30, self_ensamble=True)
#     losses = np.concatenate((losses, np.array([loss_history])), axis=0)
#     psnrs = np.concatenate((psnrs, np.array([val_psnr])), axis=0)
#     test_psnrs.append(test_psnr)

def plot_denoised_images():
    with torch.no_grad():
        # for i in range(1, 400, 20):
        for i in range(30, 210, 20):
            image_number = i
            model_DnCNN.eval()
            model_DnCNN_skip.eval()
            model_DnCNN_noens.eval()
            model_ParallelUNet.eval()
            model_ResUnet_incr_ch.eval()
            model_UNet_lean.eval()
            model_UNet_paper.eval()
            fig, axs = plt.subplots(1, 8, figsize=(30,900))
            axs[0].imshow((np.squeeze(noisy_val[image_number].cpu().permute(1 , 2 , 0 ))))
            axs[0].set_title("Noisy psnr: {:.2f}".format(compute_psnr(noisy_val[image_number].float().view(1, 3, 32, 32), clean_val[image_number].view(1, 3, 32, 32), max_range=255)))
            axs[1].imshow(  (np.squeeze(clean_val[image_number].rot90(2, [1,2]).rot90(-2, [1,2]).cpu().permute(1 , 2 , 0 ))))
            axs[1].set_title("Clean")
            axs[2].imshow((np.squeeze(ensamble(model_DnCNN, noisy_val[image_number].float().view(1, 3, 32, 32)).int().view(3, 32, 32).cpu().permute(1 , 2 , 0 ))))
            axs[2].set_title("DnCNN psnr: {:.2f}".format(compute_psnr(ensamble(model_DnCNN, noisy_val[image_number].float().view(1, 3, 32, 32)), clean_val[image_number].view(1, 3, 32, 32), max_range=255)))
            axs[3].imshow((np.squeeze(ensamble(model_DnCNN_skip, noisy_val[image_number].float().view(1, 3, 32, 32)).int().view(3, 32, 32).cpu().permute(1 , 2 , 0 ))))
            axs[3].set_title("DnCNN skip psnr: {:.2f}".format(compute_psnr(ensamble(model_DnCNN_skip, noisy_val[image_number].float().view(1, 3, 32, 32)), clean_val[image_number].view(1, 3, 32, 32), max_range=255)))
            axs[4].imshow((np.squeeze((ensamble(model_ParallelUNet, (noisy_val[image_number].float()/255.0).view(1, 3, 32, 32))*255).int().view(3, 32, 32).cpu().permute(1 , 2 , 0 ))))
            axs[4].set_title("Parallel U-Net psnr: {:.2f}".format(compute_psnr(ensamble(model_ParallelUNet, (noisy_val[image_number].float()/255.0).view(1, 3, 32, 32)), (clean_val[image_number].float()/255.0).view(1, 3, 32, 32), max_range=1)))
            axs[5].imshow((np.squeeze((ensamble(model_ResUnet_incr_ch, (noisy_val[image_number].float()/255.0).view(1, 3, 32, 32))*255).int().view(3, 32, 32).cpu().permute(1 , 2 , 0 ))))
            axs[5].set_title("Deep ResUnet psnr: {:.2f}".format(compute_psnr(ensamble(model_ResUnet_incr_ch, (noisy_val[image_number]/255).float().view(1, 3, 32, 32)), (clean_val[image_number].float()/255.0).view(1, 3, 32, 32), max_range=1)))
            axs[6].imshow((np.squeeze((ensamble(model_UNet_lean, (noisy_val[image_number].float()/255.0).view(1, 3, 32, 32))*255).int().view(3, 32, 32).cpu().permute(1 , 2 , 0 ))))
            axs[6].set_title("Shallow Unet psnr: {:.2f}".format(compute_psnr(ensamble(model_UNet_lean, (noisy_val[image_number]/255).float().view(1, 3, 32, 32)), (clean_val[image_number].float()/255.0).view(1, 3, 32, 32), max_range=1)))
            axs[7].imshow((np.squeeze((ensamble(model_UNet_paper, (noisy_val[image_number].float()/255.0).view(1, 3, 32, 32))*255).int().view(3, 32, 32).cpu().permute(1 , 2 , 0 ))))
            axs[7].set_title("Unet psnr: {:.2f}".format(compute_psnr(ensamble(model_UNet_paper, (noisy_val[image_number]/255).float().view(1, 3, 32, 32)), (clean_val[image_number].float()/255.0).view(1, 3, 32, 32), max_range=1)))

# Uncomment the following lines to plot the denoised output AFTER FITTING EVERY MODEL
# plot_denoised_images()