# FOr the tuning we used the optuna library, please install this package to run the hyperparameters by running:
# ! pip install optuna

# If you want to experiment with Madgrad and Adabelief please install them using the following lines and uncomment
# the corresponding lines in the code:
# ! pip install madgrad
# ! pip install adabelief-pytorch==0.2.0

import torch
import torch.nn as nn
import torch.nn.functional as F
from models import *
from losses import *
# from madgrad import MADGRAD
# from adabelief_pytorch import AdaBelief
import optuna
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
def cross_validation_train_model(model, train_input, train_target, criterion, optimizer, mini_batch_size=4, epochs=500, normalize=False, self_ensamble=False):
    if torch.cuda.is_available():
        model.to("cuda")
        train_input, train_target = train_input.to("cuda"), train_target.to("cuda")
    if normalize:
        mu, std = train_input.mean(), train_input.std()
        train_input.sub_(mu).div_(std)

    for e in range(epochs):
        avg_loss = 0
        model.train()
        for b in range(0, train_input.size(0), mini_batch_size):
            output = model(train_input.narrow(0, b, mini_batch_size))
            loss = criterion(output, train_target.narrow(0, b, mini_batch_size))
            avg_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            print(f"Loss at {e} is {avg_loss / (train_input.size(0) / mini_batch_size)}")
            if self_ensamble:
                output = ensamble(model, noisy_val_validation)
            else:
                output = model(noisy_val_validation.float())

            if model_name in ["Deep ResUnet", "UNet", "Shallow Unet", "Parallel Unet"]:
                psnr_val = compute_psnr(output, clean_val_validation, max_range=1)
            else:
                psnr_val = compute_psnr(output, clean_val_validation, max_range=255)
            print(f"PSNR: {psnr_val}")

    return psnr_val

# Define the training function for the case of the L0 loss
def cross_validation_train_model_l0(model, train_input, train_target, criterion, optimizer, mini_batch_size=4, epochs=500, normalize=False, self_ensamble=False):
    if torch.cuda.is_available():
        model.to("cuda")
        train_input, train_target = train_input.to("cuda"), train_target.to("cuda")
    if normalize:
        mu, std = train_input.mean(), train_input.std()
        train_input.sub_(mu).div_(std)

    for e in range(epochs):
        avg_loss = 0
        model.train()
        for b in range(0, train_input.size(0), mini_batch_size):
            output = model(train_input.narrow(0, b, mini_batch_size))
            loss = criterion(output, train_target.narrow(0, b, mini_batch_size), epochs, e)
            avg_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            print(f"Loss at {e} is {avg_loss / (train_input.size(0) / mini_batch_size)}")
            if self_ensamble:
                output = ensamble(model, noisy_val_validation)
            else:
                output = model(noisy_val_validation.float())

            if model_name in ["Deep ResUnet", "UNet", "Shallow Unet", "Parallel Unet"]:
                psnr_val = compute_psnr(output, clean_val_validation, max_range=1)
            else:
                psnr_val = compute_psnr(output, clean_val_validation, max_range=255)
            print(f"PSNR: {psnr_val}")

    return psnr_val

# HYperparameter tuninig
def initialize_validation(model, parameters):
    if parameters['optimizer'] == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), betas=(parameters['momentum'], 0.999), lr=parameters['learning_rate'], weight_decay=parameters['weight_decay'])
    # elif parameters['optimizer'] == "Madgrad":
    #     optimizer = MADGRAD(model.parameters(), lr=parameters['learning_rate'], weight_decay=parameters['weight_decay'], momentum=parameters['momentum'])
    # elif parameters['optimizer'] == "AdaBelief":
    #     optimizer = AdaBelief(model.parameters(), lr=parameters['learning_rate'], eps=1e-8, weight_decay=parameters['weight_decay'], betas=(parameters['momentum'],0.999), weight_decouple = False, rectify = False)
    elif parameters['optimizer'] == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=parameters['learning_rate'], weight_decay=parameters['weight_decay'], momentum=parameters['momentum'])

    if parameters['loss'] == "L0":
        val_psnr  = cross_validation_train_model_l0(model, noisy_imgs1.float(), noisy_imgs2.float(), L0Loss(2), optimizer, mini_batch_size=parameters['batchsize'], epochs=10, self_ensamble=True)
    elif parameters['loss'] == "L1":
        val_psnr  = cross_validation_train_model(model, noisy_imgs1.float(), noisy_imgs2.float(), torch.nn.L1Loss(), optimizer, mini_batch_size=parameters['batchsize'], epochs=10, self_ensamble=True)
    elif parameters['loss'] == "L2":
        val_psnr  = cross_validation_train_model(model, noisy_imgs1.float(), noisy_imgs2.float(), torch.nn.MSELoss(), optimizer, mini_batch_size=parameters['batchsize'], epochs=10, self_ensamble=True)
    elif parameters['loss'] == "HDR":
        val_psnr  = cross_validation_train_model(model, noisy_imgs1.float(), noisy_imgs2.float(), HDRLoss(), optimizer, mini_batch_size=parameters['batchsize'], epochs=10, self_ensamble=True)

    return val_psnr

"""Hyperparameter tuning code: Optuna"""
def objective(trial):
    params = {
              'learning_rate': trial.suggest_float('learning_rate', low = 1e-4, high =5e-3, step = 0.0005),
              'optimizer': trial.suggest_categorical("optimizer", ["Adam", "SGD"]),#, "Madgrad", "AdaBelief"]),
              'momentum': trial.suggest_float('momentum', low=0.85, high=0.95, step=0.025),
              'beta2': trial.suggest_float('beta2', low=0.979, high=0.999, step=0.002),
              'activation': trial.suggest_categorical('activation', ["Relu", "Leaky_Relu", "PRelu"]),
              'loss': trial.suggest_categorical('loss', ["L2"]),#"L0", "L1", ]),#, "HDR"]),
              'encoder_channels': trial.suggest_int('encoder_channels', 32, 64, 32),
              'weight_decay': trial.suggest_loguniform('weight_decay', 1e-9, 1e-3),
              'batchsize': trial.suggest_int('batchsize', 25, 50, 25),
              }

    if model_name == "DnCNN":
        model = DnCNN(activation=params['activation'], encoder_ch=params['encoder_channels'])
    elif model_name == "DnCNN skip":
        model = DnCNN_skip(activation=params['activation'], encoder_ch=params['encoder_channels'])
    elif model_name == "Deep ResUnet":
        model = ResUnet_incr_ch(activation=params['activation'])  # gives nan as loss
    elif model_name == "UNet":
        model = UNet_paper(params['activation'])
    elif model_name == "Shallow UNet":
        model = UNet_lean(activation=params['activation'])
    elif model_name == "Parallel Unet":
        model = ParallelUNet(activation=params['activation'], encoder_ch=params['encoder_channels'])

    val_psnr = initialize_validation(model, params)

    return val_psnr

# Initizialize the tuning
study = optuna.create_study(study_name='DnCNN', direction="maximize", sampler=optuna.samplers.TPESampler(seed=559))
study.optimize(objective, n_trials=10)
with open('drive/My Drive/DnCNN.pkl','wb') as f:
    pickle.dump(study, f)
for i in range(10):
  with open(r"drive/My Drive/DnCNN.pkl", "rb") as input_file:
    study = pickle.load(input_file)
  study.optimize(objective, n_trials=10)
  with open("drive/My Drive/DnCNN.pkl","wb") as f:
    pickle.dump(study, f)