import torch

from model import *


def train_model(model, train_input, train_target, criterion, optimizer, mini_batch_size=1, epochs=1):
    for e in range(epochs):
        print(f"Epoch: {e}")
        avg_loss = 0
        for b in range(0, train_input.size(0), mini_batch_size):
            output = model.forward(train_input.narrow(0, b, mini_batch_size)) #* 255
            # print(f"My Output: {output}")
            loss = criterion.forward(output, train_target.narrow(0, b, mini_batch_size))
            # print(f"My Loss: {loss}")
            avg_loss += loss.item()
            model.zero_grad()
            model.backward(criterion.backward())
            optimizer.step()
        print(avg_loss / (train_input.size(0) // mini_batch_size))
        print(validate(model, val_noisy.float(), val_clean.float()))


def train_model_torch(model, train_input, train_target, criterion, optimizer, mini_batch_size=1, epochs=500):
    for e in range(epochs):
        print(f"Epoch: {e}")
        avg_loss = 0
        for b in range(0, train_input.size(0), mini_batch_size):
            output = model(train_input.narrow(0, b, mini_batch_size))# * 255
            # print(f"Torch Output: {output}")
            loss = criterion(output, train_target.narrow(0, b, mini_batch_size))
            # print(f"Torch Loss: {loss}")
            avg_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(avg_loss / (train_input.size(0) // mini_batch_size))
        print(validate(model, val_noisy.float(), val_clean.float()))


def psnr(x, y, max_range=255):
    assert x.shape == y.shape and x.ndim == 4
    return 20 * torch.log10(torch.tensor(max_range)) - 10 * torch.log10(((x - y) ** 2).mean((1, 2, 3))).mean()

def validate(model, noise_img, ground_truth):
    psnr_tot = 0
    for i in range(noise_img.size(0)):
        denoised = model.forward(noise_img[i].view(1, 3, 32, 32)) #* 255
        psnr_val = psnr(denoised, ground_truth[i].unsqueeze(0)).item()
        psnr_tot += psnr_val
    psnr_tot /= noise_img.size(0)
    return psnr_tot


model_new = Sequential(Conv2d(3, 32, stride=2, padding=1),
                       ReLU(),
                       Conv2d(32, 3, stride=2, padding=1),
                       ReLU(),
                       NearestUpsampling(),
                       ReLU(),
                       NearestUpsampling(),
                       Sigmoid())

loss = MSE()
optimizer_new = SGD(model_new.param(), lr=0.00001)

model2 = torch.nn.Sequential(torch.nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=1), torch.nn.ReLU(),
                             torch.nn.Conv2d(32, 3, kernel_size=(3, 3), stride=(2, 2), padding=1), torch.nn.ReLU(),
                             torch.nn.UpsamplingNearest2d(scale_factor=2), torch.nn.ReLU(),
                             torch.nn.UpsamplingNearest2d(scale_factor=2), torch.nn.Sigmoid())

criterion = torch.nn.MSELoss()
optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.00001)

noisy_imgs1, noisy_imgs2 = torch.load('../train_data.pkl')  # 50000 x 3 x 32 x 32
val_noisy, val_clean = torch.load('../val_data.pkl')  # 50000 x 3 x 32 x 32
noisy_imgs1 = noisy_imgs1[:500].float()
noisy_imgs2 = noisy_imgs2[:500].float()
val_noisy = val_noisy[:100].float()
val_clean = val_clean[:100].float()

model_new.modules[0].weights[:] = model2._modules['0'].weight.clone()
model_new.modules[0].biases[:] = model2._modules['0'].bias.clone()
model_new.modules[2].weights[:] = model2._modules['2'].weight.clone()
model_new.modules[2].biases[:] = model2._modules['2'].bias.clone()

train_model_torch(model2, noisy_imgs1, noisy_imgs2, criterion, optimizer2, epochs=1, mini_batch_size=250)
with torch.no_grad():
    train_model(model_new, noisy_imgs1, noisy_imgs2, loss, optimizer_new, epochs=1, mini_batch_size=250)


'''
from model import Model

noisy_imgs1, noisy_imgs2 = torch.load('../train_data.pkl')  # 50000 x 3 x 32 x 32
val_noisy, val_clean = torch.load('../val_data.pkl')  # 50000 x 3 x 32 x 32
#noisy_imgs1 = noisy_imgs1.float().to("cuda")
#noisy_imgs2 = noisy_imgs2.float().to("cuda")
#val_noisy = val_noisy.float().to("cuda")
#val_clean = val_clean.float().to("cuda")
#model2 = torch.nn.Sequential(torch.nn.Conv2d(3, 8, kernel_size=(3, 3), stride=(2, 2), padding=1), torch.nn.ReLU(),
                             #torch.nn.Conv2d(8, 8, kernel_size=(3, 3), stride=(2, 2), padding=1), torch.nn.ReLU(),
                             #torch.nn.UpsamplingNearest2d(scale_factor=2), torch.nn.Conv2d(8, 8, kernel_size=(3, 3), padding=1),
                             #torch.nn.UpsamplingNearest2d(scale_factor=2), torch.nn.Conv2d(8, 3, kernel_size=(3, 3), padding=1), torch.nn.Sigmoid())

#model2 = model2.to("cuda")
#criterion = torch.nn.MSELoss()
#optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.000005)
model = Model()
#train_model_torch(model2, noisy_imgs1, noisy_imgs2, criterion, optimizer2, epochs=1, mini_batch_size=25)
with torch.no_grad():
    model.train(noisy_imgs1.float(), noisy_imgs2.float(), 10)
#print(validate(model, val_noisy.float(), val_clean.float()))
#print(validate(model2, val_noisy, val_clean))
#print(val_noisy[0])
#print(val_clean[0])
#print(model2(val_noisy[0].float()) * 255)
'''