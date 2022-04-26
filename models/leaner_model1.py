# PSNR: 24.442471048355102 in 10 epochs
# train_model(model, noisy_imgs1.float(), noisy_imgs2.float(), nn.MSELoss(), torch.optim.Adam(model.parameters()), 500, 10)
# SGD diverges apparently and gives NaN as loss 
# decresing mini_batch size to 100 improves a lot (leaner_model1_2) --> PSNR: 25.28 after 7 epochs
# decresing mini_batch size to 50 improves  --> PSNR: 25.41 after 10 epochs with betas = (0.9, 0.999) (leaner_model1_3) and PSNR: 25.48 after 10 with betas = (0.9, 0.99) (model not saved)
# increasing learning rate to 0.01 makes it diverge after a few epochs --> error -> inf
# decreasing mini_batch size to 25 improves to PSNR of 25.50 and terminates in 7 minutes on colab leaving betas (0.9, 0.99) (leaner_model1_4)

class LeanerModel(torch.nn.Module):
    def __init__(self, transposed_conv=False):
        super().__init__()
        self.enc_conv0 = nn.Conv2d(3, 16, (3, 3), padding='same')
        self.enc_conv1 = nn.Conv2d(16, 32, (3, 3), padding='same')
        self.enc_conv3 = nn.Conv2d(32, 32, (3, 3), padding='same')
                  # concatenation
        self.upsample2 = nn.UpsamplingNearest2d(scale_factor=2)
                  # concatenatio
        self.dec_conv2a = nn.Conv2d(64, 64, (3, 3), padding='same')
        self.dec_conv2b = nn.Conv2d(64, 64, (3, 3), padding='same')
        self.upsample1 = nn.UpsamplingNearest2d(scale_factor=2)
                  # concatenation
        self.dec_conv1a = nn.Conv2d(67, 32, (3, 3), padding='same')
        self.dec_conv1b = nn.Conv2d(32, 16, (3, 3), padding='same')
        self.dec_conv1c = nn.Conv2d(16, 3, (3, 3), padding='same')
                  #what does linear activation mean (output unchanged?)

    def forward(self, x):
      input = x.clone()
      x = F.leaky_relu(self.enc_conv0(x), negative_slope=0.1)
      x = F.leaky_relu(self.enc_conv1(x), negative_slope=0.1)
      pool1 = F.max_pool2d(x, 2) #32x16x16
      x = F.leaky_relu(self.enc_conv3(pool1), negative_slope=0.1)
      #pool3 = F.max_pool2d(x, 2) #48x8x8
      #x = self.upsample2(x) #48x32x32
      x = torch.cat((x, pool1), dim=1)#64x16x16
      x = F.leaky_relu(self.dec_conv2a(x), negative_slope=0.1)
      #x = F.leaky_relu(self.dec_conv2b(x), negative_slope=0.1)
      x = self.upsample1(x)  #64x32x32
      x = torch.cat((x, input), dim=1) #67x32x32
      x = F.leaky_relu(self.dec_conv1a(x), negative_slope=0.1)
      x = F.leaky_relu(self.dec_conv1b(x), negative_slope=0.1)
      x = self.dec_conv1c(x)
      return x