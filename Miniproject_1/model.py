import torch
import torch.nn as nn
import torch.nn.functional as F


class bn_lrelu(nn.Module):
    def __init__(self, input_ch, activation="Relu"):
        super().__init__()
        self.bn = nn.BatchNorm2d(input_ch)
        if activation == "Relu":
            self.relu = nn.ReLU()
        elif activation == "Leaky_Relu":
            self.relu = nn.LeakyReLU(negative_slope=0.1)
        elif activation == "PRelu":
            self.relu = nn.PReLU()

    def forward(self, inputs):
        x = self.bn(inputs)
        x = self.relu(x)
        return x

class DnCNN(torch.nn.Module):
    def __init__(self, activation="PRelu", encoder_ch=32):
        super().__init__()
        """Encoder 1"""
        self.conv1 = nn.Conv2d(3, 64, (3, 3), padding='same')
        self.bnlr1 = bn_lrelu(64)
        self.conv2 = nn.Conv2d(64, 64, (3, 3), padding='same')
        self.bnlr2 = bn_lrelu(64)
        self.conv3 = nn.Conv2d(64, 64, (3, 3), padding='same')
        self.bnlr3 = bn_lrelu(64)
        self.conv4 = nn.Conv2d(64, 64, (3, 3), padding='same')
        self.bnlr4 = bn_lrelu(64)
        self.conv5 = nn.Conv2d(64, 64, (3, 3), padding='same')
        self.bnlr5 = bn_lrelu(64)
        self.conv6 = nn.Conv2d(64, 64, (3, 3), padding='same')
        self.bnlr6 = bn_lrelu(64)
        self.output = nn.Conv2d(64, 3, (3, 3), padding='same')

    def forward(self, inputs):
        """Remove the skip steps to try the simple DcCNN"""
        x = self.conv1(inputs)
        x = self.bnlr1(x)
        x = self.conv2(x)
        x = self.bnlr2(x)
        x = self.conv3(x)
        x = self.bnlr3(x)
        x = self.conv4(x)
        x = self.bnlr4(x)
        x = self.conv5(x)
        x = self.bnlr5(x)
        x = self.conv6(x)
        x = self.bnlr6(x)
        x = self.output(x)

        return x + inputs

class Model():
    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = DnCNN().to(device=self.device)
        self.optimizer = torch.optim.Adam(params= self.model.parameters(), lr=0.003535281772231024, betas=(0.88, 0.999), weight_decay=4.936803349899099e-07)
        self.criterion = nn.MSELoss().to(device=self.device)

    def predict(self, test_input):
        test_input = test_input.float().to(self.device)
        self.model.eval()
        with torch.no_grad():
            output = torch.zeros(test_input.shape, device=self.device, dtype=torch.float)
            for i in range(4):
                # output += self.model(test_input.rot90(i, [2, 3]).view(-1, 3, 32, 32)).rot90(-i, [2, 3])
                output += self.model(test_input.rot90(i, [2, 3]).view(test_input.shape)).rot90(-i, [2, 3])

            for i in range(4):
                # noisy_val_filp = input.flip(3)
                output += self.model(test_input.flip(3).rot90(i, [2, 3]).view(test_input.shape)).rot90(-i, [2, 3]).flip(3)
            output = output / 8

        return output.clamp(min=0, max=255)

    def load_pretrained_model( self ):
        from pathlib import Path
        model_path = Path(__file__).parent / "bestmodel.pth"
        self.model.load_state_dict(torch.load(model_path))

    def train(self , train_input , train_target , num_epochs=10):
        mini_batch_size = 50
        self.model.train()
        # train_input, train_target = train_input.float().to(device), train_target.float().to(device)
        for e in range(num_epochs):
            for b in range(0, train_input.size(0), mini_batch_size):
                output = self.model(train_input.narrow(0, b, mini_batch_size).float().to(self.device))
                loss = self.criterion(output, train_target.narrow(0, b, mini_batch_size).float().to(self.device))
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


