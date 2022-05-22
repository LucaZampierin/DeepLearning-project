import torch
from torch.nn.functional import fold

# torch.set_grad_enabled(False)


class Module(object):
    def __init__(self):
        self.input = torch.empty(0)
        self.output = torch.empty(0)

    def forward(self, *input: torch.Tensor):
        raise NotImplementedError

    def backward(self, *grad_wrt_output: torch.Tensor):
        raise NotImplementedError

    def zero_grad(self):
        pass

    def param(self):
        return []


class Activation(Module):
    def __init__(self, act, act_derivative):
        super().__init__()
        self.activation = act
        self.derivative = act_derivative

    def forward(self, input: torch.Tensor):
        self.input = input
        return self.activation(self.input)

    def backward(self, grad_wrt_output: torch.Tensor):
        return grad_wrt_output * self.derivative(self.input)


class ReLU(Activation):
    def __init__(self):
        def relu(input):
            return input.clamp(0, None)

        def relu_prime(input):
            res = input.where(input <= 0, torch.empty(input.size()).fill_(1)).clamp(0, None)
            return res

        super().__init__(relu, relu_prime)


class Sigmoid(Activation):
    def __init__(self):
        def logistic_func(input):
            return input.sigmoid()

        def logistic_func_prime(input):
            # maybe do input.sigmoid() * (1 - input.sigmoid())
            return input.sigmoid() * (torch.empty(input.size()).fill_(1) - input.sigmoid())

        super().__init__(logistic_func, logistic_func_prime)


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules
        self.params = []

        for module in self.modules:
            self.params.extend(module.param())

    def zero_grad(self):
        for module in self.modules:
            module.zero_grad()

    def forward(self, input: torch.Tensor):
        for module in self.modules:
            input = module.forward(input)
        return input

    def backward(self, grad_wrt_output: torch.Tensor):
        for i in reversed(range(len(self.modules))):
            grad_wrt_output = self.modules[i].backward(grad_wrt_output)
        return grad_wrt_output

    def param(self):
        return self.params


class MSE(Module):
    def __init__(self):
        super().__init__()

    def forward(self, *input: torch.Tensor):
        assert len(input) == 2
        input, target = input[0], input[1]
        self.input = (target - input)  # NxCxHxW
        return (self.input ** 2).mean()

    def backward(self, *grad_wrt_output: torch.Tensor):
        # added terms to make it equal to pytorch
        return -(2 / (self.input.size(0) * self.input.size(1) * self.input.size(2) * self.input.size(3))) * self.input


class SGD:
    def __init__(self, network_parameters, lr=0.01):
        self.network_parameters = network_parameters
        self.lr = lr

    def step(self):
        '''Network Parameters can be a list of tuples (param tensor, grad tensor)'''
        for p in self.network_parameters:
            p[0] -= self.lr * p[1]


# Nearest Neighbour upsampling
class Upsampling(Module):
    def __init__(self, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor

    def zero_grad(self):
        pass

    def forward(self, input: torch.Tensor):
        self.input = input
        # supports input of arbitrary dimension
        output_size = [self.input.size(0), self.input.size(1)] + [int(float(input.size(i + 2)) * self.scale_factor) for
                                                                  i in range(self.input.dim() - 2)]
        self.output = torch.empty(output_size)

        for y in range(self.output.size(dim=3)):
            for x in range(self.output.size(dim=2)):
                self.output[:, :, x, y] = self.input[:, :, int(x / self.scale_factor), int(y / self.scale_factor)]
        return self.output

    # Sum of patches of gradients
    def backward(self, grad_wrt_output: torch.Tensor):
        res = torch.empty((grad_wrt_output.size(0), grad_wrt_output.size(1),
                           grad_wrt_output.size(2) // self.scale_factor,
                           grad_wrt_output.size(3) // self.scale_factor)).fill_(0)
        for c in range(grad_wrt_output.size(1)):
            unfolded = torch.nn.functional.unfold(
                grad_wrt_output[:, c, :, :].view(grad_wrt_output.size(0), 1, grad_wrt_output.size(2),
                                                 grad_wrt_output.size(3)), kernel_size=self.scale_factor,
                stride=self.scale_factor)
            res[:, c, :, :] += unfolded.sum(dim=1).view(grad_wrt_output.size(0), 1,
                                                        grad_wrt_output.size(2) // self.scale_factor,
                                                        grad_wrt_output.size(3) // self.scale_factor).squeeze()
        return res



class Conv2d_new(Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=1, padding=0):
        assert stride > 0 and 0 < kernel_size[0] == kernel_size[1]
        if padding == "same":
            assert stride == 1
            padding = (kernel_size[0] - 1) // 2
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Glorot Initialization
        std_weights = 6.0 / (in_channels + out_channels)
        std_bias = 1
        self.weights = torch.empty((out_channels, in_channels, *kernel_size)).uniform_(-std_weights, std_weights)
        self.weights_derivatives = torch.empty(self.weights.size()).fill_(0)
        self.biases = torch.empty(out_channels).uniform_(-std_weights, std_weights)
        self.biases_derivatives = torch.empty(self.biases.size()).fill_(0)

        self.params = [[self.weights, self.weights_derivatives], [self.biases, self.biases_derivatives]]

    def cross_correlation(self, input: torch.Tensor, kernel: torch.Tensor, stride=1, padding=0, dilation=1):
        kernel_size = (kernel.size(-2), kernel.size(-1))
        unfolded = torch.nn.functional.unfold(input, kernel_size=kernel_size, stride=stride, padding=padding,
                                              dilation=dilation)
        wxb = kernel.view(kernel.size(0), kernel.size(1), kernel.size(2), 1, -1) @ unfolded.view(input.size(0), 1, input.size(1), -1, unfolded.size(-1))
        # samples x 1-channel x image size
        return wxb.view(input.size(0), kernel.size(1), input.size(1),
                        (input.shape[2] - dilation * (kernel_size[0] - 1) - 1 + 2 * padding) // stride + 1,
                        (input.shape[3] - dilation * (kernel_size[1] - 1) - 1 + 2 * padding) // stride + 1)

    def forward(self, input: torch.Tensor):
        assert input.size(1) == self.in_channels
        self.input = input

        res = self.cross_correlation(input, self.weights.unsqueeze(0), stride=self.stride, padding=self.padding)
        self.output = res.sum(2) + self.biases.view(1, -1, 1, 1)
        return self.output

    def zero_grad(self):
        self.weights_derivatives.fill_(0)
        self.biases_derivatives.fill_(0)

    # @profile
    def cross_correlation_test(self, input: torch.Tensor, kernel: torch.Tensor, stride=1, padding=0, dilation=1):
        kernel_size = (kernel.size(-2), kernel.size(-1))
        unfolded = torch.nn.functional.unfold(input, kernel_size=kernel_size, stride=stride, padding=padding,
                                              dilation=dilation)
        wxb = kernel.view(kernel.size(0), kernel.size(1), 1, 1, -1) @ unfolded.view(input.size(0), 1, input.size(1), -1,
                                                                                    unfolded.size(2))
        # samples x 1-channel x image size
        return wxb.view(input.size(0), kernel.size(1), input.size(1),
                        (input.shape[2] - dilation * (kernel_size[0] - 1) - 1 + 2 * padding) // stride + 1,
                        (input.shape[3] - dilation * (kernel_size[1] - 1) - 1 + 2 * padding) // stride + 1)

    # @profile
    def cross_correlation_test_2(self, input: torch.Tensor, kernel: torch.Tensor, stride=1, padding=0, dilation=1):
        kernel_size = (kernel.size(-2), kernel.size(-1))
        unfolded = torch.nn.functional.unfold(input, kernel_size=kernel_size, stride=stride, padding=padding,
                                              dilation=dilation)
        wxb = kernel.view(kernel.size(0), kernel.size(1), kernel.size(2), 1, -1) @ unfolded.view(input.size(0),
                                                                                                 input.size(1), 1, -1,
                                                                                                 unfolded.size(2))
        # samples x 1-channel x image size
        return wxb.view(input.size(0), kernel.size(1), kernel.size(2),
                        (input.shape[2] - dilation * (kernel_size[0] - 1) - 1 + 2 * padding) // stride + 1,
                        (input.shape[3] - dilation * (kernel_size[1] - 1) - 1 + 2 * padding) // stride + 1)

    # @profile
    # N x OUT_channels x H x W -> OUT x
    def backward(self, grad_wrt_output: torch.Tensor):
        self.biases_derivatives += grad_wrt_output.sum((0, 2, 3))
        dilated_grad = dilate_efficient(grad_wrt_output, factor=self.stride)
        res = self.cross_correlation_test(self.input, dilated_grad, padding=self.padding).sum(dim=0).squeeze()
        self.weights_derivatives += res
        res = self.cross_correlation_test_2(dilated_grad, rot180(self.weights).unsqueeze(0),
                                            padding=(self.weights.size(2) - 1) // 2)
        return res.sum(dim=1).squeeze()

    def param(self):
        return self.params


def rot180(input: torch.Tensor):
    factor = [i for i in range(len(input.shape))][-2:]
    return input.rot90(2, factor)



# Dilation of 1 means no dilation
# Interleaves each column and row with (factor - 1) zeros
def dilate_efficient(input: torch.Tensor, factor: int):
    assert factor > 0
    if factor == 1:
        return input
    dilated_size = (factor * input.size(-2), factor * input.size(-1))
    dilator = torch.empty(dilated_size).fill_(0).fill_diagonal_(1)[::factor]
    res = (input.mT @ dilator.reshape(1, -1, factor * input.size(-2))).mT @ dilator
    return res


def train_model(model, train_input, train_target, criterion, optimizer, mini_batch_size=1, epochs=1):
    for e in range(epochs):
        print(f"Epoch: {e}")
        avg_loss = 0
        for b in range(0, train_input.size(0), mini_batch_size):
            output = model.forward(train_input.narrow(0, b, mini_batch_size))
            # print(f"My Output: {output}")
            loss = criterion.forward(output, train_target.narrow(0, b, mini_batch_size))
            # print(f"My Loss: {loss}")
            avg_loss += loss.item()
            model.zero_grad()
            model.backward(criterion.backward())
            optimizer.step()
        print(avg_loss)


def train_model_torch(model, train_input, train_target, criterion, optimizer, mini_batch_size=1, epochs=500):
    for e in range(epochs):
        print(f"Epoch: {e}")
        avg_loss = 0
        for b in range(0, train_input.size(0), mini_batch_size):
            output = model(train_input.narrow(0, b, mini_batch_size))
            # print(f"Torch Output: {output}")
            loss = criterion(output, train_target.narrow(0, b, mini_batch_size))
            # print(f"Torch Loss: {loss}")
            avg_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(avg_loss)


def psnr(denoised, ground_truth):
    mse = torch.mean((denoised - ground_truth) ** 2)
    return -10 * torch.log10(mse + 10 ** -8)


#test = torch.ones((2, 2, 2, 2))
#print(dilate_efficient_stride2(test, factor=2))
#print(dilate_efficient_stride2(test, factor=3))
'''
input = torch.arange(1, 65, dtype=torch.float32).view(2, 2, 4, 4).requires_grad_()
target = torch.empty((2, 2, 8, 8)).fill_(2)

up = torch.nn.UpsamplingNearest2d(scale_factor=2)
criterion = torch.nn.MSELoss()
output = up(input)
print(f"Torch_out: {output}")

up2 = Upsampling()
criterion2 = MSE()
output2 = up2.forward(input)
print(f"Our_out: {output2}")

loss = criterion(output, target)
print(f"Torch_loss: {loss}")
loss2 = criterion2.forward(output2, target)
print(f"Our_loss: {loss2}")

print(torch.autograd.grad(loss, input))
print(up2.backward(criterion2.backward()))
'''


#
def validate(model, noise_img, ground_truth):
    psnr_tot = 0
    for i in range(noise_img.size(0)):
        denoised = model.forward(noise_img[i].view(1, 3, 32, 32))
        psnr_val = psnr(denoised, ground_truth[i]).item()
        psnr_tot += psnr_val
    psnr_tot /= noise_img.size(0)
    return psnr_tot


'''
model_new = Sequential(Conv2d_new(3, 32, stride=2, padding=1),
                       ReLU(),
                       Conv2d_new(32, 3, stride=2, padding=1),
                       ReLU(),
                       Upsampling(),
                       ReLU(),
                       Upsampling(),
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

train_model_torch(model2, noisy_imgs1, noisy_imgs2, criterion, optimizer2, epochs=1, mini_batch_size=50)
with torch.no_grad():
    train_model(model_new, noisy_imgs1, noisy_imgs2, loss, optimizer_new, epochs=1, mini_batch_size=50)

train_input = torch.empty((4, 1, 8, 8)).fill_(1).requires_grad_()
train_target = torch.empty((4, 1, 8, 8)).fill_(2)
model.modules[0].weights[:] = model2._modules['0'].weight.clone()
model.modules[0].biases[:] = model2._modules['0'].bias.clone()
model.modules[2].weights[:] = model2._modules['2'].weight.clone()
model.modules[2].biases[:] = model2._modules['2'].bias.clone()
train_model(model, train_input, train_target, loss, optimizer, epochs=10, mini_batch_size=2)
train_model_torch(model2, train_input, train_target, criterion, optimizer2, epochs=10, mini_batch_size=2)

'''
# print(model.forward(train_input.clone()))
# print(model2(train_input.clone()))
'''

criterion = torch.nn.MSELoss()
sig = torch.nn.ReLU()
loss = MSE()
sig2 = ReLU()
train_input = torch.empty((2, 1, 4, 4)).fill_(1).requires_grad_()
train_target = torch.empty((2, 1, 4, 4)).fill_(2)
l = loss.forward(train_input.detach(), train_target.detach())
l2 = criterion(train_input, train_target)
print(l)
print(loss.backward())
print(l2)
print(torch.autograd.grad(l2, train_input))

#s = sig2.forward(train_input)
#s2 = sig(train_input)
#print(s)
#print(s2)

#print(sig2.backward(torch.empty(train_target.size()).fill_(1)))
#print(torch.autograd.grad(s2.sum(), train_input))

'''

# TODO: get rid of loop upsampling
# TODO: adapt to test.py

conv2 = torch.nn.Conv2d(5, 1, kernel_size=(3, 3), stride=(1, 1), padding=1)
convnew = Conv2d_new(5, 1,padding=1, stride=1)
convnew.weights = conv2.weight.clone()
convnew.biases = conv2.bias.clone()

criterion = torch.nn.MSELoss()
lossnew = MSE()
train_input = torch.empty((5, 5, 4, 4)).fill_(1).requires_grad_()
train_target = torch.empty((5, 1, 4, 4)).fill_(2)

l2 = criterion(conv2(train_input), train_target)
#l2.backward()
print(l2)
print(torch.autograd.grad(l2, train_input))

with torch.no_grad():
    l_new = lossnew.forward(convnew.forward(train_input), train_target)
    print(l_new)
    print(convnew.backward(lossnew.backward()))


'''
print(x)
print(cross_correlation(x, kernel, 2, 1, 1))
'''

'''
input = torch.empty((3, 5)).normal_().requires_grad_()
relu = torch.nn.ReLU()
mine_relu = ReLU()
act_relu = relu(input)
act_mine = mine_relu.forward(input)

print(f"Activations: \n\t{act_relu} \n\t{act_mine}")

act_relu.backward()
print(act_relu.grad)

#target = torch.empty((3, 5)).normal_().requires_grad_()
'''
