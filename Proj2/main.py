import torch
from torch import empty, cat, arange
from torch.nn.functional import fold, unfold

#torch.set_grad_enabled(False)


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
            return input.where(input > 0, torch.empty(input.size()).fill_(1)).clamp(0, None)

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
        self.input = (target - input) #NxCxHxW
        return (self.input ** 2).mean()

    def backward(self, *grad_wrt_output: torch.Tensor):
        return -(2/self.input.size(0)) * self.input


class SGD:
    def __init__(self, network_parameters, lr=0.01):
        self.network_parameters = network_parameters
        self.lr = lr

    def step(self):
        '''Network Parameters can be a list of tuples (param tensor, grad tensor)'''
        for p in self.network_parameters:
            p[0] -= self.lr * p[1]


class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=1, padding=0, xavier=True):
        assert stride > 0 and kernel_size[0] > 0 and kernel_size[1] > 0
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # TODO: xavier initialization?
        std_weights = 1
        std_bias = 1
        self.weights = torch.empty((out_channels, in_channels, *kernel_size)).normal_(0, std_weights)
        self.weights_derivatives = torch.empty(self.weights.size(0), self.weights.size(1)).fill_(0)
        self.biases = torch.empty(out_channels).normal_(0, std_bias)
        self.biases_derivatives = torch.empty(self.biases.size()).fill_(0)

        self.params = [(self.weights, self.weights_derivatives), (self.biases, self.biases_derivatives)]

    '''
        Expects the input of Nx1xHxW
    '''
    def cross_correlation(self, input: torch.Tensor, kernel: torch.Tensor):
        unfolded = torch.nn.functional.unfold(input, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        wxb = kernel.view(1, -1) @ unfolded
        # samples x 1-channel x image size
        return wxb.view(input.size(0), 1, (input.shape[2] - self.kernel_size[0] + 2*self.padding) // self.stride + 1, (input.shape[3] - self.kernel_size[1] + 2*self.padding) // self.stride + 1)

    def forward(self, input: torch.Tensor):
        assert input.size(1) == self.in_channels
        output_size = ((input.shape[2] - self.kernel_size[0] + 2*self.padding) // self.stride + 1, (input.shape[3] - self.kernel_size[1] + 2*self.padding) // self.stride + 1)
        self.output = torch.empty((input.size(0), self.out_channels, *output_size)).fill_(0)
        self.input = input

        for i in range(self.out_channels):
            for j in range(self.in_channels):
                res = self.cross_correlation(input[:,j].view(input.size(0), 1, input.size(2), -1), self.weights[i, j])
                slice = self.output[:,i].view(self.output.size(0), 1, self.output.size(2), self.output.size(3))
                slice += res
            self.output[:,i] += self.biases[i]
        return self.output

    def zero_grad(self):
        self.weights_derivatives.fill_(0)
        self.biases_derivatives.fill_(0)

    def backward(self, grad_wrt_output: torch.Tensor):
        self.biases_derivatives = grad_wrt_output.clone()
        dl_dx = torch.empty((self.input.size())).fill_(0)

        for i in range(self.out_channels):
            for j in range(self.in_channels):
                self.weights_derivatives[i, j] = self.cross_correlation(self.input[:, j].view(self.input.size(0), 1, self.input.size(2), -1), grad_wrt_output[:, i].view(self.input.size(0), 1, grad_wrt_output.size(2), -1))
                dl_dx += self.cross_correlation(grad_wrt_output[i].view(self.input.size(0), 1,self.input.size(2), -1), rot180(self.weights[i, j]))
        return dl_dx

    def param(self):
        return self.params


def rot180(input: torch.Tensor):
    assert len(input.shape) == 2
    return input.rot90(2, [0, 1])


def train_model(model, train_input, train_target, criterion, optimizer, mini_batch_size=1, epochs=1):
    for e in range(epochs):
      avg_loss = 0
      for b in range(0, train_input.size(0), mini_batch_size):
          output = model.forward(train_input.narrow(0, b, mini_batch_size))
          loss = criterion.forward(output, train_target.narrow(0, b, mini_batch_size))
          avg_loss += loss.item()
          model.zero_grad()
          model.backward(criterion.backward())
          optimizer.step()


model = Sequential(Conv2d(1, 1, padding=1), ReLU())
loss = MSE()
optimizer = SGD(model.param())

train_input = torch.empty((1, 1, 4, 4)).normal_()
train_target = torch.empty((1, 1, 4, 4)).fill_(2)
train_model(model, train_input, train_target, loss, optimizer, epochs=10)

'''
x = torch.empty((2, 2, 4, 4)).fill_(2)
conv2 = torch.nn.Conv2d(2, 2, kernel_size=(3, 3), stride=(2, 2))
print(conv2(x))
conv = Conv2d(2, 2,padding=0,stride=2)
conv.weights = conv2.weight
conv.biases = conv2.bias
print(conv.forward(x))
'''
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



