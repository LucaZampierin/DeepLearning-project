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
            return input.sigmoid() * (torch.empty(input.size()).fill_(1) - input.sigmoid())

        super().__init__(logistic_func, logistic_func_prime)


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules
        self.params = []

        for module in self.modules:
            self.params.extend(module.param())

    def forward(self, input: torch.Tensor):
        for module in self.modules:
            input = module.forward(input)
        return input

    def backward(self, grad_wrt_output: torch.Tensor):
        grad_wrt_output = grad_wrt_output[0]
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
        self.input = ((target - input) ** 2).mean()
        return self.input

    def backward(self, *grad_wrt_output: torch.Tensor):
        return 2 * self.input


class SGD:
    def __init__(self, network_parameters, lr=0.01):
        self.network_parameters = network_parameters
        self.lr = lr

    def step(self):
        '''Network Parameters can be a list of tuples (param tensor, grad tensor)'''
        for p in self.network_parameters:
            p[0] -= self.lr * p[1]


class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=0):
        assert stride > 0 and kernel_size > 0
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation

        # TODO: xavier initialization
        self.weights = torch.empty((out_channels, in_channels, kernel_size, kernel_size)).normal_()
        self.weights_derivatives = torch.empty(self.weights.size())
        # TODO: finish this
        self.biases = torch.empty((out_channels, ))
        self.biases_derivatives = torch.empty(self.biases.size())

        self.params = [(self.weights, self.weights_derivatives), (self.biases, self.biases_derivatives)]

    def forward(self, *input: torch.Tensor):
        pass

    def backward(self, *grad_wrt_output: torch.Tensor):
        pass

    def param(self):
        return self.params


container = Sequential(ReLU(), Sigmoid())
container2 = torch.nn.Sequential(torch.nn.ReLU(), torch.nn.Sigmoid())
input = torch.empty((3, 5)).normal_().requires_grad_()

print(f"Results:")
print(container.forward(input))
print(container2(input))


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



