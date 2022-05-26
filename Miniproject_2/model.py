import torch
from .others.utils import *


# torch.set_grad_enabled(False)

class Module(object):
    """
    Base module class that is inherited from by every component of the framework.
    """

    def __init__(self, device="cpu"):
        self.input = torch.empty(0)
        self.output = torch.empty(0)
        self.device = device

    def forward(self, *input: torch.Tensor):
        raise NotImplementedError

    def backward(self, *grad_wrt_output: torch.Tensor):
        raise NotImplementedError

    def zero_grad(self):
        pass

    def set_params(self, val):
        pass

    def param(self):
        return []


class Activation(Module):
    """
    Base activation class that is extended by activation implementations.
    Each activation function must define its forward method (the function itself) and its derivative.
    All activation functions share the same gradient structure.
    """

    def __init__(self, act, act_derivative, device="cpu"):
        super().__init__(device=device)
        self.activation = act
        self.derivative = act_derivative

    def forward(self, input: torch.Tensor):
        self.input = input
        return self.activation(self.input)

    def backward(self, grad_wrt_output: torch.Tensor):
        return grad_wrt_output * self.derivative(self.input)


class ReLU(Activation):
    """
    ReLU activation function.
    It introduces a non-linearity by returning the max(0, x) element-wise.
    """

    def __init__(self, device="cpu"):
        def relu(input: torch.Tensor):
            """
            Clamps all negative values to 0.
            """
            return input.clamp(0, None)

        def relu_prime(input: torch.Tensor):
            """
            Derivative of ReLU is 0 for negative values and 1 for positive values.
            """
            res = input.where(input <= 0, torch.empty(input.size()).fill_(1).to(self.device)).clamp(0, None)
            return res

        super().__init__(relu, relu_prime, device=device)


class Sigmoid(Activation):
    """
    Sigmoid activation function.
    It introduces a non-linearity by returning the sigmoid of the given tensor.
    """

    def __init__(self, device="cpu"):
        def logistic_func(input: torch.Tensor):
            """
            Applies the sigmoid function to the input element-wise.

            :param input: input Tensor
            :return: e^(input) / ( e^(input) + 1 )
            """
            return input.sigmoid()

        def logistic_func_prime(input: torch.Tensor):
            """
            Derivative of sigmoid is S(x) * (1 - S(x))

            :param input: input Tensor
            :return: S(x) * (1 - S(x)) element-wise
            """
            return input.sigmoid() * (1 - input.sigmoid())

        super().__init__(logistic_func, logistic_func_prime, device=device)


class Sequential(Module):
    """
    Basic container to implement simple sequential networks.
    Allows to specify an arbitrary number of modules and saves internally the
    list of all parameters to be passed to an optimizer.
    """

    def __init__(self, *modules):
        super().__init__()
        self.modules = modules
        self.params = []

    def zero_grad(self):
        """
        Invokes the zero_grad method of all methods to clear out the grad field.
        """
        for module in self.modules:
            module.zero_grad()

    def forward(self, input: torch.Tensor):
        """
        Computes the result of the forward pass of the network by invoking sequentially all modules
        with the output of the previous one.

        :param input: input Tensor to the network
        :return: result of the forward pass
        """
        for module in self.modules:
            input = module.forward(input)
        return input

    def backward(self, grad_wrt_output: torch.Tensor):
        """
        Computes the result of the backward pass of the network by invoking sequentially all modules
        in reverse order to provide the derivative of the loss with respect to the inputs of
        module i as input to the backward pass of module i - 1.

        :param input: derivative of the predicted output with respect to the input to the loss
        :return: result of the backward pass (derivative of loss with respect to input tensor to network)
        """
        for i in reversed(range(len(self.modules))):
            grad_wrt_output = self.modules[i].backward(grad_wrt_output)
        return grad_wrt_output

    def param(self):
        """
        :return: returns the parameter list of the whole network.
        """
        self.params = []
        for module in self.modules:
            self.params.extend(module.param())
        return self.params


class MSE(Module):
    """
    Mean-Square Error loss implementation.
    """

    def __init__(self, device="cpu"):
        super().__init__(device=device)

    def forward(self, *input: torch.Tensor):
        """
        The loss is computed as the mean of the squared difference between target and input
        :param input: tuple of (input, target)
        :return: MSE loss
        """
        assert len(input) == 2
        input, target = input[0], input[1]
        self.input = (target - input)  # NxCxHxW
        return (self.input ** 2).mean()

    def backward(self, *grad_wrt_output: torch.Tensor):
        """
        :return: returns the derivative of the loss with respect to (input - target)
        """
        return -(2 / (self.input.size(0) * self.input.size(1) * self.input.size(2) * self.input.size(3))) * self.input


class Upsampling(Module):
    """
    Wrapper around NNUpsampling and normal convolution.
    Since no dilation support has been added to Conv2d, no dilation is supported for Upsampling either
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=0, padding=0, scale_factor=2, device="cpu"):
        super().__init__(device=device)
        self.upsamp = NNUpsampling(scale_factor, device=device)
        self.conv = Conv2d(in_channels, out_channels, kernel_size,padding=padding, device=device)

    def zero_grad(self):
        self.upsamp.zero_grad()
        self.conv.zero_grad()

    def param(self):
        return self.conv.param()

    def set_params(self, val):
        self.conv.set_params(val)

    def forward(self, input: torch.Tensor):
        input = self.upsamp.forward(input)
        return self.conv.forward(input)

    def backward(self, grad_wrt_output: torch.Tensor):
        res = self.conv.backward(grad_wrt_output)
        return self.upsamp.backward(res)


class NNUpsampling(Module):
    """
    Implementation of a Nearest Neighbor Upsampling.
    """

    def __init__(self, scale_factor=2, device="cpu"):
        super().__init__(device=device)
        self.scale_factor = scale_factor

    def zero_grad(self):
        pass

    def forward(self, input: torch.Tensor):
        """
        Returns the upsampled version of the input tensor according to nearest-neighbor policy.
        :param input: (N x C x H x W) input tensor
        :return: (N x C x H*factor x W*factor) upsampled output
        """
        assert len(input.shape) == 4
        self.input = input

        # Duplicates and interleaves the tensor horizontally and the vertically wrt to image plane
        #                   123         112233
        #       123         123         112233
        #       456 -->     456 -->     445566
        #       789         456         445566
        #                   789         778899
        #                   789         778899

        self.output = self.input.repeat_interleave(self.scale_factor, dim=-2).repeat_interleave(self.scale_factor,
                                                                                                dim=-1)
        return self.output

    # Sum of patches of gradients
    def backward(self, grad_wrt_output: torch.Tensor):
        """
        Returns the gradient of the output wrt to input by summing across patches of size (factor, factor).
        :param grad_wrt_output: gradient of loss wrt output
        :return: gradient of loss wrt to input to upsampling
        """
        unfolded = torch.nn.functional.unfold(grad_wrt_output, kernel_size=self.scale_factor, stride=self.scale_factor)
        res = unfolded.view(unfolded.size(0), grad_wrt_output.size(1), self.scale_factor ** 2, -1).sum(dim=2).view(
            unfolded.size(0), grad_wrt_output.size(1), grad_wrt_output.size(2) // self.scale_factor,
                                                       grad_wrt_output.size(3) // self.scale_factor)
        return res


class Conv2d(Module):
    """
    Implementation of a Convolutional Layer of arbitrary kernel, stride, and padding.
    Padding of 'same' is computed before applying any stride and aims at keeping unchanged the output without considering
    potential effects of a strided convolution.
    NOTE: **Kernel is square**
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, device="cpu"):
        assert stride > 0 and kernel_size > 0
        kernel_size = (kernel_size, kernel_size)
        if padding == "same":
            assert stride == 1
            padding = (kernel_size[0] - 1) // 2
        super().__init__(device=device)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # init.kaiming_uniform_
        fan_in = in_channels * kernel_size[0] * kernel_size[1]
        gain = (2.0 / (1 + (5 ** 0.5) ** 2)) ** 0.5
        std_weights = (3.0 ** 0.5) * (gain / (fan_in ** 0.5))
        std_bias = 1 / ((fan_in)**0.5)
        self.weight = torch.empty((out_channels, in_channels, *kernel_size)).uniform_(-std_weights, std_weights).to(self.device)
        self.weight_derivatives = torch.empty(self.weight.size()).fill_(0).to(self.device)
        self.bias = torch.empty(out_channels).uniform_(-std_bias, std_bias).to(self.device)
        self.bias_derivatives = torch.empty(self.bias.size()).fill_(0).to(self.device)

        self.params = [[self.weight, self.weight_derivatives], [self.bias, self.bias_derivatives]]

    def set_params(self, val):
        assert len(val) == 2
        assert val[0][0].shape == self.weight.shape and val[1][0].shape == self.bias.shape
        self.weight = val[0][0].clone().to(self.device)
        self.bias = val[1][0].clone().to(self.device)
        self.params = [[self.weight, self.weight_derivatives], [self.bias, self.bias_derivatives]]

    def cross_correlation(self, input: torch.Tensor, kernel: torch.Tensor, stride=1, padding=0, dilation=1):
        kernel_size = (kernel.size(-2), kernel.size(-1))
        unfolded = torch.nn.functional.unfold(input.squeeze(1), kernel_size=kernel_size, stride=stride, padding=padding,
                                              dilation=dilation)
        wxb = kernel.view(kernel.size(0), kernel.size(1), kernel.size(2), 1, -1) @ unfolded.view(input.size(0), input.size(1),
                                                                                                 input.size(2), -1,
                                                                                                 unfolded.size(-1))
        return wxb.view(input.size(0), kernel.size(1), input.size(2),
                        (input.shape[-2] - dilation * (kernel_size[0] - 1) - 1 + 2 * padding) // stride + 1,
                        (input.shape[-1] - dilation * (kernel_size[1] - 1) - 1 + 2 * padding) // stride + 1)

    def forward(self, input: torch.Tensor):
        """
        Computes the forward pass with the internal weights of the given input
        :param input: tensor of shape (N x C_in x H x W)
        :return: input convolved with internal weights + bias
        """
        assert input.size(1) == self.in_channels
        self.input = input

        res = self.cross_correlation(input.unsqueeze(1), self.weight.unsqueeze(0), stride=self.stride, padding=self.padding)
        self.output = res.sum(2) + self.bias.view(1, -1, 1, 1)
        return self.output

    def zero_grad(self):
        """
        Zeroes the gradients of the weights and biases.
        """
        self.weight_derivatives.fill_(0)
        self.bias_derivatives.fill_(0)

    def cross_correlation_backward(self, input: torch.Tensor, kernel: torch.Tensor, stride=1, padding=0, dilation=1):
        kernel_size = (kernel.size(-2), kernel.size(-1))
        unfolded = torch.nn.functional.unfold(input, kernel_size=kernel_size, stride=stride, padding=padding,
                                              dilation=dilation)
        wxb = kernel.view(kernel.size(0), kernel.size(1), kernel.size(2), 1, -1) @ unfolded.view(input.size(0),
                                                                                                 input.size(1), 1, -1,
                                                                                                 unfolded.size(-1))
        return wxb.view(input.size(0), kernel.size(1), kernel.size(2),
                        (input.shape[-2] - dilation * (kernel_size[0] - 1) - 1 + 2 * padding) // stride + 1,
                        (input.shape[-1] - dilation * (kernel_size[1] - 1) - 1 + 2 * padding) // stride + 1)

    def backward(self, grad_wrt_output: torch.Tensor):
        """
        Computes the gradient of the loss wrt to the input tensor as well as the accumulation of gradient of loss
        with respect to the internal weights and biases.
        :param grad_wrt_output: gradient of loss wrt to output tensor of the forward pass
        :return: gradient of loss wrt to input to the layer
        """
        self.bias_derivatives += grad_wrt_output.sum((0, 2, 3))
        dilated_grad = dilate_efficient(grad_wrt_output, factor=self.stride, device=self.device)
        res = self.cross_correlation(self.input.unsqueeze(1), dilated_grad.unsqueeze(2), padding=self.padding).sum(dim=0)#.squeeze()
        self.weight_derivatives += res
        res = self.cross_correlation_backward(dilated_grad, rot180(self.weight).unsqueeze(0),
                                            padding=(self.weight.size(2) - 1) // 2)
        return res.sum(dim=1).squeeze()

    def param(self):
        """
        :return: parameter list to allow for optimizer's step as tuple of [(weights, derivative_weights), (bias,
        derivative_bias)]
        """
        return self.params



class SGD:
    """
    Implementation of a Stochastic Gradient Descent Optimizer with no momentum.
    """

    def __init__(self, network_parameters, lr=0.01, momentum=0):
        self.network_parameters = network_parameters
        self.lr = lr
        self.momentum = momentum
        self.last_factor = [0 for _ in range(len(self.network_parameters))]

    def step(self):
        """
        Applies a step towards the line of maximum slope to all parameters of the network given their derivatives.
        Parameters are in the form (parameter, gradient_wrt_loss_of_parameter)
        """
        for i, p in enumerate(self.network_parameters):
            self.last_factor[i] = self.momentum * self.last_factor[i] + self.lr * p[1]
            p[0] -= self.last_factor[i]


class Model:
    def __init__(self) -> None:
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.model = Sequential(Conv2d(3, 16, padding=1, device=self.device),
                                ReLU(device=self.device),
                                Conv2d(16, 32, stride=2, padding=1,device=self.device),
                                ReLU(device=self.device),
                                Upsampling(32, 16, padding=1,device=self.device),
                                ReLU(device=self.device),
                                Conv2d(16, 3, padding=1,device=self.device),
                                Sigmoid(device=self.device))

        self.criterion = MSE()
        self.optimizer = SGD(self.model.param(), lr=0.00001, momentum=0.9)
        self.mini_batch_size = 50

    def load_pretrained_model(self) -> None:
        from pathlib import Path
        model_path = Path(__file__).parent / "bestmodel.pth"
        model = torch.load(model_path)
        for param in model:
            i, val = param
            if len(val) == 0:continue
            self.model.modules[i].set_params(val)
        self.optimizer = SGD(self.model.param(), lr = 0.00001, momentum=0.9)
        

    def train(self, train_input: torch.Tensor, train_target: torch.Tensor, num_epochs: int):
        train_input = train_input.to(self.device).float()
        train_target = train_target.to(self.device).float()
        for e in range(num_epochs):
            print(f"Epoch: {e}")
            avg_loss = 0
            for b in range(0, train_input.size(0), self.mini_batch_size):
                output = self.model.forward(train_input.narrow(0, b, self.mini_batch_size)) * 255
                loss = self.criterion.forward(output, train_target.narrow(0, b, self.mini_batch_size))
                avg_loss += loss.item()
                self.model.zero_grad()
                self.model.backward(self.criterion.backward())
                self.optimizer.step()
            print(avg_loss/(train_input.size(0) // self.mini_batch_size))

    def predict(self, test_input: torch.Tensor) -> torch.Tensor:
        test_input = test_input.to(self.device).float()
        # Check it is in the correct range
        return self.model.forward(test_input) * 255
