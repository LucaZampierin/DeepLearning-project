import torch
from torch.nn.functional import fold
from utils import *

# torch.set_grad_enabled(False)


class Module(object):
    """
    Base module class that is inherited from by every component of the framework.
    """
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
    """
    Base activation class that is extended by activation implementations.
    Each activation function must define its forward method (the function itself) and its derivative.
    All activation functions share the same gradient structure.
    """
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
    """
    ReLU activation function.
    It introduces a non-linearity by returning the max(0, x) element-wise.
    """
    def __init__(self):
        def relu(input: torch.Tensor):
            """
            Clamps all negative values to 0.
            """
            return input.clamp(0, None)

        def relu_prime(input: torch.Tensor):
            """
            Derivative of ReLU is 0 for negative values and 1 for positive values.
            """
            res = input.where(input <= 0, torch.empty(input.size()).fill_(1)).clamp(0, None)
            return res

        super().__init__(relu, relu_prime)


class Sigmoid(Activation):
    """
    Sigmoid activation function.
    It introduces a non-linearity by returning the sigmoid of the given tensor.
    """
    def __init__(self):
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

        super().__init__(logistic_func, logistic_func_prime)


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

        for module in self.modules:
            self.params.extend(module.param())

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
        return self.params


class MSE(Module):
    """
    Mean-Square Error loss implementation.
    """
    def __init__(self):
        super().__init__()

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


class SGD:
    """
    Implementation of a Stochastic Gradient Descent Optimizer with no momentum.
    """
    def __init__(self, network_parameters, lr=0.01):
        self.network_parameters = network_parameters
        self.lr = lr

    def step(self):
        """
        Applies a step towards the line of maximum slope to all parameters of the network given their derivatives.
        Parameters are in the form (parameter, gradient_wrt_loss_of_parameter)
        """
        for p in self.network_parameters:
            p[0] -= self.lr * p[1]


class Upsampling(Module):
    """
    Implementation of a Nearest Neighbor Upsampling.
    """
    def __init__(self, scale_factor=2):
        super().__init__()
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

        self.output = self.input.repeat_interleave(self.scale_factor, dim=-2).repeat_interleave(self.scale_factor, dim=-1)
        return self.output

    # Sum of patches of gradients
    def backward(self, grad_wrt_output: torch.Tensor):
        """
        Returns the gradient of the output wrt to input by summing across patches of size (factor, factor).
        :param grad_wrt_output: gradient of loss wrt output
        :return: gradient of loss wrt to input to upsampling
        """
        unfolded = torch.nn.functional.unfold(grad_wrt_output, kernel_size=self.scale_factor, stride=self.scale_factor)
        res = unfolded.view(unfolded.size(0), grad_wrt_output.size(1), self.scale_factor ** 2, -1).sum(dim=2).view(unfolded.size(0), grad_wrt_output.size(1), grad_wrt_output.size(2) // self.scale_factor, grad_wrt_output.size(3) // self.scale_factor)
        return res


class Conv2d(Module):
    """
    Implementation of a Convolutional Layer of arbitrary kernel, stride, and padding.
    Padding of 'same' is computed before applying any stride and aims at keeping unchanged the output without considering
    potential effects of a strided convolution.
    NOTE: **Kernel is square**
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        assert stride > 0 and kernel_size > 0
        kernel_size = (kernel_size, kernel_size)
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
        """
        Computes the forward pass with the internal weights of the given input
        :param input: tensor of shape (N x C_in x H x W)
        :return: input convolved with internal weights + bias
        """
        assert input.size(1) == self.in_channels
        self.input = input

        res = self.cross_correlation(input, self.weights.unsqueeze(0), stride=self.stride, padding=self.padding)
        self.output = res.sum(2) + self.biases.view(1, -1, 1, 1)
        return self.output

    def zero_grad(self):
        """
        Zeroes the gradients of the weights and biases.
        """
        self.weights_derivatives.fill_(0)
        self.biases_derivatives.fill_(0)

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

    def backward(self, grad_wrt_output: torch.Tensor):
        """
        Computes the gradient of the loss wrt to the input tensor as well as the accumulation of gradient of loss
        with respect to the internal weights and biases.
        :param grad_wrt_output: gradient of loss wrt to output tensor of the forward pass
        :return: gradient of loss wrt to input to the layer
        """
        self.biases_derivatives += grad_wrt_output.sum((0, 2, 3))
        dilated_grad = dilate_efficient(grad_wrt_output, factor=self.stride)
        res = self.cross_correlation_test(self.input, dilated_grad, padding=self.padding).sum(dim=0).squeeze()
        self.weights_derivatives += res
        res = self.cross_correlation_test_2(dilated_grad, rot180(self.weights).unsqueeze(0),
                                            padding=(self.weights.size(2) - 1) // 2)
        return res.sum(dim=1).squeeze()

    def param(self):
        """
        :return: parameter list to allow for optimizer's step as tuple of [(weights, derivative_weights), (bias,
        derivative_bias)]
        """
        return self.params


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

def validate(model, noise_img, ground_truth):
    psnr_tot = 0
    for i in range(noise_img.size(0)):
        denoised = model.forward(noise_img[i].view(1, 3, 32, 32))
        psnr_val = psnr(denoised, ground_truth[i]).item()
        psnr_tot += psnr_val
    psnr_tot /= noise_img.size(0)
    return psnr_tot


model_new = Sequential(Conv2d(3, 32, stride=2, padding=1),
                       ReLU(),
                       Conv2d(32, 3, stride=2, padding=1),
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

train_model_torch(model2, noisy_imgs1, noisy_imgs2, criterion, optimizer2, epochs=1, mini_batch_size=250)
with torch.no_grad():
    train_model(model_new, noisy_imgs1, noisy_imgs2, loss, optimizer_new, epochs=1, mini_batch_size=250)




# TODO: adapt to test.py
# TODO: finish documentation
# TODO: clean up code and separate modules
