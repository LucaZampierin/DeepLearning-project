import torch.nn


class Module(object):
    def forward(self, input):
        raise NotImplementedError

    def backward(self, grad_wrt_output):
        raise NotImplementedError

    def param(self):
        return []


class ReLU(Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.clamp_min(input, 0)

    def backward(self, grad_wrt_output):
        grad_wrt_output[]


class Sigmoid(Module):
    def forward(self, *input):
        for x in input:
            x =
        return input

    def backward(self, *grad_wrt_output):
        for x in grad_wrt_output:
            x = 0 if x < 0 else 1
        return grad_wrt_output
