import torch


def rot180(input: torch.Tensor):
    """
    Rotates the input tensor by 180 degrees
    :param input: tensor to be rotated
    :return: rotated tensor of 180 degrees (vertical flip followed by horizontal flip)
    """
    factor = [i for i in range(len(input.shape))][-2:]
    return input.rot90(2, factor)


def dilate_efficient(input: torch.Tensor, factor: int):
    """
    Interleaves the input tensor's columns and rows with (factor - 1) zeros.
    Factor of 1 means no dilation.

    :param input: input tensor
    :param factor: number of zeros + 1 to interleave the input
    :return: interleaved tensor
    """
    assert factor > 0
    if factor == 1:
        return input
    dilated_size = (factor * input.size(-2), factor * input.size(-1))
    dilator = torch.empty(dilated_size).fill_(0).fill_diagonal_(1)[::factor]
    res = (input.mT @ dilator.reshape(1, -1, factor * input.size(-2))).mT @ dilator
    return res

