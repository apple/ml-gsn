import math

import torch
from torch import nn
import torch.nn.functional as F

from .op import FusedLeakyReLU, fused_leaky_relu, upfirdn2d


class PixelNorm(nn.Module):
    """Pixel normalization layer.

    Normalizes feature maps along the pixel dimension. Used to prevent explosion of pixel magnitude.
    """

    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


class ConstantInput(nn.Module):
    """Constant input layer.

    A learned constant input used to start the generation.

    Args:
    ----
    channel: int
        Number of channels.
    size: int
        Spatial dimension of constant input.

    """

    def __init__(self, channel, size=4, ndim=2):
        super().__init__()

        res = (size,) * ndim
        self.input = nn.Parameter(torch.randn(1, channel, *res))

    def forward(self, input):
        batch = input.shape[0]
        # out = self.input.repeat(batch, 1, 1, 1)
        out = torch.repeat_interleave(self.input, batch, dim=0)
        return out


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k


class Blur(nn.Module):
    """Blur layer.

    Applies a blur kernel to input image using finite impulse response filter. Blurring feature maps after
    convolutional upsampling or before convolutional downsampling helps produces models that are more robust to
    shifting inputs (https://richzhang.github.io/antialiased-cnns/). In the context of GANs, this can provide
    cleaner gradients, and therefore more stable training.

    Args:
    ----
    kernel: list, int
        A list of integers representing a blur kernel. For exmaple: [1, 3, 3, 1].
    pad: tuple, int
        A tuple of integers representing the number of rows/columns of padding to be added to the top/left and
        the bottom/right respectively.
    upsample_factor: int
        Upsample factor.

    """

    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)

        self.register_buffer("kernel", kernel)
        self.pad = pad

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, pad=self.pad)
        return out


class Upsample(nn.Module):
    """Upsampling layer.

    Perform upsampling using a blur kernel.

    Args:
    ----
    kernel: list, int
        A list of integers representing a blur kernel. For exmaple: [1, 3, 3, 1].
    factor: int
        Upsampling factor.

    """

    def __init__(self, kernel=[1, 3, 3, 1], factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel) * (factor ** 2)
        self.register_buffer("kernel", kernel)

        p = kernel.shape[0] - factor
        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2
        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=self.factor, down=1, pad=self.pad)
        return out


class Downsample(nn.Module):
    """Downsampling layer.

    Perform downsampling using a blur kernel.

    Args:
    ----
    kernel: list, int
        A list of integers representing a blur kernel. For exmaple: [1, 3, 3, 1].
    factor: int
        Downsampling factor.

    """

    def __init__(self, kernel=[1, 3, 3, 1], factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel)
        self.register_buffer("kernel", kernel)

        p = kernel.shape[0] - factor
        pad0 = (p + 1) // 2
        pad1 = p // 2
        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=1, down=self.factor, pad=self.pad)
        return out


class EqualLinear(nn.Module):
    """Linear layer with equalized learning rate.

    During the forward pass the weights are scaled by the inverse of the He constant (i.e. sqrt(in_dim)) to
    prevent vanishing gradients and accelerate training. This constant only works for ReLU or LeakyReLU
    activation functions.

    Args:
    ----
    in_channel: int
        Input channels.
    out_channel: int
        Output channels.
    bias: bool
        Use bias term.
    bias_init: float
        Initial value for the bias.
    lr_mul: float
        Learning rate multiplier. By scaling weights and the bias we can proportionally scale the magnitude of
        the gradients, effectively increasing/decreasing the learning rate for this layer.
    activate: bool
        Apply leakyReLU activation.

    """

    def __init__(self, in_channel, out_channel, bias=True, bias_init=0, lr_mul=1, activate=False):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_channel, in_channel).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel).fill_(bias_init))
        else:
            self.bias = None

        self.activate = activate
        self.scale = (1 / math.sqrt(in_channel)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activate:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)
        else:
            out = F.linear(input, self.weight * self.scale, bias=self.bias * self.lr_mul)
        return out

    def __repr__(self):
        return f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})"


class EqualConv2d(nn.Module):
    """2D convolution layer with equalized learning rate.

    During the forward pass the weights are scaled by the inverse of the He constant (i.e. sqrt(in_dim)) to
    prevent vanishing gradients and accelerate training. This constant only works for ReLU or LeakyReLU
    activation functions.

    Args:
    ----
    in_channel: int
        Input channels.
    out_channel: int
        Output channels.
    kernel_size: int
        Kernel size.
    stride: int
        Stride of convolutional kernel across the input.
    padding: int
        Amount of zero padding applied to both sides of the input.
    bias: bool
        Use bias term.

    """

    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_channel, in_channel, kernel_size, kernel_size))
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))
        else:
            self.bias = None

    def forward(self, input):
        out = F.conv2d(input, self.weight * self.scale, bias=self.bias, stride=self.stride, padding=self.padding)
        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},"
            f" {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})"
        )


class EqualConvTranspose2d(nn.Module):
    """2D transpose convolution layer with equalized learning rate.

    During the forward pass the weights are scaled by the inverse of the He constant (i.e. sqrt(in_dim)) to
    prevent vanishing gradients and accelerate training. This constant only works for ReLU or LeakyReLU
    activation functions.

    Args:
    ----
    in_channel: int
        Input channels.
    out_channel: int
        Output channels.
    kernel_size: int
        Kernel size.
    stride: int
        Stride of convolutional kernel across the input.
    padding: int
        Amount of zero padding applied to both sides of the input.
    output_padding: int
        Extra padding added to input to achieve the desired output size.
    bias: bool
        Use bias term.

    """

    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, output_padding=0, bias=True):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(in_channel, out_channel, kernel_size, kernel_size))
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))
        else:
            self.bias = None

    def forward(self, input):
        out = F.conv_transpose2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
        )
        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[0]}, {self.weight.shape[1]},'
            f' {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})'
        )


class ConvLayer2d(nn.Sequential):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size=3,
        upsample=False,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        bias=True,
        activate=True,
    ):
        assert not (upsample and downsample), 'Cannot upsample and downsample simultaneously'
        layers = []

        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            layers.append(
                EqualConvTranspose2d(
                    in_channel, out_channel, kernel_size, padding=0, stride=2, bias=bias and not activate
                )
            )
            layers.append(Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor))

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            layers.append(Blur(blur_kernel, pad=(pad0, pad1)))
            layers.append(
                EqualConv2d(in_channel, out_channel, kernel_size, padding=0, stride=2, bias=bias and not activate)
            )

        if (not downsample) and (not upsample):
            padding = kernel_size // 2

            layers.append(
                EqualConv2d(in_channel, out_channel, kernel_size, padding=padding, stride=1, bias=bias and not activate)
            )

        if activate:
            layers.append(FusedLeakyReLU(out_channel, bias=bias))

        super().__init__(*layers)


class ConvResBlock2d(nn.Module):
    """2D convolutional residual block with equalized learning rate.

    Residual block composed of 3x3 convolutions and leaky ReLUs.

    Args:
    ----
    in_channel: int
        Input channels.
    out_channel: int
        Output channels.
    upsample: bool
        Apply upsampling via strided convolution in the first conv.
    downsample: bool
        Apply downsampling via strided convolution in the second conv.

    """

    def __init__(self, in_channel, out_channel, upsample=False, downsample=False):
        super().__init__()

        assert not (upsample and downsample), 'Cannot upsample and downsample simultaneously'
        mid_ch = in_channel if downsample else out_channel

        self.conv1 = ConvLayer2d(in_channel, mid_ch, upsample=upsample, kernel_size=3)
        self.conv2 = ConvLayer2d(mid_ch, out_channel, downsample=downsample, kernel_size=3)

        if (in_channel != out_channel) or upsample or downsample:
            self.skip = ConvLayer2d(
                in_channel,
                out_channel,
                upsample=upsample,
                downsample=downsample,
                kernel_size=1,
                activate=False,
                bias=False,
            )

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)

        if hasattr(self, 'skip'):
            skip = self.skip(input)
            out = (out + skip) / math.sqrt(2)
        else:
            out = (out + input) / math.sqrt(2)
        return out


class ModulationLinear(nn.Module):
    """Linear modulation layer.

    This layer is inspired by the modulated convolution layer from StyleGAN2, but adapted to linear layers.

    Args:
    ----
    in_channel: int
        Input channels.
    out_channel: int
        Output channels.
    z_dim: int
        Latent dimension.
    demodulate: bool
        Demudulate layer weights.
    activate: bool
        Apply LeakyReLU activation to layer output.
    bias: bool
        Add bias to layer output.

    """

    def __init__(
        self,
        in_channel,
        out_channel,
        z_dim,
        demodulate=True,
        activate=True,
        bias=True,
    ):
        super().__init__()

        self.eps = 1e-8
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.z_dim = z_dim
        self.demodulate = demodulate

        self.scale = 1 / math.sqrt(in_channel)
        self.weight = nn.Parameter(torch.randn(out_channel, in_channel))
        self.modulation = EqualLinear(z_dim, in_channel, bias_init=1, activate=False)

        if activate:
            # FusedLeakyReLU includes a bias term
            self.activate = FusedLeakyReLU(out_channel, bias=bias)
        elif bias:
            self.bias = nn.Parameter(torch.zeros(1, out_channel))

    def __repr__(self):
        return f'{self.__class__.__name__}({self.in_channel}, {self.out_channel}, z_dim={self.z_dim})'

    def forward(self, input, z):
        # feature modulation
        gamma = self.modulation(z)  # B, in_ch
        input = input * gamma

        weight = self.weight * self.scale

        if self.demodulate:
            # weight is out_ch x in_ch
            # here we calculate the standard deviation per input channel
            demod = torch.rsqrt(weight.pow(2).sum([1]) + self.eps)
            weight = weight * demod.view(-1, 1)

            # also normalize inputs
            input_demod = torch.rsqrt(input.pow(2).sum([1]) + self.eps)
            input = input * input_demod.view(-1, 1)

        out = F.linear(input, weight)

        if hasattr(self, 'activate'):
            out = self.activate(out)

        if hasattr(self, 'bias'):
            out = out + self.bias

        return out


class ModulatedConv2d(nn.Module):
    """2D convolutional modulation layer.

    This layer was originally proposed in StyleGAN2 (https://arxiv.org/pdf/1912.04958.pdf) as a replacement for
    Adaptive Instance Normalization (AdaIN), which was shown to produce artifacts in generated samples.

    Args:
    ----
    in_channel: int
        Input channels.
    out_channel: int
        Output channels.
    kernel_size: int
        Size of the convolutional kernel.
    z_dim: int
        Dimension of the latent code.
    demodulate: bool
        Demodulate layer weights.
    upsample: bool
        Output will be 2x scale of inputs.
    downsample: bool
        Outputs will be 0.5x scale of inputs.
    blur_kernel: list, int
        List of ints representing the blur kernel to use for blurring before/after convolution.
    activate: bool
        Apply LeakyReLU activation to layer output.
    bias: bool
        Add bias to layer output.

    """

    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        z_dim,
        demodulate=True,
        upsample=False,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        activate=True,
        bias=True,
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.z_dim = z_dim
        self.upsample = upsample
        self.downsample = downsample

        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            self.blur = Blur(blur_kernel, pad=(pad0, pad1))

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(torch.randn(1, out_channel, in_channel, kernel_size, kernel_size))
        self.modulation = EqualLinear(z_dim, in_channel, bias_init=1)
        self.demodulate = demodulate

        if activate:
            # FusedLeakyReLU includes a bias term
            self.activate = FusedLeakyReLU(out_channel, bias=bias)
        elif bias:
            self.bias = nn.Parameter(torch.zeros(1, out_channel, 1, 1))

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.in_channel}, {self.out_channel}, kernel_size={self.kernel_size}, "
            f"z_dim={self.z_dim}, upsample={self.upsample}, downsample={self.downsample})"
        )

    def forward(self, input, z):
        batch, in_channel, height, width = input.shape

        gamma = self.modulation(z).view(batch, 1, in_channel, 1, 1)
        weight = self.scale * self.weight * gamma

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

        weight = weight.view(batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size)

        if self.upsample:
            input = input.view(1, batch * in_channel, height, width)
            weight = weight.view(batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size)
            weight = weight.transpose(1, 2).reshape(
                batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size
            )
            out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
            out = self.blur(out)
        elif self.downsample:
            input = self.blur(input)
            _, _, height, width = input.shape
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
        else:
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=self.padding, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        if hasattr(self, 'activate'):
            out = self.activate(out)

        if hasattr(self, 'bias'):
            out = out + self.bias

        return out


class ToRGB(nn.Module):
    """Output aggregation layer.

    In the original StyleGAN2 this layer aggregated RGB predictions across all resolutions, but it's been slightly
    adjusted here to work with outputs of any dimension.

    Args:
    ----
    in_channel: int
        Input channels.
    out_channel: int
        Output channels.
    z_dim: int
        Latent code dimension.
    upsample: bool
        Upsample the aggregated outputs.

    """

    def __init__(self, in_channel, out_channel, z_dim, upsample=True):
        super().__init__()

        if upsample:
            self.upsample = Upsample()

        self.conv = ModulatedConv2d(
            in_channel=in_channel,
            out_channel=out_channel,
            kernel_size=1,
            z_dim=z_dim,
            demodulate=False,
            activate=False,
            bias=True,
        )

    def forward(self, input, z, skip=None):
        out = self.conv(input, z)

        if skip is not None:
            skip = self.upsample(skip)
            out = out + skip
        return out


class ConvRenderBlock2d(nn.Module):
    """2D convolutional neural rendering block.

    This block takes a feature map generated from a NeRF-style MLP and upsamples it to a higher resolultion
    image, as done in GIRAFFE (https://arxiv.org/pdf/2011.12100.pdf). Inspired by StyleGAN2, this module uses
    skip connections (by summing RGB outputs at each layer) to improve gradient flow. GIRAFFE specifically uses
    small convolutional kernels a single conv layer per block to "avoid entangling global scene properties".

    Args:
    ----
    in_channel: int
        Input channels.
    out_channel: int
        Output channels.
    mode: str
        Whether to use original GIRAFFE implementation or the modified implementation. Modified implementation
        uses conv transpose + blur as a learnable upsampling kernel, and replaces bilinear upsampling with blur
        upsampling. This is mainly so that we have consistency with the rest of the model.
    deep: bool
        Apply two convolutional layers in succession, as in StyleGAN2. Otherwise only apply a single convolution
        layer, as in GIRAFFE.

    """

    def __init__(self, in_channel, out_channel, mode='blur', deep=False):
        super().__init__()
        self.mode = mode
        self.deep = deep

        # the first conv layer doesn't have bias because it is fused with the leakyReLU activation
        if mode == 'original':
            self.conv = EqualConv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            self.conv = EqualConvTranspose2d(in_channel, out_channel, kernel_size=3, stride=2, padding=0, bias=False)
            self.blur = Blur(kernel=[1, 3, 3, 1], pad=(1, 1), upsample_factor=2)
            self.skip_upsample = Upsample(kernel=[1, 3, 3, 1], factor=2)

        if deep:
            self.conv2 = EqualConv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)

        self.activation = FusedLeakyReLU(out_channel, bias=True)
        self.toRGB = EqualConv2d(out_channel, 3, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x, skip=None):
        if self.mode == 'original':
            x = torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')
            x = self.conv(x)
        else:
            x = self.conv(x)
            x = self.blur(x)
        x = self.activation(x)

        if self.deep:
            x = self.conv2(x)
            x = self.activation(x)

        rgb = self.toRGB(x)

        if skip is not None:
            if self.mode == 'original':
                skip = torch.nn.functional.interpolate(skip, scale_factor=2, mode='bilinear', align_corners=False)
            else:
                skip = self.skip_upsample(skip)
            rgb = rgb + skip
        return x, rgb


class PositionalEncoding(nn.Module):
    """Positional encoding layer.

    Positionally encode inputs by projecting them through sinusoidal functions at multiple frequencies.
    Frequencies are scaled logarithmically. The original input is also included in the output so that the
    absolute position information is not lost.

    Args:
    ----
    in_dim: int
        Input dimension.
    frequency_bands: int
        Number of frequencies to encode input into.

    """

    def __init__(self, in_dim, frequency_bands=6, include_input=True):
        super().__init__()
        self.in_dim = in_dim
        if include_input:
            self.out_dim = in_dim + (2 * frequency_bands * in_dim)
        else:
            self.out_dim = 2 * frequency_bands * in_dim
        self.frequency_bands = frequency_bands
        self.include_input = include_input

        freqs = 2.0 ** torch.linspace(0.0, frequency_bands - 1, frequency_bands, dtype=torch.float)
        self.freqs = torch.nn.Parameter(freqs, requires_grad=False)

    def forward(self, x):
        if self.include_input:
            encoding = [x]
        else:
            encoding = []

        for freq in self.freqs:
            for func in [torch.sin, torch.cos]:
                encoding.append(func(x * freq))
        encoding = torch.cat(encoding, dim=-1)
        return encoding
