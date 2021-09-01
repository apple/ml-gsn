try:
    from .fused_act import FusedLeakyReLU, fused_leaky_relu
    from .upfirdn2d import upfirdn2d

    print('Using custom CUDA kernels')
except Exception as e:
    print(str(e))
    print('There was something wrong with the CUDA kernels')
    print('Reverting to native PyTorch implementation')
    from .native_ops import FusedLeakyReLU, fused_leaky_relu, upfirdn2d
