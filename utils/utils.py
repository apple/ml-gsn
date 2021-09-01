import torch
import argparse
import importlib


def get_obj_from_str(string):
    # From https://github.com/CompVis/taming-transformers
    module, cls = string.rsplit(".", 1)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    # From https://github.com/CompVis/taming-transformers
    if "target" not in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def chunk_tensor_dict(input, chunks, dim=0):
    x1 = {}
    x2 = {}

    for key in input.keys():
        x1[key], x2[key] = torch.chunk(input[key], chunks=chunks, dim=dim)
        x1[key], x2[key] = x1[key].contiguous(), x2[key].contiguous()
    return x1, x2


def exclusive_mean(x, dim):
    # take mean across axis, but only for nonzero elements
    mask = x != 0
    count = torch.sum(mask, dim=dim)
    out = torch.sum(x, dim=dim) / torch.clamp(count, min=1)
    return out


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
