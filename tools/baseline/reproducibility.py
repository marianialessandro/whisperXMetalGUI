"""Opt-in process RNG initialization for paired diagnostics; default legacy workers remain unseeded."""


def seed_process(seed):
    if type(seed) is not int or not 0 <= seed < 2 ** 32:
        raise ValueError("diagnostic_seed_invalid")
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available(): torch.mps.manual_seed(seed)
    try:
        import mlx.core as mx
        mx.random.seed(seed)
    except ImportError:
        pass
    return {"seed": seed, "deterministic_algorithms_forced": False}
