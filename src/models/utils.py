import gc
import torch

def cleanup():
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()

cleanup()
