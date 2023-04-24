import torch
import os
import gc
torch.cuda.empty_cache()
print("cache cleared")
gc.collect()

torch.cuda.memory_summary(device=None, abbreviated=False)
print("done")

