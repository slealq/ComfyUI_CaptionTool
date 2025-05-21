from PIL import Image
import numpy as np
import torch

def tensor2pil(tensor : torch.Tensor):
    return Image.fromarray(np.clip(255.0 * tensor.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))