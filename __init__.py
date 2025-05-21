"""
@author: Stuart Leal
@title: Caption using Llama Vision Model
@nickname: slealq
@description: Caption using Llama Vision Model
"""

from .nodes.VisionModel import LlamaVisionModelLoader
from .nodes.Caption import CaptionTool
from .nodes.CaptionResponseVisualizer import CaptionResponseVisualizer
from .nodes.CaptionResponseSave import CaptionResponseSave
from .nodes.Images import BatchImagesLoader

NODE_CLASS_MAPPINGS = { 
    "LlamaVisionModelLoader": LlamaVisionModelLoader,
    "CaptionTool": CaptionTool,
    "CaptionResponseVisualizer": CaptionResponseVisualizer,
    "CaptionResponseSave" : CaptionResponseSave,
    "BatchImagesLoader": BatchImagesLoader,
}
