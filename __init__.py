"""
@author: Stuart Leal
@title: Caption using Llama Vision Model
@nickname: slealq
@description: Caption using Llama Vision Model
"""

from .nodes.VisionModel import LlamaVisionModelLoader
from .nodes.Caption import CaptionTool

NODE_CLASS_MAPPINGS = { 
    "LlamaVisionModelLoader": LlamaVisionModelLoader,
    "CaptionTool": CaptionTool
}
