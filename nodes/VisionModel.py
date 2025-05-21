import torch
import os
from transformers import MllamaForConditionalGeneration, AutoProcessor
import folder_paths

class LlamaVisionModel:
    def __init__(self):
        self.model = None
        self.processor = None

    def clearCache(self):
        self.model = None
        self.processor = None

class LlamaVisionModelLoader:
    def __init__(self):
        self.llamaModelPath = None
        self.modelName = None
        self.authToken = None
        self.visionModel = LlamaVisionModel()
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "modelName": (["unsloth/Llama-3.2-11B-Vision-Instruct", "meta-llama/Llama-3.2-11B-Vision-Instruct"],), 
                "llamaModelDirectory": (["LLAMA"],),
            },
            "optional": {
                "authToken": ("STRING", {"default": None}),
            }
        }
    
    CATEGORY = "CaptionTool/Loaders"
    RETURN_TYPES = ("LlamaVisionModel",)
    FUNCTION = "load"

    def load(self, modelName, llamaModelDirectory, authToken=None):
        basename = os.path.basename(modelName)
        model_checkpoint = os.path.join(folder_paths.models_dir, llamaModelDirectory, basename)

        if not os.path.exists(model_checkpoint):
            model = MllamaForConditionalGeneration.from_pretrained(
                modelName,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                use_auth_token=authToken
            )

            model.save_pretrained(model_checkpoint)

        else:
            model = MllamaForConditionalGeneration.from_pretrained(
                model_checkpoint,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                use_auth_token=authToken
            )

        processor = AutoProcessor.from_pretrained(modelName)

        self.visionModel.model = model
        self.visionModel.processor = processor

        return (self.visionModel,)