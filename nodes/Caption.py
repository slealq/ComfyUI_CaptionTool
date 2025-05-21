import torch
from PIL import Image
import numpy as np

class CaptionTool:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vision_model": ("LlamaVisionModel",),
                "image": ("IMAGE",),
                "max_new_tokens": ("INT", {"default": 1024, "min": 10, "max": 4096, "step": 1}),
                "prompt":   ("STRING", {"multiline": True, "default": "A descriptive caption for this image"},),
            }
        }

    CATEGORY = "CaptionTool/Loaders"
    RETURN_TYPES = ("STRING",)
    FUNCTION = "gen"
    def gen(self, vision_model, image, max_new_tokens, prompt): 

        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]}
        ]
        input_text = vision_model.processor.apply_chat_template(messages, add_generation_prompt=True)

        input_image = self.tensor2pil(image)

        inputs = vision_model.processor(
            input_image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(vision_model.model.device)

        output = vision_model.model.generate(**inputs, max_new_tokens=max_new_tokens)
        text_response = vision_model.processor.decode(output[0])

        # Extract just the assistant's response by finding the text between assistant header and eot_id
        start_idx = text_response.find("<|start_header_id|>assistant<|end_header_id|>\n\n") + len("<|start_header_id|>assistant<|end_header_id|>\n\n")
        end_idx = text_response.rfind("<|eot_id|>")
        text_response = text_response[start_idx:end_idx].strip()

        return (text_response,)
    
    def tensor2pil(self, t_image: torch.Tensor)  -> Image:
        return Image.fromarray(np.clip(255.0 * t_image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))