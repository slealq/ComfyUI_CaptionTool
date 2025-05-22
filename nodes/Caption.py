import torch
from PIL import Image
from .Images import ImageBatch
from .utils import tensor2pil

class ImageCaption:
    def __init__(self, image, caption):
        self.image = image
        self.caption = caption

class CaptionResponse:
    def __init__(self):
        self.image_captions = []

    def add_caption(self, image : Image, caption : str):
        imageCaption = ImageCaption(image, caption)

        print("Added one caption")

        self.image_captions.append(imageCaption)

    def get_captions(self):
        return self.image_captions

    def clearCache(self):
        self.image_captions = []

class CaptionTool:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            # Using LlamaVisionModel for the parameter name to avoid confusion with the model variable name
            "required": {
                "LlamaVisionModel": ("LlamaVisionModel",),
                "max_new_tokens": ("INT", {"default": 1024, "min": 10, "max": 4096, "step": 1}),
                "prompt":   ("STRING", {"multiline": True, "default": "A descriptive caption for this image"},),
            },
            "optional": {
                "image": ("IMAGE",),
                "batch_images": ("ImageBatch",),
            }
        }

    CATEGORY = "CaptionTool/Caption"
    RETURN_TYPES = ("CaptionResponse",)
    FUNCTION = "gen"
    def gen(self, LlamaVisionModel, max_new_tokens, prompt, image=None, batch_images=None): 

        image_batch = self._load_images(image, batch_images)

        caption_response = CaptionResponse()

        count = 0
        total_images = len(image_batch.images)

        for image in image_batch.images:
            pilImage = tensor2pil(image)

            text_response = self._generate_caption(LlamaVisionModel, pilImage, max_new_tokens, prompt)
            caption_response.add_caption(image, text_response)

            print(f"Have calculated {count} captions from the {total_images} total ones. Progress is {(count*100.00)/total_images} %")
            count += 1

        return (caption_response,)
    
    def _load_images(self, image : torch.Tensor | None, batch_images : ImageBatch | None):
        if image is not None:
            image_batch = ImageBatch()
            image_batch.images = [image]
        elif batch_images is not None:
            image_batch = batch_images
        else:
            raise ValueError("No image or batch_images provided")
        
        return image_batch
        
    def _generate_caption(self, vision_model, input_image : Image, max_new_tokens, prompt):
        
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]}
        ]
        input_text = vision_model.processor.apply_chat_template(messages, add_generation_prompt=True)

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

        return text_response

