import folder_paths
import os
from .utils import tensor2pil

class CaptionResponseSave:
    def __init__(self):
        self.captionResponse = None
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"

    def clearCache(self):
        self.captionResponse = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "outputPath": ("STRING", {"default": None}),
                "CaptionResponse": ("CaptionResponse",),
            }
        }
    
    RETURN_TYPES = ()
    FUNCTION = "save_images"

    OUTPUT_NODE = True

    CATEGORY = "CaptionTool/Savers"
    DESCRIPTION = "Saves the input images to your ComfyUI output directory."

    def save_images(self, outputPath, CaptionResponse):

        if CaptionResponse is not None:
            self.captionResponse = CaptionResponse

        captions = self.captionResponse.get_captions()

        count = 0

        print(f"Found {len(captions)} captions")

        for caption in captions:
            image = tensor2pil(caption.image)

            image.save(os.path.join(outputPath, f"{count}.png"), compress_level=4)

            # Save caption text to a .txt file
            caption_text = caption.caption
            text_file_path = os.path.join(outputPath, f"{count}.txt")
            with open(text_file_path, "w", encoding="utf-8") as f:
                f.write(caption_text)
            
            count += 1

        return {"ui": {"text": ("done",)}}