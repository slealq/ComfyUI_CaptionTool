import torch
import os
import glob
from PIL import Image
import numpy as np
import node_helpers
from PIL import Image, ImageOps, ImageSequence

class ImageBatch:
    def __init__(self):
        self.images = None

    def clearCache(self):
        self.images = None

class BatchImagesLoader:
    def __init__(self):
        self.imagesPath = None
        self.imageBatch = ImageBatch()
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "imagesPath": ("STRING", {"default": None}),
            },
        }
    
    CATEGORY = "CaptionTool/Loaders"
    RETURN_TYPES = ("ImageBatch",)
    FUNCTION = "load"

    def load(self, imagesPath):

        if not os.path.exists(imagesPath):
            raise ValueError(f"Path {imagesPath} does not exist")

        # Get all files in the directory
        image_files = []

        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']

        for ext in extensions:
            image_files.extend(glob.glob(os.path.join(imagesPath, ext)))

        if not image_files:
            raise ValueError(f"No image files found in {imagesPath}")

        # Load and verify images
        images = []
        for img_path in image_files:
            (image, mask) = self.load_like_comfy(img_path)

            images.append(image)

        if not images:
            raise ValueError("No valid images could be loaded") 
        
        print(f"[BatchImagesLoader] loaded {len(images)} images")

        # Stack all images into a single batch tensor
        self.imageBatch.images = images
        return (self.imageBatch,)
    
    def custom_image_load(self, image_path):

        img_tensor = None

        try:
            img = Image.open(image_path)
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            # Convert PIL Image to numpy array and then to tensor
            img_array = np.array(img)
            img_tensor = torch.from_numpy(img_array).float() / 255.0
        except Exception as e:
            print(f"Error loading image {image_path}: {str(e)}")
            
        return img_tensor

    def load_like_comfy(self, image_path):
        img = node_helpers.pillow(Image.open, image_path)

        output_images = []
        output_masks = []
        w, h = None, None

        excluded_formats = ['MPO']

        for i in ImageSequence.Iterator(img):
            i = node_helpers.pillow(ImageOps.exif_transpose, i)

            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")

            if len(output_images) == 0:
                w = image.size[0]
                h = image.size[1]

            if image.size[0] != w or image.size[1] != h:
                continue

            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            elif i.mode == 'P' and 'transparency' in i.info:
                mask = np.array(i.convert('RGBA').getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1 and img.format not in excluded_formats:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        return (output_image, output_mask)