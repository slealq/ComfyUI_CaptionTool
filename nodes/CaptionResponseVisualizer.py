class CaptionResponseVisualizer:
    def __init__(self):
        self.captionResponse = None

    def clearCache(self):
        self.captionResponse = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "index": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
                "CaptionResponse": ("CaptionResponse",),
            }
        }
    
    CATEGORY = "CaptionTool/Visualizers"
    RETURN_TYPES = ("IMAGE", "TEXT")
    FUNCTION = "visualize"

    def visualize(self, index, CaptionResponse):

        if CaptionResponse is not None:
            self.captionResponse = CaptionResponse

        captions = self.captionResponse.get_captions()

        if len(captions) < index:
            raise IndexError("Index is out of range of the available captions")
        
        return (captions[index].image, captions[index].caption)