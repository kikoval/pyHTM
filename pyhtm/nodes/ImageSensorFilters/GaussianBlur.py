from PIL import (Image, ImageEnhance)

from pyhtm.nodes.ImageSensorFilters.BaseFilter import BaseFilter


class GaussianBlur(BaseFilter):
    """Apply a Gaussian blur to the image.
    """

    def __init__(self, level=1):
    """
    Args:
        level: Number of times to blur.
    """
    BaseFilter.__init__(self)

    self.level = level

    def process(self, image):
    """
    Args:
        image: The image to process.

    Returns:
        a single image, or a list containing one or more images.
    """
    BaseFilter.process(self, image)

    mask = image.split()[1]
    for i in range(self.level):
        sharpness_enhancer = ImageEnhance.Sharpness(image.split()[0])
        image = sharpness_enhancer.enhance(0.0)
    image.putalpha(mask)
    return image
