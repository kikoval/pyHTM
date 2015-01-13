from PIL import (Image,
                 ImageChops,
                 ImageEnhance)

from nupic.pynodes.ImageSensorFilters.BaseFilter import BaseFilter


class Brightness(BaseFilter):
    """Modify the brightness of the image.
    """

    def __init__(self, factor=1.0):
    """
    Args:
        factor: Factor by which to brighten the image, a nonnegative number.
                0.0 returns a black image, 1.0 returns the original image, and
                higher values return brighter images.
    """
    BaseFilter.__init__(self)

    if factor < 0:
        raise ValueError("'factor' must be a nonnegative number")

    self.factor = factor

    def process(self, image):
    """
    Args:
        image: The image to process.

    Returns:
        a single image, or a list containing one or more images.
    """
    BaseFilter.process(self, image)

    brightnessEnhancer = ImageEnhance.Brightness(image.split()[0])
    newImage = brightnessEnhancer.enhance(self.factor)
    newImage.putalpha(image.split()[1])
    return newImage
