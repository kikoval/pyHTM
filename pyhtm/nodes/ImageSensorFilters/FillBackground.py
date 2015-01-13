from PIL import (Image, ImageChops)

from pyhtm.nodes.ImageSensorFilters.BaseFilter import BaseFilter
from pyhtm.image import (createMask, isSimpleBBox)


class FillBackground(BaseFilter):
    """Fill in the background (around the mask or around the bounding box).
    """

    def __init__(self, value=None, threshold=10, maskScale=1.0, blurRadius=0.0):
    """
    Args:
        value: If None, the background is filled in with the background color.
               Otherwise, it is filled with value. If value is a list, then
               this filter will return multiple images, one for each value
    """
    BaseFilter.__init__(self)

    if hasattr(value, '__len__'):
        self._values = value
    else:
        self._values = [value]
    self._threshold = threshold
    self._maskScale = maskScale
    self._blurRadius = blurRadius

    def getOutputCount(self):
    """Return the number of images returned by each call to process().

    If the filter creates multiple simultaneous outputs, return a tuple:
    (outputCount, simultaneousOutputCount).
    """
    return len(self._values)

    def process(self, image):
    """
    Args:
        image: The image to process.

    Returns:
        a single image, or a list containing one or more images.
    """
    BaseFilter.process(self, image)

    # Create the mask around the source image
    mask = image.split()[-1]
    if image.mode[-1] != 'A' or isSimpleBBox(mask):
        mask = createMask(image, threshold=self._threshold, fillHoles=True,
                          backgroundColor=self.background, blurRadius=self._blurRadius,
                          maskScale=self._maskScale)

    # Process each value
    newImages = []
    for value in self._values:
        if value is None:
            value = self.background

        bg = ImageChops.constant(image, value)
        newImage = Image.composite(image.split()[0], bg, mask)
        newImage.putalpha(image.split()[-1])
        newImages.append(newImage)

    if len(newImages) == 1:
        return newImages[0]
    else:
        return newImages
