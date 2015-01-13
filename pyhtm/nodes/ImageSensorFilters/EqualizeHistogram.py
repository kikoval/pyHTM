from PIL import (Image, ImageOps)
import numpy

from pyhtm.nodes.ImageSensorFilters.BaseFilter import BaseFilter


class EqualizeHistogram(BaseFilter):
    """Equalize the image's histogram.
    """

    def __init__(self, region='all', mode=None):
    """
    Args:
        region: Options are 'all' (equalize the entire image), 'bbox'
      (equalize just the portion of the image within the bounding box), and
      'mask' (equalize just the portion of the image within the mask).
      mode: ** DEPRECATED ** Alias for 'region'.
    """
    BaseFilter.__init__(self)

    if mode is not None:
        region = mode

    if region not in ('all', 'bbox', 'mask'):
        raise RuntimeError("Not a supported region (options are 'all', 'bbox', and 'mask')")

    self.region = region

    def process(self, image):
    """
    Args:
        image: The image to process.

    Returns:
        a single image, or a list containing one or more images.
    """
    BaseFilter.process(self, image)

    if self.mode != 'gray':
        raise RuntimeError("EqualizeHistogram only supports grayscale images.")

    if self.region == 'bbox':
        bbox = image.split()[1].getbbox()
        croppedImage = image.crop(bbox)
        croppedImage.load()
        alpha = croppedImage.split()[1]
        croppedImage = ImageOps.equalize(croppedImage.split()[0])
        croppedImage.putalpha(alpha)
        image.paste(croppedImage, bbox)
    elif self.region == 'mask':
        bbox = image.split()[1].getbbox()
        croppedImage = image.crop(bbox)
        croppedImage.load()
        alpha = croppedImage.split()[1]
        # Fill in the part of the cropped image outside the bounding box with
        # uniformly-distributed noise
        noiseArray = \
          numpy.random.randint(0, 255, croppedImage.size[0]*croppedImage.size[1])
        noiseImage = Image.new('L', croppedImage.size)
        noiseImage.putdata(noiseArray)
        compositeImage = Image.composite(croppedImage, noiseImage, alpha)
        # Equalize the composite image
        compositeImage = ImageOps.equalize(compositeImage.split()[0])
        # Paste the part of the equalized image within the mask back
        # into the cropped image
        croppedImage = Image.composite(compositeImage, croppedImage, alpha)
        croppedImage.putalpha(alpha)
        # Paste the cropped image back into the full image
        image.paste(croppedImage, bbox)
    elif self.region == 'all':
        alpha = image.split()[1]
        image = ImageOps.equalize(image.split()[0])
        image.putalpha(alpha)
    return image
