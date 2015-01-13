from PIL import (Image,
                 ImageChops,
                 ImageOps)

from pyhtm.nodes.ImageSensorFilters.BaseFilter import BaseFilter


class NormalizeContrast(BaseFilter):
    """Perform contrast normalization on the image.
    """

    def __init__(self, region='all', cutoff=0):
    """
    Args:
        region: Options are 'all' (equalize the entire image), 'bbox'
                (equalize just the portion of the image within the bounding
                box), and 'mask' (equalize just the portion of the image within
                the mask)
        cutoff: Number of pixels to clip from each end of the histogram
                before rescaling it
    """
    BaseFilter.__init__(self)

    if region not in ('all', 'bbox', 'mask'):
        raise RuntimeError("Not a supported region (options are 'all', 'bbox', and 'mask')")
    if type(cutoff) != int or cutoff < 0:
        raise RuntimeError("'cutoff' must be a positive integer")

    self.region = region
    self.cutoff = cutoff

    def process(self, image):
    """
    Args:
        image: The image to process

    Returns:
        a single image, or a list containing one or more images
    """
    BaseFilter.process(self, image)

    if self.mode != 'gray':
        raise RuntimeError("NormalizeContrast only supports grayscale images.")

    if self.region == 'bbox':
        bbox = image.split()[1].getbbox()
        croppedImage = image.crop(bbox)
        croppedImage.load()
        alpha = croppedImage.split()[1]
        croppedImage = \
        ImageOps.autocontrast(croppedImage.split()[0], cutoff=self.cutoff)
        croppedImage.putalpha(alpha)
        image.paste(croppedImage, image.bbox)
    elif self.region == 'mask':
        bbox = image.split()[1].getbbox()
        croppedImage = image.crop(bbox)
        croppedImage.load()
        alpha = croppedImage.split()[1]
        # Fill in the part of the cropped image outside the bounding box with a
        # uniform shade of gray
        grayImage = ImageChops.constant(croppedImage, 128)
        compositeImage = Image.composite(croppedImage, grayImage, alpha)
        # Equalize the composite image
        compositeImage = ImageOps.autocontrast(compositeImage.split()[0], cutoff=self.cutoff)
        # Paste the part of the equalized image within the mask back
        # into the cropped image
        croppedImage = Image.composite(compositeImage, croppedImage, alpha)
        croppedImage.putalpha(alpha)
        # Paste the cropped image back into the full image
        image.paste(croppedImage, bbox)
    elif self.region == 'all':
        alpha = image.split()[1]
        image = ImageOps.autocontrast(image.split()[0], cutoff=self.cutoff)
        image.putalpha(alpha)
    return image
