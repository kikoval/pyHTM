from pyhtm.nodes.ImageSensorFilters.BaseFilter import BaseFilter


class Crop(BaseFilter):
    """ Crop the image.
    """

    def __init__(self, box):
    """
    Args:
        box: 4-tuple specifying the left, top, right, and bottom coords.
    """
    BaseFilter.__init__(self)

    if box[2] <= box[0] or box[3] <= box[1]:
        raise RuntimeError('Specified box has zero width or height')

    self.box = box

    def process(self, image):
    """
    Args:
        image: The image to process.

    Returns:
        a single image, or a list containing one or more images.
    """
    BaseFilter.process(self, image)

    if self.box[2] > image.size[0] or self.box[3] > image.size[1]:
        raise RuntimeError('Crop coordinates exceed image bounds')

    return image.crop(self.box)
