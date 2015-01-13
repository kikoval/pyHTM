from PIL import Image

from pyhtm.nodes.ImageSensorFilters.BaseFilter import BaseFilter


class Translation(BaseFilter):
    """Create translated version of the image.
    """

    def __init__(self, x_axis=0, y_axis=0):

        BaseFilter.__init__(self)

        self.x_axis = x_axis  # horizontal
        self.y_axis = y_axis  # vertical

    def process(self, image):
        """
        TODO check bounding box
        """
        BaseFilter.process(self, image)

        image = image.convert('LA')

        matrix = [1, 0, self.x_axis, 0, 1, self.y_axis, 0, 0, 1]
        translated_image = image.transform(image.size, Image.AFFINE, matrix)

        # Create a new larger image to hold the translated image
        # It is filled with the background color and an alpha value of 0
        outputImage = Image.new('LA', translated_image.size, (self.background, 0))

        # Paste the translated image into the new image, using the image's
        # alpha channel as a mask
        # This effectively just fills the area around the rotation with the
        # background color, and imports the alpha channel from the translated image
        outputImage.paste(translated_image, None, translated_image.split()[1])
        outputImage = outputImage.convert('L')

        return outputImage

    def getOutputCount(self):
        """
        Return the number of images returned by each call to process().

        If the filter creates multiple simultaneous outputs, return a tuple:
        (outputCount, simultaneousOutputCount).
        """
        return 1
