from PIL import (Image, ImageChops)

from pyhtm.nodes.ImageSensorFilters.BaseFilter import BaseFilter
import numpy


class AddNoise(BaseFilter):
    """Add noise to the image.
    """

    def __init__(self, noiseLevel=0.0, doForeground=True, doBackground=False,
                 dynamic=True):
    """
    Args:
        noiseLevel: Amount of noise to add, from 0 to 1.0
        doForeground: If true, add noise to the foreground
        doBackground: If true, add noise to the background
    """
    BaseFilter.__init__(self)

    self.noiseLevel = noiseLevel
    self.doForeground = doForeground
    self.doBackground = doBackground
    self.dynamic = dynamic

    # Generate and save our random state
    saveState = numpy.random.get_state()
    numpy.random.seed(0)
    self._randomState = numpy.random.get_state()
    numpy.random.set_state(saveState)


    def process(self, image):
    """
    Args:
        image: The image to process

    Returns:
        a single image, or a list containing one or more images
    """
    # Get our random state back
    saveState = numpy.random.get_state()
    numpy.random.set_state(self._randomState)

    # Send through parent class first
    BaseFilter.process(self, image)

    alpha = image.split()[1]
    # -----------------------------------------------------------------------
    # black and white
    if self.mode == 'bw':
        # For black and white images, our doBackground pixels are 255 and our figure pixels
        #  are 0.
        pixels = numpy.array(image.split()[0].getdata(), dtype=int)

        noise = numpy.random.random(len(pixels))  # get array of floats from 0 to 1

        if self.doForeground and self.doBackground:
            noise = numpy.array(noise < self.noiseLevel, dtype=int) * 255
            pixels -= noise
            pixels = numpy.abs(pixels)

        else:
            # "Flip" self.noiseLevel percent of the foreground pixels
            # We only want to add noise to the figure, so we will flip some percent of the
            #  0 pixels.
            if self.doForeground:
                noise = numpy.array(noise < self.noiseLevel, dtype=int) * 255
                pixels |= noise

            # "Flip" self.noiseLevel percent of the background pixels
            # We only want to add noise to the background, so we will flip some percent of the
            #  255 pixels.
            if self.doBackground:
                noise = numpy.array(noise > self.noiseLevel, dtype=int) * 255
                pixels &= noise

    # -----------------------------------------------------------------------
    # gray-scale
    elif self.mode == 'gray':
        pixels = numpy.array(image.split()[0].getdata(), dtype=int)
        noise = numpy.random.random(len(pixels))  # get array of floats from 0 to 1

        # Add +/- self.noiseLevel to each pixel
        noise = (noise - 0.5) * 2 * self.noiseLevel * 256
        mask = numpy.array(alpha.getdata(), dtype=int) != 0
        if self.doForeground and self.doBackground:
            pixels += noise
        elif self.doForeground:
            pixels[mask != 0] += noise[mask != 0]
        elif self.doBackground:
            pixels[mask == 0] += noise[mask == 0]
        pixels = pixels.clip(min=0, max=255)

    else:
        raise "AddNoise Filter: this image mode not supported"

    # write out the new pixels
    newimage = Image.new(image.mode, image.size)
    newimage.putdata(pixels)
    newimage.putalpha(alpha)

    # If generating dynamic noise, change our random state each time.
    if self.dynamic:
        self._randomState = numpy.random.get_state()

    # Restore random state
    numpy.random.set_state(saveState)

    return newimage
