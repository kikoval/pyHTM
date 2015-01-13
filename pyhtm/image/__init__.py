"""
This module contains functions for working with Python Imaging Library (PIL)
images, as well as a list of supported image extensions.
"""

import math
from io import BytesIO as BytesIO

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageOps
import numpy


# List of extensions that we have tested with PIL
imageExtensions = ('.bmp', '.gif', '.jpg', '.jpeg', '.pgm', '.png',
                   '.tif', '.tiff')


def scaleToFitPIL(image, size):
    """Scale to fit the size, preserving aspect ratio."""

    targetRatio = size[0] / float(size[1])
    imageRatio = image.size[0] / float(image.size[1])
    if imageRatio > targetRatio:
        xSize = size[0]
        scale = size[0] / float(image.size[0])
        ySize = int(scale * image.size[1])
    else:
        ySize = size[1]
        scale = size[1] / float(image.size[1])
        xSize = int(scale * image.size[0])
    if scale < 1:
        return image.resize((xSize, ySize), Image.ANTIALIAS)
    else:
        return image.resize((xSize, ySize), Image.BICUBIC)


def cropToFit(image, width, height):
    """
    Scale to fit the size, preserving aspect ratio, cropping if necessary.

    Only works on PIL images.
    """

    targetRatio = width / float(height)
    imageRatio = image.size[0] / float(image.size[1])
    if imageRatio > targetRatio:
        # Original image is too wide
        scale = height / float(image.size[1])
        newSize = (int(scale * image.size[0]), height)
        cropStart = ((newSize[0] - width) / 2, 0)
    else:
        # Original image is too tall
        scale = width / float(image.size[0])
        newSize = (width, int(scale * image.size[1]))
        cropStart = (0, (newSize[1] - height) / 2)
    image = image.resize(newSize)
    # Crop if necessary
    if newSize != (width, height):
        image = image.crop((cropStart[0], cropStart[1],
          cropStart[0] + width, cropStart[1] + height))
    return image


def cropToAspectRatio(image, ratio):
    """
    Crop the PIL image to have the specified aspect ratio.
    """

    width, height = image.size
    imageRatio = width / float(height)
    x = y = 0
    if imageRatio > ratio:
        # Original image is too wide
        x = int((width - height * ratio) / 2)
    else:
        # Original image is too tall
        y = int((height - width * 1 / ratio) / 2)
    image = image.resize((width, height))
    # Crop if necessary
    if x or y:
        image = image.crop((x, y, width - x, height - y))
    return image


def thresholdBW(image, threshold=128):
    """
    Threshold the image to make it black and white.

    Returned image is mode 'L', but all values are 0 or 255.

    PIL's convert() method dithers rather than thresholds.
    """

    if image.mode != 'L':
        raise ValueError("Image must be mode 'L'")

    data = numpy.array(image)
    data[data > threshold] = 255
    data[data < 255] = 0
    image.putdata(data.flatten())
    return image


def blur(image, radius, edgeColor=None):
    """
    Gaussian blur with variable-sized kernel.

    radius -- Kernel radius.
    edgeColor -- Color with which to pad the edges for blurring. If not
      specified, the edge of the returned image is not blurred.
    """

    if type(radius) is not int:
        raise TypeError("'radius' must be an integer")

    if image.mode == 'LA':
        blurredImage = blur(image.split()[0], radius, edgeColor)
        blurredAlpha = blur(image.split()[1], radius, 0)
        blurredImage.putalpha(blurredAlpha)
        return blurredImage

    elif image.mode != 'L':
        raise ValueError("Image must be mode 'L' or 'LA'")

    sigma = radius / 3.0

    # Convert image to numpy array
    array = numpy.array(image)

    if edgeColor is not None:
        # Pad image with edge color
        shape = [(d + (radius - 1) * 2) for d in array.shape]
        if edgeColor == 0:
            paddedArray = numpy.zeros(shape)
        else:
            paddedArray = numpy.ones(shape) * edgeColor
        paddedArray[radius - 1:-(radius - 1), radius - 1:-(radius - 1)] = array
        array = paddedArray

    # Build kernel
    length = radius * 2 - 1
    kernel = numpy.zeros(length)
    for x in range(radius):
        val = 1 / (sigma * 2 * math.pi)
        val *= math.exp(-1 * x ** 2 / (2 * sigma ** 2))
        kernel[radius - 1 + x] = val
        kernel[radius - 1 - x] = val
    kernel /= kernel.sum()

    # Convolve with array
    # Rows
    padded1 = numpy.zeros((array.shape[0], array.shape[1] + radius - 1))
    padded1[:, :array.shape[1]] = array
    padded2 = numpy.convolve(padded1.flatten(), kernel, mode='same')
    padded2.resize(padded1.shape)
    array2 = padded2[:, :array.shape[1]]
    # Columns
    padded3 = numpy.zeros((array2.shape[0] + radius - 1, array2.shape[1]))
    padded3[:array2.shape[0], :] = array2
    padded3 = padded3.transpose()
    padded4 = numpy.convolve(padded3.flatten(), kernel, mode='same')
    padded4.resize(padded3.shape)
    padded4 = padded4.transpose()
    result = padded4[:array.shape[0], :]

    blurredImage = Image.new('L', (array.shape[1], array.shape[0]))
    blurredImage.putdata(result.flatten())
    if edgeColor is None:
        # Paste the blurred image into the original image, cutting off the edge
        # of the blurred image, to avoid edge effects
        newImage = image.copy()
        croppedImage = blurredImage.crop((radius - 1,
                                          radius - 1,
                                          image.size[0] - radius + 1,
                                          image.size[1] - radius + 1))
        newImage.paste(croppedImage, (radius - 1, radius - 1))
        return newImage
    else:
        # Crop out the padding
        return blurredImage.crop((radius - 1, radius - 1,
                                  blurredImage.size[0] - (radius - 1),
                                  blurredImage.size[1] - (radius - 1)))

def serializeImage(image):
    s = BytesIO()
    format = 'png'
    if hasattr(image, 'format') and image.format:
        format = image.format
    try:
        image.save(s, format=format)
    except:
        image.save(s, format='png')
    return s.getvalue()


def deserializeImage(s, info=None):
    image = Image.open(BytesIO(s))
#    image.load()
    if info:
        image.info.update(info)
    return image


def createMask(imageIn, threshold=10, fillHoles=True, backgroundColor=255, blurRadius=0.0,
                maskScale=1.0):
    """
    Given an image, create a mask by locating the pixels that are not the backgroundColor
    (within a threshold).

    @param threshold  How far away from the backgroundColor a pixel must be to be included
                        in the mask
    @param fillHoles  If true, the inside of the mask will be filled in. This is useful if
                        the inside of objects might contain the background color
    @param backgroundColor the background color.
    @param blurRadius If set to some fraction > 0.0, then the edges of the mask will be blurred
                        using a blur radius which is this fraction of the image size.
    @param maskScale  If set to < 1.0, then the effective size of the object (the area where
                        the mask includes the object) will be scaled down by this
                        amount. This can be useful when the outside of the object contains
                        some noise that you want to trim out and not include in the mask.

    @retval the mask as a PIL 'L' image, where 255 is areas that include the object, and 0
                      are areas that are background. If blurRadius is > 0, then it will
                      also contain values between 0 and 255 which act as compositing values.

    """

    image = imageIn.convert('L')
    bwImage = image.point(lambda x: (abs(x - backgroundColor) > threshold) * 255)

    if not fillHoles:
        mask = bwImage
    else:
        bwImage = ImageOps.expand(bwImage, 1, fill=0)
        maskColor = 128
        ImageDraw.floodfill(bwImage, (0, 0), maskColor)
        mask = bwImage.point(lambda x: (x != maskColor) * 255)
        mask = ImageOps.crop(mask, 1)

    # Are we reducing the object size?
    if maskScale < 1.0:
        newSize = [int(x * maskScale) for x in mask.size]
        reducedMask = mask.resize(newSize, Image.ANTIALIAS)
        sizeDiff = numpy.array(mask.size) - numpy.array(newSize)
        pos = [int(x / 2) for x in sizeDiff]
        mask = ImageChops.constant(mask, 0)
        mask.paste(reducedMask, tuple(pos))

    # Blur the mask
    if blurRadius > 0.0:
        radius = int(round(blurRadius * (mask.size[0] + mask.size[1]) / 2))
        if radius > 1:
            mask = blur(mask, radius=radius, edgeColor=0)
        else:
            import pdb; pdb.set_trace()

    return mask


def isSimpleBBox(alpha):
    """
    Return true if the passed in alpha channel is a simple bbox
    """

    # Get the data
    alphaData = numpy.array(alpha.getdata())

    # if it were a simple bbox, what would the data be?
    bbox = list(alpha.getbbox())
    simple = ImageChops.constant(alpha, 0)
    dc = ImageDraw.Draw(simple)
    bbox[2] -= 1
    bbox[3] -= 1
    dc.rectangle(bbox, fill=255)
    bboxData = numpy.array(simple.getdata())

    return numpy.array_equal(alphaData, bboxData)


def erode(image, iterations=1):
    """
    Erode (or dilate) the image.

    image -- A grayscale PIL image.
    iterations -- Number of times to erode. Use a negative value to dilate.

    Returns the eroded/dilated image.
    """

    if image.mode != 'L':
        raise ValueError("'image' should be mode 'L' (grayscale, single-band)")

    if iterations > 0:
        # Erosion
        f = ImageFilter.MinFilter(3)
    else:
        # Dialation
        iterations = -iterations
        f = ImageFilter.MaxFilter(3)
    for i in range(iterations):
        image = image.filter(f)
    return image
