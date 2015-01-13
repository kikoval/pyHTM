import os
import re
import shutil
import inspect
import pickle
import copy
from base64 import b64encode, b64decode
from unicodedata import normalize

import numpy
from PIL import (Image,
                 ImageChops,
                 ImageDraw,
                 ImageFilter)

from pyhtm.support.getsetstate import (GetSomeVars,
                                       UpdateMembers)

from pyhtm.image import (serializeImage,
                         deserializeImage,
                         imageExtensions)


class ImageSensor:
    """ImageSensor is an extensible sensor for grayscale and black and white images.
    It uses 'filter' and 'explorer' plugins to do advanced image processing and
    training.

    It loads images either from files on disk or directly from the Numenta tools.
    There are several commands for loading images:
    - loadSingleImage, for loading a single image file from disk
    - loadMultipleImages, for loading multiple image files from disk
    - loadSerializedImage, for receiving a serialized image directly
    The loadSingleImage and loadMultipleImage commands don't actually load images
    into memory until the images are needed. Furthermore, the filters (see below)
    are not run until needed. This keeps ImageSensor's memory usage low, making
    it possible to use large datasets and run many filters.

    There is also a 'memoryLimit' parameter, which caps the total amount of
    memory to be used for storing images. ImageSensor will automatically unload
    images and filter outputs as necessary to stay within the limit.

    ImageSensor does not necessarily present each image to the bottom nodes of
    the network once; rather, the explorer plugin dictates the movement of the
    sensor across the image. Typically, the sensor will sweep over each image for
    many iterations, in order to help the network generate invariance.

    The filter plugins are located in pynodes/ImageSensorFilters. Bundled filters
    include scaling, contrast normalization, and Gabor filters. To make a new
    filter, subclass BaseFilter (using the other filters as examples), and drop
    the new filter in the ImageSensorFilters directory.

    The explorer plugins, located in pynodes/ImageSensorExplorers, control how
    the sensor moves through the set of input images, their filtered versions,
    and possible locations of the sensor window on the image. You may create new
    explorers by subclassing BaseExplorer and putting them in the
    ImageSensorExplorers directory. The documentation for the BaseExplorer class
    contains detailed information on explorers.

    Only the 'width', 'height', and 'depth' parameters need to be set when the
    sensor is first constructed, though other parameters may be set at that time.
    Some ImageSensor parameters may need to be changed for each level of
    level-by-level training. For example, Pictures trains a single node at each
    level and then clones the trained node to all other nodes at the same level.
    Only some bottom-level nodes are enabled, except when training the top node.
    Thus, the Pictures example changes ImageSensor's enabledWidth and
    enabledHeight parameters with each level of training. In many cases, users
    may wish to change the explorer for each level of training.

    Some explorers can calculate ahead of time how many iterations will be
    necessary to explorer all the images, though explorers that move randomly
    cannot. If you are using one of the deterministic explorers, such as
    ExhaustiveSweep, you can use ImageSensor's numIterations parameter to get the
    total number of iterations for the loaded images. Changing the explorer or
    images will change the number of iterations, so make sure to check the
    parameter right before running the network.

    The simplest explorer, 'Flash', presents each image once without any
    sweeping. If you have trained a network with a different explorer and wish to
    perform "flash inference" for testing, just set the explorer to 'Flash'.

    All of ImageSensor's public commands and parameters are documented through
    its NodeSpec. You can view the NodeSpec with the following Python commands:
    from nupic.network import nodeHelp
    nodeHelp("py.ImageSensor")
    """

    def __init__(self, width=1, height=1, depth=1, mode='gray',
        blankWithReset=False, background=255, invertOutput=False,
        filters=[], postFilters=[], explorer="Flash",
        categoryOutputFile="", logText=False, logOutputImages=False,
        logOriginalImages=False, logFilteredImages=False,
        logLocationImages=False, logLocationOnOriginalImage=False,
        logBoundingBox=False, logDir="imagesensor_log",
        automaskingTolerance=0, automaskingPadding=0, memoryLimit=100,
        minimalBoundingBox=False, dataOut=None, categoryOut=None,
        partitionOut=None, resetOut=None, bboxOut=None, alphaOut=None, **keywds):
        """
        width -- Width of the sensor's output to the network (pixels).
        height -- Height of the sensor's output to the network (pixels).
        depth -- Optional parameter used to send multiple versions of an image
          out at the same time.
        mode -- Current options are 'gray' (8-bit grayscale) and 'bw' (1-bit
          black and white).
        blankWithReset -- ** DEPRECATED ** Whether to send a blank output every
          time the explorer generates a reset signal (such as when beginning
          a new sweep). Turning on blanks increases the number of iterations.
        background -- Pixel value of background, used for padding a cropped
          image, and also for finding the bounding box in the absence of a mask.
        invertOutput -- Inverts the output of the node (e.g. white pixels
          become black).
        filters -- List of filters to apply to each image. Each element in
          the list should be either a string (just the filter name) or a list
          containing both the filter name and a dictionary specifying its
          arguments.
        explorer -- Explorer (used to move the sensor through the input
          space). Specify as a string (just the explorer name) or a list
          containing both the explorer name and a dictionary specifying its
          arguments.
        categoryOutputFile -- Name of file to which to write category number
          on each compute (useful for analyzing network accuracy after inference).
        logText -- Toggle for verbose logging to imagesensor_log.txt.
        logOutputImages -- Toggle for writing each output to disk (as an image)
          on each iteration.
        logOriginalImages -- Toggle for writing the original, unfiltered version
          of the current image to disk on each iteration.
        logFilteredImages -- Toggle for writing the intermediate versions of
          images to disk as they pass through the filter chain.
        logLocationImages -- Toggle for writing an image to disk on each
          iteration which shows the location of the sensor window.
        logLocationOnOriginalImage -- Whether to overlay the location rectangle
          on the original image instead of the filtered image. Does not work if
          the two images do not have the same size, and may be nonsensical
          even if they do (for example, if a filter moved the object within the
          image).
        logBoundingBox -- Toggle for writing a log containing the bounding
          box information for each output image.
        automaskingTolerance -- Affects the process by which bounding box masks
          are automatically generated from images based on similarity to the
          specified 'background' pixel value.  The bounding box will enclose all
          pixels in the image that differ from 'background' by more than
          the value specified in 'automaskingTolerance'.  Default is 0, which
          generates bounding boxes that enclose all pixels that differ at all
          from the background.  In general, increasing the value of
          'automaskingTolerance' will produce tighter (smaller) bounding box masks.
        automaskingPadding -- Affects the process by which bounding box masks
          are automatically generated from images.  After computing the
          bounding box based on image similarity with respect to the background,
          the box will be expanded by 'automaskPadding' pixels in all four
          directions (constrained by the original size of the image.)
        memoryLimit -- Maximum amount of memory that ImageSensor should use
          for storing images, in megabytes. ImageSensor will unload images and
          filter outputs to stay beneath this ceiling. Set to -1 for no limit.
        minimalBoundingBox -- Whether the bounding box found by looking at the
          image background should be set even if it touches one of the sides of
          the image. Set to False to avoid chopping edges off certain images, or
          True if that is not an issue and you wish to use a sweeping explorer.
        dataOut -- The output element count of the 'dataOut' output.
        categoryOut -- The output element count of the 'categoryOut' output.
        resetOut -- The output element count of the 'resetOut' output.
        bboxOut -- The output element count of the 'bboxOut' output.
        alphaOut -- The output element count of the 'alphaOut' output.
        """

        # Validate the output element counts
        if dataOut:
            if hasattr(dataOut, "__iter__"):
                if ([1] * (3 - len(dataOut)) + list(dataOut)) == [depth, height, width]:
                    pass
            elif dataOut == (depth * height * width):
                pass

        if categoryOut and categoryOut != 1:
            raise RuntimeError("The 'categoryOut' output element count must be 1.")
        if partitionOut and partitionOut != 1:
            raise RuntimeError("The 'partitionOut' output element count must be 1.")
        if resetOut and resetOut != 1:
            raise RuntimeError("The 'resetOut' output element count must be 1.")
        if bboxOut and bboxOut != 4:
            raise RuntimeError("The 'bboxOut' output element count must be 4.")
        if alphaOut and alphaOut != width * height:
            raise RuntimeError("The 'alphaOut' output element count must be equal "
                               "to width * height")

        self.width = width
        self.height = height
        self.depth = depth
        self.mode = mode
        self.blankWithReset = blankWithReset
        self.background = background
        self.automaskingTolerance = automaskingTolerance
        self.automaskingPadding = automaskingPadding
        if self.mode == 'bw' and self.background != 0:
            self.background = 255
        self.invertOutput = invertOutput
        self.categoryOutputFile = categoryOutputFile
        self.logText = logText
        self.logOutputImages = logOutputImages
        self.logOriginalImages = logOriginalImages
        self.logFilteredImages = logFilteredImages
        self.logLocationImages = logLocationImages
        self.logLocationOnOriginalImage = logLocationOnOriginalImage
        self.logBoundingBox = logBoundingBox
        self.logDir = logDir
        self.memoryLimit = memoryLimit
        self.minimalBoundingBox = minimalBoundingBox
        self.enabledWidth = self.width
        self.enabledHeight = self.height

        # The imageList data structure contains all the information about all the
        #   images which have been loaded via and of the load* methods. Some images
        #   may not be in memory, but their metadata is always kept in imageList.
        # imageList[imageIndex] returns all the information about the image with
        #   the specified index, in a dictionary. The keys in the dictionary are:
        #   'image': The unfiltered image.
        #   'imagePath': The path from which the image was loaded.
        #   'maskPath': The path from which the mask was loaded.
        #   'categoryName': The name of the image's category.
        #   'categoryIndex': The index of the image's category.
        #   'filtered': A dictionary of filtered images created from this image.
        # In general, images are only loaded once they are needed. But if an image
        #   is loaded via loadSerializedImage, then its entry in imageList has an
        #   'image' value but no 'imagePath' value. Thus, it will never be deleted
        #   from memory because it cannot be recovered. All other images are fair
        #   game.
        # The 'filtered' dictionary requires more explanation. Each key in the
        #   dictionary is a tuple specifying the positions of the filters that
        #   generated the image. (Filters can generate multiple outputs, so an
        #   image that comes out of the filter pipeline must be referenced by its
        #   position in the outputs of each filter in the pipeline). The dictionary
        #   also contains images that have been run through only part of the filter
        #   pipeline, which are kept around for us as inputs for the remaining
        #   filters.
        # Here is an example with 3 filters in the pipeline:
        #   0: A Resize filter that generates 3 outputs (small, medium, large)
        #   1: An EqualizeHistogram filter that generates 1 output
        #   2: A Rotation2D filter that generates 5 outputs (5 rotation angles)
        # A typical key for an image would be (0, 0, 2), specifying the smallest
        #   scale from the Resize filter (0), the only output from the
        #   EqualizeHistogram filter (0), and the middle rotation angle (2).
        # Another valid key would be (1), specifying an image that has gone through
        #   the Resize filter (the middle scale), but which has not been through
        #   the other filters yet. This image would neven be shown to the network,
        #   but it would be used by ImageSensor to compute other images.
        # The _getFilteredImages method is the only method which directly accesses
        #   the filtered images in imageList. Filtering is only done on-demand.
        #   If _getFilteredImages is called and the requested images have not yet
        #   been created, _applyFilter is called to run each filter, and the
        #   resulting images are stored in imageList for later use. They may be
        #   deleted due to the memoryLimit parameter, in which case they will be
        #   recreated later if necessary.
        self._imageList = []
        self.categoryInfo = []  # (categoryName, canonicalImage) for each category
        self._imageQueue = []  # Queue of image indices for managing memory
        self._filterQueue = []  # Queue of filter outputs for mananging memory
        self._pixelCount = 0  # Count of total loaded pixels for mananging memory
        self.outputImage = None  # Copy of the last image sent to the network
        self.locationImage = None  # Copy of the location image for the last output
        self.prevPosition = None  # Position used for the last compute iteration
        self._categoryOutputFile = None  # To write the category on each iteration
        self._iteration = 0  # Internal iteration counter
        self._setFilters(filters)
        self._setPostFilters(postFilters)
        self._setExplorer(explorer)

    def loadSingleImage(self, imagePath, maskPath=None, categoryName=None,
        clearImageList=True, skipExplorerUpdate=False):
        """
        Add the specified image to the list of images.

        Images are not loaded into memory until they are needed.

        imagePath -- Path to the image to load.
        maskPath -- Path to the mask to load with this image.
        categoryName -- Name of the category of this image.
        clearImageList -- If True, all loaded images are removed before this
          image is loaded. If False, this image is appended to the list of
          images.
        """

        if categoryName is not None and type(categoryName) is not str:
            categoryName = str(categoryName)

        if clearImageList:
            self.clearImageList(skipExplorerUpdate=True)

        self._addImage(imagePath=imagePath, maskPath=maskPath,
          categoryName=categoryName)

        if not skipExplorerUpdate:
            self.explorer[2].update(numImages=len(self._imageList))

        self._logCommand([('index', len(self._imageList)-1)])

        if clearImageList:
            self.explorer[2].first()

        return self.getParameter('numImages'), self.getParameter('numMasks')

    def loadSpecificImages(self, imagePaths, categoryNames=None,
        clearImageList=True):
        """
        Add multiple images to the list of images.

        See the loadMultipleImages to load images which have been organized by
        category on disk.

        This command is equivalent to calling loadSingleImage repeatedly, but it
        is faster because it avoids updating the explorer between each image, and
        because it only involves one call to the runtime engine.

        imagePaths -- List with the path of each image.
        categoryNames -- Category name for each image (or can be a single string
          with the category name that should be applied to all images).
        clearImageList -- If True, all loaded images are removed before this
          image is loaded. If False, this image is appended to the list of
          images.
        """

        if categoryNames is not None and isinstance(categoryNames, basestring):
            categoryNames = [categoryNames] * len(imagePaths)

        if clearImageList:
            self.clearImageList(skipExplorerUpdate=True)

        for i in range(len(imagePaths)):
            if categoryNames is not None:
                categoryName = categoryNames[i]
            else:
                categoryName = None
            self.loadSingleImage(imagePath=imagePaths[i],
                                 categoryName=categoryName,
                                 clearImageList=False,
                                 skipExplorerUpdate=True)

        self.explorer[2].update(numImages=len(self._imageList))

        return self.getParameter('numImages'), self.getParameter('numMasks')

    def _walk(self, top):
        """
        Directory tree generator lifted from python 2.6 and then
        stripped down.  It improves on the 2.5 os.walk() by adding
        the 'followlinks' capability.
        """

        try:
            # Note that listdir and error are globals in this module due
            # to earlier import-*.
            names = os.listdir(top)
        except:
            return

        dirs, nondirs = [], []
        for name in names:
            if os.path.isdir(os.path.join(top, name)):
                dirs.append(name)
            else:
                nondirs.append(name)

        yield top, dirs, nondirs
        for name in dirs:
            path = os.path.join(top, name)
            for x in self._walk(path):
                yield x

    def loadMultipleImages(self, imagePath, extension=None, maskPath=None,
        first=None, last=None, subsample=1, clearImageList=True,
        strictMaskLocations=True, categoryNameFilter=None, pattern=None,
        skipInterval=None, skipOffset=None, useCategories=True):
        """Add images from multiple categories to the list of images.

        Images are not loaded into memory until they are needed.

        imagePath -- Path from which to load the images (see note below).
        extension -- Extension of images files to accept (or None to accept all
          valid images).
        maskPath -- Path from which to load masks that correspond to the loaded
          images (see note below).
        first -- Index of the first image in each category to load. If
          first == 1, ImageSensor skips the first image and loads the rest.
        last -- Index of the last image in each category to load. If
          last == 1, ImageSensor loads the first two images.
        subsample -- ImageSensor loads 1/subsample of the images in each
          category. If subsample == 3, loads the first image, the fourth, the
          seventh, and so on.
        clearImageList -- If True, ImageSensor removes all loaded images when
          loading these new images. If False, the images loaded by this
          method will be appended to the existing list of images.
        strictMaskLocations -- If True, ImageSensor will only load masks whose
          path (in the mask directory) exactly parallels the path to the image
          (in the image directory). If False, ImageSensor will attempt to find
          the masks even if there aren't category subdirectories, or all the
          mask files are in the root mask directory, etc.
        categoryNameFilter -- String or list of strings that will be matched
          against category names. Only categories that match one of the strings
          will be processed. Each string can be a regular expression.
        pattern -- Regular expression for filtering images. Only images which
          match the regular expression (via re.search()) will be accepted.
          The path provided to pattern is the absolute path to the image file.
        skipInterval -- The inverse of 'subsample'; this parameter directs
          ImageSensor to skip every Nth image that it would otherwise load.
          For example, if 'skipInterval' is 2, then ImageSensor will load
          only every other image.  The default is None, which directs
          ImageSensor to skip no images.  Note that a 'skipInterval' of 1
          implies to skip every image, which is nonsensical; therefore,
          non-None values of 'skipInterval' which are less than 2 cause
          RuntimeErrors to be raised.
        skipOffset -- Operates in conjunction with 'skipInterval'.  Specifies
          an offset to use for the purpose of skipping.  For example, if 'skipInterval'
          was 10 (skip every 10th image) and 'skipOffset' was 0 (or None), then
          the first 9 images would be loaded, the 10th would be skipped, etc.
          But if 'skipOffset' were 2, then the first 7 images would be loaded,
          the 8th skipped, the next 9 loaded, the 17th skipped, etc.  Defaults
          to None (equivalent to zero.)  If both 'skipOffset' and 'skipInterval'
          are non-None, then 'skipOffset' must be non-negative and less than
          'skipInterval'.
        useCategories -- True for normal behavior, or False to load any image found
          in imagePath, without looking for nested directory folders.

        Returns a tuple containing the number of images loaded and the number of
        masks loaded.

        This method expects a directory structure like the following:
        imagePath/categoryName1/image01.ext
                                image02.ext
                  categoryName2/image01.ext
                                image02.ext

        Optionally, images can be nested arbitrarily deep. For instance:
        imagePath/categoryName1/objectName1/image01.ext
                                            image02.ext
                                objectName2/image01.ext
                                            image02.ext
                  categoryName2/objectName1/image01.ext
        A depth-first search is performed to find images.

        Directories and images are processed in sorted order.

        The nested directory structure with category names is necessary, but the
        names of the image files are unimportant.
        """

        if first is not None and type(first) != int:
            raise RuntimeError("'first' must be None or a nonnegative integer")
        if last is not None and type(last) != int:
            raise RuntimeError("'last' must be None or a nonnegative integer")
        if subsample is not None and type(subsample) != int:
            raise RuntimeError("'subsample' must be None or a positive integer")
        if skipInterval is not None and (type(skipInterval) != int
                                         or skipInterval < 2):
            raise RuntimeError("'skipInterval' must be None or an integer >= 2")
        if skipOffset is not None and skipInterval is not None and \
            (type(skipOffset) != int or skipOffset < 0
                                     or skipOffset >= skipInterval):
            raise RuntimeError("'skipOffset' must be None or a non-negative integer "
                               "< 'skipInterval'")

        self._logCommand()

        filterLogDir = os.path.join(self.logDir, 'output_from_filters')
        if self.logFilteredImages:
            if clearImageList and os.path.exists(filterLogDir):
                shutil.rmtree(filterLogDir)
            if self.filters and not os.path.exists(filterLogDir):
                os.makedirs(filterLogDir)

        if clearImageList:
            self.clearImageList(skipExplorerUpdate=True)

        if extension:
            # Only look for the extension specified by the user
            if not extension.startswith('.'):
                extension = '.' + extension
            extensions = [extension]
        else:
            extensions = imageExtensions

        imagePath = os.path.abspath(imagePath)
        if maskPath:
            maskPath = os.path.abspath(maskPath)
            if not os.path.exists(maskPath):
                maskPath = None

        # Convert 'first', 'last', and 'subsample' to proper Python names/format
        start = first
        stop = None
        if last is not None:
            stop = last + 1
        step = subsample

        # Handle skipping images that would otherwise be loaded
        if skipOffset is None:
            skipOffset = 0

        images = []
        categoryList = [None]
        if useCategories:
            # Assume each directory in imagePath is its own category
            categoryList = [c for c in sorted(os.listdir(imagePath))
                            if c[0] != '.' and
                            os.path.isdir(os.path.join(imagePath, c))]
            if categoryList:
                # Filter categories if specified
                if categoryNameFilter:
                    # Need to convert to NFC and re-encode to UTF-8 or else paths may not
                    # match the category filter
                    categoryList = [normalize('NFC', unicode(c, 'utf8')).encode('utf8')
                                    for c in categoryList]
                    if isinstance(categoryNameFilter, basestring):
                        categoryNameFilter = [categoryNameFilter]
                    else:
                        categoryNameFilter = list(categoryNameFilter)
                    for i, r in enumerate(categoryNameFilter):
                        if r[-1] != '$':
                            categoryNameFilter[i] += '$'
                    categoryList = [c for c in categoryList if True in
                      [re.match(r, c) is not None for r in categoryNameFilter]]

        for category in categoryList:
            skipCounter = skipOffset

            # Call loadSingleImage on every image with the correct extension at any
            # depth, using a depth first search
            categoryFilenames = []
            if category:
                walkPath = os.path.join(imagePath, category)
            else:
                walkPath = imagePath
                category = os.path.split(imagePath)[1]

            #if float(".".join([str(x) for x in sys.version_info[:2]])) >= 2.6:
            #  w = os.walk(walkPath, followlinks=True)
            #else:
            #  w = os.walk(walkPath)
            for dirpath, dirnames, filenames in os.walk(walkPath):
                # Don't enter directories that begin with '.'
                for d in dirnames[:]:
                    if d.startswith('.'):
                        dirnames.remove(d)
                dirnames.sort()
                # Ignore files that begin with '.'
                filenames = [f for f in filenames if not f.startswith('.')]
                # Only load images with the right extension
                filenames = [f for f in filenames
                  if os.path.splitext(f)[1].lower() in extensions]
                if pattern:
                    # Filter images with regular expression
                    filenames = [f for f in filenames
                                 if re.search(pattern, os.path.join(dirpath, f))]
                filenames.sort()
                imageFilenames = [os.path.join(dirpath, f) for f in filenames]
                # Get the corresponding path to the masks
                if maskPath:
                    maskdirpath = os.path.join(maskPath, dirpath[len(imagePath)+1:])
                    maskFilenames = [os.path.join(maskdirpath, f) for f in filenames]
                    if strictMaskLocations:
                        # Only allow masks with parallel filenames
                        for i, filename in enumerate(maskFilenames):
                            if not os.path.exists(filename):
                                maskFilenames[i] = None
                    else:
                        # Find the masks even if the path does not match exactly
                        for i, filename in enumerate(maskFilenames):
                            while True:
                                if os.path.exists(filename):
                                    maskFilenames[i] = filename
                                    break
                                if os.path.split(filename)[0] == maskPath:
                                    # Failed to find the mask
                                    maskFilenames[i] = None
                                    break
                                # Try to find the mask by eliminating subdirectories
                                body, tail = os.path.split(filename)
                                head, body = os.path.split(body)
                                while not os.path.exists(head):
                                    tail = os.path.join(body, tail)
                                    head, body = os.path.split(head)
                                filename = os.path.join(head, tail)
                else:
                    maskFilenames = [None for f in filenames]
                # Add our new images and masks to the list for this category
                categoryFilenames.extend(zip(imageFilenames, maskFilenames))
            # We have the full list of filenames for this category
            for f in categoryFilenames[start:stop:step]:
                skipCounter += 1
                if not skipInterval or skipCounter % skipInterval:
                    images.append((f[0], f[1], category))

        # Load all images and masks
        for i in range(len(images)):
            self.loadSingleImage(imagePath=images[i][0], maskPath=images[i][1],
                                 categoryName=images[i][2], clearImageList=False,
                                 skipExplorerUpdate=True)

        self.explorer[2].update(numImages=len(self._imageList))

        return self.getParameter('numImages'), self.getParameter('numMasks')

    def clearImageList(self, skipExplorerUpdate=False):
        """
        Clear the list of images.
        """

        self._imageList = []
        self._imageQueue = []
        self._filterQueue = []
        self._pixelCount = 0
        self.prevPosition = None
        if not skipExplorerUpdate:
            self.explorer[2].update(numImages=0)

    def seek(self, iteration=None, image=None, filters=None, offset=None,
             reset=None):
        """Seek to the specified iteration, image, filter position, or offset.

        Args:
            iteration: Iteration number.
            image: Image number.
            filters: Tuple specifying a position for each filter.
            offset: Tuple of integers specifying the offset as (x,y).

        Examples:
        seek(0) -- Reset to the first iteration.
        seek(image=100, filters=(0,0,..,0)) -- Seek to image 100 and position
          0 for each filter.
        seek(offset=(0,0)) -- Seek to the original position of the current image.

        The 'iteration' parameter cannot be combined with the other parameters.
        """
        self._logCommand()

        # Combine image, filters, and offset into position
        position = None
        if image is not None or filters is not None or offset is not None \
            or reset is not None:
            position = {'image': image, 'filters': filters, 'offset': offset,
                        'reset': reset}

        # Validate inputs
        if iteration is not None and position is not None:
            raise RuntimeError("Cannot combine 'iteration' with other arguments")
        if iteration is None and position is None:
            raise RuntimeError("Must specify at least one argument")
        if position is not None:
            if position['offset'] and type(position['offset']) is tuple:
                position['offset'] = list(position['offset'])
            if position['image'] is not None:
                if position['image'] < 0:
                    raise RuntimeError("'image' must be nonnegative")
                if position['image'] >= len(self._imageList):
                    raise RuntimeError("'image' exceeds number of loaded images")
            if position['filters'] is not None:
                if type(position['filters']) != list:
                    raise RuntimeError("'filters' must be a list of nonnegative values")
                if len(position['filters']) != len(self.filters):
                    raise RuntimeError("Length of 'filters' does not match numFilters")

        self.explorer[2].seek(iteration=iteration, position=position)

    def getNumIterations(self, image=None):
        """Calculate how many samples the explorer will provide.

        Args:
            image: If None, returns the sum of the iterations for all the loaded
                   images. Otherwise, image should be an integer specifying the
                   image for which to calculate iterations.
        """

        if image is not None and type(image) != int:
            raise RuntimeError("'image' must be None or a nonnegative integer")

        return self.explorer[2].getNumIterations(image)

    def saveImagesToFile(self, filename):
        """Save imageList, categoryInfo, and filters to the specified file.

        Loads all images and runs all filters first.

        This method can be used to speed up image loading when expensive filters
        are run. After loading images once and passing them through the filters,
        use saveImagesToFile to dump the filtered versions to disk. On subsequent
        runs with the same images and filters, call loadImagesFromFile to load
        the filtered images and avoid rerunning the filters.
        """

        # Load all images and run all filters
        for i in range(len(self._imageList)):
            self._applyAllFilters(i)

        # Create serializable versions for pickling
        sImageList = _serializeImageList(self._imageList)
        filters = self.getParameter('filters')
        sCategoryInfo = self.getParameter('categoryInfo')

        # Pickle serializable objects to file
        f = open(filename, 'wb')
        pickle.dump((sImageList, filters, sCategoryInfo), f,
          protocol=pickle.HIGHEST_PROTOCOL)
        f.close()

    def loadImagesFromFile(self, filename):
        """Load from a file created with saveImagesToFile.

        Loads imageList and categoryInfo. Also loads the filters used to create
        the saved images, and overwrites ImageSensor.filters.
        """

        f = open(filename, 'rb')
        sImageList, filters, sCategoryInfo = pickle.load(f)
        f.close()

        self.setParameter('filters', filters)

        self._imageList = _deserializeImageList(sImageList)
        self.explorer[2].update(numImages=len(self._imageList))

        self.setParameter('categoryInfo', sCategoryInfo)

        return self.getParameter('numImages'), self.getParameter('numMasks')

    def _addImage(self, image=None, imagePath=None, maskPath=None,
                  categoryName=None, erode=None):
        """Create a dictionary for an image and metadata and add to the
        imageList.
        """
        item = {'image': image,
                'imagePath': imagePath,
                'maskPath': maskPath,
                'erode': True,
                'categoryName': categoryName,
                'categoryIndex': None,
                'partitionID': None,
                'filtered': {}}
        self._imageList.append(item)

        if erode is not None:
            item['erode'] = erode
            setErodeFlag = False
        else:
            setErodeFlag = True

        # Look up category index from name
        if item['categoryName'] is None:
            # Unspecified category
            item['categoryName'] = ""
            item['categoryIndex'] = -1
        else:
            # Look up the category in categoryInfo
            for i in range(len(self.categoryInfo)):
                if self.categoryInfo[i][0] == item['categoryName']:
                    item['categoryIndex'] = i
                    break
        if item['categoryIndex'] is None:
            # This is the first image of this category (blank categories ignored)
            item['categoryIndex'] = len(self.categoryInfo)
            # Load the image in order to use it for categoryInfo
            original = self._loadImage(len(self._imageList) - 1, returnOriginal=True,
                                       setErodeFlag=setErodeFlag)
            if not image:
                self._imageQueue.insert(0, len(self._imageList) - 1)
            # Append this category to categoryInfo
            self.categoryInfo.append((item['categoryName'], original))
        elif image:
            # Image is already present, just prepare it
            # Not necessary if it was already loaded for categoryInfo
            self._loadImage(len(self._imageList) - 1, setErodeFlag=setErodeFlag)

    def _loadImage(self, index, returnOriginal=False, setErodeFlag=True):
        """Load an image that exists in the imageList but is not loaded into memory.

        Args:
            index: Index of the image to load.
            returnOriginal: Whether to return an unmodified version of the image
                            for categoryInfo.
        """
        item = self._imageList[index]

        if not item['image']:
            # Load the image from disk
            f = open(item['imagePath'], 'rb')
            item['image'] = Image.open(f)
            item['image'].load()
            f.close()
            # Update the pixel count
            self._pixelCount += item['image'].size[0] * item['image'].size[1]

        # Extract partition ID if it exists
        partitionID = item['image'].info.get('partitionID')
        if partitionID is None:
            partitionID = -1
        item['partitionID'] = int(partitionID)

        # Convert to grayscale
        if item['image'].mode not in ('L', 'LA'):
            if 'A' in item['image'].getbands():
                # Convert to grayscale but preserve alpha channel
                item['image'] = item['image'].convert('LA')
            else:
                item['image'] = item['image'].convert('L')

        if returnOriginal:
            # Keep copy of original image
            original = item['image'].copy()

        bbox = None
        if item['maskPath'] is not None:
            # Load the mask image and add it to the image as the alpha channel
            # If the image already has an alpha channel, it will be overwritten
            f = open(item['maskPath'], 'rb')
            mask = Image.open(f)
            mask.load()
            if mask.mode != 'L':
                mask = mask.convert('L')
            f.close()
            item['image'].putalpha(mask)
        elif item['image'].mode != 'LA':
            diffImage = ImageChops.difference(item['image'],
                                              ImageChops.constant(item['image'],
                                              self.background))
            if self.automaskingTolerance:
                diffImage = ImageChops.subtract(diffImage,
                                                ImageChops.constant(item['image'],
                                                self.automaskingTolerance))
            bbox = diffImage.getbbox()
            if not bbox:
                bbox = (0, 0, item['image'].size[0], item['image'].size[1])
            elif self.automaskingPadding:
                bbox = (max(0, bbox[0] - self.automaskingPadding),
                        max(0, bbox[1] - self.automaskingPadding),
                        min(item['image'].size[0], bbox[2] + self.automaskingPadding),
                        min(item['image'].size[1], bbox[3] + self.automaskingPadding),
                        )
            if not self.minimalBoundingBox:
                # Do not use the bounding box found from the background color unless
                # it does not touch any of the sides of the image
                if not (bbox[0] > 0
                        and bbox[1] > 0
                        and bbox[2] < item['image'].size[0]
                        and bbox[3] < item['image'].size[1]):
                    # Bounding box was not brought in on all four sides
                    # Set it back to the full image
                    bbox = (0, 0, item['image'].size[0], item['image'].size[1])
            mask = ImageChops.constant(item['image'], 0)
            mask.paste(255, bbox)
            item['image'].putalpha(mask)

        if setErodeFlag:
            # Check if the image has a nonuniform alpha channel
            # If so, set the 'erode' option to False, indicating that the alpha
            # channel is meaningful and it does not need to be eroded by GaborNode
            # to avoid "phantom edges"
            # If a bounding box was used to generated the alpha channel, use the box
            # directly to avoid the expense of scanning the pixels
            if bbox:
                # Bounding box was used
                # Set to dilate mode if the bounding box doesn't touch any of the edges
                if bbox[0] != 0 \
                    and bbox[1] != 0 \
                    and bbox[2] != item['image'].size[0] \
                    and bbox[3] != item['image'].size[1]:
                    # Nonuniform alpha channel (from bounding box)
                    item['erode'] = False
            else:
                extrema = item['image'].split()[1].getextrema()
                if extrema[0] != extrema[1]:
                    # Nonuniform alpha channel
                    item['erode'] = False

        if returnOriginal:
            return original

    def _applyFilter(self, image, imageIndex, filterIndex):
        """Apply the specified filter to the image."""

        filtered = self.filters[filterIndex][2].process(image)

        if type(filtered) is not list:
            filtered = [filtered]

        for i, item in enumerate(filtered):
            if type(item) is not list:
                filtered[i] = [item]

        # Verify that the filter produced the correct number of outputs
        outputCount = self.filters[filterIndex][2].getOutputCount()
        if type(outputCount) not in (tuple, list):
            outputCount = (outputCount, 1)
        if len(filtered) != outputCount[0] or \
            False in [len(outputs) == outputCount[1] for outputs in filtered]:
            raise RuntimeError("The %s filter " % self.filters[filterIndex][0] +
              "did not return the correct number of outputs. The number of images " +
              "that it returned does not match the return value of the filter's " +
              "getOutputCount() method.")

        for item in filtered:
            for image in item:
                # Verify that the image has the correct mode
                if image.mode != 'LA':
                    s = """Filtered image returned by the "%s" filter (index %d) has
                      illegal mode '%s'. Images must be mode 'LA' (grayscale with alpha
                      channel containing the mask).""" % (self.filters[filterIndex][0],
                      filterIndex, image.mode)
                    if image.mode == 'L':
                        s += " The filter may have removed the alpha channel."
                    raise RuntimeError(s)
                # Update the pixel count
                self._pixelCount += image.size[0] * image.size[1]

        if self.logFilteredImages:
            # Save filter output to disk
            filterLogDir = os.path.join(self.logDir, 'output_from_filters')
            path = os.path.join(filterLogDir, '%02d_' % filterIndex +
              self.filters[filterIndex][0], '%09d' % imageIndex)
            # Create the output directory if it does not exist
            if not os.path.exists(path):
                os.makedirs(path)
            index = 0
            pathContents = [x for x in sorted(os.listdir(path)) if re.match('\d', x)]
            if pathContents:
                index = int(re.match('(\d*)', pathContents[-1]).groups()[0]) + 1
            for f in filtered:
                if len(f) > 1:
                    # Simultaneous outputs
                    for i, image in enumerate(f):
                        filename = os.path.join(path, '%02d_%02d.png' % (index, i))
                        image.split()[0].save(filename)
                else:
                    # Single output
                    filename = os.path.join(path, '%02d.png' % index)
                    f[0].split()[0].save(filename)
                index += 1

        return filtered

    def _applyPostFilters(self, image, filterIndex=0):
        """Recursively apply the postFilters to the image and return a list of
        images.
        """
        # Filter the image
        raw_output = None
        filtered = self.postFilters[filterIndex][2].process(image)

        # Handle special case where the post filter wants to control the output
        # of the image sensor (e.g convolution post filters)
        if type(filtered) is tuple:
            assert len(filtered) == 2
            raw_output = filtered[1]
            assert type(raw_output) == numpy.ndarray
            filtered = filtered[0][0]

        # Flatten all responses into a single list
        if type(filtered) is not list:
            # One response
            filtered = [filtered]
        else:
            if type(filtered[0]) is list:
                # Simultaneous responses
                filtered2 = []
                for responses in filtered:
                    filtered2.extend(responses)
                filtered = filtered2

        # Verify that the filter produced the correct number of outputs
        outputCount = self.postFilters[filterIndex][2].getOutputCount()
        if type(outputCount) in (tuple, list):
            if len(outputCount) == 1:
                outputCount = outputCount[0]
            else:
                outputCount = outputCount[0] * outputCount[1]
        if len(filtered) != outputCount:
            raise RuntimeError("%s postFilter " % self.postFilters[filterIndex][0] +
              "did not return the correct number of outputs")

        for image in filtered:
            # Verify that the image has the correct mode
            if image.mode != 'LA':
                s = """Filtered image returned by the "%s" postFilter (index %d) has
                  illegal mode '%s'. Images must be mode 'LA' (grayscale with alpha
                  channel containing the mask).""" % (self.postFilters[filterIndex][0],
                  filterIndex, image.mode)
                if image.mode == 'L':
                    s += " The filter may have removed the alpha channel."
                raise RuntimeError(s)

        if self.logFilteredImages:
            # Save intermediate outputs to disk
            filterLogDir = os.path.join(self.logDir, 'output_from_post_filters')
            path = os.path.join(filterLogDir, '%02d_' % filterIndex +
                                self.postFilters[filterIndex][0])
            # Create the output directory if it does not exist
            if not os.path.exists(path):
                os.makedirs(path)
            # Save the images
            if len(filtered) > 1:
                for i, image in enumerate(filtered):
                    name = os.path.join(path, "%09d_%02d.png" % (self._iteration, i))
                    image.save(name)
            else:
                name = os.path.join(path, "%09d.png" % self._iteration)
                filtered[0].save(name)

        if filterIndex == len(self.postFilters) - 1:
            return filtered, raw_output

        # Concatenate all responses into one flat list of simultaneous responses
        responses = []
        for image in filtered:
            response = self._applyPostFilters(image, filterIndex+1)
            if raw_output is not None:
                assert (response[1] is None) # Only one post-filter can determine raw_output

            responses.extend(response[0])

        return responses, raw_output

    def _applyAllFilters(self, image=None):
        """Run all filters on the specified image or all images.
        """

        numFilterOutputs = self._getNumFilterOutputs(self.filters)
        if image is None:
            images = range(len(self._imageList))
        else:
            images = [image]
        for image in images:
            filterPosition = [0] * len(self.filters)
            while True:
                self._getFilteredImages({'image': image, 'filters': filterPosition})
                for i in range(len(self.filters)-1, -1, -1):
                    filterPosition[i] += 1
                    if filterPosition[i] == numFilterOutputs[i]:
                        filterPosition[i] = 0
                    else:
                        break
                if filterPosition == [0] * len(self.filters):
                    break

    def _getOriginalImage(self, index=None):
        """Get the specified image, loading it if necessary.

        index -- Index of the image to retrieve. Retrieves the current image if
          not specified.
        """

        if index is None:
            index = self.explorer[2].position['image']

        if not self._imageList[index]['image']:
            # Image needs to be loaded
            self._loadImage(index)

        return self._imageList[index]['image']

    def _getFilteredImages(self, position=None):
        """Get the filtered images specified by the position.

        position -- Position to use. Uses current position if not specified.
        """

        if not position:
            position = self.explorer[2].position

        if not self._imageList[position['image']]['image']:
            # Image needs to be loaded
            self._loadImage(position['image'])

        if not self.filters:
            # No filters - return original version
            return [self._imageList[position['image']]['image']]

        # Iterate through the specified list of filter positions
        # Run filters as necessary
        allFilteredImages = self._imageList[position['image']]['filtered']
        filterPosition = tuple()
        for filterIndex, pos in enumerate(position['filters']):
            filterPosition += (pos,)
            if filterPosition not in allFilteredImages:
                # Run the filter
                if len(filterPosition) > 1:
                    # Use the first of the simultaneous responses
                    imageToFilter = allFilteredImages[filterPosition[:-1]][0]
                else:
                    imageToFilter = self._imageList[position['image']]['image']
                    # Inject the original image path to the Image's info
                    # dict in case the filter wants to use it.
                    imageToFilter.info['path'] = self._imageList[position['image']]['imagePath']
                newFilteredImages = self._applyFilter(imageToFilter, position['image'],
                                                      filterIndex)
                for j, image in enumerate(newFilteredImages):
                    # Store in the dictionary of filtered images
                    thisFilterPosition = filterPosition[:-1] + (j,)
                    allFilteredImages[thisFilterPosition] = image
                    # Update the filter queue
                    thisFilterTuple = (position['image'], thisFilterPosition)
                    if thisFilterTuple in self._filterQueue:
                        self._filterQueue.remove(thisFilterTuple)
                    self._filterQueue.insert(0, thisFilterTuple)

        # Update the queues to mark this image as recently accessed
        # Only mark the original image if it could be loaded from disk again
        if self._imageList[position['image']]['imagePath']:
            if position['image'] in self._imageQueue:
                self._imageQueue.remove(position['image'])
            self._imageQueue.insert(0, position['image'])
        # Mark all precursors to the current filter
        for i in range(1, len(position['filters']) + 1):
            partialFilterTuple = (position['image'], tuple(position['filters'][:i]))
            if partialFilterTuple in self._filterQueue:
                self._filterQueue.remove(partialFilterTuple)
            self._filterQueue.insert(0, partialFilterTuple)

        self._meetMemoryLimit()

        return allFilteredImages[filterPosition]

    def _getImageInfo(self, imageIndex=None):
        """Get the dictionary of info for the image, excluding actual PIL images.

        Args:
            image: Image index to use. Uses current position if not specified.
        """

        if imageIndex is None:
            imageIndex = self.explorer[2].position['image']
        item = self._imageList[imageIndex].copy()
        item.pop('image')
        item.pop('filtered')
        return item

    def _getOutputImages(self):
        """Get the current image(s) to send out, based on the current position.

        A post filter may want to provide the finall output of the node. In
        this case it will return a non-None final output that the ImageSensor will
        use as the output of the node regardless of the output images.
        """

        if self.prevPosition['reset'] and self.blankWithReset:
            # Blank
            return ([Image.new('LA', (self.enabledWidth, self.enabledHeight))] \
              * self.depth, None)

        else:
            # Get the image(s) to send out
            allImages = self._getFilteredImages()

            # Calculate a scale factor in each dimension for adjusting the offset
            scaleX = [image.size[0] / float(allImages[0].size[0])
                      for image in allImages]
            scaleY = [image.size[1] / float(allImages[0].size[1])
                      for image in allImages]
            offset = self.explorer[2].position['offset']

            # Normally, the enabledSize is smaller than the sensor size. But, there are some
            #  configurations where the user might want to explore in a larger size, then run
            #  it through a post-filter to get the end sensor size (for example, when using a
            #  fish-eye post filter). If we detect that the enabledSize is greater than the
            #  sensor size, then change our crop bounds
            dstImgWidth = max(self.width, self.enabledWidth)
            dstImgHeight = max(self.height, self.enabledHeight)

            # Cut out the relevant part of each image
            newImages = []
            for i, image in enumerate(allImages):
                x = int(offset[0] * scaleX[i])
                y = int(offset[1] * scaleY[i])
                cropBounds = (max(0, x),
                              max(0, y),
                              min(x + dstImgWidth, image.size[0]),
                              min(y + dstImgHeight, image.size[1]))
                croppedImage = image.crop(cropBounds)
                newImage = Image.new(croppedImage.split()[0].mode,
                                     (dstImgWidth, dstImgHeight),
                                     self.background)
                if newImage.mode == 'L':
                    newImage.putalpha(Image.new('L', newImage.size))
                newImage.paste(croppedImage, (max(0, -x), max(0, -y)))
                newImages.append(newImage)

            # Crop the shifted images back to the enabled size
            croppedImages = [image.crop((0, 0,
                                         int(round(self.enabledWidth * scaleX[i])),
                                         int(round(self.enabledHeight * scaleY[i]))))
                              for i, image in enumerate(newImages)]

            # Filter through the post filters
            final_output = None
            if self.postFilters:

                newCroppedImages = []
                for i in range(len(croppedImages)):
                    (responses, raw_output) = self._applyPostFilters(croppedImages[i])
                    if raw_output is not None:
                        assert final_output is None
                        final_output = raw_output
                    while type(responses[0]) == list:
                        responses = responses[0]
                    newCroppedImages.extend(responses)
                croppedImages = newCroppedImages

            # Check that the number of images matches the depth
            if len(croppedImages) != self.depth:
                raise RuntimeError("The filters and postFilters created %d images to"
                  " send out simultaneously, which does not match ImageSensor's"
                  " depth parameter, set to %d." % (len(croppedImages), self.depth))

            # Invert output if necessary
            if self.invertOutput:
                if croppedImages[0].mode == '1':
                    croppedImages = [ImageChops.invert(image) for image in croppedImages]
                else:
                    for i, croppedImage in enumerate(croppedImages):
                        grayscale = croppedImage.split()[0]
                        alpha = croppedImage.split()[1]
                        inverted = ImageChops.invert(grayscale)
                        inverted.putalpha(alpha)
                        croppedImages[i] = inverted

            return (croppedImages, final_output)

    def _logCommand(self, reportList=None, argList='auto'):
        """Print information about the calling command to the ImageSensor log file.

        Without arguments, prints the calling command's name and arguments. Add
        extra information to print with reportList. If necessary, override the
        list of arguments with argList.

        reportList -- Extra data to print, as a list of tuples (like an
          ordered dictionary).
        argList -- Arguments to print, as a list of tuples. Default value
          'auto' specifies that they should be obtained automatically.

        ImageSensor has a very specific log file format that is machine-readable.
        A typical section looks like this:

        ('seek', {'iteration': 0, 'image': None, 'position': None}, {})
        ('compute', {}, {'iteration': 0, 'position': {'image': 0,
          'filters': [0,0,0], 'offset': [0,0]}, 'isBlank': False})

        The log snippet above indicates that the 'seek' command was issued, with
        the argument iteration=0. The command executed and has nothing extra to
        report. Then the runtime engine called 'compute'. The compute command
        reports back that this was iteration 0, and the explorer chose image 0,
        filter position [0,0,0], and offset [0,0].

        Since each call generates one line of properly-formatted Python code, it
        is easy to read in the report file and parse it with a Python script.

        Does not print if self.logText is False. Opens the file if necessary.
        """

        if not self.logText:
            return

        # Set up the log directory and log file if necessary
        if not os.path.exists(self.logDir):
            os.makedirs(self.logDir)
        if not hasattr(self, 'logFile'):
            self.logFile = open(os.path.join(self.logDir, 'imagesensor_log.txt'), 'w')

        # Get the caller's name
        callerInfo = inspect.stack()[1]
        callerName = callerInfo[3]

        # Automatically get the caller's arguments, unless they were specified
        if argList == 'auto':
            callerFrame = callerInfo[0]
            callerArgs, a, k, callerLocals = inspect.getargvalues(callerFrame)
            argList = [(name, callerLocals[name]) for name in callerArgs
                       if name != 'self']

        # Create strings to print
        # argList and reportList are lists of tuples
        # Convert each into a string form of a dictionary, but preserve the order
        argStr = reportStr = {}
        if argList:
            argStr = '{'
            for key, value in argList:
                argStr += "'%s': %s, " % (key, repr(value))
            argStr = argStr[:-2] + '}'
        if reportList:
            reportStr = '{'
            for key, value in reportList:
                reportStr += "'%s': %s, " % (key, repr(value))
            reportStr = reportStr[:-2] + '}'

        # Print to the file
        print >>self.logFile, '(%s, %s, %s)' \
            % (repr(callerName), argStr, reportStr) + os.linesep
        self.logFile.flush()

    def _logOutputImages(self):
        """
        Save the output images to disk.
        """

        # Create output directory if necessary
        outputLogDir = os.path.join(self.logDir, 'output_to_network')
        if not os.path.exists(outputLogDir):
            os.makedirs(outputLogDir)
        # Save the sensor's output images
        if self.depth > 1:
            for i in range(self.depth):
                outputImageName = "%09d_%02d.png" % (self._iteration, i)
                name = os.path.join(outputLogDir, outputImageName)
                self.outputImage[i].split()[0].save(name)
        else:
            outputImageName = "%09d.png" % self._iteration
            name = os.path.join(outputLogDir, outputImageName)
            self.outputImage.split()[0].save(name)

    def _logBoundingBox(self, bbox):
        """
        Log the current bounding box
        """

        # Create the log directory and log file if necessary
        if not os.path.exists(self.logDir):
            os.makedirs(self.logDir)
        if not hasattr(self, 'bboxLogFile'):
            self.bboxLogFile = open(os.path.join(self.logDir, 'imagesensor_bbox_log.txt'), 'w')

        # Log the bounding box
        print >>self.bboxLogFile, '%d %d %d %d' % (bbox[0], bbox[1], bbox[2], bbox[3])
        self.bboxLogFile.flush()

    def _logOriginalImage(self):
        """
        Save the original, unfiltered image to disk.
        """

        # Create output directory if necessary
        originalLogDir = os.path.join(self.logDir, 'original_images')
        if not os.path.exists(originalLogDir):
            os.makedirs(originalLogDir)

        # Save the original image
        originalImageName = "%09d.png" % self._iteration
        image = self._getOriginalImage().split()[0]
        image.save(os.path.join(originalLogDir, originalImageName))

    def _logLocationImage(self):
        """Save the location of the sensor window to disk (as an image).
        """

        # Create output directory if necessary
        locationLogDir = os.path.join(self.logDir, 'output_locations')
        if not os.path.exists(locationLogDir):
            os.makedirs(locationLogDir)

        # Save the location image
        if not self.locationImage:
            self.locationImage = self._createLocationImage()
        locationImageName = "%09d.png" % self._iteration
        self.locationImage.save(os.path.join(locationLogDir, locationImageName))

    def _createLocationImage(self):
        """Create the 'location' image, with a rectangle around the sensor window.
        """

        if self.prevPosition['reset'] and self.blankWithReset:
            # Create a blank image
            locationImage = Image.new('1', (self.width, self.height))
            if self.invertOutput:
                locationImage = ImageChops.invert(locationImage)
        else:
            # Get the filtered image
            firstImage = self._getFilteredImages(self.prevPosition)[0]
            # Select backdrop upon which sensor position will be overlaid
            if self.logLocationOnOriginalImage:
                filteredImage = firstImage
                firstImage = self._getOriginalImage(self.prevPosition['image'])
                if firstImage.size != filteredImage.size:
                    raise RuntimeError("logLocationOnOriginalImage is True, but the"
                      " filtered image does not match the size of the original"
                      " image, so the location image would be invalid")
            locationImage = Image.new('RGB', firstImage.size)
            locationImage.paste(firstImage, (0,0))
            locationImageDraw = ImageDraw.Draw(locationImage)
            x, y = self.prevPosition['offset']
            x2, y2 = x + self.enabledWidth - 1, y + self.enabledHeight - 1
            locationImageDraw.rectangle((x-1, y-1, x2+1, y2+1), outline='red')
            if locationImage.size[0] > 32 or locationImage.size[1] > 32:
                # Draw again to create a thicker border
                locationImageDraw.rectangle((x-2, y-2, x2+2, y2+2), outline='red')

        return locationImage

    def _writeCategoryToFile(self, category):
        """Write the specified category index to the file at self.categoryOutputFile.

        Args:
            category: Category index (integer).
        """

        if self.categoryOutputFile:  # Only write if we have a valid filename
            if not self._categoryOutputFile:
                self._categoryOutputFile = open(self.categoryOutputFile, 'w')
                # Write a 1 to the first line to specify one entry per line
                self._categoryOutputFile.write('1' + os.linesep)
            self._categoryOutputFile.write(str(category) + os.linesep)
            self._categoryOutputFile.flush()

    def _setFilters(self, filters):
        """
        Change one or more filters, and recompute the ones that changed.

        Args:
            filters: List of filters, where each filter is a list [classname,
                     parameters] (or just a string with the class name, if the
                     filter needs no parameters).

        Filters should be located in the pynodes/ImageSensorFilters directory.
        """

        if filters:
            if not isinstance(filters, list):
                raise TypeError("'filters' must be a list of one or more filters")
            if isinstance(filters, list) and len(filters) == 2 \
                and isinstance(filters[1], dict):
                raise TypeError("'filters' must be a _list_ of filters. If you "
                                "are specifying a filter with the [name, {args}] "
                                "syntax, nest it within a list: [[name, {args}]]")

        filters = copy.deepcopy(filters)

        if self.logFilteredImages:
            # Remove the filter log directory if it exists
            filterLogDir = os.path.join(self.logDir, 'output_from_filters')
            if os.path.exists(filterLogDir):
                shutil.rmtree(filterLogDir)

        if filters is None:
            filters = []
        elif type(filters) is tuple:
            filters = list(filters)
        for i, filter in enumerate(filters):
            if type(filter) is str:
                filters[i] = [filter, {}]
            elif type(filter) is tuple:
                filters[i] = list(filters[i])
            if len(filters[i]) == 1:
                filters[i].append({})
        self.filters = filters
        self._importFilters(self.filters)

        # Validate no filter except the last returns simultaneous responses
        for i in range(len(self.filters)-1):
            outputCount = self.filters[i][2].getOutputCount()
            if type(outputCount) in (tuple, list) and len(outputCount) > 1 \
                and outputCount[1] > 1:
                raise RuntimeError("Only the last filter can return a nested list of "
                  "images (multiple simultaneous responses). "
                  "The %s filter, " % self.filters[i][0] +
                  "index %d of %d, " % (i, len(self.filters)-1) +
                  "creates %d simultaneous responses." % outputCount[1])

        # Invalidate the filtered versions of all images
        for item in self._imageList:
            if item['filtered']:
                item['filtered'] = {}
        self._filterQueue = []
        # Update the pixel count to only count to the original images
        self._pixelCount = 0
        for i in self._imageQueue:
            image = self._imageList[i]['image']
            self._pixelCount += image.size[0] * image.size[1]

        # Tell the explorer about these new filters
        if hasattr(self, 'explorer') and type(self.explorer) == list \
            and len(self.explorer) > 2:
            self.explorer[2].update(numFilters=len(filters),
              numFilterOutputs=self._getNumFilterOutputs(self.filters))

    def _setPostFilters(self, postFilters):
        """Change one or more postFilters, and recompute the ones that changed.

        postFilters -- List of filters, where each filter is a list
          [classname, parameters] (or just a string with the class name, if the
          filter needs no parameters).

        Filters should be located in the pynodes/ImageSensorFilters directory.
        """
        if postFilters:
            if not isinstance(postFilters, list):
                raise TypeError("'postFilters' must be a list of one or more filters")
            if isinstance(postFilters, list) and len(postFilters) == 2 \
                and isinstance(postFilters[1], dict):
                raise TypeError("'postFilters' must be a _list_ of filters. If you "
                                "are specifying a filter with the [name, {args}] "
                                "syntax, nest it within a list: [[name, {args}]]")

        postFilters = copy.deepcopy(postFilters)

        if postFilters is None:
            postFilters = []
        elif type(postFilters) is tuple:
            postFilters = list(postFilters)
        for i, filter in enumerate(postFilters):
            if type(filter) is str:
                postFilters[i] = [filter, {}]
            elif type(filter) is tuple:
                postFilters[i] = list(postFilters[i])
            if len(postFilters[i]) == 1:
                postFilters[i].append({})
        self.postFilters = postFilters
        self._importFilters(self.postFilters)

    def _getNumFilterOutputs(self, filters):
        """Return the number of outputs for each filter.

        Ignores simultaneous outputs.
        """
        numFilterOutputs = []
        for f in filters:
            n = f[2].getOutputCount()
            if type(n) in (tuple, list):
                numFilterOutputs.append(n[0])
            elif type(n) is int:
                numFilterOutputs.append(n)
            else:
                raise RuntimeError("%s filter must return an int or a " % f[0]
                  + "list/tuple of two ints from getOutputCount()")
        return numFilterOutputs

    def _importFilters(self, filters):
        """Import and instantiate all the specified filters.

        This method lives on its own so that it can be used by both _setFilters
        and _setPostFilters.
        """
        for i in range(len(filters)):
            # Import the filter
            # If name is just the class name, such as 'PadToFit', we assume the same
            # name for the module: names = ['PadToFit', 'PadToFit']
            # If name is of the form 'ModuleName.ClassName' (useful to try multiple
            # versions of the same filter): names = ['ModuleName', 'ClassName']
            # By default, ImageSensor searches for filters in
            # nupic.pynodes.ImageSensorFilters. If the import fails, it tries the
            # import unmodified - so you may use filters that are located anywhere
            # that Python knows about.
            if not '.' in filters[i][0]:
                moduleName = className = filters[i][0]
            else:
                components = filters[i][0].split('.')
                moduleName = '.'.join(components[:-1])
                className = components[-1]
            try:
                # Search in ImageSensorFilters first
                filterModule = __import__('pyhtm.nodes.ImageSensorFilters.%s'
                  % moduleName, {}, {}, className)
            except:
                try:
                    filterModule = __import__(moduleName, {}, {}, className)
                except:
                    raise RuntimeError('Could not find filter "%s"' % filters[i][0])
            filterClass = getattr(filterModule, className)
            # Instantiate the filter
            filters[i].append(filterClass(**copy.deepcopy(filters[i][1])))
            filters[i][2].update(mode=self.mode, background=self.background)

    def _setExplorer(self, explorer):
        """Set the explorer (algorithm used to explore the input space).

        Args:
            explorer: List containing the explorer name and parameters.
        """
        if explorer is None:
            raise RuntimeError("Must specify explorer (try 'Flash' for no sweeping)")

        explorer = copy.deepcopy(explorer)

        if type(explorer) is str:
            explorer = [explorer, {}]
        elif type(explorer) is tuple:
            explorer = list(explorer)
        if len(explorer) == 1:
            explorer.append({})

        # Import the explorer
        # If name is just the class name, such as 'Flash', we assume the same
        # name for the module: names = ['Flash', 'Flash']
        # If name is of the form 'ModuleName.ClassName' (useful to try multiple
        # versions of the same explorer): names = ['ModuleName', 'ClassName']
        # By default, ImageSensor searches for explorers in
        # nupic.pynodes.ImageSensorExplorers. If the import fails, it tries the
        # import unmodified - so you may use explorers that are located anywhere
        # that Python knows about.
        if '.' not in explorer[0]:
            moduleName = className = explorer[0]
        else:
            components = explorer[0].split('.')
            moduleName = '.'.join(components[:-1])
            className = components[-1]
        try:
            # Search in ImageSensorExplorers first
            explorerModule = __import__('pyhtm.nodes.ImageSensorExplorers.%s'
              % moduleName, {}, {}, className)
        except ImportError:
            try:
                explorerModule = __import__(moduleName, {}, {}, className)
            except ImportError:
                raise RuntimeError('Could not find explorer "%s"' % explorer[0])
        explorerClass = getattr(explorerModule, className)
        explorerArgs = copy.deepcopy(explorer[1])
        # Append the image accessor methods to the arguments
        explorerArgs.update({
          'getOriginalImage': self._getOriginalImage,
          'getFilteredImages': self._getFilteredImages,
          'getImageInfo': self._getImageInfo
        })

        # Instantiate the explorer
        self.explorer = explorer
        self.explorer.append(explorerClass(**explorerArgs))
        self.explorer[2].update(numImages=len(self._imageList),
          numFilters=len(self.filters),
          numFilterOutputs=self._getNumFilterOutputs(self.filters),
          enabledWidth=self.enabledWidth, enabledHeight=self.enabledHeight,
          blankWithReset=self.blankWithReset)

    def _meetMemoryLimit(self):
        """Unload images as necessary to stay within the memory limit.
        """
        if self.memoryLimit < 0:
            return
        while self._pixelCount * 4 / 1000000.0 > self.memoryLimit:
            if len(self._filterQueue) > 1:
                # Unload the filtered image used least recently
                imageIndex, filterPosition = self._filterQueue.pop()
                filtered = self._imageList[imageIndex]['filtered'][filterPosition]
                for i in range(len(filtered)):
                    self._pixelCount -= filtered[i].size[0] * filtered[i].size[1]
                self._imageList[imageIndex]['filtered'].pop(filterPosition)
            elif self._imageQueue:
                if len(self._imageQueue) == 1 and not self.filters:
                    # No filters and this is the current image - don't unload it
                    break
                # Unload the original image used least recently
                imageIndex = self._imageQueue.pop()
                size = self._imageList[imageIndex]['image'].size
                self._pixelCount -= size[0] * size[1]
                self._imageList[imageIndex]['image'] = None
            else:
                break

    def _updatePrevPosition(self):
        """Deep copy position to self.prevPosition.
        """
        position = self.explorer[2].position
        self.prevPosition = {
          'image': position['image'],
          'filters': copy.copy(position['filters']),
          'offset': copy.copy(position['offset']),
          'reset': position['reset']
        }

    def compute(self, nodeInfo=None, inputs=None):
        """Generate the next sensor output and send it out.

        This method is called by the runtime engine.
        """
        if len(self._imageList) == 0:
            raise RuntimeError("ImageSensor can't run compute: no images loaded")

        self._updatePrevPosition()

        # Get the image(s) to send out
        outputImages, final_output = self._getOutputImages()

        # Compile information about this iteration and log it
        imageInfo = self._getImageInfo()
        if imageInfo['imagePath'] is None:
            filename = ""
        else:
            filename = os.path.split(imageInfo['imagePath'])[1]
        category = imageInfo['categoryIndex']
        if category == -1:
            categoryName = ""
        else:
            categoryName = self.categoryInfo[category][0]
        self._logCommand([
          ('iteration', self._iteration),
          ('position', self.explorer[2].position),
          ('filename', filename),
          ('categoryIndex', category),
          ('categoryName', categoryName),
          ('erode', imageInfo['erode']),
          ('blank', bool(self.prevPosition['reset'] and self.blankWithReset))
        ], None)

        # If we don't have a partition ID at this point (e.g., because
        # of memory limits), then we need to try and pull from the
        # just-loaded image
        if imageInfo['partitionID'] is None:
            imgPosn = self.explorer[2].position['image']
            imageInfo['partitionID'] = self._imageList[imgPosn].get('partitionID')

        if self.depth == 1:
            self.outputImage = outputImages[0]
        else:
            self.outputImage = outputImages

        # Invalidate the old location image
        self.locationImage = None

        # Log the images and locations if specified
        if self.logOutputImages:
            self._logOutputImages()
        if self.logOriginalImages:
            self._logOriginalImage()
        if self.logLocationImages:
            self._logLocationImage()

        self.explorer[2].next()
        self._iteration += 1

        # Save category to file
        self._writeCategoryToFile(category)

        outputs = numpy.array([[]])

        # Convert the output images to a numpy vector
        croppedArrays = [numpy.asarray(image.split()[0])
          for image in outputImages]
        # Pad the images to fit the full output size if necessary generating
        # a stack of images, each of them self.width X self.height
        pad = (self.depth > 1 or \
              croppedArrays[0].shape != (self.height, self.width))
        if pad:
            fullArrays = [numpy.zeros((self.height, self.width))
              for i in range(self.depth)]
            for i in range(self.depth):
                fullArrays[i][:croppedArrays[i].shape[0],:croppedArrays[i].shape[1]] \
                  = croppedArrays[i]
        else:
            fullArrays = croppedArrays
        # Flatten and concatenate the arrays
        outputArray = numpy.concatenate([a.flat for a in fullArrays])

        # Send black and white images as (0, 1) instead of (0, 255)
        if self.mode == 'bw':
            outputArray = numpy.multiply(outputArray, 1/255)
#            outputArray /= 255

        # dataOut - main output
        if final_output is None:
            outputs = outputArray
        else:
            outputs = final_output

        # outputs /= 255.

        # apply filter for finding edges
        if False:
            image_data = numpy.reshape(outputs, (self.width, self.height))
            im = Image.fromarray(image_data)
            im = im.filter(ImageFilter.FIND_EDGES)
            im = im.filter(ImageFilter.SMOOTH)
            outputs = numpy.asarray(im.getdata())

        # TODO use namedtuple
        return {
                'data': numpy.reshape(outputs, (self.width, self.height)),
                'category': category,
                'category_name': categoryName,
                'is_reset': bool(self.prevPosition['reset'])
               }

    def getParameter(self, parameterName, nodeSet=""):
        """Get the value of an ImageSensor parameter."""

        if parameterName == 'filters':
            # Remove filter objects
            return [filter[:2] for filter in self.filters]

        elif parameterName == 'postFilters':
            # Remove filter objects
            return [filter[:2] for filter in self.postFilters]

        elif parameterName == 'explorer':
            # Remove explorer object
            return self.explorer[:2]

        elif parameterName == 'numImages':
            return len(self._imageList)

        elif parameterName == 'numMasks':
            return len([True for x in self._imageList if x['maskPath']])

        elif parameterName in ('numIterations', 'maxOutputVectorCount'):
            return self.getNumIterations()

        elif parameterName == 'activeOutputCount':
            return self.width * self.height * self.depth

        elif parameterName == 'position':
            return self.explorer[2].position

        elif parameterName == 'imageInfo':
            return [self._getImageInfo(i) for i in range(len(self._imageList))]

        elif parameterName == 'prevImageInfo':
            if self.prevPosition and self._imageList:
                return self._getImageInfo(self.prevPosition['image'])
            else:
                return None

        elif parameterName == 'nextImageInfo':
            if self.explorer[2].position and self._imageList:
                return self._getImageInfo()
            else:
                return None

        elif parameterName == 'categoryInfo':
            return serializeCategoryInfo(self.categoryInfo)

        elif parameterName == 'outputImage':
            if self._iteration == 0:
                return
            if self.depth == 1:
                return serializeImage(self.outputImage.split()[0])
            else:
                return [serializeImage(image.split()[0]) for image in self.outputImage]

        elif parameterName == 'outputImageWithAlpha':
            if self._iteration == 0:
                return
            if self.depth == 1:
                return serializeImage(self.outputImage)
            else:
                return [serializeImage(image) for image in self.outputImage]

        elif parameterName == 'originalImage':
            if not self._imageList or self._iteration == 0:
                return
            return serializeImage(
              self._getOriginalImage(self.prevPosition['image']).split()[0])

        elif parameterName == 'locationImage':
            if not self._imageList or self._iteration == 0 or not self.prevPosition:
                return
            if not self.locationImage:
                self.locationImage = self._createLocationImage()
            return serializeImage(self.locationImage)

        elif parameterName == 'background':
            if self.mode == 'bw':
                return self.background / 255
            else:
                return self.background

    def setParameter(self, parameterName, parameterValue, nodeSet=""):
        """Set the value of an ImageSensor parameter."""

        if parameterName == 'filters':
            self._setFilters(parameterValue)

        elif parameterName == 'postFilters':
            self._setPostFilters(parameterValue)

        elif parameterName == 'explorer':
            self._setExplorer(parameterValue)

        elif parameterName == 'enabledWidth':
            self.enabledWidth = parameterValue
            self.explorer[2].update(enabledWidth=parameterValue)

        elif parameterName == 'enabledHeight':
            self.enabledHeight = parameterValue
            self.explorer[2].update(enabledHeight=parameterValue)

        elif parameterName == 'blankWithReset':
            self.blankWithReset = parameterValue
            self.explorer[2].update(blankWithReset=parameterValue)

        elif parameterName == 'categoryOutputFile':
            if self._categoryOutputFile:
                self._categoryOutputFile.close()
                self._categoryOutputFile = None
            self.categoryOutputFile = parameterValue

        elif parameterName == 'categoryInfo':
            self.categoryInfo = deserializeCategoryInfo(parameterValue)
            # TODO change the names and indices of the loaded image?

        elif parameterName == 'background':
            self.background = parameterValue
            if self.mode == 'bw':
                self.background *= 255
            for filter in self.filters + self.postFilters:
                filter[2].update(background=self.background)

        elif parameterName == 'logDir':
            if hasattr(self, 'logFile') and self.logDir != parameterValue:
                self.logFile.close()
                del self.logFile
            if hasattr(self, 'bboxLogFile') and self.logDir != parameterValue:
                self.bboxLogFile.close()
                del self.bboxLogFile
            self.logDir = parameterValue

        elif parameterName == 'logText':
            self.logText = parameterValue
            if hasattr(self, 'logFile') and not self.logText:
                self.logFile.close()
                del self.logFile

        elif parameterName == 'memoryLimit':
            self.memoryLimit = parameterValue
            self._meetMemoryLimit()

    def __getstate__(self):
        """Get serializable state."""

        # Serialize images stored in categoryInfo
        serializedCategoryInfo = [(name, b64encode(imageStr)) for name, imageStr
                                  in self.getParameter('categoryInfo')]

        # Get the object-less filters and explorer
        resetFilters = self.getParameter('filters')
        resetPostFilters = self.getParameter('postFilters')
        resetExplorer = self.getParameter('explorer')

        # Compile a dictionary of attributes to save
        state = GetSomeVars(self, ('width', 'height', 'depth', 'mode',
          'blankWithReset', 'enabledWidth', 'enabledHeight', 'invertOutput',
          'background', 'automaskingTolerance', 'automaskingPadding',
          'memoryLimit', 'minimalBoundingBox'))

        # Add attributes that have been manipulated
        state.update({'serializedCategoryInfo': serializedCategoryInfo,
          'resetExplorer': resetExplorer, 'resetFilters': resetFilters,
          'resetPostFilters': resetPostFilters})

        # Save a version number
        state['version'] = 1.7

        return state

    def __setstate__(self, state):
        """Set state from serialized state."""

        if type(state) is tuple:
            raise RuntimeError("Cannot convert legacy ImageSensor state")

        # Get the version number
        version = state.pop('version')

        if version < 1.6:
            raise RuntimeError("Cannot convert legacy ImageSensor state")

        # Get attributes that need to be manipulated
        serializedCategoryInfo = state.pop('serializedCategoryInfo')
        resetFilters = state.pop('resetFilters')
        resetPostFilters = state.pop('resetPostFilters')
        resetExplorer = state.pop('resetExplorer')

        UpdateMembers(self, state)

        # Deserialize images stored in categoryInfo (not base64-encoded)
        if version >= 1.64:
            # Undo base64 encoding
            serializedCategoryInfo = [(name, b64decode(imageStr)) for name, imageStr
                                      in serializedCategoryInfo]
        self.setParameter('categoryInfo', serializedCategoryInfo)

        # Set variables that weren't saved
        self._imageList = []
        self._imageQueue = []
        self._filterQueue = []
        self._pixelCount = 0
        self._iteration = 0
        self.logText = False
        self.logOutputImages = False
        self.logOriginalImages = False
        self.logFilteredImages = False
        self.logLocationImages = False
        self.logLocationOnOriginalImage = False
        self.logBoundingBox = False
        self.logDir = "imagesensor_log"
        self.categoryOutputFile = None
        self._categoryOutputFile = None
        self.outputImage = None
        self.locationImage = None
        self.prevPosition = None

        # Set up the filters and explorer
        self._setFilters(resetFilters)
        self._setPostFilters(resetPostFilters)
        self._setExplorer(resetExplorer)

        # Backward compatibility
        if version < 1.63:
            if not hasattr(self, 'automaskingTolerance'):
                self.automaskingTolerance = 0
            if not hasattr(self, 'automaskingPadding'):
                self.automaskingPadding = 0

        if version < 1.65:
            # Set to True, the old behavior, though it is set to False by default
            # in new networks
            self.minimalBoundingBox = True


def serializeCategoryInfo(categoryInfo):
    return [[name, serializeImage(image)] for name, image in categoryInfo]


def deserializeCategoryInfo(sCategoryInfo):
    if sCategoryInfo is None: return []
    return [[name, (deserializeImage(sImage) if sImage is not None else None)]
            for name, sImage in sCategoryInfo]


def _serializeImageList(imageList):
    sImageList = []
    for i in range(len(imageList)):
        sImageList.append(imageList[i].copy())
        if sImageList[i]['image']:
            sImageList[i]['image'] = serializeImage(sImageList[i]['image'])
        if sImageList[i]['filtered']:
            sImageList[i]['filtered'] = _serializeAllImages(sImageList[i]['filtered'])
    return sImageList


def _deserializeImageList(sImageList):
    imageList = sImageList
    for i in range(len(imageList)):
        if imageList[i]['image']:
            imageList[i]['image'] = deserializeImage(imageList[i]['image'])
        if imageList[i]['filtered']:
            imageList[i]['filtered'] = _deserializeAllImages(imageList[i]['filtered'])
    return imageList


def _serializeAllImages(old):
    new = {}
    for key in old:
        new[key] = [serializeImage(image) for image in old[key]]
    return new


def _deserializeAllImages(old):
    new = {}
    for key in old:
        new[key] = [deserializeImage(sImage) for sImage in old[key]]
    return new
