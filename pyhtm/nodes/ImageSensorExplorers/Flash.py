from pyhtm.nodes.ImageSensorExplorers.BaseExplorer import BaseExplorer


class Flash(BaseExplorer):
    """This explorer flashes each filtered image without sweeping.

    It centers each image, but does not resize them. If an image is larger than
    the sensor's size, only the center portion of it will be visible.

    Use this explorer for flash inference or any other time you want your
    images to be shown in order with no sweeping.

    This explorer does not use reset signals.
    """

    def next(self, seeking=False):
        """Go to the next position (next iteration).

        Args:
            seeking: Boolean that indicates whether the explorer is calling next()
                     from seek(). If True, the explorer should avoid unnecessary
                     computation that would not affect the seek command. The
                     last call to next() from seek() will be with seeking=False.
        """
        # Iterate through the filters
        for i in range(self.numFilters):
            self.position['filters'][i] += 1
            if self.position['filters'][i] < self.numFilterOutputs[i]:
                if not seeking:
                    # Center the image
                    self.centerImage()
                return
            self.position['filters'][i] = 0
        # Go to the next image
        self.position['image'] += 1
        if self.position['image'] == self.numImages:
            self.position['image'] = 0
        if not seeking:
            # Center the image
            self.centerImage()

    def seek(self, iteration=None, position=None):
        """Seek to the specified position or iteration.

        Args:
            iteration: Target iteration number (or None)
            position: Target position (or None)

        BaseExplorer checks validity of inputs, checks that one (but not both) of
        position and iteration are None, and checks that if position is not None,
        at least one of its values is not None.

        Updates value of position.
        """
        if iteration is not None:
            self.first()
            self.position['image'] = (iteration / self.numFilteredVersionsPerImage) \
                                     % self.numImages
            remainingIterations = iteration % self.numFilteredVersionsPerImage
            if remainingIterations:
                for i in range(remainingIterations - 1):
                    self.next(seeking=True)
                self.next()
        else:
            if position['image'] is not None:
                self.position['image'] = position['image']
            if position['filters'] is not None:
                self.position['filters'] = position['filters']
            # Ignore 'offset' and 'reset', which are not used by this explorer

    def getNumIterations(self, image):
        """Get the number of iterations required to completely explore the
        input space.

        Explorers that do not wish to support this method should not override it.

        Args:
            image: If None, returns the sum of the iterations for all the loaded
                   images. Otherwise, image should be an integer specifying the
                   image for which to calculate iterations.

        ImageSensor takes care of the input validation.
        """
        if image is not None:
            return self.numFilteredVersionsPerImage
        else:
            return self.numFilteredVersionsPerImage * self.numImages
