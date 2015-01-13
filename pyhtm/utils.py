import numpy as np
import PIL


class RingBuffer:
    def __init__(self, size):
        assert(size > 0)
        self.size = size
        self.data = [None] * self.size
        self.counter = 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

    def append(self, x):
        """Append new item and remove the oldest one (to keep the same size).

        Args:
            x: item to be stored in the buffer
        """
        self.data.pop(0)
        self.data.append(x)
        if self.counter < len(self.data):
            self.counter += 1

    def get(self):
        return self.data

    def filled_items(self):
        """Return the number of filled items.
        """
        i = 0
        for d in self.data:
            if d is not None:
                i += 1
        return i

    def is_filled(self):
        return self.counter == len(self.data)

    def reset(self):
        """Reset buffer to the inital empty state.
        """
        for i in range(self.size):
            self.data[i] = None
        self.counter = 0


def scale_to_unit_interval(ndar, eps=1e-8):
    """Scale all values in the ndarray ndar to be between 0 and 1.

    Args:
        ndar: numpy array
        eps: numerical precission

    Returns:
        new numpy array having each value scaled to (0, 1)
    """
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max() + eps)
    return ndar


def fill_default_params(params, default_params):
    """Add key-value pairs to a dict from a dict with default values if it does
    not exist.

    Args:
        params: dict
        default_params: dict with default values

    Returns:
        dict having set all keys as the dict with default parameters
    """
    for k, v in default_params.items():
        if k not in params.keys():
            params[k] = v
    return params


def crop_array(array, box):
    """Crop a rectangle from a 2D array.

    Args:
        array: 2D array
        box: defines coordinates of the top left and bottom right corners
             box = [top_left_x, top_left_y, bottom_right_x, bottom_right_y]

    Returns:
        2D array
    """
    box = [int(x) for x in box]
    return array[box[0]:box[2], box[1]:box[3]]


def is_list(l):
    """Check for list-like structure (list, tuple, numpy array).

    Args:
        l: data structure

    Returns:
        boolean
    """
    return type(l) in [list, tuple, np.ndarray]


def is_nested_list(l):
    """Check for nested list-like structure.

    Args:
        l: list-like structure

    Returns:
        boolean
    """
    return is_list(l) and len(l) > 0 and is_list(l[0])


def flatten_list(l):
    """Flatten a nested list.
    from [[x1], [x2], ..., [xn]] makes = [x1, x2, ...,xn]

    Args:
        l: nested list

    Returns:
        flattened list
    """
    if not is_nested_list(l):# FIXME or not isinstance(l, numpy.ndarray):
        return l
    return [j for i in l for j in i]


def count_patches(im_size, patch_size, overlap_size=[0, 0]):
    """Determine the number of patches along x an y axis.
    """
    if overlap_size[0] > 0:
        assert(im_size[0] % overlap_size[0] == 0)
        assert(im_size[1] % overlap_size[1] == 0)

        x_range = (im_size[0] - overlap_size[0]) / (patch_size[0] - overlap_size[0])
        y_range = (im_size[1] - overlap_size[1]) / (patch_size[1] - overlap_size[1])
    else:
        x_range = int(im_size[0] / patch_size[0])
        y_range = int(im_size[1] / patch_size[1])

    return (int(x_range), int(y_range))


def extract_patches(im, patch_size=[8, 8], overlap=[0, 0]):
    """Extract data_sequence of given size from the image.
    Image is swiped horizontally starting from the top left corner.
    Image is a numpy array.

    Args:
        im: a numpy array representing the image bitmap
        patch_size: a list defining the size of the image patch in pixels
        overlap: a list defining the size of the overlap in pixels

    Returns:
        a list of numpy arrays representing the image patches
    """
    if not isinstance(im, np.ndarray):
        if isinstance(im, PIL.Image.Image):
            im = np.array(im)
        else:
            raise TypeError("Parameter 'im' is not a numpy array.")
    x_range, y_range = count_patches(im.shape, patch_size, overlap)

    data_sequence = []

    ps_x = patch_size[0] - overlap[0]
    ps_y = patch_size[1] - overlap[1]

    patch_pixels = patch_size[0] * patch_size[1]  # number of pixels in a patch

    for i in range(x_range):
        for j in range(y_range):
            box = (i * ps_x, j * ps_y,
                   i * ps_x + patch_size[0], j * ps_y + patch_size[1])

            data_sequence.append(crop_array(im, box).reshape(patch_pixels, 1))
    return np.asarray(data_sequence)


def symmetrize(a):
    """Make a matrix symmetric.

    Args:
        a: 2D array
    Returns:
        2D array
    """
    return a + a.T - np.diag(a.diagonal())


def normalize_to_one(x):
    """Normalize vector to sum up to 1.

    Args:
        x: a numpy vector

    Returns:
        a numpy vector
    """
    x_sum = x.sum()
    if x_sum == 0:
        raise ValueError('The sum of the array elements must not be zero.')
    else:
        return x / x_sum


def flatten_patches(patches, patch_size, image_size):
    """Flatten list of 2D arrays (image patches).

    Args:
        patches: list of patches in the order starting from the top left corner
        image: a list or tuple of image dimmensions

    Retuns:
        an image array
    """
    if len(patches) == 0 or np.sum(image_size) == 0:
        return None

    row = np.array([])
    im = np.array([])
    num_width_patches = int(image_size[0] / patch_size[0])

    for i in range(len(patches)):
        p = patches[i].reshape(patch_size)
        if len(row) == 0:
            row = p
        else:
            row = np.hstack((row, p))
        if i > 0 and (i+1) % num_width_patches == 0:
            if len(im) == 0:
                im = row.copy()
            else:
                im = np.vstack((im, row))
            row = np.array([])

    return im


def multi_delete(list_, indices):
    indices = sorted(list(indices), reverse=True)
    for index in indices:
        del list_[index]
    return list_


def get_max_indices(l):
    """Compute indices of maximal values for all vectors in a list.

    Example:
        input: [[23, 1, 4], [2.4, 5, 11], [44, 22, 105, 18]]
        output: [0, 2, 2]

    Args:
        l: a list of numpy vectors

    Returns:
        a numpy vector
    """
    return np.asarray([np.argmax(m) for m in l])


def save_data_to_weka(data, labels, filename, label_names=None):
    """Save numpy array to a file in WEKA format.
    http://weka.wikispaces.com/ARFF+%28stable+version%29

    Args:
        data: a numpy array
        labels: a numpy array of label indices
        filename: string
        label_names: list of label names (their order has to match the label
                     indices)
    """
    comment = ''
    relation_name = 'htm'
    n_rows, n_cols = data.shape
    assert(n_rows == len(labels))

    f = open(filename, 'w')
    f.write('% '+comment+'\n')
    f.write('\n')
    f.write('@RELATION '+relation_name+'\n')

    attributenames = ['attr%d' % i for i in range(n_cols)]

    for a in attributenames:
        f.write('@ATTRIBUTE '+str(a)+' NUMERIC\n')  # assume values are numeric

    # class labels attribute
    if label_names is None:
        label_names = [str(i) for i in set(labels)]
    f.write('@ATTRIBUTE class {%s}\n' % ','.join(label_names))

    f.write('\n')
    f.write('@DATA\n')  # write the data, one attribute vector per line
    for i in range(n_rows):
        for j in range(n_cols):
            f.write(str(data[i, j]))
            f.write(',')
        f.write(str(label_names[labels[i]]))
        f.write('\n')

    f.close()


def flatten_array(a):
    return a.reshape((np.prod(a.shape),))
