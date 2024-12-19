# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 University of Dundee & Open Microscopy Environment.
# All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

import numpy as np
from collections import defaultdict
from omero.gateway import ColorHolder
from omero.model import MaskI
from omero.rtypes import (
    rdouble,
    rint,
    rstring,
)


class NoMaskFound(ValueError):
    """
    Exception thrown when no foreground pixels were found in a mask
    """

    def __init__(self, msg="No mask found"):
        super(Exception, self).__init__(msg)


class InvalidBinaryImage(ValueError):
    """
    Exception thrown when invalid labels are found
    """

    def __init__(self, msg="Invalid labels found"):
        super(Exception, self).__init__(msg)


def mask_from_binary_image(
    binim, rgba=None, z=None, c=None, t=None, text=None, raise_on_no_mask=True
):
    """
    Create a mask shape from a binary image (background=0)

    :param numpy.array binim: Binary 2D array, must contain values [0, 1] only
    :param rgba int-4-tuple: Optional (red, green, blue, alpha) colour
    :param z: Optional Z-index for the mask
    :param c: Optional C-index for the mask
    :param t: Optional T-index for the mask
    :param text: Optional text for the mask
    :param raise_on_no_mask: If True (default) throw an exception if no mask
           found, otherwise return an empty Mask
    :return: An OMERO mask
    :raises NoMaskFound: If no labels were found
    :raises InvalidBinaryImage: If the maximum labels is greater than 1
    """

    # Find bounding box to minimise size of mask
    xmask = binim.sum(0).nonzero()[0]
    ymask = binim.sum(1).nonzero()[0]
    if any(xmask) and any(ymask):
        x0 = min(xmask)
        w = max(xmask) - x0 + 1
        y0 = min(ymask)
        h = max(ymask) - y0 + 1
        submask = binim[y0 : (y0 + h), x0 : (x0 + w)]
        if not np.array_equal(np.unique(submask), [0, 1]) and not np.array_equal(
            np.unique(submask), [1]
        ):
            raise InvalidBinaryImage()
    else:
        if raise_on_no_mask:
            raise NoMaskFound()
        x0 = 0
        w = 0
        y0 = 0
        h = 0
        submask = []

    mask = MaskI()
    # BUG in older versions of Numpy:
    # https://github.com/numpy/numpy/issues/5377
    # Need to convert to an int array
    # mask.setBytes(np.packbits(submask))
    mask.setBytes(np.packbits(np.asarray(submask, dtype=int)))
    mask.setWidth(rdouble(w))
    mask.setHeight(rdouble(h))
    mask.setX(rdouble(x0))
    mask.setY(rdouble(y0))

    if rgba is not None:
        ch = ColorHolder.fromRGBA(*rgba)
        mask.setFillColor(rint(ch.getInt()))
    if z is not None:
        mask.setTheZ(rint(z))
    if c is not None:
        mask.setTheC(rint(c))
    if t is not None:
        mask.setTheT(rint(t))
    if text is not None:
        mask.setTextValue(rstring(text))

    return mask


def masks_from_label_image(
    labelim, rgba=None, z=None, c=None, t=None, text=None, raise_on_no_mask=True
):
    """
    Create mask shapes from a label image (background=0)

    :param numpy.array labelim: 2D label array
    :param rgba int-4-tuple: Optional (red, green, blue, alpha) colour
    :param z: Optional Z-index for the mask
    :param c: Optional C-index for the mask
    :param t: Optional T-index for the mask
    :param text: Optional text for the mask
    :param raise_on_no_mask: If True (default) throw an exception if no mask
           found, otherwise return an empty Mask
    :return: A list of OMERO masks in label order ([] if no labels found)

    """
    masks = []
    for i in range(1, labelim.max() + 1):
        mask = mask_from_binary_image(
            labelim == i, rgba, z, c, t, text, raise_on_no_mask
        )
        masks.append(mask)
    return masks


def masks_from_3d_label_image(
      planes, rgba=None, z=None, c=None, z_stack=True, t=None, text=None):
    """
    Create mask shapes from a 3d label image, either z stack or timepoints,
    grouped by label ID (background=0)

    :param list of numpy.array planes: list of 2D label arrays in z order
    :param rgba int-4-tuple: Optional (red, green, blue, alpha) colour
    :param z: Optional Z-index for the mask
    :param c: Optional C-index for the mask
    :param t: Optional T-index for the mask
    :param z_stack: Flag if the planes represent a z stack, timepoints
                    otherwise (default: True)
    :param text: Optional text for the mask
    :return: A dictionary of OMERO masks with the labels as keys
           ({} if no labels found)

    """
    masks = defaultdict(list)
    for i, plane in enumerate(planes):
        if z_stack:
            plane_masks = masks_from_label_image(plane, rgba, i, c, t,
                                                 text, False)
        else:
            plane_masks = masks_from_label_image(plane, rgba, z, c, i,
                                                 text, False)
        for label, mask in enumerate(plane_masks):
            if mask.getBytes().any():
                masks[label].append(mask)
    return masks
