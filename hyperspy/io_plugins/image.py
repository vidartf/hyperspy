# -*- coding: utf-8 -*-
# Copyright 2007-2011 The HyperSpy developers
#
# This file is part of  HyperSpy.
#
#  HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
#  HyperSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with  HyperSpy.  If not, see <http://www.gnu.org/licenses/>.

import os

from scipy.misc import imread, imsave

from hyperspy.misc.rgb_tools import regular_array2rgbx

# Plugin characteristics
# ----------------------
format_name = 'Image'
description = 'Import/Export standard image formats using PIL or freeimage'
full_support = False
file_extensions = ['png', 'bmp', 'dib', 'gif', 'jpeg', 'jpe', 'jpg',
                   'msp', 'pcx', 'ppm', "pbm", "pgm", 'xbm', 'spi', ]
default_extension = 0  # png


# Writing features
writes = [(2, 0), ]
# ----------------------


# TODO Extend it to support SI
def file_writer(filename, signal, file_format='png', **kwds):
    '''Writes data to any format supported by PIL

        Parameters
        ----------
        filename: str
        signal: a Signal instance
        file_format : str
            The fileformat defined by its extension that is any one supported by
            PIL.
    '''
    imsave(filename, signal.data)


def file_reader(filename, **kwds):
    '''Read data from any format supported by PIL.

    Parameters
    ----------
    filename: str

    '''
    dc = imread(filename)
    if len(dc.shape) > 2:
        # It may be a grayscale image that was saved in the RGB or RGBA
        # format
        if (dc[:, :, 1] == dc[:, :, 2]).all() and \
                (dc[:, :, 1] == dc[:, :, 2]).all():
            dc = dc[:, :, 0]
        else:
            dc = regular_array2rgbx(dc)
    return [{'data': dc,
             'metadata':
             {
                 'General': {'original_filename': os.path.split(filename)[1]},
                 "Signal": {'signal_type': "",
                            'record_by': 'image', },
             }
             }]
