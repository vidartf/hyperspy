# -*- coding: utf-8 -*-
# Copyright 2010 Stefano Mazzucco
# Copyright 2011 The HyperSpy developers
#
# This file is part of  HyperSpy. It is a fork of the original PIL dm3 plugin
# written by Stefano Mazzucco.
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

# Plugin to read the Gatan Digital Micrograph(TM) file format

from __future__ import with_statement  # for Python versions < 2.6
from __future__ import division

import os

import numpy as np
import traits.api as t

from hyperspy.misc.io.utils_readfile import *
from hyperspy.exceptions import *
import hyperspy.misc.io.tools
from hyperspy.misc.utils import DictionaryTreeBrowser


# Plugin characteristics
# ----------------------
format_name = 'Digital Micrograph dm3'
description = 'Read data from Gatan Digital Micrograph (TM) files'
full_suport = False
# Recognised file extension
file_extensions = ('dm3', 'DM3', 'dm4', 'DM4')
default_extension = 0

# Writing features
writes = False
# ----------------------


class DigitalMicrographReader(object):

    """ Class to read Gatan Digital Micrograph (TM) files.

    Currently it supports versions 3 and 4.

    Attributes
    ----------
    dm_version, endian, tags_dict

    Methods
    -------
    parse_file, parse_header, get_image_dictionaries

    """

    _complex_type = (15, 18, 20)
    simple_type = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)

    def __init__(self, f, verbose=False):
        self.verbose = verbose
        self.dm_version = None
        self.endian = None
        self.tags_dict = None
        self.f = f

    def parse_file(self):
        self.f.seek(0)
        self.parse_header()
        self.tags_dict = {"root": {}}
        number_of_root_tags = self.parse_tag_group()[2]
        if self.verbose is True:
            print('Total tags in root group:', number_of_root_tags)
        self.parse_tags(
            number_of_root_tags,
            group_name="root",
            group_dict=self.tags_dict)

    def parse_header(self):
        self.dm_version = read_long(self.f, "big")
        if self.dm_version not in (3, 4):
            print('File address:', dm_version[1])
            raise NotImplementedError(
                "Currently we only support reading DM versions 3 and 4 but "
                "this file "
                "seems to be version %s " % self.dm_version)
        self.skipif4()
        filesizeB = read_long(self.f, "big")
        is_little_endian = read_long(self.f, "big")

        if self.verbose is True:
            # filesizeMB = filesizeB[3] / 2.**20
            print('DM version: %i' % self.dm_version)
            print('size %i B' % filesizeB)
            print('Is file Little endian? %s' % bool(is_little_endian))
        if bool(is_little_endian):
            self.endian = 'little'
        else:
            self.endian = 'big'

    def parse_tags(self, ntags, group_name='root', group_dict={}):
        """Parse the DM file into a dictionary.

        """
        unnammed_data_tags = 0
        unnammed_group_tags = 0
        for tag in xrange(ntags):
            if self.verbose is True:
                print('Reading tag name at address:', self.f.tell())
            tag_header = self.parse_tag_header()
            tag_name = tag_header['tag_name']

            skip = True if (group_name == "ImageData" and
                            tag_name == "Data") else False
            if self.verbose is True:
                print('Tag name:', tag_name[:20])
                print('Tag ID:', tag_header['tag_id'])

            if tag_header['tag_id'] == 21:  # it's a TagType (DATA)
                if not tag_name:
                    tag_name = 'Data%i' % unnammed_data_tags
                    unnammed_data_tags += 1

                if self.verbose is True:
                    print('Reading data tag at address:', self.f.tell())

                # Start reading the data
                # Raises IOError if it is wrong
                self.check_data_tag_delimiter()
                self.skipif4()
                infoarray_size = read_long(self.f, 'big')
                if self.verbose:
                    print("Infoarray size ", infoarray_size)
                self.skipif4()
                if infoarray_size == 1:  # Simple type
                    if self.verbose:
                        print("Reading simple data")
                    etype = read_long(self.f, "big")
                    data = self.read_simple_data(etype)
                elif infoarray_size == 2:  # String
                    if self.verbose:
                        print("Reading string")
                    enctype = read_long(self.f, "big")
                    if enctype != 18:
                        raise IOError("Expected 18 (string), got %i" % enctype)
                    string_length = self.parse_string_definition()
                    data = self.read_string(string_length, skip=skip)
                elif infoarray_size == 3:  # Array of simple type
                    if self.verbose:
                        print("Reading simple array")
                    # Read array header
                    enctype = read_long(self.f, "big")
                    if enctype != 20:  # Should be 20 if it is an array
                        raise IOError("Expected 20 (string), got %i" % enctype)
                    size, enc_eltype = self.parse_array_definition()
                    data = self.read_array(size, enc_eltype, skip=skip)
                elif infoarray_size > 3:
                    enctype = read_long(self.f, "big")
                    if enctype == 15:  # It is a struct
                        if self.verbose:
                            print("Reading struct")
                        definition = self.parse_struct_definition()
                        if self.verbose:
                            print("Struct definition ", definition)
                        data = self.read_struct(definition, skip=skip)
                    elif enctype == 20:  # It is an array of complex type
                        # Read complex array info
                        # The structure is
                        # 20 <4>, ?  <4>, enc_dtype <4>, definition <?>,
                        # size <4>
                        self.skipif4()
                        enc_eltype = read_long(self.f, "big")
                        if enc_eltype == 15:  # Array of structs
                            if self.verbose:
                                print("Reading array of structs")
                            definition = self.parse_struct_definition()
                            self.skipif4()  # Padding?
                            size = read_long(self.f, "big")
                            if self.verbose:
                                print("Struct definition: ", definition)
                                print("Array size: ", size)
                            data = self.read_array(
                                size=size,
                                enc_eltype=enc_eltype,
                                extra={"definition": definition},
                                skip=skip)
                        elif enc_eltype == 18:  # Array of strings
                            if self.verbose:
                                print("Reading array of strings")
                            string_length = \
                                self.parse_string_definition()
                            size = read_long(self.f, "big")
                            data = self.read_array(
                                size=size,
                                enc_eltype=enc_eltype,
                                extra={"length": string_length},
                                skip=skip)
                        elif enc_eltype == 20:  # Array of arrays
                            if self.verbose:
                                print("Reading array of arrays")
                            el_length, enc_eltype = \
                                self.parse_array_definition()
                            size = read_long(self.f, "big")
                            data = self.read_array(
                                size=size,
                                enc_eltype=enc_eltype,
                                extra={"size": el_length},
                                skip=skip)

                else:  # Infoarray_size < 1
                    raise IOError("Invalided infoarray size ", infoarray_size)

                if self.verbose:
                    print("Data: %s" % str(data)[:70])
                group_dict[tag_name] = data

            elif tag_header['tag_id'] == 20:  # it's a TagGroup (GROUP)
                if not tag_name:
                    tag_name = 'TagGroup%i' % unnammed_group_tags
                    unnammed_group_tags += 1
                if self.verbose is True:
                    print('Reading Tag group at address:', self.f.tell())
                ntags = self.parse_tag_group(skip4=3)[2]
                group_dict[tag_name] = {}
                self.parse_tags(
                    ntags=ntags,
                    group_name=tag_name,
                    group_dict=group_dict[tag_name])
            else:
                print('File address:', self.f.tell())
                raise DM3TagIDError(tag_header['tag_id'])

    def get_data_reader(self, enc_dtype):
    # _data_type dictionary.
    # The first element of the InfoArray in the TagType
    # will always be one of _data_type keys.
    # the tuple reads: ('read bytes function', 'number of bytes', 'type')

        dtype_dict = {
            2: (read_short, 2, 'h'),
            3: (read_long, 4, 'l'),
            4: (read_ushort, 2, 'H'),  # dm3 uses ushorts for unicode chars
            5: (read_ulong, 4, 'L'),
            6: (read_float, 4, 'f'),
            7: (read_double, 8, 'd'),
            8: (read_boolean, 1, 'B'),
            # dm3 uses chars for 1-Byte signed integers
            9: (read_char, 1, 'b'),
            10: (read_byte, 1, 'b'),   # 0x0a
            11: (read_double, 8, 'l'),  # Unknown, new in DM4
            12: (read_double, 8, 'l'),  # Unknown, new in DM4
            15: (self.read_struct, None, 'struct',),  # 0x0f
            18: (self.read_string, None, 'c'),  # 0x12
            20: (self.read_array, None, 'array'),  # 0x14
        }
        return dtype_dict[enc_dtype]

    def skipif4(self, n=1):
        if self.dm_version == 4:
            self.f.seek(4 * n, 1)

    def parse_array_definition(self):
        """Reads and returns the element type and length of the array.

        The position in the file must be just after the
        array encoded dtype.

        """
        self.skipif4()
        enc_eltype = read_long(self.f, "big")
        self.skipif4()
        length = read_long(self.f, "big")
        return length, enc_eltype

    def parse_string_definition(self):
        """Reads and returns the length of the string.

        The position in the file must be just after the
        string encoded dtype.
        """
        self.skipif4()
        return read_long(self.f, "big")

    def parse_struct_definition(self):
        """Reads and returns the struct definition tuple.

        The position in the file must be just after the
        struct encoded dtype.

        """
        self.f.seek(4, 1)  # Skip the name length
        self.skipif4(2)
        nfields = read_long(self.f, "big")
        definition = ()
        for ifield in xrange(nfields):
            self.f.seek(4, 1)
            self.skipif4(2)
            definition += (read_long(self.f, "big"),)

        return definition

    def read_simple_data(self, etype):
        """Parse the data of the given DM3 file f
        with the given endianness (byte order).
        The infoArray iarray specifies how to read the data.
        Returns the tuple (file address, data).
        The tag data is stored in the platform's byte order:
        'little' endian for Intel, PC; 'big' endian for Mac, Motorola.
        If skip != 0 the data is actually skipped.
        """
        data = self.get_data_reader(etype)[0](self.f, self.endian)
        if isinstance(data, str):
            data = hyperspy.misc.utils.ensure_unicode(data)
        return data

    def read_string(self, length, skip=False):
        """Read a string defined by the infoArray iarray from
         file f with a given endianness (byte order).
        endian can be either 'big' or 'little'.

        If it's a tag name, each char is 1-Byte;
        if it's a tag data, each char is 2-Bytes Unicode,
        """
        if skip is True:
            offset = self.f.tell()
            self.f.seek(length, 1)
            return {'size': length,
                    'size_bytes': size_bytes,
                    'offset': offset,
                    'endian': self.endian, }
        data = ''
        if self.endian == 'little':
            s = L_char
        elif self.endian == 'big':
            s = B_char
        for char in xrange(length):
            data += s.unpack(self.f.read(1))[0]
        try:
            data = data.decode('utf8')
        except:
            # Sometimes the dm3 file strings are encoded in latin-1
            # instead of utf8
            data = data.decode('latin-1', errors='ignore')
        return data

    def read_struct(self, definition, skip=False):
        """Read a struct, defined by iarray, from file f
        with a given endianness (byte order).
        Returns a list of 2-tuples in the form
        (fieldAddress, fieldValue).
        endian can be either 'big' or 'little'.

        """
        field_value = []
        size_bytes = 0
        offset = self.f.tell()
        for dtype in definition:
            if dtype in self.simple_type:
                if skip is False:
                    data = self.get_data_reader(dtype)[0](self.f, self.endian)
                    field_value.append(data)
                else:
                    sbytes = self.get_data_reader(dtype)[1]
                    self.f.seek(sbytes, 1)
                    size_bytes += sbytes
            else:
                raise DM3DataTypeError(dtype)
        if skip is False:
            return tuple(field_value)
        else:
            return {'size': len(definition),
                    'size_bytes': size_bytes,
                    'offset': offset,
                    'endian': self.endian, }

    def read_array(self, size, enc_eltype, extra=None, skip=False):
        """Read an array, defined by iarray, from file f
        with a given endianness (byte order).
        endian can be either 'big' or 'little'.

        """
        eltype = self.get_data_reader(enc_eltype)[0]  # same for all elements
        if skip is True:
            if enc_eltype not in self._complex_type:
                size_bytes = self.get_data_reader(enc_eltype)[1] * size
                data = {"size": size,
                        "endian": self.endian,
                        "size_bytes": size_bytes,
                        "offset": self.f.tell()}
                self.f.seek(size_bytes, 1)  # Skipping data
            else:
                data = eltype(skip=skip, **extra)
                self.f.seek(data['size_bytes'] * (size - 1), 1)
                data['size'] = size
                data['size_bytes'] *= size
        else:
            if enc_eltype in self.simple_type:  # simple type
                data = [eltype(self.f, self.endian)
                        for element in xrange(size)]
                if enc_eltype == 4 and data:  # it's actually a string
                    data = "".join([unichr(i) for i in data])
            elif enc_eltype in self._complex_type:
                data = [eltype(**extra)
                        for element in xrange(size)]
        return data

    def parse_tag_group(self, skip4=1):
        """Parse the root TagGroup of the given DM3 file f.
        Returns the tuple (is_sorted, is_open, n_tags).
        endian can be either 'big' or 'little'.
        """
        is_sorted = read_byte(self.f, "big")
        is_open = read_byte(self.f, "big")
        self.skipif4(n=skip4)
        n_tags = read_long(self.f, "big")
        return bool(is_sorted), bool(is_open), n_tags

    def find_next_tag(self):
        while read_byte(self.f, "big") not in (20, 21):
            continue
        location = self.f.tell() - 1
        self.f.seek(location)
        tag_id = read_byte(self.f, "big")
        self.f.seek(location)
        tag_header = self.parse_tag_header()
        if tag_id == 20:
            print("Tag header length", tag_header['tag_name_length'])
            if not 20 > tag_header['tag_name_length'] > 0:
                print("Skipping id 20")
                self.f.seek(location + 1)
                self.find_next_tag()
            else:
                self.f.seek(location)
                return
        else:
            try:
                self.check_data_tag_delimiter()
                self.f.seek(location)
                return
            except DM3TagTypeError:
                self.f.seek(location + 1)
                print("Skipping id 21")
                self.find_next_tag()

    def find_next_data_tag(self):
        while read_byte(self.f, "big") != 21:
            continue
        position = self.f.tell() - 1
        self.f.seek(position)
        tag_header = self.parse_tag_header()
        try:
            self.check_data_tag_delimiter()
            self.f.seek(position)
        except DM3TagTypeError:
            self.f.seek(position + 1)
            self.find_next_data_tag()

    def parse_tag_header(self):
        tag_id = read_byte(self.f, "big")
        tag_name_length = read_short(self.f, "big")
        tag_name = self.read_string(tag_name_length)
        return {'tag_id': tag_id,
                'tag_name_length': tag_name_length,
                'tag_name': tag_name, }

    def check_data_tag_delimiter(self):
        self.skipif4(2)
        delimiter = self.read_string(4)
        if delimiter != '%%%%':
            raise DM3TagTypeError(delimiter)

    def get_image_dictionaries(self):
        """Returns the image dictionaries of all images in the file except
        the thumbnails.

        Returns
        -------
        dict, None

        """
        if 'ImageList' not in self.tags_dict:
            return None
        if "Thumbnails" in self.tags_dict:
            thumbnail_idx = [t['ImageIndex'] for key, t in
                             self.tags_dict['Thumbnails'].iteritems()]
        else:
            thumbnail_idx = []
        images = [image for key, image in
                  self.tags_dict['ImageList'].iteritems()
                  if not int(key.replace("TagGroup", "")) in
                  thumbnail_idx]
        return images


class ImageObject(object):

    def __init__(self, imdict, file, order="C", record_by=None):
        self.imdict = DictionaryTreeBrowser(imdict)
        self.file = file
        self._order = order if order else "C"
        self._record_by = record_by

    @property
    def shape(self):
        dimensions = self.imdict.ImageData.Dimensions
        shape = tuple([dimension[1] for dimension in dimensions])
        return shape[::-1]  # DM uses image indexing X, Y, Z...

    @property
    def offsets(self):
        dimensions = self.imdict.ImageData.Calibrations.Dimension
        origins = np.array([dimension[1].Origin for dimension in dimensions])
        return (-1 * origins[::-1] * self.scales)

    @property
    def scales(self):
        dimensions = self.imdict.ImageData.Calibrations.Dimension
        return np.array([dimension[1].Scale for dimension in dimensions])[::-1]

    @property
    def units(self):
        dimensions = self.imdict.ImageData.Calibrations.Dimension
        return tuple([dimension[1].Units
                      if dimension[1].Units else ""
                      for dimension in dimensions])[::-1]

    @property
    def names(self):
        names = [t.Undefined] * len(self.shape)
        indices = range(len(self.shape))
        if self.signal_type == "EELS":
            if "eV" in self.units:
                names[indices.pop(self.units.index("eV"))] = "Energy loss"
        elif self.signal_type in ("EDS", "EDX"):
            if "keV" in self.units:
                names[indices.pop(self.units.index("keV"))] = "Energy"
        for index, name in zip(indices[::-1], ("x", "y", "z")):
            names[index] = name
        return names

    @property
    def title(self):
        if "Name" in self.imdict:
            return self.imdict.Name
        else:
            return ''

    @property
    def record_by(self):
        if self._record_by is not None:
            return self._record_by
        if len(self.scales) == 1:
            return "spectrum"
        elif (('ImageTags.Meta_Data.Format' in self.imdict and
               self.imdict.ImageTags.Meta_Data.Format in ("Spectrum image",
                                                          "Spectrum")) or (
                "ImageTags.spim" in self.imdict)) and len(self.scales) == 2:
            return "spectrum"
        else:
            return "image"

    @property
    def to_spectrum(self):
        if (('ImageTags.Meta_Data.Format' in self.imdict and
                self.imdict.ImageTags.Meta_Data.Format == "Spectrum image") or (
                "ImageTags.spim" in self.imdict)) and len(self.scales) > 2:
            return True
        else:
            return False

    @property
    def order(self):
        return self._order

    @property
    def intensity_calibration(self):
        ic = self.imdict.ImageData.Calibrations.Brightness.as_dictionary()
        if not ic['Units']:
            ic['Units'] = ""
        return ic

    @property
    def dtype(self):
        # Image data types (Image Object chapter on DM help)#
        # key = DM data type code
        # value = numpy data type
        if self.imdict.ImageData.DataType == 4:
            raise NotImplementedError(
                "Reading data of this type is not implemented.")

        imdtype_dict = {
            0: 'not_implemented',  # null
            1: 'int16',
            2: 'float32',
            3: 'complex64',
            5: 'float32',  # not numpy: 8-Byte packed complex (FFT data)
            6: 'uint8',
            7: 'int32',
            8: np.dtype({'names': ['B', 'G', 'R', 'A'],
                         'formats': ['u1', 'u1', 'u1', 'u1']}),
            9: 'int8',
            10: 'uint16',
            11: 'uint32',
            12: 'float64',
            13: 'complex128',
            14: 'bool',
            23: np.dtype({'names': ['B', 'G', 'R', 'A'],
                          'formats': ['u1', 'u1', 'u1', 'u1']}),
            27: 'complex64',  # not numpy: 8-Byte packed complex (FFT data)
            28: 'complex128',  # not numpy: 16-Byte packed complex (FFT data)
        }
        return imdtype_dict[self.imdict.ImageData.DataType]

    @property
    def signal_type(self):
        if 'ImageTags.Meta_Data.Signal' in self.imdict:
            if self.imdict.ImageTags.Meta_Data.Signal == "X-ray":
                return "EDS_TEM"
            return self.imdict.ImageTags.Meta_Data.Signal
        elif 'ImageTags.spim.eels' in self.imdict:  # Orsay's tag group
            return "EELS"
        else:
            return ""

    def _get_data_array(self):
        self.file.seek(self.imdict.ImageData.Data.offset)
        count = self.imdict.ImageData.Data.size
        if self.imdict.ImageData.DataType in (27, 28):  # Packed complex
            count = int(count / 2)
        return np.fromfile(self.file,
                           dtype=self.dtype,
                           count=count)

    @property
    def size(self):
        if self.imdict.ImageData.DataType in (27, 28):  # Packed complex
            if self.imdict.ImageData.Data.size % 2:
                raise IOError(
                    "ImageData.Data.size should be an even integer for "
                    "this datatype.")
            else:
                return int(self.imdict.ImageData.Data.size / 2)
        else:
            return self.imdict.ImageData.Data.size

    def get_data(self):
        if isinstance(self.imdict.ImageData.Data, np.ndarray):
            return self.imdict.ImageData.Data
        data = self._get_data_array()
        if self.imdict.ImageData.DataType in (27, 28):  # New packed complex
            return self.unpack_new_packed_complex(data)
        elif self.imdict.ImageData.DataType == 5:  # Old packed compled
            return self.unpack_packed_complex(data)
        elif self.imdict.ImageData.DataType in (8, 23):  # ABGR
            # Reorder the fields
            data = np.hstack((data[["B", "G", "R"]].view(("u1", 3))[..., ::-1],
                              data["A"].reshape(-1, 1))).view(
                {"names": ("R", "G", "B", "A"),
                 "formats": ("u1",) * 4}).copy()
        return data.reshape(self.shape, order=self.order)

    def unpack_new_packed_complex(self, data):
        packed_shape = (self.shape[0], int(self.shape[1] / 2 + 1))
        data = data.reshape(packed_shape, order=self.order)
        return np.hstack((data[:, ::-1], np.conjugate(data[:, 1:-1])))

    def unpack_packed_complex(self, tmpdata):
        shape = self.shape
        if shape[0] != shape[1] or len(shape) > 2:
            msg = "Packed complex format works only for a 2Nx2N image"
            msg += " -> width == height"
            print msg
            raise IOError(
                'Unable to read this DM file in packed complex format. '
                'Pleare report the issue to the HyperSpy developers providing'
                ' the file if possible')
        N = int(self.shape[0] / 2)      # think about a 2Nx2N matrix
        # create an empty 2Nx2N ndarray of complex
        data = np.zeros(shape, dtype="complex64")

        # fill in the real values:
        data[N, 0] = tmpdata[0]
        data[0, 0] = tmpdata[1]
        data[N, N] = tmpdata[2 * N ** 2]  # Nyquist frequency
        data[0, N] = tmpdata[2 * N ** 2 + 1]  # Nyquist frequency

        # fill in the non-redundant complex values:
        # top right quarter, except 1st column
        for i in xrange(N):  # this could be optimized
            start = 2 * i * N + 2
            stop = start + 2 * (N - 1) - 1
            step = 2
            realpart = tmpdata[start:stop:step]
            imagpart = tmpdata[start + 1:stop + 1:step]
            data[i, N + 1:2 * N] = realpart + imagpart * 1j
        # 1st column, bottom left quarter
        start = 2 * N
        stop = start + 2 * N * (N - 1) - 1
        step = 2 * N
        realpart = tmpdata[start:stop:step]
        imagpart = tmpdata[start + 1:stop + 1:step]
        data[N + 1:2 * N, 0] = realpart + imagpart * 1j
        # 1st row, bottom right quarter
        start = 2 * N ** 2 + 2
        stop = start + 2 * (N - 1) - 1
        step = 2
        realpart = tmpdata[start:stop:step]
        imagpart = tmpdata[start + 1:stop + 1:step]
        data[N, N + 1:2 * N] = realpart + imagpart * 1j
        # bottom right quarter, except 1st row
        start = stop + 1
        stop = start + 2 * N * (N - 1) - 1
        step = 2
        realpart = tmpdata[start:stop:step]
        imagpart = tmpdata[start + 1:stop + 1:step]
        complexdata = realpart + imagpart * 1j
        data[
            N +
            1:2 *
            N,
            N:2 *
            N] = complexdata.reshape(
            N -
            1,
            N,
            order=self.order)

        # fill in the empty pixels: A(i)(j) = A(2N-i)(2N-j)*
        # 1st row, top left quarter, except 1st element
        data[0, 1:N] = np.conjugate(data[0, -1:-N:-1])
        # 1st row, bottom left quarter, except 1st element
        data[N, 1:N] = np.conjugate(data[N, -1:-N:-1])
        # 1st column, top left quarter, except 1st element
        data[1:N, 0] = np.conjugate(data[-1:-N:-1, 0])
        # 1st column, top right quarter, except 1st element
        data[1:N, N] = np.conjugate(data[-1:-N:-1, N])
        # top left quarter, except 1st row and 1st column
        data[1:N, 1:N] = np.conjugate(data[-1:-N:-1, -1:-N:-1])
        # bottom left quarter, except 1st row and 1st column
        data[N + 1:2 * N, 1:N] = np.conjugate(data[-N - 1:-2 * N:-1, -1:-N:-1])

        return data

    def get_axes_dict(self):
        return [{'name': name,
                 'size': size,
                 'index_in_array': i,
                 'scale': scale,
                 'offset': offset,
                 'units': unicode(units), }
                for i, (name, size, scale, offset, units) in enumerate(
                    zip(self.names, self.shape, self.scales, self.offsets,
                        self.units))]

    def get_metadata(self, metadata={}):
        if "General" not in metadata:
            metadata['General'] = {}
        if "Signal" not in metadata:
            metadata['Signal'] = {}
        metadata['General']['title'] = self.title
        metadata["Signal"]['record_by'] = self.record_by
        metadata["Signal"]['signal_type'] = self.signal_type
        return metadata

mapping = {
    "ImageList.TagGroup0.ImageTags.EELS.Experimental_Conditions.Collection_semi_angle_mrad": ("Acquisition_instrument.TEM.Detector.EELS.collection_angle", None),
    "ImageList.TagGroup0.ImageTags.EELS.Experimental_Conditions.Convergence_semi_angle_mrad": ("Acquisition_instrument.TEM.convergence_angle", None),
    "ImageList.TagGroup0.ImageTags.Acquisition.Parameters.Detector.exposure_s": ("Acquisition_instrument.TEM.dwell_time", None),
    "ImageList.TagGroup0.ImageTags.Microscope_Info.Voltage": ("Acquisition_instrument.TEM.beam_energy", lambda x: x / 1e3),
    "ImageList.TagGroup0.ImageTags.EDS.Detector_Info.Azimuthal_angle": ("Acquisition_instrument.TEM.Detector.EDS.azimuth_angle", None),
    "ImageList.TagGroup0.ImageTags.EDS.Detector_Info.Elevation_angle": ("Acquisition_instrument.TEM.Detector.EDS.elevation_angle", None),
    "ImageList.TagGroup0.ImageTags.EDS.Detector_Info.Stage_tilt": ("Acquisition_instrument.TEM.tilt_stage", None),
    "ImageList.TagGroup0.ImageTags.EDS.Solid_angle": ("Acquisition_instrument.TEM.Detector.EDS.solid_angle", None),
    "ImageList.TagGroup0.ImageTags.EDS.Live_time": ("Acquisition_instrument.TEM.Detector.EDS.live_time", None),
    "ImageList.TagGroup0.ImageTags.EDS.Real_time": ("Acquisition_instrument.TEM.Detector.EDS.real_time", None),
}


def file_reader(filename, record_by=None, order=None, verbose=False):
    """Reads a DM3 file and loads the data into the appropriate class.
    data_id can be specified to load a given image within a DM3 file that
    contains more than one dataset.

    Parameters
    ----------
    record_by: Str
        One of: SI, Image
    order: Str
        One of 'C' or 'F'

    """

    with open(filename, "rb") as f:
        dm = DigitalMicrographReader(f, verbose=verbose)
        dm.parse_file()
        images = [ImageObject(imdict, f, order=order, record_by=record_by)
                  for imdict in dm.get_image_dictionaries()]
        imd = []
        del dm.tags_dict['ImageList']
        dm.tags_dict['ImageList'] = {}

        for image in images:
            dm.tags_dict['ImageList'][
                'TagGroup0'] = image.imdict.as_dictionary()
            axes = image.get_axes_dict()
            mp = image.get_metadata()
            mp['General']['original_filename'] = os.path.split(filename)[1]
            post_process = []
            if image.to_spectrum is True:
                post_process.append(lambda s: s.to_spectrum())
            post_process.append(lambda s: s.squeeze())
            imd.append(
                {'data': image.get_data(),
                 'axes': axes,
                 'metadata': mp,
                 'original_metadata': dm.tags_dict,
                 'post_process': post_process,
                 'mapping': mapping,
                 })

    return imd
