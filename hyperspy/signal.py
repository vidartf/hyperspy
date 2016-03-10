# -*- coding: utf-8 -*-
# Copyright 2007-2016 The HyperSpy developers
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

import copy
import h5py
import os.path
import warnings
import math
import inspect
from contextlib import contextmanager
from datetime import datetime
import logging

import numpy as np
import numpy.ma as ma
import scipy.interpolate
try:
    from scipy.signal import savgol_filter
    savgol_imported = True
except ImportError:
    savgol_imported = False
import scipy as sp
from matplotlib import pyplot as plt
try:
    from statsmodels.nonparametric.smoothers_lowess import lowess
    statsmodels_installed = True
except:
    statsmodels_installed = False

from hyperspy.axes import AxesManager
from hyperspy import io
from hyperspy.drawing import mpl_hie, mpl_hse, mpl_he
from hyperspy.learn.mva import MVA, LearningResults
import hyperspy.misc.utils
from hyperspy.misc.utils import DictionaryTreeBrowser
from hyperspy.drawing import signal as sigdraw
from hyperspy.decorators import auto_replot
from hyperspy.defaults_parser import preferences
from hyperspy.misc.io.tools import ensure_directory
from hyperspy.external.progressbar import progressbar
from hyperspy.gui.tools import (
    SpectrumCalibration,
    SmoothingSavitzkyGolay,
    SmoothingLowess,
    SmoothingTV,
    ButterworthFilter)
from hyperspy.misc.tv_denoise import _tv_denoise_1d
from hyperspy.gui.egerton_quantification import BackgroundRemoval
from hyperspy.decorators import only_interactive
from hyperspy.decorators import interactive_range_selector
from scipy.ndimage.filters import gaussian_filter1d
from hyperspy.misc.spectrum_tools import find_peaks_ohaver
from hyperspy.misc.image_tools import (shift_image, estimate_image_shift)
from hyperspy.misc.math_tools import symmetrize, antisymmetrize
from hyperspy.exceptions import SignalDimensionError, DataDimensionError
from hyperspy.misc import array_tools
from hyperspy.misc import spectrum_tools
from hyperspy.misc import rgb_tools
from hyperspy.gui.tools import IntegrateArea
from hyperspy import components
from hyperspy.misc.utils import underline
from hyperspy.external.astroML.histtools import histogram
from hyperspy.drawing.utils import animate_legend
from hyperspy.misc.slicing import SpecialSlicers, FancySlicing
from hyperspy.misc.utils import slugify
from hyperspy.docstrings.signal import (
    ONE_AXIS_PARAMETER, MANY_AXIS_PARAMETER, OUT_ARG)
from hyperspy.events import Events, Event
from hyperspy.interactive import interactive
from hyperspy.misc.signal_tools import are_signals_aligned

_logger = logging.getLogger(__name__)


class ModelManager(object):

    """Container for models
    """

    class ModelStub(object):

        def __init__(self, mm, name):
            self._name = name
            self._mm = mm
            self.restore = lambda: mm.restore(self._name)
            self.remove = lambda: mm.remove(self._name)
            self.pop = lambda: mm.pop(self._name)
            self.restore.__doc__ = "Returns the stored model"
            self.remove.__doc__ = "Removes the stored model"
            self.pop.__doc__ = \
                "Returns the stored model and removes it from storage"

        def __repr__(self):
            return repr(self._mm._models[self._name])

    def __init__(self, signal, dictionary=None):
        self._signal = signal
        self._models = DictionaryTreeBrowser()
        self._add_dictionary(dictionary)

    def _add_dictionary(self, dictionary=None):
        if dictionary is not None:
            for k, v in dictionary.items():
                if k.startswith('_') or k in ['restore', 'remove']:
                    raise KeyError("Can't add dictionary with key '%s'" % k)
                k = slugify(k, True)
                self._models.set_item(k, v)
                setattr(self, k, self.ModelStub(self, k))

    def _set_nice_description(self, node, names):
        ans = {'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
               'dimensions': self._signal.axes_manager._get_dimension_str(),
               }
        node.add_dictionary(ans)
        for n in names:
            node.add_node('components.' + n)

    def _save(self, name, dictionary):

        from itertools import product
        _abc = 'abcdefghijklmnopqrstuvwxyz'

        def get_letter(models):
            howmany = len(models)
            if not howmany:
                return 'a'
            order = int(np.log(howmany) / np.log(26)) + 1
            letters = [_abc, ] * order
            for comb in product(*letters):
                guess = "".join(comb)
                if guess not in models.keys():
                    return guess

        if name is None:
            name = get_letter(self._models)
        else:
            name = self._check_name(name)

        if name in self._models:
            self.remove(name)

        self._models.add_node(name)
        node = self._models.get_item(name)
        names = [c['name'] for c in dictionary['components']]
        self._set_nice_description(node, names)

        node.set_item('_dict', dictionary)
        setattr(self, name, self.ModelStub(self, name))

    def store(self, model, name=None):
        """If the given model was created from this signal, stores it

        Parameters
        ----------
        model : model
            the model to store in the signal
        name : {string, None}
            the name for the model to be stored with

        See Also
        --------
        remove
        restore
        pop
        """
        if model.signal is self._signal:
            self._save(name, model.as_dictionary())
        else:
            raise ValueError("The model is created from a different signal, "
                             "you should store it there")

    def _check_name(self, name, existing=False):
        if not isinstance(name, str):
            raise KeyError('Name has to be a string')
        if name.startswith('_'):
            raise KeyError('Name cannot start with "_" symbol')
        if '.' in name:
            raise KeyError('Name cannot contain dots (".")')
        name = slugify(name, True)
        if existing:
            if name not in self._models:
                raise KeyError(
                    "Model named '%s' is not currently stored" %
                    name)
        return name

    def remove(self, name):
        """Removes the given model

        Parameters
        ----------
        name : string
            the name of the model to remove

        See Also
        --------
        restore
        store
        pop
        """
        name = self._check_name(name, True)
        delattr(self, name)
        self._models.__delattr__(name)

    def pop(self, name):
        """Returns the restored model and removes it from storage

        Parameters
        ----------
        name : string
            the name of the model to restore and remove

        See Also
        --------
        restore
        store
        remove
        """
        name = self._check_name(name, True)
        model = self.restore(name)
        self.remove(name)
        return model

    def restore(self, name):
        """Returns the restored model

        Parameters
        ----------
        name : string
            the name of the model to restore

        See Also
        --------
        remove
        store
        pop
        """
        name = self._check_name(name, True)
        d = self._models.get_item(name + '._dict').as_dictionary()
        return self._signal.create_model(dictionary=copy.deepcopy(d))

    def __repr__(self):
        return repr(self._models)

    def __len__(self):
        return len(self._models)

    def __getitem__(self, name):
        name = self._check_name(name, True)
        return getattr(self, name)


class Signal2DTools(object):

    def estimate_shift2D(self,
                         reference='current',
                         correlation_threshold=None,
                         chunk_size=30,
                         roi=None,
                         normalize_corr=False,
                         sobel=True,
                         medfilter=True,
                         hanning=True,
                         plot=False,
                         dtype='float',
                         show_progressbar=None):
        """Estimate the shifts in a image using phase correlation

        This method can only estimate the shift by comparing
        bidimensional features that should not change position
        between frames. To decrease the memory usage, the time of
        computation and the accuracy of the results it is convenient
        to select a region of interest by setting the roi keyword.

        Parameters
        ----------

        reference : {'current', 'cascade' ,'stat'}
            If 'current' (default) the image at the current
            coordinates is taken as reference. If 'cascade' each image
            is aligned with the previous one. If 'stat' the translation
            of every image with all the rest is estimated and by
            performing statistical analysis on the result the
            translation is estimated.
        correlation_threshold : {None, 'auto', float}
            This parameter is only relevant when `reference` is 'stat'.
            If float, the shift estimations with a maximum correlation
            value lower than the given value are not used to compute
            the estimated shifts. If 'auto' the threshold is calculated
            automatically as the minimum maximum correlation value
            of the automatically selected reference image.
        chunk_size: {None, int}
            If int and `reference`=='stat' the number of images used
            as reference are limited to the given value.
        roi : tuple of ints or floats (left, right, top bottom)
             Define the region of interest. If int(float) the position
             is given axis index(value).
        sobel : bool
            apply a sobel filter for edge enhancement
        medfilter :  bool
            apply a median filter for noise reduction
        hanning : bool
            Apply a 2d hanning filter
        plot : bool
            If True plots the images after applying the filters and
            the phase correlation
        dtype : str or dtype
            Typecode or data-type in which the calculations must be
            performed.
        show_progressbar : None or bool
            If True, display a progress bar. If None the default is set in
            `preferences`.

        Returns
        -------

        list of applied shifts

        Notes
        -----

        The statistical analysis approach to the translation estimation
        when using `reference`='stat' roughly follows [1]_ . If you use
        it please cite their article.

        References
        ----------

        .. [1] Schaffer, Bernhard, Werner Grogger, and Gerald
        Kothleitner. “Automated Spatial Drift Correction for EFTEM
        Image Series.”
        Ultramicroscopy 102, no. 1 (December 2004): 27–36.

        """
        if show_progressbar is None:
            show_progressbar = preferences.General.show_progressbar
        self._check_signal_dimension_equals_two()
        if roi is not None:
            # Get the indices of the roi
            yaxis = self.axes_manager.signal_axes[1]
            xaxis = self.axes_manager.signal_axes[0]
            roi = tuple([xaxis._get_index(i) for i in roi[2:]] +
                        [yaxis._get_index(i) for i in roi[:2]])

        ref = None if reference == 'cascade' else \
            self.__call__().copy()
        shifts = []
        nrows = None
        images_number = self.axes_manager._max_index + 1
        if reference == 'stat':
            nrows = images_number if chunk_size is None else \
                min(images_number, chunk_size)
            pcarray = ma.zeros((nrows, self.axes_manager._max_index + 1,
                                ),
                               dtype=np.dtype([('max_value', np.float),
                                               ('shift', np.int32,
                                                (2,))]))
            nshift, max_value = estimate_image_shift(
                self(),
                self(),
                roi=roi,
                sobel=sobel,
                medfilter=medfilter,
                hanning=hanning,
                normalize_corr=normalize_corr,
                plot=plot,
                dtype=dtype)
            np.fill_diagonal(pcarray['max_value'], max_value)
            pbar = progressbar(maxval=nrows * images_number,
                               disabled=not show_progressbar)
        else:
            pbar = progressbar(maxval=images_number,
                               disabled=not show_progressbar)

        # Main iteration loop. Fills the rows of pcarray when reference
        # is stat
        for i1, im in enumerate(self._iterate_signal()):
            if reference in ['current', 'cascade']:
                if ref is None:
                    ref = im.copy()
                    shift = np.array([0, 0])
                nshift, max_val = estimate_image_shift(
                    ref, im, roi=roi, sobel=sobel, medfilter=medfilter,
                    hanning=hanning, plot=plot,
                    normalize_corr=normalize_corr, dtype=dtype)
                if reference == 'cascade':
                    shift += nshift
                    ref = im.copy()
                else:
                    shift = nshift
                shifts.append(shift.copy())
                pbar.update(i1 + 1)
            elif reference == 'stat':
                if i1 == nrows:
                    break
                # Iterate to fill the columns of pcarray
                for i2, im2 in enumerate(
                        self._iterate_signal()):
                    if i2 > i1:
                        nshift, max_value = estimate_image_shift(
                            im,
                            im2,
                            roi=roi,
                            sobel=sobel,
                            medfilter=medfilter,
                            hanning=hanning,
                            normalize_corr=normalize_corr,
                            plot=plot,
                            dtype=dtype)

                        pcarray[i1, i2] = max_value, nshift
                    del im2
                    pbar.update(i2 + images_number * i1 + 1)
                del im
        if reference == 'stat':
            # Select the reference image as the one that has the
            # higher max_value in the row
            sqpcarr = pcarray[:, :nrows]
            sqpcarr['max_value'][:] = symmetrize(sqpcarr['max_value'])
            sqpcarr['shift'][:] = antisymmetrize(sqpcarr['shift'])
            ref_index = np.argmax(pcarray['max_value'].min(1))
            self.ref_index = ref_index
            shifts = (pcarray['shift'] +
                      pcarray['shift'][ref_index, :nrows][:, np.newaxis])
            if correlation_threshold is not None:
                if correlation_threshold == 'auto':
                    correlation_threshold = \
                        (pcarray['max_value'].min(0)).max()
                    _logger.info("Correlation threshold = %1.2f",
                          correlation_threshold)
                shifts[pcarray['max_value'] <
                       correlation_threshold] = ma.masked
                shifts.mask[ref_index, :] = False

            shifts = shifts.mean(0)
        else:
            shifts = np.array(shifts)
            del ref
        return shifts

    def align2D(self, crop=True, fill_value=np.nan, shifts=None, expand=False,
                roi=None,
                sobel=True,
                medfilter=True,
                hanning=True,
                plot=False,
                normalize_corr=False,
                reference='current',
                dtype='float',
                correlation_threshold=None,
                chunk_size=30,
                interpolation_order=1):
        """Align the images in place using user provided shifts or by
        estimating the shifts.

        Please, see `estimate_shift2D` docstring for details
        on the rest of the parameters not documented in the following
        section

        Parameters
        ----------
        crop : bool
            If True, the data will be cropped not to include regions
            with missing data
        fill_value : int, float, nan
            The areas with missing data are filled with the given value.
            Default is nan.
        shifts : None or list of tuples
            If None the shifts are estimated using
            `estimate_shift2D`.
        expand : bool
            If True, the data will be expanded to fit all data after alignment.
            Overrides `crop`.
        interpolation_order: int, default 1.
            The order of the spline interpolation. Default is 1, linear
            interpolation.

        Returns
        -------
        shifts : np.array
            The shifts are returned only if `shifts` is None

        Notes
        -----

        The statistical analysis approach to the translation estimation
        when using `reference`='stat' roughly follows [1]_ . If you use
        it please cite their article.

        References
        ----------

        .. [1] Schaffer, Bernhard, Werner Grogger, and Gerald
        Kothleitner. “Automated Spatial Drift Correction for EFTEM
        Image Series.”
        Ultramicroscopy 102, no. 1 (December 2004): 27–36.

        """
        self._check_signal_dimension_equals_two()
        if shifts is None:
            shifts = self.estimate_shift2D(
                roi=roi,
                sobel=sobel,
                medfilter=medfilter,
                hanning=hanning,
                plot=plot,
                reference=reference,
                dtype=dtype,
                correlation_threshold=correlation_threshold,
                normalize_corr=normalize_corr,
                chunk_size=chunk_size)
            return_shifts = True
        else:
            return_shifts = False
        if not np.any(shifts):
            # The shift array if filled with zeros, nothing to do.
            return

        if expand:
            # Expand to fit all valid data
            left, right = (int(np.floor(shifts[:, 1].min())) if
                           shifts[:, 1].min() < 0 else 0,
                           int(np.ceil(shifts[:, 1].max())) if
                           shifts[:, 1].max() > 0 else 0)
            top, bottom = (int(np.floor(shifts[:, 0].min())) if
                           shifts[:, 0].min() < 0 else 0,
                           int(np.ceil(shifts[:, 0].max())) if
                           shifts[:, 0].max() > 0 else 0)
            xaxis = self.axes_manager.signal_axes[0]
            yaxis = self.axes_manager.signal_axes[1]
            padding = []
            for i in range(self.data.ndim):
                if i == xaxis.index_in_array:
                    padding.append((right, -left))
                elif i == yaxis.index_in_array:
                    padding.append((bottom, -top))
                else:
                    padding.append((0, 0))
            self.data = np.pad(self.data, padding, mode='constant',
                               constant_values=(fill_value,))
            if left < 0:
                xaxis.offset += left * xaxis.scale
            if np.any((left < 0, right > 0)):
                xaxis.size += right - left
            if top < 0:
                yaxis.offset += top * yaxis.scale
            if np.any((top < 0, bottom > 0)):
                yaxis.size += bottom - top

        # Translate with sub-pixel precision if necesary
        for im, shift in zip(self._iterate_signal(),
                             shifts):
            if np.any(shift):
                shift_image(im, -shift,
                            fill_value=fill_value,
                            interpolation_order=interpolation_order)
                del im

        if crop and not expand:
            # Crop the image to the valid size
            shifts = -shifts
            bottom, top = (int(np.floor(shifts[:, 0].min())) if
                           shifts[:, 0].min() < 0 else None,
                           int(np.ceil(shifts[:, 0].max())) if
                           shifts[:, 0].max() > 0 else 0)
            right, left = (int(np.floor(shifts[:, 1].min())) if
                           shifts[:, 1].min() < 0 else None,
                           int(np.ceil(shifts[:, 1].max())) if
                           shifts[:, 1].max() > 0 else 0)
            self.crop_image(top, bottom, left, right)
            shifts = -shifts

        self.events.data_changed.trigger(obj=self)
        if return_shifts:
            return shifts

    def crop_image(self, top=None, bottom=None,
                   left=None, right=None):
        """Crops an image in place.

        top, bottom, left, right : int or float

            If int the values are taken as indices. If float the values are
            converted to indices.

        See also:
        ---------
        crop

        """
        self._check_signal_dimension_equals_two()
        self.crop(self.axes_manager.signal_axes[1].index_in_axes_manager,
                  top,
                  bottom)
        self.crop(self.axes_manager.signal_axes[0].index_in_axes_manager,
                  left,
                  right)


class Signal1DTools(object):

    def shift1D(self,
                shift_array,
                interpolation_method='linear',
                crop=True,
                expand=False,
                fill_value=np.nan,
                show_progressbar=None):
        """Shift the data in place over the signal axis by the amount specified
        by an array.

        Parameters
        ----------
        shift_array : numpy array
            An array containing the shifting amount. It must have
            `axes_manager._navigation_shape_in_array` shape.
        interpolation_method : str or int
            Specifies the kind of interpolation as a string ('linear',
            'nearest', 'zero', 'slinear', 'quadratic, 'cubic') or as an
            integer specifying the order of the spline interpolator to
            use.
        crop : bool
            If True automatically crop the signal axis at both ends if
            needed.
        expand : bool
            If True, the data will be expanded to fit all data after alignment.
            Overrides `crop`.
        fill_value : float
            If crop is False fill the data outside of the original
            interval with the given value where needed.
        show_progressbar : None or bool
            If True, display a progress bar. If None the default is set in
            `preferences`.

        Raises
        ------
        SignalDimensionError if the signal dimension is not 1.

        """
        if not np.any(shift_array):
            # Nothing to do, the shift array if filled with zeros
            return
        if show_progressbar is None:
            show_progressbar = preferences.General.show_progressbar
        self._check_signal_dimension_equals_one()
        axis = self.axes_manager.signal_axes[0]
        pbar = progressbar(
            maxval=self.axes_manager.navigation_size,
            disabled=not show_progressbar)

        # Figure out min/max shifts, and translate to shifts in index as well
        minimum, maximum = np.nanmin(shift_array), np.nanmax(shift_array)
        if minimum < 0:
            ihigh = 1 + axis.value2index(
                axis.high_value + minimum,
                rounding=math.floor)
        else:
            ihigh = axis.high_index + 1
        if maximum > 0:
            ilow = axis.value2index(axis.offset + maximum,
                                    rounding=math.ceil)
        else:
            ilow = axis.low_index
        if expand:
            padding = []
            for i in range(self.data.ndim):
                if i == axis.index_in_array:
                    padding.append(
                        (axis.high_index - ihigh + 1, ilow - axis.low_index))
                else:
                    padding.append((0, 0))
            self.data = np.pad(self.data, padding, mode='constant',
                               constant_values=(fill_value,))
            axis.offset += minimum
            axis.size += axis.high_index - ihigh + 1 + ilow - axis.low_index
        offset = axis.offset
        original_axis = axis.axis.copy()
        for i, (dat, shift) in enumerate(zip(
                self._iterate_signal(),
                shift_array.ravel(()))):
            if np.isnan(shift):
                continue
            si = sp.interpolate.interp1d(original_axis,
                                         dat,
                                         bounds_error=False,
                                         fill_value=fill_value,
                                         kind=interpolation_method)
            axis.offset = float(offset - shift)
            dat[:] = si(axis.axis)
            pbar.update(i + 1)

        axis.offset = offset

        if crop and not expand:
            self.crop(axis.index_in_axes_manager,
                      ilow,
                      ihigh)

        self.events.data_changed.trigger(obj=self)

    def interpolate_in_between(self, start, end, delta=3,
                               show_progressbar=None, **kwargs):
        """Replace the data in a given range by interpolation.

        The operation is performed in place.

        Parameters
        ----------
        start, end : {int | float}
            The limits of the interval. If int they are taken as the
            axis index. If float they are taken as the axis value.

        delta : {int | float}
            The windows around the (start, end) to use for interpolation

        show_progressbar : None or bool
            If True, display a progress bar. If None the default is set in
            `preferences`.

        All extra keyword arguments are passed to
        scipy.interpolate.interp1d. See the function documentation
        for details.

        Raises
        ------
        SignalDimensionError if the signal dimension is not 1.

        """
        if show_progressbar is None:
            show_progressbar = preferences.General.show_progressbar
        self._check_signal_dimension_equals_one()
        axis = self.axes_manager.signal_axes[0]
        i1 = axis._get_index(start)
        i2 = axis._get_index(end)
        if isinstance(delta, float):
            delta = int(delta / axis.scale)
        i0 = int(np.clip(i1 - delta, 0, np.inf))
        i3 = int(np.clip(i2 + delta, 0, axis.size))
        pbar = progressbar(
            maxval=self.axes_manager.navigation_size,
            disabled=not show_progressbar)
        for i, dat in enumerate(self._iterate_signal()):
            dat_int = sp.interpolate.interp1d(
                list(range(i0, i1)) + list(range(i2, i3)),
                dat[i0:i1].tolist() + dat[i2:i3].tolist(),
                **kwargs)
            dat[i1:i2] = dat_int(list(range(i1, i2)))
            pbar.update(i + 1)
        self.events.data_changed.trigger(obj=self)

    def _check_navigation_mask(self, mask):
        if mask is not None:
            if not isinstance(mask, Signal):
                raise ValueError("mask must be a Signal instance.")
            elif mask.axes_manager.signal_dimension not in (0, 1):
                raise ValueError("mask must be a Signal with signal_dimension "
                                 "equal to 1")
            elif (mask.axes_manager.navigation_dimension !=
                  self.axes_manager.navigation_dimension):
                raise ValueError("mask must be a Signal with the same "
                                 "navigation_dimension as the current signal.")

    def estimate_shift1D(self,
                         start=None,
                         end=None,
                         reference_indices=None,
                         max_shift=None,
                         interpolate=True,
                         number_of_interpolation_points=5,
                         mask=None,
                         show_progressbar=None):
        """Estimate the shifts in the current signal axis using
         cross-correlation.

        This method can only estimate the shift by comparing
        unidimensional features that should not change the position in
        the signal axis. To decrease the memory usage, the time of
        computation and the accuracy of the results it is convenient to
        select the feature of interest providing sensible values for
        `start` and `end`. By default interpolation is used to obtain
        subpixel precision.

        Parameters
        ----------
        start, end : {int | float | None}
            The limits of the interval. If int they are taken as the
            axis index. If float they are taken as the axis value.
        reference_indices : tuple of ints or None
            Defines the coordinates of the spectrum that will be used
            as eference. If None the spectrum at the current
            coordinates is used for this purpose.
        max_shift : int
            "Saturation limit" for the shift.
        interpolate : bool
            If True, interpolation is used to provide sub-pixel
            accuracy.
        number_of_interpolation_points : int
            Number of interpolation points. Warning: making this number
            too big can saturate the memory
        mask : Signal of bool data type.
            It must have signal_dimension = 0 and navigation_shape equal to the
            current signal. Where mask is True the shift is not computed
            and set to nan.
        show_progressbar : None or bool
            If True, display a progress bar. If None the default is set in
            `preferences`.

        Returns
        -------
        An array with the result of the estimation in the axis units.

        Raises
        ------
        SignalDimensionError if the signal dimension is not 1.

        """
        if show_progressbar is None:
            show_progressbar = preferences.General.show_progressbar
        self._check_signal_dimension_equals_one()
        ip = number_of_interpolation_points + 1
        axis = self.axes_manager.signal_axes[0]
        self._check_navigation_mask(mask)
        if reference_indices is None:
            reference_indices = self.axes_manager.indices

        i1, i2 = axis._get_index(start), axis._get_index(end)
        shift_array = np.zeros(self.axes_manager._navigation_shape_in_array,
                               dtype=float)
        ref = self.inav[reference_indices].data[i1:i2]
        if interpolate is True:
            ref = spectrum_tools.interpolate1D(ip, ref)
        pbar = progressbar(
            maxval=self.axes_manager.navigation_size,
            disabled=not show_progressbar)
        for i, (dat, indices) in enumerate(zip(
                self._iterate_signal(),
                self.axes_manager._array_indices_generator())):
            if mask is not None and bool(mask.data[indices]) is True:
                shift_array[indices] = np.nan
            else:
                dat = dat[i1:i2]
                if interpolate is True:
                    dat = spectrum_tools.interpolate1D(ip, dat)
                shift_array[indices] = np.argmax(
                    np.correlate(ref, dat, 'full')) - len(ref) + 1
            pbar.update(i + 1)
        pbar.finish()

        if max_shift is not None:
            if interpolate is True:
                max_shift *= ip
            shift_array.clip(-max_shift, max_shift)
        if interpolate is True:
            shift_array /= ip
        shift_array *= axis.scale
        return shift_array

    def align1D(self,
                start=None,
                end=None,
                reference_indices=None,
                max_shift=None,
                interpolate=True,
                number_of_interpolation_points=5,
                interpolation_method='linear',
                crop=True,
                expand=False,
                fill_value=np.nan,
                also_align=[],
                mask=None,
                show_progressbar=None):
        """Estimate the shifts in the signal axis using
        cross-correlation and use the estimation to align the data in place.

        This method can only estimate the shift by comparing
        unidimensional
        features that should not change the position.
        To decrease memory usage, time of computation and improve
        accuracy it is convenient to select the feature of interest
        setting the `start` and `end` keywords. By default interpolation is
        used to obtain subpixel precision.

        Parameters
        ----------
        start, end : {int | float | None}
            The limits of the interval. If int they are taken as the
            axis index. If float they are taken as the axis value.
        reference_indices : tuple of ints or None
            Defines the coordinates of the spectrum that will be used
            as eference. If None the spectrum at the current
            coordinates is used for this purpose.
        max_shift : int
            "Saturation limit" for the shift.
        interpolate : bool
            If True, interpolation is used to provide sub-pixel
            accuracy.
        number_of_interpolation_points : int
            Number of interpolation points. Warning: making this number
            too big can saturate the memory
        interpolation_method : str or int
            Specifies the kind of interpolation as a string ('linear',
            'nearest', 'zero', 'slinear', 'quadratic, 'cubic') or as an
            integer specifying the order of the spline interpolator to
            use.
        crop : bool
            If True automatically crop the signal axis at both ends if
            needed.
        expand : bool
            If True, the data will be expanded to fit all data after alignment.
            Overrides `crop`.
        fill_value : float
            If crop is False fill the data outside of the original
            interval with the given value where needed.
        also_align : list of signals
            A list of Signal instances that has exactly the same
            dimensions
            as this one and that will be aligned using the shift map
            estimated using the this signal.
        mask : Signal of bool data type.
            It must have signal_dimension = 0 and navigation_shape equal to the
            current signal. Where mask is True the shift is not computed
            and set to nan.
        show_progressbar : None or bool
            If True, display a progress bar. If None the default is set in
            `preferences`.

        Returns
        -------
        An array with the result of the estimation. The shift will be

        Raises
        ------
        SignalDimensionError if the signal dimension is not 1.

        See also
        --------
        estimate_shift1D

        """
        if also_align is None:
            also_align = []
        self._check_signal_dimension_equals_one()
        shift_array = self.estimate_shift1D(
            start=start,
            end=end,
            reference_indices=reference_indices,
            max_shift=max_shift,
            interpolate=interpolate,
            number_of_interpolation_points=number_of_interpolation_points,
            mask=mask,
            show_progressbar=show_progressbar)
        for signal in also_align + [self]:
            signal.shift1D(shift_array=shift_array,
                           interpolation_method=interpolation_method,
                           crop=crop,
                           fill_value=fill_value,
                           expand=expand,
                           show_progressbar=show_progressbar)

    def integrate_in_range(self, signal_range='interactive'):
        """ Sums the spectrum over an energy range, giving the integrated
        area.

        The energy range can either be selected through a GUI or the command
        line.

        Parameters
        ----------
        signal_range : {a tuple of this form (l, r), "interactive"}
            l and r are the left and right limits of the range. They can be
            numbers or None, where None indicates the extremes of the interval.
            If l and r are floats the `signal_range` will be in axis units (for
            example eV). If l and r are integers the `signal_range` will be in
            index units. When `signal_range` is "interactive" (default) the
            range is selected using a GUI.

        Returns
        -------
        integrated_spectrum : Signal subclass

        See Also
        --------
        integrate_simpson

        Examples
        --------

        Using the GUI

        >>> s.integrate_in_range()

        Using the CLI

        >>> s_int = s.integrate_in_range(signal_range=(560,None))

        Selecting a range in the axis units, by specifying the
        signal range with floats.

        >>> s_int = s.integrate_in_range(signal_range=(560.,590.))

        Selecting a range using the index, by specifying the
        signal range with integers.

        >>> s_int = s.integrate_in_range(signal_range=(100,120))

        """

        if signal_range == 'interactive':
            self_copy = self.deepcopy()
            ia = IntegrateArea(self_copy, signal_range)
            ia.edit_traits()
            integrated_spectrum = self_copy
        else:
            integrated_spectrum = self._integrate_in_range_commandline(
                signal_range)
        return integrated_spectrum

    def _integrate_in_range_commandline(self, signal_range):
        e1 = signal_range[0]
        e2 = signal_range[1]
        integrated_spectrum = self.isig[e1:e2].integrate1D(-1)
        return integrated_spectrum

    @only_interactive
    def calibrate(self):
        """Calibrate the spectral dimension using a gui.

        It displays a window where the new calibration can be set by:
        * Setting the offset, units and scale directly
        * Selection a range by dragging the mouse on the spectrum figure
         and
        setting the new values for the given range limits

        Notes
        -----
        For this method to work the output_dimension must be 1. Set the
        view
        accordingly

        Raises
        ------
        SignalDimensionError if the signal dimension is not 1.

        """
        self._check_signal_dimension_equals_one()
        calibration = SpectrumCalibration(self)
        calibration.edit_traits()

    def smooth_savitzky_golay(self,
                              polynomial_order=None,
                              window_length=None,
                              differential_order=0):
        """Apply a Savitzky-Golay filter to the data in place.

        If `polynomial_order` or `window_length` or `differential_order` are
        None the method is run in interactive mode.

        Parameters
        ----------
        window_length : int
            The length of the filter window (i.e. the number of coefficients).
            `window_length` must be a positive odd integer.
        polynomial_order : int
            The order of the polynomial used to fit the samples.
            `polyorder` must be less than `window_length`.
        differential_order: int, optional
            The order of the derivative to compute.  This must be a
            nonnegative integer.  The default is 0, which means to filter
            the data without differentiating.

        Notes
        -----
        More information about the filter in `scipy.signal.savgol_filter`.

        """
        if not savgol_imported:
            raise ImportError("scipy >= 0.14 needs to be installed to use"
                              "this feature.")
        self._check_signal_dimension_equals_one()
        if (polynomial_order is not None and
                window_length is not None):
            axis = self.axes_manager.signal_axes[0]
            self.data = savgol_filter(
                x=self.data,
                window_length=window_length,
                polyorder=polynomial_order,
                deriv=differential_order,
                delta=axis.scale,
                axis=axis.index_in_array)
            self.events.data_changed.trigger(obj=self)
        else:
            # Interactive mode
            smoother = SmoothingSavitzkyGolay(self)
            smoother.differential_order = differential_order
            if polynomial_order is not None:
                smoother.polynomial_order = polynomial_order
            if window_length is not None:
                smoother.window_length = window_length
            smoother.edit_traits()

    def smooth_lowess(self,
                      smoothing_parameter=None,
                      number_of_iterations=None,
                      show_progressbar=None):
        """Lowess data smoothing in place.

        If `smoothing_parameter` or `number_of_iterations` are None the method
        is run in interactive mode.

        Parameters
        ----------
        smoothing_parameter: float or None
            Between 0 and 1. The fraction of the data used
            when estimating each y-value.
        number_of_iterations: int or None
            The number of residual-based reweightings
            to perform.
        show_progressbar : None or bool
            If True, display a progress bar. If None the default is set in
            `preferences`.

        Raises
        ------
        SignalDimensionError if the signal dimension is not 1.
        ImportError if statsmodels is not installed.

        Notes
        -----
        This method uses the lowess algorithm from statsmodels. statsmodels
        is required for this method.

        """
        if not statsmodels_installed:
            raise ImportError("statsmodels is not installed. This package is "
                              "required for this feature.")
        self._check_signal_dimension_equals_one()
        if smoothing_parameter is None or number_of_iterations is None:
            smoother = SmoothingLowess(self)
            if smoothing_parameter is not None:
                smoother.smoothing_parameter = smoothing_parameter
            if number_of_iterations is not None:
                smoother.number_of_iterations = number_of_iterations
            smoother.edit_traits()
        else:
            self.map(lowess,
                     exog=self.axes_manager[-1].axis,
                     frac=smoothing_parameter,
                     it=number_of_iterations,
                     is_sorted=True,
                     return_sorted=False,
                     show_progressbar=show_progressbar)

    def smooth_tv(self, smoothing_parameter=None, show_progressbar=None):
        """Total variation data smoothing in place.

        Parameters
        ----------
        smoothing_parameter: float or None
           Denoising weight relative to L2 minimization. If None the method
           is run in interactive mode.
        show_progressbar : None or bool
            If True, display a progress bar. If None the default is set in
            `preferences`.

        Raises
        ------
        SignalDimensionError if the signal dimension is not 1.

        """
        self._check_signal_dimension_equals_one()
        if smoothing_parameter is None:
            smoother = SmoothingTV(self)
            smoother.edit_traits()
        else:
            self.map(_tv_denoise_1d, weight=smoothing_parameter,
                     show_progressbar=show_progressbar)

    def filter_butterworth(self,
                           cutoff_frequency_ratio=None,
                           type='low',
                           order=2):
        """Butterworth filter in place.

        Raises
        ------
        SignalDimensionError if the signal dimension is not 1.

        """
        self._check_signal_dimension_equals_one()
        smoother = ButterworthFilter(self)
        if cutoff_frequency_ratio is not None:
            smoother.cutoff_frequency_ratio = cutoff_frequency_ratio
            smoother.apply()
        else:
            smoother.edit_traits()

    def _remove_background_cli(
            self, signal_range, background_estimator, estimate_background=True,
            show_progressbar=None):
        from hyperspy.models.model1D import Model1D
        model = Model1D(self)
        model.append(background_estimator)
        if estimate_background:
            background_estimator.estimate_parameters(
                self,
                signal_range[0],
                signal_range[1],
                only_current=False)
        else:
            model.set_signal_range(signal_range[0], signal_range[1])
            model.multifit(show_progressbar=show_progressbar)
        return self - model.as_signal(show_progressbar=show_progressbar)

    def remove_background(
            self,
            signal_range='interactive',
            background_type='PowerLaw',
            polynomial_order=2,
            estimate_background=True,
            show_progressbar=None):
        """Remove the background, either in place using a gui or returned as a new
        spectrum using the command line.

        Parameters
        ----------
        signal_range : tuple, optional
            If this argument is not specified, the signal range has to be
            selected using a GUI. And the original spectrum will be replaced.
            If tuple is given, the a spectrum will be returned.
        background_type : string
            The type of component which should be used to fit the background.
            Possible components: PowerLaw, Gaussian, Offset, Polynomial
            If Polynomial is used, the polynomial order can be specified
        polynomial_order : int, default 2
            Specify the polynomial order if a Polynomial background is used.
        estimate_background : bool
            If True, estimate the background. If False, the signal is fitted
            using a full model. This is slower compared to the estimation but
            possibly more accurate.
        show_progressbar : None or bool
            If True, display a progress bar. If None the default is set in
            `preferences`.

        Examples
        --------

        Using gui, replaces spectrum s

        >>>> s.remove_background()

        Using command line, returns a spectrum

        >>>> s = s.remove_background(signal_range=(400,450), background_type='PowerLaw')

        Using a full model to fit the background

        >>>> s = s.remove_background(signal_range=(400,450), estimate_background=False)

        Raises
        ------
        SignalDimensionError if the signal dimension is not 1.

        """
        self._check_signal_dimension_equals_one()
        if signal_range == 'interactive':
            br = BackgroundRemoval(self)
            br.edit_traits()
        else:
            if background_type == 'PowerLaw':
                background_estimator = components.PowerLaw()
            elif background_type == 'Gaussian':
                background_estimator = components.Gaussian()
            elif background_type == 'Offset':
                background_estimator = components.Offset()
            elif background_type == 'Polynomial':
                background_estimator = components.Polynomial(polynomial_order)
            else:
                raise ValueError(
                    "Background type: " +
                    background_type +
                    " not recognized")

            spectra = self._remove_background_cli(
                signal_range, background_estimator, estimate_background,
                show_progressbar=show_progressbar)
            return spectra

    @interactive_range_selector
    def crop_spectrum(self, left_value=None, right_value=None,):
        """Crop in place the spectral dimension.

        Parameters
        ----------
        left_value, righ_value: {int | float | None}
            If int the values are taken as indices. If float they are
            converted to indices using the spectral axis calibration.
            If left_value is None crops from the beginning of the axis.
            If right_value is None crops up to the end of the axis. If
            both are
            None the interactive cropping interface is activated
            enabling
            cropping the spectrum using a span selector in the signal
            plot.

        Raises
        ------
        SignalDimensionError if the signal dimension is not 1.

        """
        self._check_signal_dimension_equals_one()
        self.crop(
            axis=self.axes_manager.signal_axes[0].index_in_axes_manager,
            start=left_value, end=right_value)

    @auto_replot
    def gaussian_filter(self, FWHM):
        """Applies a Gaussian filter in the spectral dimension in place.

        Parameters
        ----------
        FWHM : float
            The Full Width at Half Maximum of the gaussian in the
            spectral axis units

        Raises
        ------
        ValueError if FWHM is equal or less than zero.

        SignalDimensionError if the signal dimension is not 1.

        """
        self._check_signal_dimension_equals_one()
        if FWHM <= 0:
            raise ValueError(
                "FWHM must be greater than zero")
        axis = self.axes_manager.signal_axes[0]
        FWHM *= 1 / axis.scale
        self.data = gaussian_filter1d(
            self.data,
            axis=axis.index_in_array,
            sigma=FWHM / 2.35482)
        self.events.data_changed.trigger(obj=self)

    @auto_replot
    def hanning_taper(self, side='both', channels=None, offset=0):
        """Apply a hanning taper to the data in place.

        Parameters
        ----------
        side : {'left', 'right', 'both'}
        channels : {None, int}
            The number of channels to taper. If None 5% of the total
            number of channels are tapered.
        offset : int

        Returns
        -------
        channels

        Raises
        ------
        SignalDimensionError if the signal dimension is not 1.

        """
        # TODO: generalize it
        self._check_signal_dimension_equals_one()
        if channels is None:
            channels = int(round(len(self()) * 0.02))
            if channels < 20:
                channels = 20
        dc = self.data
        if side == 'left' or side == 'both':
            dc[..., offset:channels + offset] *= (
                np.hanning(2 * channels)[:channels])
            dc[..., :offset] *= 0.
        if side == 'right' or side == 'both':
            if offset == 0:
                rl = None
            else:
                rl = -offset
            dc[..., -channels - offset:rl] *= (
                np.hanning(2 * channels)[-channels:])
            if offset != 0:
                dc[..., -offset:] *= 0.
        self.events.data_changed.trigger(obj=self)
        return channels

    def find_peaks1D_ohaver(self, xdim=None, slope_thresh=0, amp_thresh=None,
                            subchannel=True, medfilt_radius=5, maxpeakn=30000,
                            peakgroup=10):
        """Find peaks along a 1D line (peaks in spectrum/spectra).

        Function to locate the positive peaks in a noisy x-y data set.

        Detects peaks by looking for downward zero-crossings in the
        first derivative that exceed 'slope_thresh'.

        Returns an array containing position, height, and width of each
        peak.

        'slope_thresh' and 'amp_thresh', control sensitivity: higher
        values will
        neglect smaller features.


        peakgroup is the number of points around the top peak to search
        around

        Parameters
        ---------


        slope_thresh : float (optional)
                       1st derivative threshold to count the peak
                       default is set to 0.5
                       higher values will neglect smaller features.

        amp_thresh : float (optional)
                     intensity threshold above which
                     default is set to 10% of max(y)
                     higher values will neglect smaller features.

        medfilt_radius : int (optional)
                     median filter window to apply to smooth the data
                     (see scipy.signal.medfilt)
                     if 0, no filter will be applied.
                     default is set to 5

        peakgroup : int (optional)
                    number of points around the "top part" of the peak
                    default is set to 10

        maxpeakn : int (optional)
                   number of maximum detectable peaks
                   default is set to 5000

        subpix : bool (optional)
                 default is set to True

        Returns
        -------
        peaks : structured array of shape _navigation_shape_in_array in which
        each cell contains an array that contains as many structured arrays as
        peaks where found at that location and which fields: position, height,
        width, contains position, height, and width of each peak.

        Raises
        ------
        SignalDimensionError if the signal dimension is not 1.

        """
        # TODO: add scipy.signal.find_peaks_cwt
        self._check_signal_dimension_equals_one()
        axis = self.axes_manager.signal_axes[0].axis
        arr_shape = (self.axes_manager._navigation_shape_in_array
                     if self.axes_manager.navigation_size > 0
                     else [1, ])
        peaks = np.zeros(arr_shape, dtype=object)
        for y, indices in zip(self._iterate_signal(),
                              self.axes_manager._array_indices_generator()):
            peaks[indices] = find_peaks_ohaver(
                y,
                axis,
                slope_thresh=slope_thresh,
                amp_thresh=amp_thresh,
                medfilt_radius=medfilt_radius,
                maxpeakn=maxpeakn,
                peakgroup=peakgroup,
                subchannel=subchannel)
        return peaks

    def estimate_peak_width(self,
                            factor=0.5,
                            window=None,
                            return_interval=False,
                            show_progressbar=None):
        """Estimate the width of the highest intensity of peak
        of the spectra at a given fraction of its maximum.

        It can be used with asymmetric peaks. For accurate results any
        background must be previously substracted.
        The estimation is performed by interpolation using cubic splines.

        Parameters
        ----------
        factor : 0 < float < 1
            The default, 0.5, estimates the FWHM.
        window : None, float
            The size of the window centred at the peak maximum
            used to perform the estimation.
            The window size must be chosen with care: if it is narrower
            than the width of the peak at some positions or if it is
            so wide that it includes other more intense peaks this
            method cannot compute the width and a NaN is stored instead.
        return_interval: bool
            If True, returns 2 extra signals with the positions of the
            desired height fraction at the left and right of the
            peak.
        show_progressbar : None or bool
            If True, display a progress bar. If None the default is set in
            `preferences`.

        Returns
        -------
        width or [width, left, right], depending on the value of
        `return_interval`.

        """
        if show_progressbar is None:
            show_progressbar = preferences.General.show_progressbar
        self._check_signal_dimension_equals_one()
        if not 0 < factor < 1:
            raise ValueError("factor must be between 0 and 1.")

        left, right = (self._get_navigation_signal(),
                       self._get_navigation_signal())
        # The signals must be of dtype float to contain np.nan
        left.change_dtype('float')
        right.change_dtype('float')
        axis = self.axes_manager.signal_axes[0]
        x = axis.axis
        maxval = self.axes_manager.navigation_size
        if maxval > 0:
            pbar = progressbar(maxval=maxval,
                               disabled=not show_progressbar)
        for i, spectrum in enumerate(self):
            if window is not None:
                vmax = axis.index2value(spectrum.data.argmax())
                spectrum = spectrum.isig[vmax - window / 2.:vmax + window / 2.]
                x = spectrum.axes_manager[0].axis
            spline = scipy.interpolate.UnivariateSpline(
                x,
                spectrum.data - factor * spectrum.data.max(),
                s=0)
            roots = spline.roots()
            if len(roots) == 2:
                left.isig[self.axes_manager.indices] = roots[0]
                right.isig[self.axes_manager.indices] = roots[1]
            else:
                left.isig[self.axes_manager.indices] = np.nan
                right.isig[self.axes_manager.indices] = np.nan
            if maxval > 0:
                pbar.update(i)
        if maxval > 0:
            pbar.finish()
        width = right - left
        if factor == 0.5:
            width.metadata.General.title = (
                self.metadata.General.title + " FWHM")
            left.metadata.General.title = (
                self.metadata.General.title + " FWHM left position")

            right.metadata.General.title = (
                self.metadata.General.title + " FWHM right position")
        else:
            width.metadata.General.title = (
                self.metadata.General.title +
                " full-width at %.1f maximum" % factor)
            left.metadata.General.title = (
                self.metadata.General.title +
                " full-width at %.1f maximum left position" % factor)
            right.metadata.General.title = (
                self.metadata.General.title +
                " full-width at %.1f maximum right position" % factor)
        if return_interval is True:
            return [width, left, right]
        else:
            return width


class MVATools(object):
    # TODO: All of the plotting methods here should move to drawing

    def _plot_factors_or_pchars(self, factors, comp_ids=None,
                                calibrate=True, avg_char=False,
                                same_window=None, comp_label='PC',
                                img_data=None,
                                plot_shifts=True, plot_char=4,
                                cmap=plt.cm.gray, quiver_color='white',
                                vector_scale=1,
                                per_row=3, ax=None):
        """Plot components from PCA or ICA, or peak characteristics

        Parameters
        ----------

        comp_ids : None, int, or list of ints
            if None, returns maps of all components.
            if int, returns maps of components with ids from 0 to given
            int.
            if list of ints, returns maps of components with ids in
            given list.
        calibrate : bool
            if True, plots are calibrated according to the data in the
            axes
            manager.
        same_window : bool
            if True, plots each factor to the same window.  They are
            not scaled.
        comp_label : string, the label that is either the plot title
        (if plotting in
            separate windows) or the label in the legend (if plotting
            in the
            same window)
        cmap : a matplotlib colormap
            The colormap used for factor images or
            any peak characteristic scatter map
            overlay.

        Parameters only valid for peak characteristics (or pk char factors):
        --------------------------------------------------------------------

        img_data - 2D numpy array,
            The array to overlay peak characteristics onto.  If None,
            defaults to the average image of your stack.

        plot_shifts - bool, default is True
            If true, plots a quiver (arrow) plot showing the shifts for
            each
            peak present in the component being plotted.

        plot_char - None or int
            If int, the id of the characteristic to plot as the colored
            scatter plot.
            Possible components are:
               4: peak height
               5: peak orientation
               6: peak eccentricity

       quiver_color : any color recognized by matplotlib
           Determines the color of vectors drawn for
           plotting peak shifts.

       vector_scale : integer or None
           Scales the quiver plot arrows.  The vector
           is defined as one data unit along the X axis.
           If shifts are small, set vector_scale so
           that when they are multiplied by vector_scale,
           they are on the scale of the image plot.
           If None, uses matplotlib's autoscaling.

        """
        if same_window is None:
            same_window = preferences.MachineLearning.same_window
        if comp_ids is None:
            comp_ids = range(factors.shape[1])

        elif not hasattr(comp_ids, '__iter__'):
            comp_ids = range(comp_ids)

        n = len(comp_ids)
        if same_window:
            rows = int(np.ceil(n / float(per_row)))

        fig_list = []

        if n < per_row:
            per_row = n

        if same_window and self.axes_manager.signal_dimension == 2:
            f = plt.figure(figsize=(4 * per_row, 3 * rows))
        else:
            f = plt.figure()
        for i in range(len(comp_ids)):
            if self.axes_manager.signal_dimension == 1:
                if same_window:
                    ax = plt.gca()
                else:
                    if i > 0:
                        f = plt.figure()
                    ax = f.add_subplot(111)
                ax = sigdraw._plot_1D_component(
                    factors=factors,
                    idx=comp_ids[i],
                    axes_manager=self.axes_manager,
                    ax=ax,
                    calibrate=calibrate,
                    comp_label=comp_label,
                    same_window=same_window)
                if same_window:
                    plt.legend(ncol=factors.shape[1] // 2, loc='best')
            elif self.axes_manager.signal_dimension == 2:
                if same_window:
                    ax = f.add_subplot(rows, per_row, i + 1)
                else:
                    if i > 0:
                        f = plt.figure()
                    ax = f.add_subplot(111)

                sigdraw._plot_2D_component(factors=factors,
                                           idx=comp_ids[i],
                                           axes_manager=self.axes_manager,
                                           calibrate=calibrate, ax=ax,
                                           cmap=cmap, comp_label=comp_label)
            if not same_window:
                fig_list.append(f)
        try:
            plt.tight_layout()
        except:
            pass
        if not same_window:
            return fig_list
        else:
            return f

    def _plot_loadings(self, loadings, comp_ids=None, calibrate=True,
                       same_window=None, comp_label=None,
                       with_factors=False, factors=None,
                       cmap=plt.cm.gray, no_nans=False, per_row=3):
        if same_window is None:
            same_window = preferences.MachineLearning.same_window
        if comp_ids is None:
            comp_ids = range(loadings.shape[0])

        elif not hasattr(comp_ids, '__iter__'):
            comp_ids = range(comp_ids)

        n = len(comp_ids)
        if same_window:
            rows = int(np.ceil(n / float(per_row)))

        fig_list = []

        if n < per_row:
            per_row = n

        if same_window and self.axes_manager.signal_dimension == 2:
            f = plt.figure(figsize=(4 * per_row, 3 * rows))
        else:
            f = plt.figure()

        for i in range(n):
            if self.axes_manager.navigation_dimension == 1:
                if same_window:
                    ax = plt.gca()
                else:
                    if i > 0:
                        f = plt.figure()
                    ax = f.add_subplot(111)
            elif self.axes_manager.navigation_dimension == 2:
                if same_window:
                    ax = f.add_subplot(rows, per_row, i + 1)
                else:
                    if i > 0:
                        f = plt.figure()
                    ax = f.add_subplot(111)
            sigdraw._plot_loading(
                loadings, idx=comp_ids[i], axes_manager=self.axes_manager,
                no_nans=no_nans, calibrate=calibrate, cmap=cmap,
                comp_label=comp_label, ax=ax, same_window=same_window)
            if not same_window:
                fig_list.append(f)
        try:
            plt.tight_layout()
        except:
            pass
        if not same_window:
            if with_factors:
                return fig_list, self._plot_factors_or_pchars(
                    factors, comp_ids=comp_ids, calibrate=calibrate,
                    same_window=same_window, comp_label=comp_label,
                    per_row=per_row)
            else:
                return fig_list
        else:
            if self.axes_manager.navigation_dimension == 1:
                plt.legend(ncol=loadings.shape[0] // 2, loc='best')
                animate_legend()
            if with_factors:
                return f, self._plot_factors_or_pchars(factors,
                                                       comp_ids=comp_ids,
                                                       calibrate=calibrate,
                                                       same_window=same_window,
                                                       comp_label=comp_label,
                                                       per_row=per_row)
            else:
                return f

    def _export_factors(self,
                        factors,
                        folder=None,
                        comp_ids=None,
                        multiple_files=None,
                        save_figures=False,
                        save_figures_format='png',
                        factor_prefix=None,
                        factor_format=None,
                        comp_label=None,
                        cmap=plt.cm.gray,
                        plot_shifts=True,
                        plot_char=4,
                        img_data=None,
                        same_window=False,
                        calibrate=True,
                        quiver_color='white',
                        vector_scale=1,
                        no_nans=True, per_row=3):

        from hyperspy._signals.image import Image
        from hyperspy._signals.spectrum import Spectrum

        if multiple_files is None:
            multiple_files = preferences.MachineLearning.multiple_files

        if factor_format is None:
            factor_format = preferences.MachineLearning.\
                export_factors_default_file_format

        # Select the desired factors
        if comp_ids is None:
            comp_ids = range(factors.shape[1])
        elif not hasattr(comp_ids, '__iter__'):
            comp_ids = range(comp_ids)
        mask = np.zeros(factors.shape[1], dtype=np.bool)
        for idx in comp_ids:
            mask[idx] = 1
        factors = factors[:, mask]

        if save_figures is True:
            plt.ioff()
            fac_plots = self._plot_factors_or_pchars(factors,
                                                     comp_ids=comp_ids,
                                                     same_window=same_window,
                                                     comp_label=comp_label,
                                                     img_data=img_data,
                                                     plot_shifts=plot_shifts,
                                                     plot_char=plot_char,
                                                     cmap=cmap,
                                                     per_row=per_row,
                                                     quiver_color=quiver_color,
                                                     vector_scale=vector_scale)
            for idx in range(len(comp_ids)):
                filename = '%s_%02i.%s' % (factor_prefix, comp_ids[idx],
                                           save_figures_format)
                if folder is not None:
                    filename = os.path.join(folder, filename)
                ensure_directory(filename)
                _args = {'dpi': 600,
                         'format': save_figures_format}
                fac_plots[idx].savefig(filename, **_args)
            plt.ion()

        elif multiple_files is False:
            if self.axes_manager.signal_dimension == 2:
                # factor images
                axes_dicts = []
                axes = self.axes_manager.signal_axes[::-1]
                shape = (axes[1].size, axes[0].size)
                factor_data = np.rollaxis(
                    factors.reshape((shape[0], shape[1], -1)), 2)
                axes_dicts.append(axes[0].get_axis_dictionary())
                axes_dicts.append(axes[1].get_axis_dictionary())
                axes_dicts.append({'name': 'factor_index',
                                   'scale': 1.,
                                   'offset': 0.,
                                   'size': int(factors.shape[1]),
                                   'units': 'factor',
                                   'index_in_array': 0, })
                s = Image(factor_data,
                          axes=axes_dicts,
                          metadata={
                              'General': {'title': '%s from %s' % (
                                  factor_prefix,
                                  self.metadata.General.title),
                              }})
            elif self.axes_manager.signal_dimension == 1:
                axes = [self.axes_manager.signal_axes[0].get_axis_dictionary(),
                        {'name': 'factor_index',
                         'scale': 1.,
                         'offset': 0.,
                         'size': int(factors.shape[1]),
                         'units': 'factor',
                         'index_in_array': 0,
                         }]
                axes[0]['index_in_array'] = 1
                s = Spectrum(
                    factors.T, axes=axes, metadata={
                        "General": {
                            'title': '%s from %s' %
                            (factor_prefix, self.metadata.General.title), }})
            filename = '%ss.%s' % (factor_prefix, factor_format)
            if folder is not None:
                filename = os.path.join(folder, filename)
            s.save(filename)
        else:  # Separate files
            if self.axes_manager.signal_dimension == 1:

                axis_dict = self.axes_manager.signal_axes[0].\
                    get_axis_dictionary()
                axis_dict['index_in_array'] = 0
                for dim, index in zip(comp_ids, range(len(comp_ids))):
                    s = Spectrum(factors[:, index],
                                 axes=[axis_dict, ],
                                 metadata={
                                     "General": {'title': '%s from %s' % (
                                         factor_prefix,
                                         self.metadata.General.title),
                                     }})
                    filename = '%s-%i.%s' % (factor_prefix,
                                             dim,
                                             factor_format)
                    if folder is not None:
                        filename = os.path.join(folder, filename)
                    s.save(filename)

            if self.axes_manager.signal_dimension == 2:
                axes = self.axes_manager.signal_axes
                axes_dicts = [axes[0].get_axis_dictionary(),
                              axes[1].get_axis_dictionary()]
                axes_dicts[0]['index_in_array'] = 0
                axes_dicts[1]['index_in_array'] = 1

                factor_data = factors.reshape(
                    self.axes_manager._signal_shape_in_array + [-1, ])

                for dim, index in zip(comp_ids, range(len(comp_ids))):
                    im = Image(factor_data[..., index],
                               axes=axes_dicts,
                               metadata={
                                   "General": {'title': '%s from %s' % (
                                       factor_prefix,
                                       self.metadata.General.title),
                                   }})
                    filename = '%s-%i.%s' % (factor_prefix,
                                             dim,
                                             factor_format)
                    if folder is not None:
                        filename = os.path.join(folder, filename)
                    im.save(filename)

    def _export_loadings(self,
                         loadings,
                         folder=None,
                         comp_ids=None,
                         multiple_files=None,
                         loading_prefix=None,
                         loading_format=None,
                         save_figures_format='png',
                         comp_label=None,
                         cmap=plt.cm.gray,
                         save_figures=False,
                         same_window=False,
                         calibrate=True,
                         no_nans=True,
                         per_row=3):

        from hyperspy._signals.image import Image
        from hyperspy._signals.spectrum import Spectrum

        if multiple_files is None:
            multiple_files = preferences.MachineLearning.multiple_files

        if loading_format is None:
            loading_format = preferences.MachineLearning.\
                export_loadings_default_file_format

        if comp_ids is None:
            comp_ids = range(loadings.shape[0])
        elif not hasattr(comp_ids, '__iter__'):
            comp_ids = range(comp_ids)
        mask = np.zeros(loadings.shape[0], dtype=np.bool)
        for idx in comp_ids:
            mask[idx] = 1
        loadings = loadings[mask]

        if save_figures is True:
            plt.ioff()
            sc_plots = self._plot_loadings(loadings, comp_ids=comp_ids,
                                           calibrate=calibrate,
                                           same_window=same_window,
                                           comp_label=comp_label,
                                           cmap=cmap, no_nans=no_nans,
                                           per_row=per_row)
            for idx in range(len(comp_ids)):
                filename = '%s_%02i.%s' % (loading_prefix, comp_ids[idx],
                                           save_figures_format)
                if folder is not None:
                    filename = os.path.join(folder, filename)
                ensure_directory(filename)
                _args = {'dpi': 600,
                         'format': save_figures_format}
                sc_plots[idx].savefig(filename, **_args)
            plt.ion()
        elif multiple_files is False:
            if self.axes_manager.navigation_dimension == 2:
                axes_dicts = []
                axes = self.axes_manager.navigation_axes[::-1]
                shape = (axes[1].size, axes[0].size)
                loading_data = loadings.reshape((-1, shape[0], shape[1]))
                axes_dicts.append(axes[0].get_axis_dictionary())
                axes_dicts[0]['index_in_array'] = 1
                axes_dicts.append(axes[1].get_axis_dictionary())
                axes_dicts[1]['index_in_array'] = 2
                axes_dicts.append({'name': 'loading_index',
                                   'scale': 1.,
                                   'offset': 0.,
                                   'size': int(loadings.shape[0]),
                                   'units': 'factor',
                                   'index_in_array': 0, })
                s = Image(loading_data,
                          axes=axes_dicts,
                          metadata={
                              "General": {'title': '%s from %s' % (
                                  loading_prefix,
                                  self.metadata.General.title),
                              }})
            elif self.axes_manager.navigation_dimension == 1:
                cal_axis = self.axes_manager.navigation_axes[0].\
                    get_axis_dictionary()
                cal_axis['index_in_array'] = 1
                axes = [{'name': 'loading_index',
                         'scale': 1.,
                         'offset': 0.,
                         'size': int(loadings.shape[0]),
                         'units': 'comp_id',
                         'index_in_array': 0, },
                        cal_axis]
                s = Image(loadings,
                          axes=axes,
                          metadata={
                              "General": {'title': '%s from %s' % (
                                  loading_prefix,
                                  self.metadata.General.title),
                              }})
            filename = '%ss.%s' % (loading_prefix, loading_format)
            if folder is not None:
                filename = os.path.join(folder, filename)
            s.save(filename)
        else:  # Separate files
            if self.axes_manager.navigation_dimension == 1:
                axis_dict = self.axes_manager.navigation_axes[0].\
                    get_axis_dictionary()
                axis_dict['index_in_array'] = 0
                for dim, index in zip(comp_ids, range(len(comp_ids))):
                    s = Spectrum(loadings[index],
                                 axes=[axis_dict, ])
                    filename = '%s-%i.%s' % (loading_prefix,
                                             dim,
                                             loading_format)
                    if folder is not None:
                        filename = os.path.join(folder, filename)
                    s.save(filename)
            elif self.axes_manager.navigation_dimension == 2:
                axes_dicts = []
                axes = self.axes_manager.navigation_axes[::-1]
                shape = (axes[0].size, axes[1].size)
                loading_data = loadings.reshape((-1, shape[0], shape[1]))
                axes_dicts.append(axes[0].get_axis_dictionary())
                axes_dicts[0]['index_in_array'] = 0
                axes_dicts.append(axes[1].get_axis_dictionary())
                axes_dicts[1]['index_in_array'] = 1
                for dim, index in zip(comp_ids, range(len(comp_ids))):
                    s = Image(loading_data[index, ...],
                              axes=axes_dicts,
                              metadata={
                                  "General": {'title': '%s from %s' % (
                                      loading_prefix,
                                      self.metadata.General.title),
                                  }})
                    filename = '%s-%i.%s' % (loading_prefix,
                                             dim,
                                             loading_format)
                    if folder is not None:
                        filename = os.path.join(folder, filename)
                    s.save(filename)

    def plot_decomposition_factors(self,
                                   comp_ids=None,
                                   calibrate=True,
                                   same_window=None,
                                   comp_label='Decomposition factor',
                                   per_row=3):
        """Plot factors from a decomposition.

        Parameters
        ----------

        comp_ids : None, int, or list of ints
            if None, returns maps of all components.
            if int, returns maps of components with ids from 0 to given
            int.
            if list of ints, returns maps of components with ids in
            given list.

        calibrate : bool
            if True, calibrates plots where calibration is available
            from
            the axes_manager.  If False, plots are in pixels/channels.

        same_window : bool
            if True, plots each factor to the same window.  They are
            not scaled.

        comp_label : string, the label that is either the plot title
        (if plotting in
            separate windows) or the label in the legend (if plotting
            in the
            same window)

        cmap : The colormap used for the factor image, or for peak
            characteristics, the colormap used for the scatter plot of
            some peak characteristic.

        per_row : int, the number of plots in each row, when the
        same_window
            parameter is True.

        See Also
        --------
        plot_decomposition_loadings, plot_decomposition_results.

        """
        if self.axes_manager.signal_dimension > 2:
            raise NotImplementedError("This method cannot plot factors of "
                                      "signals of dimension higher than 2."
                                      "You can use "
                                      "`plot_decomposition_results` instead.")
        if same_window is None:
            same_window = preferences.MachineLearning.same_window
        factors = self.learning_results.factors
        if comp_ids is None:
            comp_ids = self.learning_results.output_dimension

        return self._plot_factors_or_pchars(factors,
                                            comp_ids=comp_ids,
                                            calibrate=calibrate,
                                            same_window=same_window,
                                            comp_label=comp_label,
                                            per_row=per_row)

    def plot_bss_factors(self, comp_ids=None, calibrate=True,
                         same_window=None, comp_label='BSS factor',
                         per_row=3):
        """Plot factors from blind source separation results.

        Parameters
        ----------

        comp_ids : None, int, or list of ints
            if None, returns maps of all components.
            if int, returns maps of components with ids from 0 to
            given int.
            if list of ints, returns maps of components with ids in
            given list.

        calibrate : bool
            if True, calibrates plots where calibration is available
            from
            the axes_manager.  If False, plots are in pixels/channels.

        same_window : bool
            if True, plots each factor to the same window.  They are
            not scaled.

        comp_label : string, the label that is either the plot title
        (if plotting in
            separate windows) or the label in the legend (if plotting
            in the
            same window)

        cmap : The colormap used for the factor image, or for peak
            characteristics, the colormap used for the scatter plot of
            some peak characteristic.

        per_row : int, the number of plots in each row, when the
        same_window
            parameter is True.

        See Also
        --------
        plot_bss_loadings, plot_bss_results.

        """
        if self.axes_manager.signal_dimension > 2:
            raise NotImplementedError("This method cannot plot factors of "
                                      "signals of dimension higher than 2."
                                      "You can use "
                                      "`plot_decomposition_results` instead.")

        if same_window is None:
            same_window = preferences.MachineLearning.same_window
        factors = self.learning_results.bss_factors
        return self._plot_factors_or_pchars(factors,
                                            comp_ids=comp_ids,
                                            calibrate=calibrate,
                                            same_window=same_window,
                                            comp_label=comp_label,
                                            per_row=per_row)

    def plot_decomposition_loadings(self,
                                    comp_ids=None,
                                    calibrate=True,
                                    same_window=None,
                                    comp_label='Decomposition loading',
                                    with_factors=False,
                                    cmap=plt.cm.gray,
                                    no_nans=False,
                                    per_row=3):
        """Plot loadings from PCA.

        Parameters
        ----------

        comp_ids : None, int, or list of ints
            if None, returns maps of all components.
            if int, returns maps of components with ids from 0 to
            given int.
            if list of ints, returns maps of components with ids in
            given list.

        calibrate : bool
            if True, calibrates plots where calibration is available
            from
            the axes_manager.  If False, plots are in pixels/channels.

        same_window : bool
            if True, plots each factor to the same window.  They are
            not scaled.

        comp_label : string,
            The label that is either the plot title (if plotting in
            separate windows) or the label in the legend (if plotting
            in the same window). In this case, each loading line can be
            toggled on and off by clicking on the legended line.

        with_factors : bool
            If True, also returns figure(s) with the factors for the
            given comp_ids.

        cmap : matplotlib colormap
            The colormap used for the factor image, or for peak
            characteristics, the colormap used for the scatter plot of
            some peak characteristic.

        no_nans : bool
            If True, removes NaN's from the loading plots.

        per_row : int
            the number of plots in each row, when the same_window
            parameter is True.

        See Also
        --------
        plot_decomposition_factors, plot_decomposition_results.

        """
        if self.axes_manager.navigation_dimension > 2:
            raise NotImplementedError("This method cannot plot loadings of "
                                      "dimension higher than 2."
                                      "You can use "
                                      "`plot_decomposition_results` instead.")
        if same_window is None:
            same_window = preferences.MachineLearning.same_window
        loadings = self.learning_results.loadings.T
        if with_factors:
            factors = self.learning_results.factors
        else:
            factors = None

        if comp_ids is None:
            comp_ids = self.learning_results.output_dimension
        return self._plot_loadings(
            loadings,
            comp_ids=comp_ids,
            with_factors=with_factors,
            factors=factors,
            same_window=same_window,
            comp_label=comp_label,
            cmap=cmap,
            no_nans=no_nans,
            per_row=per_row)

    def plot_bss_loadings(self, comp_ids=None, calibrate=True,
                          same_window=None, comp_label='BSS loading',
                          with_factors=False, cmap=plt.cm.gray,
                          no_nans=False, per_row=3):
        """Plot loadings from ICA

        Parameters
        ----------

        comp_ids : None, int, or list of ints
            if None, returns maps of all components.
            if int, returns maps of components with ids from 0 to
            given int.
            if list of ints, returns maps of components with ids in
            given list.

        calibrate : bool
            if True, calibrates plots where calibration is available
            from
            the axes_manager.  If False, plots are in pixels/channels.

        same_window : bool
            if True, plots each factor to the same window.  They are
            not scaled.

        comp_label : string,
            The label that is either the plot title (if plotting in
            separate windows) or the label in the legend (if plotting
            in the same window). In this case, each loading line can be
            toggled on and off by clicking on the legended line.

        with_factors : bool
            If True, also returns figure(s) with the factors for the
            given comp_ids.

        cmap : matplotlib colormap
            The colormap used for the factor image, or for peak
            characteristics, the colormap used for the scatter plot of
            some peak characteristic.

        no_nans : bool
            If True, removes NaN's from the loading plots.

        per_row : int
            the number of plots in each row, when the same_window
            parameter is True.

        See Also
        --------
        plot_bss_factors, plot_bss_results.

        """
        if self.axes_manager.navigation_dimension > 2:
            raise NotImplementedError("This method cannot plot loadings of "
                                      "dimension higher than 2."
                                      "You can use "
                                      "`plot_bss_results` instead.")
        if same_window is None:
            same_window = preferences.MachineLearning.same_window
        loadings = self.learning_results.bss_loadings.T
        if with_factors:
            factors = self.learning_results.bss_factors
        else:
            factors = None
        return self._plot_loadings(
            loadings,
            comp_ids=comp_ids,
            with_factors=with_factors,
            factors=factors,
            same_window=same_window,
            comp_label=comp_label,
            cmap=cmap,
            no_nans=no_nans,
            per_row=per_row)

    def export_decomposition_results(self, comp_ids=None,
                                     folder=None,
                                     calibrate=True,
                                     factor_prefix='factor',
                                     factor_format=None,
                                     loading_prefix='loading',
                                     loading_format=None,
                                     comp_label=None,
                                     cmap=plt.cm.gray,
                                     same_window=False,
                                     multiple_files=None,
                                     no_nans=True,
                                     per_row=3,
                                     save_figures=False,
                                     save_figures_format='png'):
        """Export results from a decomposition to any of the supported
        formats.

        Parameters
        ----------
        comp_ids : None, int, or list of ints
            if None, returns all components/loadings.
            if int, returns components/loadings with ids from 0 to
            given int.
            if list of ints, returns components/loadings with ids in
            given list.
        folder : str or None
            The path to the folder where the file will be saved.
            If `None` the
            current folder is used by default.
        factor_prefix : string
            The prefix that any exported filenames for
            factors/components
            begin with
        factor_format : string
            The extension of the format that you wish to save to.
        loading_prefix : string
            The prefix that any exported filenames for
            factors/components
            begin with
        loading_format : string
            The extension of the format that you wish to save to.
            Determines
            the kind of output.
                - For image formats (tif, png, jpg, etc.), plots are
                created
                  using the plotting flags as below, and saved at
                  600 dpi.
                  One plot per loading is saved.
                - For multidimensional formats (rpl, hdf5), arrays are
                saved
                  in single files.  All loadings are contained in the
                  one
                  file.
                - For spectral formats (msa), each loading is saved to a
                  separate file.
        multiple_files : Bool
            If True, on exporting a file per factor and per loading will
             be
            created. Otherwise only two files will be created, one for
            the
            factors and another for the loadings. The default value can
            be
            chosen in the preferences.
        save_figures : Bool
            If True the same figures that are obtained when using the
            plot
            methods will be saved with 600 dpi resolution

        Plotting options (for save_figures = True ONLY)
        ----------------------------------------------

        calibrate : bool
            if True, calibrates plots where calibration is available
            from
            the axes_manager.  If False, plots are in pixels/channels.
        same_window : bool
            if True, plots each factor to the same window.
        comp_label : string, the label that is either the plot title
            (if plotting in separate windows) or the label in the legend
            (if plotting in the same window)
        cmap : The colormap used for the factor image, or for peak
            characteristics, the colormap used for the scatter plot of
            some peak characteristic.
        per_row : int, the number of plots in each row, when the
        same_window
            parameter is True.
        save_figures_format : str
            The image format extension.

        See Also
        --------
        get_decomposition_factors,
        get_decomposition_loadings.

        """

        factors = self.learning_results.factors
        loadings = self.learning_results.loadings.T
        self._export_factors(
            factors,
            folder=folder,
            comp_ids=comp_ids,
            calibrate=calibrate,
            multiple_files=multiple_files,
            factor_prefix=factor_prefix,
            factor_format=factor_format,
            comp_label=comp_label,
            save_figures=save_figures,
            cmap=cmap,
            no_nans=no_nans,
            same_window=same_window,
            per_row=per_row,
            save_figures_format=save_figures_format)
        self._export_loadings(
            loadings,
            comp_ids=comp_ids, folder=folder,
            calibrate=calibrate,
            multiple_files=multiple_files,
            loading_prefix=loading_prefix,
            loading_format=loading_format,
            comp_label=comp_label,
            cmap=cmap,
            save_figures=save_figures,
            same_window=same_window,
            no_nans=no_nans,
            per_row=per_row)

    def export_bss_results(self,
                           comp_ids=None,
                           folder=None,
                           calibrate=True,
                           multiple_files=None,
                           save_figures=False,
                           factor_prefix='bss_factor',
                           factor_format=None,
                           loading_prefix='bss_loading',
                           loading_format=None,
                           comp_label=None, cmap=plt.cm.gray,
                           same_window=False,
                           no_nans=True,
                           per_row=3,
                           save_figures_format='png'):
        """Export results from ICA to any of the supported formats.

        Parameters
        ----------
        comp_ids : None, int, or list of ints
            if None, returns all components/loadings.
            if int, returns components/loadings with ids from 0 to given
             int.
            if list of ints, returns components/loadings with ids in
            iven list.
        folder : str or None
            The path to the folder where the file will be saved. If
            `None` the
            current folder is used by default.
        factor_prefix : string
            The prefix that any exported filenames for
            factors/components
            begin with
        factor_format : string
            The extension of the format that you wish to save to.
            Determines
            the kind of output.
                - For image formats (tif, png, jpg, etc.), plots are
                created
                  using the plotting flags as below, and saved at
                  600 dpi.
                  One plot per factor is saved.
                - For multidimensional formats (rpl, hdf5), arrays are
                saved
                  in single files.  All factors are contained in the one
                  file.
                - For spectral formats (msa), each factor is saved to a
                  separate file.

        loading_prefix : string
            The prefix that any exported filenames for
            factors/components
            begin with
        loading_format : string
            The extension of the format that you wish to save to.
        multiple_files : Bool
            If True, on exporting a file per factor and per loading
            will be
            created. Otherwise only two files will be created, one
            for the
            factors and another for the loadings. The default value
            can be
            chosen in the preferences.
        save_figures : Bool
            If True the same figures that are obtained when using the
            plot
            methods will be saved with 600 dpi resolution

        Plotting options (for save_figures = True ONLY)
        ----------------------------------------------
        calibrate : bool
            if True, calibrates plots where calibration is available
            from
            the axes_manager.  If False, plots are in pixels/channels.
        same_window : bool
            if True, plots each factor to the same window.
        comp_label : string
            the label that is either the plot title (if plotting in
            separate windows) or the label in the legend (if plotting
            in the
            same window)
        cmap : The colormap used for the factor image, or for peak
            characteristics, the colormap used for the scatter plot of
            some peak characteristic.
        per_row : int, the number of plots in each row, when the
        same_window
            parameter is True.
        save_figures_format : str
            The image format extension.

        See Also
        --------
        get_bss_factors,
        get_bss_loadings.

        """

        factors = self.learning_results.bss_factors
        loadings = self.learning_results.bss_loadings.T
        self._export_factors(factors,
                             folder=folder,
                             comp_ids=comp_ids,
                             calibrate=calibrate,
                             multiple_files=multiple_files,
                             factor_prefix=factor_prefix,
                             factor_format=factor_format,
                             comp_label=comp_label,
                             save_figures=save_figures,
                             cmap=cmap,
                             no_nans=no_nans,
                             same_window=same_window,
                             per_row=per_row,
                             save_figures_format=save_figures_format)

        self._export_loadings(loadings,
                              comp_ids=comp_ids,
                              folder=folder,
                              calibrate=calibrate,
                              multiple_files=multiple_files,
                              loading_prefix=loading_prefix,
                              loading_format=loading_format,
                              comp_label=comp_label,
                              cmap=cmap,
                              save_figures=save_figures,
                              same_window=same_window,
                              no_nans=no_nans,
                              per_row=per_row,
                              save_figures_format=save_figures_format)

    def _get_loadings(self, loadings):
        from hyperspy.api import signals
        data = loadings.T.reshape(
            (-1,) + self.axes_manager.navigation_shape[::-1])
        signal = signals.Signal(
            data,
            axes=(
                [{"size": data.shape[0], "navigate": True}] +
                self.axes_manager._get_navigation_axes_dicts()))
        signal.set_signal_origin(self.metadata.Signal.signal_origin)
        for axis in signal.axes_manager._axes[1:]:
            axis.navigate = False
        return signal

    def _get_factors(self, factors):
        signal = self.__class__(
            factors.T.reshape((-1,) + self.axes_manager.signal_shape[::-1]),
            axes=[{"size": factors.shape[-1], "navigate": True}] +
            self.axes_manager._get_signal_axes_dicts())
        signal.set_signal_origin(self.metadata.Signal.signal_origin)
        signal.set_signal_type(self.metadata.Signal.signal_type)
        for axis in signal.axes_manager._axes[1:]:
            axis.navigate = False
        return signal

    def get_decomposition_loadings(self):
        """Return the decomposition loadings as a Signal.

        See Also
        -------
        get_decomposition_factors, export_decomposition_results.

        """
        signal = self._get_loadings(self.learning_results.loadings)
        signal.axes_manager._axes[0].name = "Decomposition component index"
        signal.metadata.General.title = "Decomposition loadings of " + \
            self.metadata.General.title
        return signal

    def get_decomposition_factors(self):
        """Return the decomposition factors as a Signal.

        See Also
        -------
        get_decomposition_loadings, export_decomposition_results.

        """
        signal = self._get_factors(self.learning_results.factors)
        signal.axes_manager._axes[0].name = "Decomposition component index"
        signal.metadata.General.title = ("Decomposition factors of " +
                                         self.metadata.General.title)
        return signal

    def get_bss_loadings(self):
        """Return the blind source separtion loadings as a Signal.

        See Also
        -------
        get_bss_factors, export_bss_results.

        """
        signal = self._get_loadings(
            self.learning_results.bss_loadings)
        signal.axes_manager[0].name = "BSS component index"
        signal.metadata.General.title = ("BSS loadings of " +
                                         self.metadata.General.title)
        return signal

    def get_bss_factors(self):
        """Return the blind source separtion factors as a Signal.

        See Also
        -------
        get_bss_loadings, export_bss_results.

        """
        signal = self._get_factors(self.learning_results.bss_factors)
        signal.axes_manager[0].name = "BSS component index"
        signal.metadata.General.title = ("BSS factors of " +
                                         self.metadata.General.title)
        return signal

    def plot_bss_results(self,
                         factors_navigator="auto",
                         loadings_navigator="auto",
                         factors_dim=2,
                         loadings_dim=2,):
        """Plot the blind source separation factors and loadings.

        Unlike `plot_bss_factors` and `plot_bss_loadings`, this method displays
        one component at a time. Therefore it provides a more compact
        visualization than then other two methods.  The loadings and factors
        are displayed in different windows and each has its own
        navigator/sliders to navigate them if they are multidimensional. The
        component index axis is syncronize between the two.

        Parameters
        ----------
        factor_navigator, loadings_navigator : {"auto", None, "spectrum",
        Signal}
            See `plot` documentation for details.
        factors_dim, loadings_dim: int
            Currently HyperSpy cannot plot signals of dimension higher than
            two. Therefore, to visualize the BSS results when the
            factors or the loadings have signal dimension greater than 2
            we can view the data as spectra(images) by setting this parameter
            to 1(2). (Default 2)

        See Also
        --------
        plot_bss_factors, plot_bss_loadings, plot_decomposition_results.

        """
        factors = self.get_bss_factors()
        loadings = self.get_bss_loadings()
        factors.axes_manager._axes[0] = loadings.axes_manager._axes[0]
        if loadings.axes_manager.signal_dimension > 2:
            loadings.axes_manager.set_signal_dimension(loadings_dim)
        if factors.axes_manager.signal_dimension > 2:
            factors.axes_manager.set_signal_dimension(factors_dim)
        loadings.plot(navigator=loadings_navigator)
        factors.plot(navigator=factors_navigator)

    def plot_decomposition_results(self,
                                   factors_navigator="auto",
                                   loadings_navigator="auto",
                                   factors_dim=2,
                                   loadings_dim=2):
        """Plot the decompostion factors and loadings.

        Unlike `plot_factors` and `plot_loadings`, this method displays
        one component at a time. Therefore it provides a more compact
        visualization than then other two methods.  The loadings and factors
        are displayed in different windows and each has its own
        navigator/sliders to navigate them if they are multidimensional. The
        component index axis is syncronize between the two.

        Parameters
        ----------
        factor_navigator, loadings_navigator : {"auto", None, "spectrum",
        Signal}
            See `plot` documentation for details.
        factors_dim, loadings_dim : int
            Currently HyperSpy cannot plot signals of dimension higher than
            two. Therefore, to visualize the BSS results when the
            factors or the loadings have signal dimension greater than 2
            we can view the data as spectra(images) by setting this parameter
            to 1(2). (Default 2)

        See Also
        --------
        plot_factors, plot_loadings, plot_bss_results.

        """
        factors = self.get_decomposition_factors()
        loadings = self.get_decomposition_loadings()
        factors.axes_manager._axes[0] = loadings.axes_manager._axes[0]
        if loadings.axes_manager.signal_dimension > 2:
            loadings.axes_manager.set_signal_dimension(loadings_dim)
        if factors.axes_manager.signal_dimension > 2:
            factors.axes_manager.set_signal_dimension(factors_dim)
        loadings.plot(navigator=loadings_navigator)
        factors.plot(navigator=factors_navigator)


class SpecialSlicersSignal(SpecialSlicers):

    def __setitem__(self, i, j):
        """x.__setitem__(i, y) <==> x[i]=y
        """
        if isinstance(j, Signal):
            j = j.data
        array_slices = self.obj._get_array_slices(i, self.isNavigation)
        self.obj.data[array_slices] = j

    def __len__(self):
        return self.obj.axes_manager.signal_shape[0]


class Signal(FancySlicing,
             MVA,
             MVATools,
             Signal1DTools,
             Signal2DTools,):

    _record_by = ""
    _signal_type = ""
    _signal_origin = ""
    _additional_slicing_targets = [
        "metadata.Signal.Noise_properties.variance",
    ]

    def __init__(self, data, **kwds):
        """Create a Signal from a numpy array.

        Parameters
        ----------
        data : numpy array
           The signal data. It can be an array of any dimensions.
        axes : dictionary (optional)
            Dictionary to define the axes (see the
            documentation of the AxesManager class for more details).
        attributes : dictionary (optional)
            A dictionary whose items are stored as attributes.
        metadata : dictionary (optional)
            A dictionary containing a set of parameters
            that will to stores in the `metadata` attribute.
            Some parameters might be mandatory in some cases.
        original_metadata : dictionary (optional)
            A dictionary containing a set of parameters
            that will to stores in the `original_metadata` attribute. It
            typically contains all the parameters that has been
            imported from the original data file.

        """
        self._create_metadata()
        self.models = ModelManager(self)
        self.learning_results = LearningResults()
        kwds['data'] = data
        self._load_dictionary(kwds)
        self._plot = None
        self.auto_replot = True
        self.inav = SpecialSlicersSignal(self, True)
        self.isig = SpecialSlicersSignal(self, False)
        self.events = Events()
        self.events.data_changed = Event("""
            Event that triggers when the data has changed

            The event trigger when the data is ready for consumption by any
            process that depend on it as input. Plotted signals automatically
            connect this Event to its `Signal.plot()`.

            Note: The event only fires at certain specific times, not everytime
            that the `Signal.data` array changes values.

            Arguments:
                obj: The signal that owns the data.
            """, arguments=['obj'])

    def _create_metadata(self):
        self.metadata = DictionaryTreeBrowser()
        mp = self.metadata
        mp.add_node("_HyperSpy")
        mp.add_node("General")
        mp.add_node("Signal")
        mp._HyperSpy.add_node("Folding")
        folding = mp._HyperSpy.Folding
        folding.unfolded = False
        folding.signal_unfolded = False
        folding.original_shape = None
        folding.original_axes_manager = None
        mp.Signal.binned = False
        self.original_metadata = DictionaryTreeBrowser()
        self.tmp_parameters = DictionaryTreeBrowser()

    def __repr__(self):
        if self.metadata._HyperSpy.Folding.unfolded:
            unfolded = "unfolded "
        else:
            unfolded = ""
        string = '<'
        string += self.__class__.__name__
        string += ", title: %s" % self.metadata.General.title
        string += ", %sdimensions: %s" % (
            unfolded,
            self.axes_manager._get_dimension_str())

        string += '>'

        return string

    def _binary_operator_ruler(self, other, op_name):
        exception_message = (
            "Invalid dimensions for this operation")
        if isinstance(other, Signal):
            # Both objects are signals
            oam = other.axes_manager
            sam = self.axes_manager
            if sam.navigation_shape == oam.navigation_shape and \
                    sam.signal_shape == oam.signal_shape:
                # They have the same signal shape.
                # The signal axes are aligned but there is
                # no guarantee that data axes area aligned so we make sure that
                # they are aligned for the operation.
                sdata = self._data_aligned_with_axes
                odata = other._data_aligned_with_axes
                if op_name in INPLACE_OPERATORS:
                    self.data = getattr(sdata, op_name)(odata)
                    self.axes_manager._sort_axes()
                    return self
                else:
                    ns = self._deepcopy_with_new_data(
                        getattr(sdata, op_name)(odata))
                    ns.axes_manager._sort_axes()
                    return ns
            else:
                # Different navigation and/or signal shapes
                if not are_signals_aligned(self, other):
                    raise ValueError(exception_message)
                else:
                    # They are broadcastable but have different number of axes
                    new_nav_axes = []
                    for saxis, oaxis in zip(
                            sam.navigation_axes, oam.navigation_axes):
                        new_nav_axes.append(saxis if saxis.size > 1 or
                                            oaxis.size == 1 else
                                            oaxis)
                    if sam.navigation_dimension != oam.navigation_dimension:
                        bigger_am = (sam
                                     if sam.navigation_dimension >
                                     oam.navigation_dimension
                                     else oam)
                        new_nav_axes.extend(
                            bigger_am.navigation_axes[len(new_nav_axes):])
                    # Because they are broadcastable and navigation axes come
                    # first in the data array, we don't need to pad the data
                    # array.
                    new_sig_axes = []
                    for saxis, oaxis in zip(
                            sam.signal_axes, oam.signal_axes):
                        new_sig_axes.append(saxis if saxis.size > 1 or
                                            oaxis.size == 1 else
                                            oaxis)
                    if sam.signal_dimension != oam.signal_dimension:
                        bigger_am = (
                            sam if sam.signal_dimension > oam.signal_dimension
                            else oam)
                        new_sig_axes.extend(
                            bigger_am.signal_axes[len(new_sig_axes):])
                    sdim_diff = abs(sam.signal_dimension -
                                    oam.signal_dimension)
                    sdata = self._data_aligned_with_axes
                    odata = other._data_aligned_with_axes
                    if len(new_nav_axes) and sdim_diff:
                        if bigger_am is sam:
                            # Pad odata
                            while sdim_diff:
                                odata = np.expand_dims(
                                    odata, oam.navigation_dimension)
                                sdim_diff -= 1
                        else:
                            # Pad sdata
                            while sdim_diff:
                                sdata = np.expand_dims(
                                    sdata, sam.navigation_dimension)
                                sdim_diff -= 1
                    if op_name in INPLACE_OPERATORS:
                        # This should raise a ValueError if the operation
                        # changes the shape of the object on the left.
                        self.data = getattr(sdata, op_name)(odata)
                        self.axes_manager._sort_axes()
                        return self
                    else:
                        ns = self._deepcopy_with_new_data(
                            getattr(sdata, op_name)(odata))
                        new_axes = new_nav_axes[::-1] + new_sig_axes[::-1]
                        ns.axes_manager._axes = [axis.copy()
                                                 for axis in new_axes]
                        if bigger_am is oam:
                            ns.metadata.Signal.record_by = \
                                other.metadata.Signal.record_by
                            ns._assign_subclass()
                        return ns

        else:
            # Second object is not a Signal
            if op_name in INPLACE_OPERATORS:
                getattr(self.data, op_name)(other)
                return self
            else:
                return self._deepcopy_with_new_data(
                    getattr(self.data, op_name)(other))

    def _unary_operator_ruler(self, op_name):
        return self._deepcopy_with_new_data(getattr(self.data, op_name)())

    def _check_signal_dimension_equals_one(self):
        if self.axes_manager.signal_dimension != 1:
            raise SignalDimensionError(self.axes_manager.signal_dimension, 1)

    def _check_signal_dimension_equals_two(self):
        if self.axes_manager.signal_dimension != 2:
            raise SignalDimensionError(self.axes_manager.signal_dimension, 2)

    def _deepcopy_with_new_data(self, data=None):
        """Returns a deepcopy of itself replacing the data.

        This method has the advantage over deepcopy that it does not
        copy the data what can save precious memory

        Parameters
        ---------
        data : {None | np.array}

        Returns
        -------
        ns : Signal

        """
        try:
            old_data = self.data
            self.data = None
            old_plot = self._plot
            self._plot = None
            old_models = self.models._models
            self.models._models = DictionaryTreeBrowser()
            ns = self.deepcopy()
            ns.data = np.atleast_1d(data)
            return ns
        finally:
            self.data = old_data
            self._plot = old_plot
            self.models._models = old_models

    def _print_summary(self):
        string = "\n\tTitle: "
        string += self.metadata.General.title.decode('utf8')
        if self.metadata.has_item("Signal.signal_type"):
            string += "\n\tSignal type: "
            string += self.metadata.Signal.signal_type
        string += "\n\tData dimensions: "
        string += str(self.axes_manager.shape)
        if self.metadata.has_item('Signal.record_by'):
            string += "\n\tData representation: "
            string += self.metadata.Signal.record_by
            string += "\n\tData type: "
            string += str(self.data.dtype)
        print(string)

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        if isinstance(value, h5py.Dataset):
            self._data = value
        else:
            self._data = np.atleast_1d(np.asanyarray(value))

    def _load_dictionary(self, file_data_dict):
        """Load data from dictionary.

        Parameters
        ----------
        file_data_dict : dictionary
            A dictionary containing at least a 'data' keyword with an array of
            arbitrary dimensions. Additionally the dictionary can contain the
            following items:
            data : numpy array
               The signal data. It can be an array of any dimensions.
            axes : dictionary (optional)
                Dictionary to define the axes (see the
                documentation of the AxesManager class for more details).
            attributes : dictionary (optional)
                A dictionary whose items are stored as attributes.
            metadata : dictionary (optional)
                A dictionary containing a set of parameters
                that will to stores in the `metadata` attribute.
                Some parameters might be mandatory in some cases.
            original_metadata : dictionary (optional)
                A dictionary containing a set of parameters
                that will to stores in the `original_metadata` attribute. It
                typically contains all the parameters that has been
                imported from the original data file.

        """

        self.data = file_data_dict['data']
        if 'models' in file_data_dict:
            self.models._add_dictionary(file_data_dict['models'])
        if 'axes' not in file_data_dict:
            file_data_dict['axes'] = self._get_undefined_axes_list()
        self.axes_manager = AxesManager(
            file_data_dict['axes'])
        if 'metadata' not in file_data_dict:
            file_data_dict['metadata'] = {}
        if 'original_metadata' not in file_data_dict:
            file_data_dict['original_metadata'] = {}
        if 'attributes' in file_data_dict:
            for key, value in file_data_dict['attributes'].items():
                if hasattr(self, key):
                    if isinstance(value, dict):
                        for k, v in value.items():
                            eval('self.%s.__setattr__(k,v)' % key)
                    else:
                        self.__setattr__(key, value)
        self.original_metadata.add_dictionary(
            file_data_dict['original_metadata'])
        self.metadata.add_dictionary(
            file_data_dict['metadata'])
        if "title" not in self.metadata.General:
            self.metadata.General.title = ''
        if (self._record_by or
                "Signal.record_by" not in self.metadata):
            self.metadata.Signal.record_by = self._record_by
        if (self._signal_origin or
                "Signal.signal_origin" not in self.metadata):
            self.metadata.Signal.signal_origin = self._signal_origin
        if (self._signal_type or
                not self.metadata.has_item("Signal.signal_type")):
            self.metadata.Signal.signal_type = self._signal_type

    def squeeze(self):
        """Remove single-dimensional entries from the shape of an array
        and the axes.

        """
        # We deepcopy everything but data
        self = self._deepcopy_with_new_data(self.data)
        for axis in self.axes_manager._axes:
            if axis.size == 1:
                self._remove_axis(axis.index_in_axes_manager)
        self.data = self.data.squeeze()
        return self

    def _to_dictionary(self, add_learning_results=True):
        """Returns a dictionary that can be used to recreate the signal.

        All items but `data` are copies.

        Parameters
        ----------
        add_learning_results : bool

        Returns
        -------
        dic : dictionary

        """
        dic = {'data': self.data,
               'axes': self.axes_manager._get_axes_dicts(),
               'metadata': self.metadata.deepcopy().as_dictionary(),
               'original_metadata':
               self.original_metadata.deepcopy().as_dictionary(),
               'tmp_parameters':
               self.tmp_parameters.deepcopy().as_dictionary()}
        if add_learning_results and hasattr(self, 'learning_results'):
            dic['learning_results'] = copy.deepcopy(
                self.learning_results.__dict__)
        return dic

    def _get_undefined_axes_list(self):
        axes = []
        for i in range(len(self.data.shape)):
            axes.append({'size': int(self.data.shape[i]), })
        return axes

    def __call__(self, axes_manager=None):
        if axes_manager is None:
            axes_manager = self.axes_manager
        return np.atleast_1d(
            self.data.__getitem__(axes_manager._getitem_tuple))

    def plot(self, navigator="auto", axes_manager=None, **kwargs):
        """Plot the signal at the current coordinates.

        For multidimensional datasets an optional figure,
        the "navigator", with a cursor to navigate that data is
        raised. In any case it is possible to navigate the data using
        the sliders. Currently only signals with signal_dimension equal to
        0, 1 and 2 can be plotted.

        Parameters
        ----------
        navigator : {"auto", None, "slider", "spectrum", Signal}
            If "auto", if navigation_dimension > 0, a navigator is
            provided to explore the data.
            If navigation_dimension is 1 and the signal is an image
            the navigator is a spectrum obtained by integrating
            over the signal axes (the image).
            If navigation_dimension is 1 and the signal is a spectrum
            the navigator is an image obtained by stacking horizontally
            all the spectra in the dataset.
            If navigation_dimension is > 1, the navigator is an image
            obtained by integrating the data over the signal axes.
            Additionaly, if navigation_dimension > 2 a window
            with one slider per axis is raised to navigate the data.
            For example,
            if the dataset consists of 3 navigation axes X, Y, Z and one
            signal axis, E, the default navigator will be an image
            obtained by integrating the data over E at the current Z
            index and a window with sliders for the X, Y and Z axes
            will be raised. Notice that changing the Z-axis index
            changes the navigator in this case.
            If "slider" and the navigation dimension > 0 a window
            with one slider per axis is raised to navigate the data.
            If "spectrum" and navigation_dimension > 0 the navigator
            is always a spectrum obtained by integrating the data
            over all other axes.
            If None, no navigator will be provided.
            Alternatively a Signal instance can be provided. The signal
            dimension must be 1 (for a spectrum navigator) or 2 (for a
            image navigator) and navigation_shape must be 0 (for a static
            navigator) or navigation_shape + signal_shape must be equal
            to the navigator_shape of the current object (for a dynamic
            navigator).
            If the signal dtype is RGB or RGBA this parameters has no
            effect and is always "slider".

        axes_manager : {None, axes_manager}
            If None `axes_manager` is used.

        **kwargs : optional
            Any extra keyword arguments are passed to the signal plot.

        """

        if self._plot is not None:
            try:
                self._plot.close()
            except:
                # If it was already closed it will raise an exception,
                # but we want to carry on...
                pass

        if axes_manager is None:
            axes_manager = self.axes_manager
        if self.is_rgbx is True:
            if axes_manager.navigation_size < 2:
                navigator = None
            else:
                navigator = "slider"
        if axes_manager.signal_dimension == 0:
            self._plot = mpl_he.MPL_HyperExplorer()
        elif axes_manager.signal_dimension == 1:
            # Hyperspectrum
            self._plot = mpl_hse.MPL_HyperSpectrum_Explorer()
        elif axes_manager.signal_dimension == 2:
            self._plot = mpl_hie.MPL_HyperImage_Explorer()
        else:
            raise ValueError('Plotting is not supported for this view')

        self._plot.axes_manager = axes_manager
        self._plot.signal_data_function = self.__call__
        if self.metadata.General.title:
            self._plot.signal_title = self.metadata.General.title
        elif self.tmp_parameters.has_item('filename'):
            self._plot.signal_title = self.tmp_parameters.filename

        def get_static_explorer_wrapper(*args, **kwargs):
            return navigator()

        def get_1D_sum_explorer_wrapper(*args, **kwargs):
            navigator = self
            # Sum over all but the first navigation axis.
            am = navigator.axes_manager
            navigator = navigator.sum(am.signal_axes + am.navigation_axes[1:])
            return np.nan_to_num(navigator.data).squeeze()

        def get_dynamic_explorer_wrapper(*args, **kwargs):
            navigator.axes_manager.indices = self.axes_manager.indices[
                navigator.axes_manager.signal_dimension:]
            navigator.axes_manager._update_attributes()
            return navigator()

        if not isinstance(navigator, Signal) and navigator == "auto":
            if (self.axes_manager.navigation_dimension == 1 and
                    self.axes_manager.signal_dimension == 1):
                navigator = "data"
            elif self.axes_manager.navigation_dimension > 0:
                if self.axes_manager.signal_dimension == 0:
                    navigator = self.deepcopy()
                else:
                    navigator = interactive(
                        self.sum,
                        self.events.data_changed,
                        self.axes_manager.events.any_axis_changed,
                        self.axes_manager.signal_axes)
                if navigator.axes_manager.navigation_dimension == 1:
                    navigator = interactive(
                        navigator.as_spectrum,
                        navigator.events.data_changed,
                        navigator.axes_manager.events.any_axis_changed, 0)
                else:
                    navigator = interactive(
                        navigator.as_image,
                        navigator.events.data_changed,
                        navigator.axes_manager.events.any_axis_changed,
                        (0, 1))
            else:
                navigator = None
        # Navigator properties
        if axes_manager.navigation_axes:
            if navigator is "slider":
                self._plot.navigator_data_function = "slider"
            elif navigator is None:
                self._plot.navigator_data_function = None
            elif isinstance(navigator, Signal):
                # Dynamic navigator
                if (axes_manager.navigation_shape ==
                        navigator.axes_manager.signal_shape +
                        navigator.axes_manager.navigation_shape):
                    self._plot.navigator_data_function = \
                        get_dynamic_explorer_wrapper

                elif (axes_manager.navigation_shape ==
                        navigator.axes_manager.signal_shape or
                        axes_manager.navigation_shape[:2] ==
                        navigator.axes_manager.signal_shape or
                        (axes_manager.navigation_shape[0],) ==
                        navigator.axes_manager.signal_shape):
                    self._plot.navigator_data_function = \
                        get_static_explorer_wrapper
                else:
                    raise ValueError(
                        "The navigator dimensions are not compatible with "
                        "those of self.")
            elif navigator == "data":
                self._plot.navigator_data_function = \
                    lambda axes_manager=None: self.data
            elif navigator == "spectrum":
                self._plot.navigator_data_function = \
                    get_1D_sum_explorer_wrapper
            else:
                raise ValueError(
                    "navigator must be one of \"spectrum\",\"auto\","
                    " \"slider\", None, a Signal instance")

        self._plot.plot(**kwargs)
        self.events.data_changed.connect(self.update_plot, [])
        if self._plot.signal_plot:
            self._plot.signal_plot.events.closed.connect(
                lambda: self.events.data_changed.disconnect(self.update_plot),
                [])

    def save(self, filename=None, overwrite=None, extension=None,
             **kwds):
        """Saves the signal in the specified format.

        The function gets the format from the extension.:
            - hdf5 for HDF5
            - rpl for Ripple (useful to export to Digital Micrograph)
            - msa for EMSA/MSA single spectrum saving.
            - unf for SEMPER unf binary format.
            - blo for Blockfile diffraction stack saving.
            - Many image formats such as png, tiff, jpeg...

        If no extension is provided the default file format as defined
        in the `preferences` is used.
        Please note that not all the formats supports saving datasets of
        arbitrary dimensions, e.g. msa only supports 1D data, and blockfiles
        only support image stacks with a navigation dimension < 2.

        Each format accepts a different set of parameters. For details
        see the specific format documentation.

        Parameters
        ----------
        filename : str or None
            If None (default) and tmp_parameters.filename and
            `tmp_paramters.folder` are defined, the
            filename and path will be taken from there. A valid
            extension can be provided e.g. "my_file.rpl", see `extension`.
        overwrite : None, bool
            If None, if the file exists it will query the user. If
            True(False) it (does not) overwrites the file if it exists.
        extension : {None, 'hdf5', 'rpl', 'msa', 'unf', 'blo', common image
                     extensions e.g. 'tiff', 'png'}
            The extension of the file that defines the file format.
            If None, the extension is taken from the first not None in the
            following list:
            i) the filename
            ii)  `tmp_parameters.extension`
            iii) `preferences.General.default_file_format` in this order.

        """
        if filename is None:
            if (self.tmp_parameters.has_item('filename') and
                    self.tmp_parameters.has_item('folder')):
                filename = os.path.join(
                    self.tmp_parameters.folder,
                    self.tmp_parameters.filename)
                extension = (self.tmp_parameters.extension
                             if not extension
                             else extension)
            elif self.metadata.has_item('General.original_filename'):
                filename = self.metadata.General.original_filename
            else:
                raise ValueError('File name not defined')
        if extension is not None:
            basename, ext = os.path.splitext(filename)
            filename = basename + '.' + extension
        io.save(filename, self, overwrite=overwrite, **kwds)

    def _replot(self):
        if self._plot is not None:
            if self._plot.is_active() is True:
                self.plot()

    def update_plot(self):
        if self._plot is not None:
            if self._plot.is_active() is True:
                if self._plot.signal_plot is not None:
                    self._plot.signal_plot.update()
                if self._plot.navigator_plot is not None:
                    self._plot.navigator_plot.update()

    @auto_replot
    def get_dimensions_from_data(self):
        """Get the dimension parameters from the data_cube. Useful when
        the data_cube was externally modified, or when the SI was not
        loaded from a file

        """
        dc = self.data
        for axis in self.axes_manager._axes:
            axis.size = int(dc.shape[axis.index_in_array])

    def crop(self, axis, start=None, end=None):
        """Crops the data in a given axis. The range is given in pixels

        Parameters
        ----------
        axis : {int | string}
            Specify the data axis in which to perform the cropping
            operation. The axis can be specified using the index of the
            axis in `axes_manager` or the axis name.
        start, end : {int | float | None}
            The beginning and end of the cropping interval. If int
            the value is taken as the axis index. If float the index
            is calculated using the axis calibration. If start/end is
            None crop from/to the low/high end of the axis.

        """
        axis = self.axes_manager[axis]
        i1, i2 = axis._get_index(start), axis._get_index(end)
        if i1 is not None:
            new_offset = axis.axis[i1]
        # We take a copy to guarantee the continuity of the data
        self.data = self.data[
            (slice(None),) * axis.index_in_array + (slice(i1, i2),
                                                    Ellipsis)]

        if i1 is not None:
            axis.offset = new_offset
        self.events.data_changed.trigger(obj=self)
        self.get_dimensions_from_data()
        self.squeeze()

    def swap_axes(self, axis1, axis2):
        """Swaps the axes.

        Parameters
        ----------
        axis1, axis2 %s

        Returns
        -------
        s : a copy of the object with the axes swapped.

        """
        axis1 = self.axes_manager[axis1].index_in_array
        axis2 = self.axes_manager[axis2].index_in_array
        s = self._deepcopy_with_new_data(self.data.swapaxes(axis1, axis2))
        c1 = s.axes_manager._axes[axis1]
        c2 = s.axes_manager._axes[axis2]
        s.axes_manager._axes[axis1] = c2
        s.axes_manager._axes[axis2] = c1
        s.axes_manager._update_attributes()
        s._make_sure_data_is_contiguous()
        return s
    swap_axes.__doc__ %= ONE_AXIS_PARAMETER

    def rollaxis(self, axis, to_axis):
        """Roll the specified axis backwards, until it lies in a given position.

        Parameters
        ----------
        axis %s The axis to roll backwards.
            The positions of the other axes do not change relative to one another.
        to_axis %s The axis is rolled until it
            lies before this other axis.

        Returns
        -------
        s : Signal or subclass
            Output signal.

        See Also
        --------
        roll : swap_axes

        Examples
        --------
        >>> s = hs.signals.Spectrum(np.ones((5,4,3,6)))
        >>> s
        <Spectrum, title: , dimensions: (3, 4, 5, 6)>
        >>> s.rollaxis(3, 1)
        <Spectrum, title: , dimensions: (3, 4, 5, 6)>
        >>> s.rollaxis(2,0)
        <Spectrum, title: , dimensions: (5, 3, 4, 6)>

        """
        axis = self.axes_manager[axis].index_in_array
        to_index = self.axes_manager[to_axis].index_in_array
        if axis == to_index:
            return self.deepcopy()
        new_axes_indices = hyperspy.misc.utils.rollelem(
            [axis_.index_in_array for axis_ in self.axes_manager._axes],
            index=axis,
            to_index=to_index)

        s = self._deepcopy_with_new_data(self.data.transpose(new_axes_indices))
        s.axes_manager._axes = hyperspy.misc.utils.rollelem(
            s.axes_manager._axes,
            index=axis,
            to_index=to_index)
        s.axes_manager._update_attributes()
        s._make_sure_data_is_contiguous()
        return s
    rollaxis.__doc__ %= (ONE_AXIS_PARAMETER, ONE_AXIS_PARAMETER)

    @property
    def _data_aligned_with_axes(self):
        """Returns a view of `data` with is axes aligned with the Signal axes.

        """
        if self.axes_manager.axes_are_aligned_with_data:
            return self.data
        else:
            am = self.axes_manager
            nav_iia_r = am.navigation_indices_in_array[::-1]
            sig_iia_r = am.signal_indices_in_array[::-1]
            # nav_sort = np.argsort(nav_iia_r)
            # sig_sort = np.argsort(sig_iia_r) + len(nav_sort)
            data = self.data.transpose(nav_iia_r + sig_iia_r)
            return data

    def rebin(self, new_shape, out=None):
        """Returns the object with the data rebinned.

        Parameters
        ----------
        new_shape: tuple of ints
            The new shape elements must be divisors of the original shape
            elements.
        %s

        Returns
        -------
        s : Signal subclass

        Raises
        ------
        ValueError
            When there is a mismatch between the number of elements in the
            signal shape and `new_shape` or `new_shape` elements are not
            divisors of the original signal shape.


        Examples
        --------
        >>> import hyperspy.api as hs
        >>> s = hs.signals.Spectrum(np.zeros((10, 100)))
        >>> s
        <Spectrum, title: , dimensions: (10|100)>
        >>> s.rebin((5, 100))
        <Spectrum, title: , dimensions: (5|100)>
        I
        """
        if len(new_shape) != len(self.data.shape):
            raise ValueError("Wrong shape size")
        new_shape_in_array = []
        for axis in self.axes_manager._axes:
            new_shape_in_array.append(
                new_shape[axis.index_in_axes_manager])
        factors = (np.array(self.data.shape) /
                   np.array(new_shape_in_array))
        s = out or self._deepcopy_with_new_data(None)
        data = array_tools.rebin(self.data, new_shape_in_array)
        if out:
            out.data[:] = data
        else:
            s.data = data
        for axis, axis_src in zip(s.axes_manager._axes,
                                  self.axes_manager._axes):
            axis.scale = axis_src.scale * factors[axis.index_in_array]
        s.get_dimensions_from_data()
        if s.metadata.has_item('Signal.Noise_properties.variance'):
            if isinstance(s.metadata.Signal.Noise_properties.variance, Signal):
                var = s.metadata.Signal.Noise_properties.variance
                s.metadata.Signal.Noise_properties.variance = var.rebin(
                    new_shape)
        if out is None:
            return s
        else:
            out.events.data_changed.trigger(obj=out)
    rebin.__doc__ %= OUT_ARG

    def split(self,
              axis='auto',
              number_of_parts='auto',
              step_sizes='auto'):
        """Splits the data into several signals.

        The split can be defined by giving the number_of_parts, a homogeneous
        step size or a list of customized step sizes. By default ('auto'),
        the function is the reverse of utils.stack().

        Parameters
        ----------
        axis : {'auto' | int | string}
            Specify the data axis in which to perform the splitting
            operation.  The axis can be specified using the index of the
            axis in `axes_manager` or the axis name.
            - If 'auto' and if the object has been created with utils.stack,
            split will return the former list of signals
            (options stored in 'metadata._HyperSpy.Stacking_history'
             else the last navigation axis will be used.
        number_of_parts : {'auto' | int}
            Number of parts in which the SI will be splitted. The
            splitting is homegenous. When the axis size is not divisible
            by the number_of_parts the reminder data is lost without
            warning. If number_of_parts and step_sizes is 'auto',
            number_of_parts equals the length of the axis,
            step_sizes equals one  and the axis is supress from each
            sub_spectra.
        step_sizes : {'auto' | list of ints | int}
            Size of the splitted parts. If 'auto', the step_sizes equals one.
            If int, the splitting is homogenous.

        Examples
        --------
        >>> s = hs.signals.Spectrum(random.random([4,3,2]))
        >>> s
            <Spectrum, title: , dimensions: (3, 4|2)>
        >>> s.split()
            [<Spectrum, title: , dimensions: (3 |2)>,
            <Spectrum, title: , dimensions: (3 |2)>,
            <Spectrum, title: , dimensions: (3 |2)>,
            <Spectrum, title: , dimensions: (3 |2)>]
        >>> s.split(step_sizes=2)
            [<Spectrum, title: , dimensions: (3, 2|2)>,
            <Spectrum, title: , dimensions: (3, 2|2)>]
        >>> s.split(step_sizes=[1,2])
            [<Spectrum, title: , dimensions: (3, 1|2)>,
            <Spectrum, title: , dimensions: (3, 2|2)>]

        Returns
        -------
        list of the splitted signals
        """

        shape = self.data.shape
        signal_dict = self._to_dictionary(add_learning_results=False)

        if axis == 'auto':
            mode = 'auto'
            if hasattr(self.metadata._HyperSpy, 'Stacking_history'):
                stack_history = self.metadata._HyperSpy.Stacking_history
                axis_in_manager = stack_history.axis
                step_sizes = stack_history.step_sizes
            else:
                axis_in_manager = \
                    self.axes_manager[-1 + 1j].index_in_axes_manager
        else:
            mode = 'manual'
            axis_in_manager = self.axes_manager[axis].index_in_axes_manager

        axis = self.axes_manager[axis_in_manager].index_in_array
        len_axis = self.axes_manager[axis_in_manager].size

        if number_of_parts is 'auto' and step_sizes is 'auto':
            step_sizes = 1
            number_of_parts = len_axis
        elif number_of_parts is not 'auto' and step_sizes is not 'auto':
            raise ValueError(
                "You can define step_sizes or number_of_parts "
                "but not both.")
        elif step_sizes is 'auto':
            if number_of_parts > shape[axis]:
                raise ValueError(
                    "The number of parts is greater than "
                    "the axis size.")
            else:
                step_sizes = ([shape[axis] // number_of_parts, ] *
                              number_of_parts)

        if isinstance(step_sizes, int):
            step_sizes = [step_sizes] * int(len_axis / step_sizes)

        splitted = []
        cut_index = np.array([0] + step_sizes).cumsum()

        axes_dict = signal_dict['axes']
        for i in range(len(cut_index) - 1):
            axes_dict[axis]['offset'] = \
                self.axes_manager._axes[axis].index2value(cut_index[i])
            axes_dict[axis]['size'] = cut_index[i + 1] - cut_index[i]
            data = self.data[
                (slice(None), ) * axis +
                (slice(cut_index[i], cut_index[i + 1]), Ellipsis)]
            signal_dict['data'] = data
            splitted += self.__class__(**signal_dict),

        if number_of_parts == len_axis \
                or step_sizes == [1] * len_axis:
            for i, spectrum in enumerate(splitted):
                spectrum.data = spectrum.data[
                    spectrum.axes_manager._get_data_slice([(axis, 0)])]
                spectrum._remove_axis(axis_in_manager)

        if mode == 'auto' and hasattr(
                self.original_metadata, 'stack_elements'):
            for i, spectrum in enumerate(splitted):
                se = self.original_metadata.stack_elements['element' + str(i)]
                spectrum.metadata = copy.deepcopy(
                    se['metadata'])
                spectrum.original_metadata = copy.deepcopy(
                    se['original_metadata'])
                spectrum.metadata.General.title = se.metadata.General.title

        return splitted

    @auto_replot
    def _unfold(self, steady_axes, unfolded_axis):
        """Modify the shape of the data by specifying the axes whose
        dimension do not change and the axis over which the remaining axes will
        be unfolded

        Parameters
        ----------
        steady_axes : list
            The indices of the axes which dimensions do not change
        unfolded_axis : int
            The index of the axis over which all the rest of the axes (except
            the steady axes) will be unfolded

        See also
        --------
        fold
        """

        # It doesn't make sense unfolding when dim < 2
        if self.data.squeeze().ndim < 2:
            return

        # We need to store the original shape and coordinates to be used
        # by
        # the fold function only if it has not been already stored by a
        # previous unfold
        folding = self.metadata._HyperSpy.Folding
        if folding.unfolded is False:
            folding.original_shape = self.data.shape
            folding.original_axes_manager = self.axes_manager
            folding.unfolded = True

        new_shape = [1] * len(self.data.shape)
        for index in steady_axes:
            new_shape[index] = self.data.shape[index]
        new_shape[unfolded_axis] = -1
        self.data = self.data.reshape(new_shape)
        self.axes_manager = self.axes_manager.deepcopy()
        uname = ''
        uunits = ''
        to_remove = []
        for axis, dim in zip(self.axes_manager._axes, new_shape):
            if dim == 1:
                uname += ',' + str(axis)
                uunits = ',' + str(axis.units)
                to_remove.append(axis)
        ua = self.axes_manager._axes[unfolded_axis]
        ua.name = str(ua) + uname
        ua.units = str(ua.units) + uunits
        ua.size = self.data.shape[unfolded_axis]
        for axis in to_remove:
            self.axes_manager.remove(axis.index_in_axes_manager)
        self.data = self.data.squeeze()
        if self.metadata.has_item('Signal.Noise_properties.variance'):
            variance = self.metadata.Signal.Noise_properties.variance
            if isinstance(variance, Signal):
                variance._unfold(steady_axes, unfolded_axis)

    def unfold(self, unfold_navigation=True, unfold_signal=True):
        """Modifies the shape of the data by unfolding the signal and
        navigation dimensions separately

        Returns
        -------
        needed_unfolding : bool


        """
        unfolded = False
        if unfold_navigation:
            if self.unfold_navigation_space():
                unfolded = True
        if unfold_signal:
            if self.unfold_signal_space():
                unfolded = True
        return unfolded

    @contextmanager
    def unfolded(self, unfold_navigation=True, unfold_signal=True):
        """Use this function together with a `with` statement to have the
        signal be unfolded for the scope of the `with` block, before
        automatically refolding when passing out of scope.

        See also
        --------
        unfold, fold

        Examples
        --------
        >>> import numpy as np
        >>> s = Signal(np.random.random((64,64,1024)))
        >>> with s.unfolded():
                # Do whatever needs doing while unfolded here
                pass
        """
        unfolded = self.unfold(unfold_navigation, unfold_signal)
        try:
            yield unfolded
        finally:
            if unfolded is not False:
                self.fold()

    def unfold_navigation_space(self):
        """Modify the shape of the data to obtain a navigation space of
        dimension 1

        Returns
        -------
        needed_unfolding : bool

        """

        if self.axes_manager.navigation_dimension < 2:
            needed_unfolding = False
        else:
            needed_unfolding = True
            steady_axes = [
                axis.index_in_array for axis in
                self.axes_manager.signal_axes]
            unfolded_axis = (
                self.axes_manager.navigation_axes[0].index_in_array)
            self._unfold(steady_axes, unfolded_axis)
        return needed_unfolding

    def unfold_signal_space(self):
        """Modify the shape of the data to obtain a signal space of
        dimension 1

        Returns
        -------
        needed_unfolding : bool

        """
        if self.axes_manager.signal_dimension < 2:
            needed_unfolding = False
        else:
            needed_unfolding = True
            steady_axes = [
                axis.index_in_array for axis in
                self.axes_manager.navigation_axes]
            unfolded_axis = self.axes_manager.signal_axes[0].index_in_array
            self._unfold(steady_axes, unfolded_axis)
            self.metadata._HyperSpy.Folding.signal_unfolded = True
        return needed_unfolding

    @auto_replot
    def fold(self):
        """If the signal was previously unfolded, folds it back"""
        folding = self.metadata._HyperSpy.Folding
        # Note that == must be used instead of is True because
        # if the value was loaded from a file its type can be np.bool_
        if folding.unfolded is True:
            self.data = self.data.reshape(folding.original_shape)
            self.axes_manager = folding.original_axes_manager
            folding.original_shape = None
            folding.original_axes_manager = None
            folding.unfolded = False
            folding.signal_unfolded = False
            if self.metadata.has_item('Signal.Noise_properties.variance'):
                variance = self.metadata.Signal.Noise_properties.variance
                if isinstance(variance, Signal):
                    variance.fold()

    def _make_sure_data_is_contiguous(self):
        if self.data.flags['C_CONTIGUOUS'] is False:
            self.data = np.ascontiguousarray(self.data)

    def _iterate_signal(self):
        """Iterates over the signal data.

        It is faster than using the signal iterator.

        """
        if self.axes_manager.navigation_size < 2:
            yield self()
            return
        self._make_sure_data_is_contiguous()
        axes = [axis.index_in_array for
                axis in self.axes_manager.signal_axes]
        unfolded_axis = (
            self.axes_manager.navigation_axes[0].index_in_array)
        new_shape = [1] * len(self.data.shape)
        for axis in axes:
            new_shape[axis] = self.data.shape[axis]
        new_shape[unfolded_axis] = -1
        # Warning! if the data is not contigous it will make a copy!!
        data = self.data.reshape(new_shape)
        for i in range(data.shape[unfolded_axis]):
            getitem = [0] * len(data.shape)
            for axis in axes:
                getitem[axis] = slice(None)
            getitem[unfolded_axis] = i
            yield(data[getitem])

    def _remove_axis(self, axes):
        am = self.axes_manager
        axes = am[axes]
        if not np.iterable(axes):
            axes = (axes,)
        if am.navigation_dimension + am.signal_dimension > len(axes):
            old_signal_dimension = am.signal_dimension
            am.remove(axes)
            if old_signal_dimension != am.signal_dimension:
                if am.signal_dimension == 2:
                    self._record_by = "image"
                elif am.signal_dimension == 1:
                    self._record_by = "spectrum"
                elif am.signal_dimension == 0:
                    self._record_by = ""
                else:
                    return
                self.metadata.Signal.record_by = self._record_by
                self._assign_subclass()
        else:
            # Create a "Scalar" axis because the axis is the last one left and
            # HyperSpy does not # support 0 dimensions
            am.remove(axes)
            am._append_axis(
                size=1,
                scale=1,
                offset=0,
                name="Scalar",
                navigate=False,)

    def _ma_workaround(self, s, function, axes, ar_axes, out):
        # TODO: Remove if and when numpy.ma accepts tuple `axis`

        # Basically perform unfolding, but only on data. We don't care about
        # the axes since the function will consume it/them.
        if not np.iterable(ar_axes):
            ar_axes = (ar_axes,)
        ar_axes = sorted(ar_axes)
        new_shape = list(self.data.shape)
        for index in ar_axes[1:]:
            new_shape[index] = 1
        new_shape[ar_axes[0]] = -1
        data = self.data.reshape(new_shape).squeeze()

        if out:
            data = np.atleast_1d(function(data, axis=ar_axes[0],))
            if data.shape == out.data.shape:
                out.data[:] = data
                out.events.data_changed.trigger(obj=out)
            else:
                raise ValueError(
                    "The output shape %s does not match  the shape of "
                    "`out` %s" % (data.shape, out.data.shape))
        else:
            s.data = function(data, axis=ar_axes[0])
            s._remove_axis([ax.index_in_axes_manager for ax in axes])
            return s

    def _apply_function_on_data_and_remove_axis(self, function, axes,
                                                out=None):
        axes = self.axes_manager[axes]
        if not np.iterable(axes):
            axes = (axes,)
        # Use out argument in numpy function when available for operations that
        # do not return scalars in numpy.
        np_out = not len(self.axes_manager._axes) == len(axes)
        ar_axes = tuple(ax.index_in_array for ax in axes)
        if len(ar_axes) == 1:
            ar_axes = ar_axes[0]

        s = out or self._deepcopy_with_new_data(None)

        if isinstance(ar_axes, tuple) and np.ma.is_masked(self.data):
            return self._ma_workaround(s=s, function=function, axes=axes,
                                       ar_axes=ar_axes, out=out)
        if out:
            if np_out and function is not np.argmax:
                function(self.data, axis=ar_axes, out=out.data,)
            else:
                data = np.atleast_1d(function(self.data, axis=ar_axes))
                if data.shape == out.data.shape:
                    out.data[:] = data
                else:
                    raise ValueError(
                        "The output shape %s does not match  the shape of "
                        "`out` %s" % (data.shape, out.data.shape))
            out.events.data_changed.trigger(obj=out)
        else:
            s.data = np.atleast_1d(
                function(self.data, axis=ar_axes,))
            s._remove_axis([ax.index_in_axes_manager for ax in axes])
            return s

    def sum(self, axis=None, out=None):
        """Sum the data over the given axes.

        Parameters
        ----------
        axis %s
        %s

        Returns
        -------
        s : Signal

        See also
        --------
        max, min, mean, std, var, indexmax, valuemax, amax

        Examples
        --------
        >>> import numpy as np
        >>> s = Signal(np.random.random((64,64,1024)))
        >>> s.data.shape
        (64,64,1024)
        >>> s.sum(-1).data.shape
        (64,64)

        """
        if axis is None:
            axis = self.axes_manager.navigation_axes
        return self._apply_function_on_data_and_remove_axis(np.sum, axis,
                                                            out=out)
    sum.__doc__ %= (MANY_AXIS_PARAMETER, OUT_ARG)

    def max(self, axis=None, out=None):
        """Returns a signal with the maximum of the signal along at least one
        axis.

        Parameters
        ----------
        axis %s
        %s

        Returns
        -------
        s : Signal

        See also
        --------
        min, sum, mean, std, var, indexmax, valuemax, amax

        Examples
        --------
        >>> import numpy as np
        >>> s = Signal(np.random.random((64,64,1024)))
        >>> s.data.shape
        (64,64,1024)
        >>> s.max(-1).data.shape
        (64,64)

        """
        if axis is None:
            axis = self.axes_manager.navigation_axes
        return self._apply_function_on_data_and_remove_axis(np.max, axis,
                                                            out=out)
    max.__doc__ %= (MANY_AXIS_PARAMETER, OUT_ARG)

    def min(self, axis=None, out=None):
        """Returns a signal with the minimum of the signal along at least one
        axis.

        Parameters
        ----------
        axis %s
        %s

        Returns
        -------
        s : Signal

        See also
        --------
        max, sum, mean, std, var, indexmax, valuemax, amax

        Examples
        --------
        >>> import numpy as np
        >>> s = Signal(np.random.random((64,64,1024)))
        >>> s.data.shape
        (64,64,1024)
        >>> s.min(-1).data.shape
        (64,64)

        """
        if axis is None:
            axis = self.axes_manager.navigation_axes
        return self._apply_function_on_data_and_remove_axis(np.min, axis,
                                                            out=out)
    min.__doc__ %= (MANY_AXIS_PARAMETER, OUT_ARG)

    def mean(self, axis=None, out=None):
        """Returns a signal with the average of the signal along at least one
        axis.

        Parameters
        ----------
        axis %s
        %s

        Returns
        -------
        s : Signal

        See also
        --------
        max, min, sum, std, var, indexmax, valuemax, amax

        Examples
        --------
        >>> import numpy as np
        >>> s = Signal(np.random.random((64,64,1024)))
        >>> s.data.shape
        (64,64,1024)
        >>> s.mean(-1).data.shape
        (64,64)

        """
        if axis is None:
            axis = self.axes_manager.navigation_axes
        return self._apply_function_on_data_and_remove_axis(np.mean, axis,
                                                            out=out)
    mean.__doc__ %= (MANY_AXIS_PARAMETER, OUT_ARG)

    def std(self, axis=None, out=None):
        """Returns a signal with the standard deviation of the signal along
        at least one axis.

        Parameters
        ----------
        axis %s
        %s

        Returns
        -------
        s : Signal

        See also
        --------
        max, min, sum, mean, var, indexmax, valuemax, amax

        Examples
        --------
        >>> import numpy as np
        >>> s = Signal(np.random.random((64,64,1024)))
        >>> s.data.shape
        (64,64,1024)
        >>> s.std(-1).data.shape
        (64,64)

        """
        if axis is None:
            axis = self.axes_manager.navigation_axes
        return self._apply_function_on_data_and_remove_axis(np.std, axis,
                                                            out=out)
    std.__doc__ %= (MANY_AXIS_PARAMETER, OUT_ARG)

    def var(self, axis=None, out=None):
        """Returns a signal with the variances of the signal along at least one
        axis.

        Parameters
        ----------
        axis %s
        %s

        Returns
        -------
        s : Signal

        See also
        --------
        max, min, sum, mean, std, indexmax, valuemax, amax

        Examples
        --------
        >>> import numpy as np
        >>> s = Signal(np.random.random((64,64,1024)))
        >>> s.data.shape
        (64,64,1024)
        >>> s.var(-1).data.shape
        (64,64)

        """
        if axis is None:
            axis = self.axes_manager.navigation_axes
        return self._apply_function_on_data_and_remove_axis(np.var, axis,
                                                            out=out)
    var.__doc__ %= (MANY_AXIS_PARAMETER, OUT_ARG)

    def diff(self, axis, order=1, out=None):
        """Returns a signal with the n-th order discrete difference along
        given axis.

        Parameters
        ----------
        axis %s
        order : int
            the order of the derivative
        %s

        See also
        --------
        max, min, sum, mean, std, var, indexmax, valuemax, amax

        Examples
        --------
        >>> import numpy as np
        >>> s = Signal(np.random.random((64,64,1024)))
        >>> s.data.shape
        (64,64,1024)
        >>> s.diff(-1).data.shape
        (64,64,1023)
        """
        s = out or self._deepcopy_with_new_data(None)
        data = np.diff(self.data, n=order,
                       axis=self.axes_manager[axis].index_in_array)
        if out is not None:
            out.data[:] = data
        else:
            s.data = data
        axis2 = s.axes_manager[axis]
        new_offset = self.axes_manager[axis].offset + (order * axis2.scale / 2)
        axis2.offset = new_offset
        s.get_dimensions_from_data()
        if out is None:
            return s
        else:
            out.events.data_changed.trigger(obj=out)
    diff.__doc__ %= (ONE_AXIS_PARAMETER, OUT_ARG)

    def derivative(self, axis, order=1, out=None):
        """Numerical derivative along the given axis.

        Currently only the first order finite difference method is implemented.

        Parameters
        ----------
        axis %s
        order: int
            The order of the derivative. (Note that this is the order of the
            derivative i.e. `order=2` does not use second order finite
            differences method.)
        %s

        Returns
        -------
        der : Signal
            Note that the size of the data on the given `axis` decreases by the
            given `order` i.e. if `axis` is "x" and `order` is 2 the
            x dimension is N, der's x dimension is N - 2.

        See also
        --------
        diff

        """

        der = self.diff(order=order, axis=axis, out=out)
        der = out or der
        axis = self.axes_manager[axis]
        der.data /= axis.scale ** order
        if out is None:
            return der
        else:
            out.events.data_changed.trigger(obj=out)
    derivative.__doc__ %= (ONE_AXIS_PARAMETER, OUT_ARG)

    def integrate_simpson(self, axis, out=None):
        """Returns a signal with the result of calculating the integral
        of the signal along an axis using Simpson's rule.

        Parameters
        ----------
        axis %s
        %s

        Returns
        -------
        s : Signal

        See also
        --------
        max, min, sum, mean, std, var, indexmax, valuemax, amax

        Examples
        --------
        >>> import numpy as np
        >>> s = Signal(np.random.random((64,64,1024)))
        >>> s.data.shape
        (64,64,1024)
        >>> s.var(-1).data.shape
        (64,64)

        """
        axis = self.axes_manager[axis]
        s = out or self._deepcopy_with_new_data(None)
        data = sp.integrate.simps(y=self.data, x=axis.axis,
                                  axis=axis.index_in_array)
        if out is not None:
            out.data[:] = data
            out.events.data_changed.trigger(obj=out)
        else:
            s.data = data
            s._remove_axis(axis.index_in_axes_manager)
            return s
    integrate_simpson.__doc__ %= (ONE_AXIS_PARAMETER, OUT_ARG)

    def integrate1D(self, axis, out=None):
        """Integrate the signal over the given axis.

        The integration is performed using Simpson's rule if
        `metadata.Signal.binned` is False and summation over the given axis if
        True.

        Parameters
        ----------
        axis %s
        %s

        Returns
        -------
        s : Signal

        See also
        --------
        integrate_simpson, diff, derivative

        Examples
        --------
        >>> import numpy as np
        >>> s = Signal(np.random.random((64,64,1024)))
        >>> s.data.shape
        (64,64,1024)
        >>> s.var(-1).data.shape
        (64,64)

        """
        if self.metadata.Signal.binned is False:
            return self.integrate_simpson(axis=axis, out=out)
        else:
            return self.sum(axis=axis, out=out)
    integrate1D.__doc__ %= (ONE_AXIS_PARAMETER, OUT_ARG)

    def indexmax(self, axis, out=None):
        """Returns a signal with the index of the maximum along an axis.

        Parameters
        ----------
        axis %s
        %s

        Returns
        -------
        s : Signal
            The data dtype is always int.

        See also
        --------
        max, min, sum, mean, std, var, valuemax, amax

        Usage
        -----
        >>> import numpy as np
        >>> s = Signal(np.random.random((64,64,1024)))
        >>> s.data.shape
        (64,64,1024)
        >>> s.indexmax(-1).data.shape
        (64,64)

        """
        return self._apply_function_on_data_and_remove_axis(np.argmax, axis,
                                                            out=out)
    indexmax.__doc__ %= (ONE_AXIS_PARAMETER, OUT_ARG)

    def valuemax(self, axis, out=None):
        """Returns a signal with the value of coordinates of the maximum along an axis.

        Parameters
        ----------
        axis %s
        %s

        Returns
        -------
        s : Signal

        See also
        --------
        max, min, sum, mean, std, var, indexmax, amax

        Usage
        -----
        >>> import numpy as np
        >>> s = Signal(np.random.random((64,64,1024)))
        >>> s.data.shape
        (64,64,1024)
        >>> s.valuemax(-1).data.shape
        (64,64)

        """
        idx = self.indexmax(axis)
        data = self.axes_manager[axis].index2value(idx.data)
        if out is None:
            idx.data = data
            return idx
        else:
            out.data[:] = data
            out.events.data_changed.trigger(obj=out)
    valuemax.__doc__ %= (ONE_AXIS_PARAMETER, OUT_ARG)

    def get_histogram(self, bins='freedman', range_bins=None, out=None,
                      **kwargs):
        """Return a histogram of the signal data.

        More sophisticated algorithms for determining bins can be used.
        Aside from the `bins` argument allowing a string specified how bins
        are computed, the parameters are the same as numpy.histogram().

        Parameters
        ----------
        bins : int or list or str, optional
            If bins is a string, then it must be one of:
            'knuth' : use Knuth's rule to determine bins
            'scotts' : use Scott's rule to determine bins
            'freedman' : use the Freedman-diaconis rule to determine bins
            'blocks' : use bayesian blocks for dynamic bin widths
        range_bins : tuple or None, optional
            the minimum and maximum range for the histogram. If not specified,
            it will be (x.min(), x.max())
        %s
        **kwargs
            other keyword arguments (weight and density) are described in
            np.histogram().

        Returns
        -------
        hist_spec : An 1D spectrum instance containing the histogram.

        See Also
        --------
        print_summary_statistics
        astroML.density_estimation.histogram, numpy.histogram : these are the
            functions that hyperspy uses to compute the histogram.

        Notes
        -----
        The number of bins estimators are taken from AstroML. Read
        their documentation for more info.

        Examples
        --------
        >>> s = hs.signals.Spectrum(np.random.normal(size=(10, 100)))
        Plot the data histogram
        >>> s.get_histogram().plot()
        Plot the histogram of the signal at the current coordinates
        >>> s.get_current_signal().get_histogram().plot()

        """
        from hyperspy import signals
        data = self.data[~np.isnan(self.data)].flatten()
        hist, bin_edges = histogram(data,
                                    bins=bins,
                                    range=range_bins,
                                    **kwargs)
        if out is None:
            hist_spec = signals.Spectrum(hist)
        else:
            hist_spec = out
            if hist_spec.data.shape == hist.shape:
                hist_spec.data[:] = hist
            else:
                hist_spec.data = hist
        if bins == 'blocks':
            hist_spec.axes_manager.signal_axes[0].axis = bin_edges[:-1]
            warnings.warn(
                "The options `bins = 'blocks'` is not fully supported in this "
                "versions of hyperspy. It should be used for plotting purpose"
                "only.")
        else:
            hist_spec.axes_manager[0].scale = bin_edges[1] - bin_edges[0]
            hist_spec.axes_manager[0].offset = bin_edges[0]
            hist_spec.axes_manager[0].size = hist.shape[-1]
        hist_spec.axes_manager[0].name = 'value'
        hist_spec.metadata.General.title = (self.metadata.General.title +
                                            " histogram")
        hist_spec.metadata.Signal.binned = True
        if out is None:
            return hist_spec
        else:
            out.events.data_changed.trigger(obj=out)
    get_histogram.__doc__ %= OUT_ARG

    def map(self, function,
            show_progressbar=None, **kwargs):
        """Apply a function to the signal data at all the coordinates.

        The function must operate on numpy arrays and the output *must have the
        same dimensions as the input*. The function is applied to the data at
        each coordinate and the result is stored in the current signal i.e.
        this method operates *in-place*.  Any extra keyword argument is passed
        to the function. The keywords can take different values at different
        coordinates. If the function takes an `axis` or `axes` argument, the
        function is assumed to be vectorial and the signal axes are assigned to
        `axis` or `axes`.  Otherwise, the signal is iterated over the
        navigation axes and a progress bar is displayed to monitor the
        progress.

        Parameters
        ----------

        function : function
            A function that can be applied to the signal.
        show_progressbar : None or bool
            If True, display a progress bar. If None the default is set in
            `preferences`.
        keyword arguments : any valid keyword argument
            All extra keyword arguments are passed to the

        Notes
        -----
        This method is similar to Python's :func:`map` that can also be utilize
        with a :class:`Signal` instance for similar purposes. However, this
        method has the advantage of being faster because it iterates the numpy
        array instead of the :class:`Signal`.

        Examples
        --------
        Apply a gaussian filter to all the images in the dataset. The sigma
        parameter is constant.

        >>> import scipy.ndimage
        >>> im = hs.signals.Image(np.random.random((10, 64, 64)))
        >>> im.map(scipy.ndimage.gaussian_filter, sigma=2.5)

        Apply a gaussian filter to all the images in the dataset. The sigmal
        parameter is variable.

        >>> im = hs.signals.Image(np.random.random((10, 64, 64)))
        >>> sigmas = hs.signals.Signal(np.linspace(2,5,10))
        >>> sigmas.axes_manager.set_signal_dimension(0)
        >>> im.map(scipy.ndimage.gaussian_filter, sigma=sigmas)

        """
        if show_progressbar is None:
            show_progressbar = preferences.General.show_progressbar
        # Sepate ndkwargs
        ndkwargs = ()
        for key, value in kwargs.items():
            if isinstance(value, Signal):
                ndkwargs += ((key, value),)

        # Check if the signal axes have inhomogenous scales and/or units and
        # display in warning if yes.
        scale = set()
        units = set()
        for i in range(len(self.axes_manager.signal_axes)):
            scale.add(self.axes_manager[i].scale)
            units.add(self.axes_manager[i].units)
        if len(units) != 1 or len(scale) != 1:
            warnings.warn(
                "The function you applied does not take into "
                "account the difference of units and of scales in-between"
                " axes.")
        # If the function has an axis argument and the signal dimension is 1,
        # we suppose that it can operate on the full array and we don't
        # interate over the coordinates.
        try:
            fargs = inspect.getargspec(function).args
        except TypeError:
            # This is probably a Cython function that is not supported by
            # inspect.
            fargs = []

        if not ndkwargs and (self.axes_manager.signal_dimension == 1 and
                             "axis" in fargs):
            kwargs['axis'] = \
                self.axes_manager.signal_axes[-1].index_in_array

            self.data = function(self.data, **kwargs)
        # If the function has an axes argument
        # we suppose that it can operate on the full array and we don't
        # interate over the coordinates.
        elif not ndkwargs and "axes" in fargs:
            kwargs['axes'] = tuple([axis.index_in_array for axis in
                                    self.axes_manager.signal_axes])
            self.data = function(self.data, **kwargs)
        else:
            # Iteration over coordinates.
            pbar = progressbar(
                maxval=self.axes_manager.navigation_size,
                disabled=not show_progressbar)
            iterators = [signal[1]._iterate_signal() for signal in ndkwargs]
            iterators = tuple([self._iterate_signal()] + iterators)
            for data in zip(*iterators):
                for (key, value), datum in zip(ndkwargs, data[1:]):
                    kwargs[key] = datum[0]
                data[0][:] = function(data[0], **kwargs)
                next(pbar)
            pbar.finish()
        self.events.data_changed.trigger(obj=self)

    def copy(self):
        try:
            backup_plot = self._plot
            self._plot = None
            return copy.copy(self)
        finally:
            self._plot = backup_plot

    def __deepcopy__(self, memo):
        dc = type(self)(**self._to_dictionary())
        if dc.data is not None:
            dc.data = dc.data.copy()

        # uncomment if we want to deepcopy models as well:

        # dc.models._add_dictionary(
        #     copy.deepcopy(
        #         self.models._models.as_dictionary()))

        # The Signal subclasses might change the view on init
        # The following code just copies the original view
        for oaxis, caxis in zip(self.axes_manager._axes,
                                dc.axes_manager._axes):
            caxis.navigate = oaxis.navigate
        return dc

    def deepcopy(self):
        return copy.deepcopy(self)

    def change_dtype(self, dtype):
        """Change the data type.

        Parameters
        ----------
        dtype : str or dtype
            Typecode or data-type to which the array is cast. In
            addition to all standard numpy dtypes HyperSpy
            supports four extra dtypes for RGB images:
            "rgb8", "rgba8", "rgb16" and "rgba16". Changing from
            and to any rgbx dtype is more constrained than most
            other dtype conversions. To change to a rgbx dtype
            the signal `record_by` must be "spectrum",
            `signal_dimension` must be 3(4) for rgb(rgba) dtypes
            and the dtype must be uint8(uint16) for rgbx8(rgbx16).
            After conversion `record_by` becomes `image` and the
            spectra dimension is removed. The dtype of images of
            dtype rgbx8(rgbx16) can only be changed to uint8(uint16)
            and the `record_by` becomes "spectrum".


        Examples
        --------
        >>> s = hs.signals.Spectrum([1,2,3,4,5])
        >>> s.data
        array([1, 2, 3, 4, 5])
        >>> s.change_dtype('float')
        >>> s.data
        array([ 1.,  2.,  3.,  4.,  5.])

        """
        if not isinstance(dtype, np.dtype):
            if dtype in rgb_tools.rgb_dtypes:
                if self.metadata.Signal.record_by != "spectrum":
                    raise AttributeError(
                        "Only spectrum signals can be converted "
                        "to RGB images.")
                if "8" in dtype and self.data.dtype.name != "uint8":
                    raise AttributeError(
                        "Only signals with dtype uint8 can be converted to "
                        "rgb8 images")
                elif "16" in dtype and self.data.dtype.name != "uint16":
                    raise AttributeError(
                        "Only signals with dtype uint16 can be converted to "
                        "rgb16 images")
                dtype = rgb_tools.rgb_dtypes[dtype]
                self.data = rgb_tools.regular_array2rgbx(self.data)
                self.axes_manager.remove(-1)
                self.metadata.Signal.record_by = "image"
                self._assign_subclass()
                return
            else:
                dtype = np.dtype(dtype)
        if rgb_tools.is_rgbx(self.data) is True:
            ddtype = self.data.dtype.fields["B"][0]

            if ddtype != dtype:
                raise ValueError(
                    "It is only possibile to change to %s." %
                    ddtype)
            self.data = rgb_tools.rgbx2regular_array(self.data)
            self.get_dimensions_from_data()
            self.metadata.Signal.record_by = "spectrum"
            self.axes_manager[-1 + 2j].name = "RGB index"
            self._assign_subclass()
            return
        else:
            self.data = self.data.astype(dtype)

    def estimate_poissonian_noise_variance(self,
                                           expected_value=None,
                                           gain_factor=None,
                                           gain_offset=None,
                                           correlation_factor=None):
        """Estimate the poissonian noise variance of the signal.

        The variance is stored in the
        ``metadata.Signal.Noise_properties.variance`` attribute.

        A poissonian noise  variance is equal to the expected value. With the
        default arguments, this method simply sets the variance attribute to
        the given `expected_value`. However, more generally (although then
        noise is not strictly poissonian), the variance may be proportional to
        the expected value. Moreover, when the noise is a mixture of white
        (gaussian) and poissonian noise, the variance is described by the
        following linear model:

            .. math::

                \mathrm{Var}[X] = (a * \mathrm{E}[X] + b) * c

        Where `a` is the `gain_factor`, `b` is the `gain_offset` (the gaussian
        noise variance) and `c` the `correlation_factor`. The correlation
        factor accounts for correlation of adjacent signal elements that can
        be modeled as a convolution with a gaussian point spread function.


        Parameters
        ----------
        expected_value : None or Signal instance.
            If None, the signal data is taken as the expected value. Note that
            this may be inaccurate where `data` is small.
        gain_factor, gain_offset, correlation_factor: None or float.
            All three must be positive. If None, take the values from
            ``metadata.Signal.Noise_properties.Variance_linear_model`` if
            defined. Otherwise suppose poissonian noise i.e. ``gain_factor=1``,
            ``gain_offset=0``, ``correlation_factor=1``. If not None, the
            values are stored in
            ``metadata.Signal.Noise_properties.Variance_linear_model``.

        """
        if expected_value is None:
            dc = self.data.copy()
        else:
            dc = expected_value.data.copy()
        if self.metadata.has_item(
                "Signal.Noise_properties.Variance_linear_model"):
            vlm = self.metadata.Signal.Noise_properties.Variance_linear_model
        else:
            self.metadata.add_node(
                "Signal.Noise_properties.Variance_linear_model")
            vlm = self.metadata.Signal.Noise_properties.Variance_linear_model

        if gain_factor is None:
            if not vlm.has_item("gain_factor"):
                vlm.gain_factor = 1
            gain_factor = vlm.gain_factor

        if gain_offset is None:
            if not vlm.has_item("gain_offset"):
                vlm.gain_offset = 0
            gain_offset = vlm.gain_offset

        if correlation_factor is None:
            if not vlm.has_item("correlation_factor"):
                vlm.correlation_factor = 1
            correlation_factor = vlm.correlation_factor

        if gain_offset < 0:
            raise ValueError("`gain_offset` must be positive.")
        if gain_factor < 0:
            raise ValueError("`gain_factor` must be positive.")
        if correlation_factor < 0:
            raise ValueError("`correlation_factor` must be positive.")

        variance = (dc * gain_factor + gain_offset) * correlation_factor
        # The lower bound of the variance is the gaussian noise.
        variance = np.clip(variance, gain_offset * correlation_factor, np.inf)
        variance = type(self)(variance)
        variance.axes_manager = self.axes_manager
        variance.metadata.General.title = ("Variance of " +
                                           self.metadata.General.title)
        self.metadata.set_item(
            "Signal.Noise_properties.variance", variance)

    def get_current_signal(self, auto_title=True, auto_filename=True):
        """Returns the data at the current coordinates as a Signal subclass.

        The signal subclass is the same as that of the current object. All the
        axes navigation attribute are set to False.

        Parameters
        ----------
        auto_title : bool
            If True an space followed by the current indices in parenthesis
            are appended to the title.
        auto_filename : bool
            If True and `tmp_parameters.filename` is defined
            (what is always the case when the Signal has been read from a
            file), the filename is modified by appending an underscore and a
            parenthesis containing the current indices.

        Returns
        -------
        cs : Signal subclass instance.

        Examples
        --------
        >>> im = hs.signals.Image(np.zeros((2,3, 32,32)))
        >>> im
        <Image, title: , dimensions: (3, 2, 32, 32)>
        >>> im.axes_manager.indices = 2,1
        >>> im.get_current_signal()
        <Image, title:  (2, 1), dimensions: (32, 32)>

        """
        cs = self.__class__(
            self(),
            axes=self.axes_manager._get_signal_axes_dicts(),
            metadata=self.metadata.as_dictionary(),)

        if auto_filename is True and self.tmp_parameters.has_item('filename'):
            cs.tmp_parameters.filename = (self.tmp_parameters.filename +
                                          '_' +
                                          str(self.axes_manager.indices))
            cs.tmp_parameters.extension = self.tmp_parameters.extension
            cs.tmp_parameters.folder = self.tmp_parameters.folder
        if auto_title is True:
            cs.metadata.General.title = (cs.metadata.General.title +
                                         ' ' + str(self.axes_manager.indices))
        cs.axes_manager._set_axis_attribute_values("navigate", False)
        return cs

    def _get_navigation_signal(self, data=None, dtype=None):
        """Return a signal with the same axes as the navigation space.

        Parameters
        ----------
        data : {None, numpy array}, optional
            If None the `Signal` data is an array of the same dtype as the
            current one filled with zeros. If a numpy array, the array must
            have the correct dimensions.

        dtype : data-type, optional
            The desired data-type for the data array when `data` is None,
            e.g., `numpy.int8`.  Default is the data type of the current signal
            data.


        """
        if data is not None:
            ref_shape = (self.axes_manager._navigation_shape_in_array
                         if self.axes_manager.navigation_dimension != 0
                         else (1,))
            if data.shape != ref_shape:
                raise ValueError(
                    ("data.shape %s is not equal to the current navigation "
                     "shape in array which is %s") %
                    (str(data.shape), str(ref_shape)))
        else:
            if dtype is None:
                dtype = self.data.dtype
            if self.axes_manager.navigation_dimension == 0:
                data = np.array([0, ], dtype=dtype)
            else:
                data = np.zeros(self.axes_manager._navigation_shape_in_array,
                                dtype=dtype)
        if self.axes_manager.navigation_dimension == 0:
            s = Signal(data)
        elif self.axes_manager.navigation_dimension == 1:
            from hyperspy._signals.spectrum import Spectrum
            s = Spectrum(data,
                         axes=self.axes_manager._get_navigation_axes_dicts())
        elif self.axes_manager.navigation_dimension == 2:
            from hyperspy._signals.image import Image
            s = Image(data,
                      axes=self.axes_manager._get_navigation_axes_dicts())
        else:
            s = Signal(np.zeros(self.axes_manager._navigation_shape_in_array,
                                dtype=self.data.dtype),
                       axes=self.axes_manager._get_navigation_axes_dicts())
            s.axes_manager.set_signal_dimension(
                self.axes_manager.navigation_dimension)
        return s

    def _get_signal_signal(self, data=None, dtype=None):
        """Return a signal with the same axes as the signal space.

        Parameters
        ----------
        data : {None, numpy array}, optional
            If None the `Signal` data is an array of the same dtype as the
            current one filled with zeros. If a numpy array, the array must
            have the correct dimensions.
        dtype : data-type, optional
            The desired data-type for the data array when `data` is None,
            e.g., `numpy.int8`.  Default is the data type of the current signal
            data.

        """

        if data is not None:
            ref_shape = (self.axes_manager._signal_shape_in_array
                         if self.axes_manager.signal_dimension != 0
                         else (1,))
            if data.shape != ref_shape:
                raise ValueError(
                    "data.shape %s is not equal to the current signal shape in"
                    " array which is %s" % (str(data.shape), str(ref_shape)))
        else:
            if dtype is None:
                dtype = self.data.dtype
            if self.axes_manager.signal_dimension == 0:
                data = np.array([0, ], dtype=dtype)
            else:
                data = np.zeros(self.axes_manager._signal_shape_in_array,
                                dtype=dtype)

        if self.axes_manager.signal_dimension == 0:
            s = Signal(data)
            s.set_signal_type(self.metadata.Signal.signal_type)
        else:
            s = self.__class__(data,
                               axes=self.axes_manager._get_signal_axes_dicts())
        return s

    def __iter__(self):
        # Reset AxesManager iteration index
        self.axes_manager.__iter__()
        return self

    def __next__(self):
        next(self.axes_manager)
        return self.get_current_signal()

    def __len__(self):
        nitem = int(self.axes_manager.navigation_size)
        nitem = nitem if nitem > 0 else 1
        return nitem

    def as_spectrum(self, spectral_axis, out=None):
        """Return the Signal as a spectrum.

        The chosen spectral axis is moved to the last index in the
        array and the data is made contiguous for effecient
        iteration over spectra.


        Parameters
        ----------
        spectral_axis %s
        %s

        Examples
        --------
        >>> img = hs.signals.Image(np.ones((3,4,5,6)))
        >>> img
        <Image, title: , dimensions: (4, 3, 6, 5)>
        >>> img.to_spectrum(-1+1j)
        <Spectrum, title: , dimensions: (6, 5, 4, 3)>
        >>> img.to_spectrum(0)
        <Spectrum, title: , dimensions: (6, 5, 3, 4)>

        """
        # Roll the spectral axis to-be to the latex index in the array
        sp = self.rollaxis(spectral_axis, -1 + 3j)
        sp.metadata.Signal.record_by = "spectrum"
        sp._assign_subclass()
        if out is None:
            return sp
        else:
            out.data[:] = sp.data
            out.events.data_changed.trigger(obj=out)
    as_spectrum.__doc__ %= (ONE_AXIS_PARAMETER, OUT_ARG)

    def as_image(self, image_axes, out=None):
        """Convert signal to image.

        The chosen image axes are moved to the last indices in the
        array and the data is made contiguous for effecient
        iteration over images.

        Parameters
        ----------
        image_axes : tuple of {int | str | axis}
            Select the image axes. Note that the order of the axes matters
            and it is given in the "natural" i.e. X, Y, Z... order.
        %s

        Examples
        --------
        >>> s = hs.signals.Spectrum(np.ones((2,3,4,5)))
        >>> s
        <Spectrum, title: , dimensions: (4, 3, 2, 5)>
        >>> s.as_image((0,1))
        <Image, title: , dimensions: (5, 2, 4, 3)>

        >>> s.to_image((1,2))
        <Image, title: , dimensions: (4, 5, 3, 2)>

        Raises
        ------
        DataDimensionError : when data.ndim < 2

        """
        if self.data.ndim < 2:
            raise DataDimensionError(
                "A Signal dimension must be >= 2 to be converted to an Image")
        axes = (self.axes_manager[image_axes[0]],
                self.axes_manager[image_axes[1]])
        iaxes = [axis.index_in_array for axis in axes]
        im = self.rollaxis(iaxes[0] + 3j, -1 + 3j).rollaxis(
            iaxes[1] - np.argmax(iaxes) + 3j, -2 + 3j)
        im.metadata.Signal.record_by = "image"
        im._assign_subclass()
        if out is None:
            return im
        else:
            out.data[:] = im.data
            out.events.data_changed.trigger(obj=out)
    as_image.__doc__ %= OUT_ARG

    def _assign_subclass(self):
        mp = self.metadata
        self.__class__ = hyperspy.io.assign_signal_subclass(
            record_by=mp.Signal.record_by
            if "Signal.record_by" in mp
            else self._record_by,
            signal_type=mp.Signal.signal_type
            if "Signal.signal_type" in mp
            else self._signal_type,
            signal_origin=mp.Signal.signal_origin
            if "Signal.signal_origin" in mp
            else self._signal_origin)
        self.__init__(**self._to_dictionary())

    def set_signal_type(self, signal_type):
        """Set the signal type and change the current class
        accordingly if pertinent.

        The signal_type attribute specifies the kind of data that the signal
        containts e.g. "EELS" for electron energy-loss spectroscopy,
        "PES" for photoemission spectroscopy. There are some methods that are
        only available for certain kind of signals, so setting this
        parameter can enable/disable features.

        Parameters
        ----------
        signal_type : {"EELS", "EDS_TEM", "EDS_SEM", "DielectricFunction"}
            Currently there are special features for "EELS" (electron
            energy-loss spectroscopy), "EDS_TEM" (energy dispersive X-rays of
            thin samples, normally obtained in a transmission electron
            microscope), "EDS_SEM" (energy dispersive X-rays of thick samples,
            normally obtained in a scanning electron microscope) and
            "DielectricFuction". Setting the signal_type to the correct acronym
            is highly advisable when analyzing any signal for which HyperSpy
            provides extra features. Even if HyperSpy does not provide extra
            features for the signal that you are analyzing, it is good practice
            to set signal_type to a value that best describes the data signal
            type.

        """
        self.metadata.Signal.signal_type = signal_type
        self._assign_subclass()

    def set_signal_origin(self, origin):
        """Set the origin of the signal and change the current class
        accordingly if pertinent.

        The signal_origin attribute specifies if the data was obtained
        through experiment or simulation. There are some methods that are
        only available for experimental or simulated data, so setting this
        parameter can enable/disable features.


        Parameters
        ----------
        origin : {'experiment', 'simulation', None, ""}
            None an the empty string mean that the signal origin is uknown.

        Raises
        ------
        ValueError if origin is not 'experiment' or 'simulation'

        """
        if origin not in ['experiment', 'simulation', "", None]:
            raise ValueError("`origin` must be one of: experiment, simulation")
        if origin is None:
            origin = ""
        self.metadata.Signal.signal_origin = origin
        self._assign_subclass()

    def print_summary_statistics(self, formatter="%.3f"):
        """Prints the five-number summary statistics of the data, the mean and
        the standard deviation.

        Prints the mean, standandard deviation (std), maximum (max), minimum
        (min), first quartile (Q1), median and third quartile. nans are
        removed from the calculations.

        Parameters
        ----------
        formatter : bool
           Number formatter.

        See Also
        --------
        get_histogram

        """
        data = self.data
        # To make it work with nans
        data = data[~np.isnan(data)]
        print(underline("Summary statistics"))
        print("mean:\t" + formatter % data.mean())
        print("std:\t" + formatter % data.std())
        print()
        print("min:\t" + formatter % data.min())
        print("Q1:\t" + formatter % np.percentile(data,
                                                  25))
        print("median:\t" + formatter % np.median(data))
        print("Q3:\t" + formatter % np.percentile(data,
                                                  75))
        print("max:\t" + formatter % data.max())

    @property
    def is_rgba(self):
        return rgb_tools.is_rgba(self.data)

    @property
    def is_rgb(self):
        return rgb_tools.is_rgb(self.data)

    @property
    def is_rgbx(self):
        return rgb_tools.is_rgbx(self.data)

    def add_marker(self, marker, plot_on_signal=True, plot_marker=True):
        """
        Add a marker to the signal or navigator plot.

        Plot the signal, if not yet plotted

        Parameters
        ----------
        marker: `hyperspy.drawing._markers`
            the marker to add. see `plot.markers`
        plot_on_signal: bool
            If True, add the marker to the signal
            If False, add the marker to the navigator
        plot_marker: bool
            if True, plot the marker

        Examples
        -------
        >>> import scipy.misc
        >>> im = hs.signals.Image(scipy.misc.ascent())
        >>> m = hs.plot.markers.rectangle(x1=150, y1=100, x2=400,
        >>>                                  y2=400, color='red')
        >>> im.add_marker(m)

        """
        if self._plot is None:
            self.plot()
        if plot_on_signal:
            self._plot.signal_plot.add_marker(marker)
        else:
            self._plot.navigator_plot.add_marker(marker)
        if plot_marker:
            marker.plot()


ARITHMETIC_OPERATORS = (
    "__add__",
    "__sub__",
    "__mul__",
    "__floordiv__",
    "__mod__",
    "__divmod__",
    "__pow__",
    "__lshift__",
    "__rshift__",
    "__and__",
    "__xor__",
    "__or__",
    "__mod__",
    "__truediv__",
)
INPLACE_OPERATORS = (
    "__iadd__",
    "__isub__",
    "__imul__",
    "__itruediv__",
    "__ifloordiv__",
    "__imod__",
    "__ipow__",
    "__ilshift__",
    "__irshift__",
    "__iand__",
    "__ixor__",
    "__ior__",
)
COMPARISON_OPERATORS = (
    "__lt__",
    "__le__",
    "__eq__",
    "__ne__",
    "__ge__",
    "__gt__",
)
UNARY_OPERATORS = (
    "__neg__",
    "__pos__",
    "__abs__",
    "__invert__",
)
for name in ARITHMETIC_OPERATORS + INPLACE_OPERATORS + COMPARISON_OPERATORS:
    exec(
        ("def %s(self, other):\n" % name) +
        ("   return self._binary_operator_ruler(other, \'%s\')\n" %
         name))
    exec("%s.__doc__ = np.ndarray.%s.__doc__" % (name, name))
    exec("setattr(Signal, \'%s\', %s)" % (name, name))
    # The following commented line enables the operators with swapped
    # operands. They should be defined only for commutative operators
    # but for simplicity we don't support this at all atm.

    # exec("setattr(Signal, \'%s\', %s)" % (name[:2] + "r" + name[2:],
    # name))

# Implement unary arithmetic operations
for name in UNARY_OPERATORS:
    exec(
        ("def %s(self):" % name) +
        ("   return self._unary_operator_ruler(\'%s\')" % name))
    exec("%s.__doc__ = int.%s.__doc__" % (name, name))
    exec("setattr(Signal, \'%s\', %s)" % (name, name))
