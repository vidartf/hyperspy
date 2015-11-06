# -*- coding: utf-8 -*-
# Copyright 2007-2015 The HyperSpy developers
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

import numpy as np
import scipy as sp
from scipy.fftpack import fftn, ifftn
from skimage.feature.register_translation import _upsampled_dft
import matplotlib.pyplot as plt


def shift_image(im, shift, fill_value=np.nan):
    fractional, integral = np.modf(shift)
    if fractional.any():
        order = 3
    else:
        # Disable interpolation
        order = 0
    im[:] = sp.ndimage.shift(im, shift, cval=fill_value, order=order)


def triu_indices_minus_diag(n):
    """Returns the indices for the upper-triangle of an (n, n) array
    excluding its diagonal

    Parameters
    ----------
    n : int
        The length of the square array

    """
    ti = np.triu_indices(n)
    isnotdiag = ti[0] != ti[1]
    return ti[0][isnotdiag], ti[1][isnotdiag]


def hanning2d(M, N):
    """
    A 2D hanning window created by outer product.
    """
    return np.outer(np.hanning(M), np.hanning(N))


def sobel_filter(im):
    sx = sp.ndimage.sobel(im, axis=0, mode='constant')
    sy = sp.ndimage.sobel(im, axis=1, mode='constant')
    sob = np.hypot(sx, sy)
    return sob


def fft_correlation(in1, in2, normalize=False):
    """Correlation of two N-dimensional arrays using FFT.

    Adapted from scipy's fftconvolve.

    Parameters
    ----------
    in1, in2 : array
    normalize: bool
        If True performs phase correlation

    """
    s1 = np.array(in1.shape)
    s2 = np.array(in2.shape)
    size = s1 + s2 - 1
    # Use 2**n-sized FFT
    fsize = 2 ** np.ceil(np.log2(size))
    fprod = fftn(in1, fsize)
    fprod *= fftn(in2, fsize).conjugate()
    if normalize is True:
        ret = ifftn(np.nan_to_num(fprod / np.absolute(fprod))).real.copy()
    else:
        ret = ifftn(fprod).real.copy()
    return ret, fprod


def estimate_image_shift(ref, image, roi=None, sobel=True,
                         medfilter=True, hanning=True, plot=False,
                         dtype='float', normalize_corr=False,
                         sub_pixel_factor=1):
    """Estimate the shift in a image using phase correlation

    This method can only estimate the shift by comparing
    bidimensional features that should not change the position
    in the given axis. To decrease the memory usage, the time of
    computation and the accuracy of the results it is convenient
    to select a region of interest by setting the roi keyword.

    Parameters
    ----------

    ref : 2D numpy.ndarray
        Reference image
    image : 2D numpy.ndarray
        Image to register
    roi : tuple of ints (top, bottom, left, right)
         Define the region of interest
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
    normalize_corr : bool
        If True use phase correlation instead of standard correlation
    sub_pixel_factor : float
        Estimate shifts with a sub-pixel accuracy of 1/sub_pixel_factor parts
        of a pixel. Default is 1, i.e. no sub-pixel accuracy.

    Returns
    -------

    shifts: np.array
        containing the estimate shifts
    max_value : float
        The maximum value of the correlation

    """

    # Make a copy of the images to avoid modifying them
    ref = ref.copy().astype(dtype)
    image = image.copy().astype(dtype)
    if roi is not None:
        top, bottom, left, right = roi
    else:
        top, bottom, left, right = [None, ] * 4

    # Select region of interest
    ref = ref[top:bottom, left:right]
    image = image[top:bottom, left:right]

    # Apply filters
    for im in (ref, image):
        if hanning is True:
            im *= hanning2d(*im.shape)
        if medfilter is True:
            im[:] = sp.signal.medfilt(im)
        if sobel is True:
            im[:] = sobel_filter(im)
    phase_correlation, image_product = fft_correlation(
        ref, image, normalize=normalize_corr)

    # Estimate the shift by getting the coordinates of the maximum
    argmax = np.unravel_index(np.argmax(phase_correlation),
                              phase_correlation.shape)
    threshold = (phase_correlation.shape[0] / 2 - 1,
                 phase_correlation.shape[1] / 2 - 1)
    shift0 = argmax[0] if argmax[0] < threshold[0] else  \
        argmax[0] - phase_correlation.shape[0]
    shift1 = argmax[1] if argmax[1] < threshold[1] else \
        argmax[1] - phase_correlation.shape[1]
    max_val = phase_correlation.max()
    shifts = np.array((shift0, shift1))

    # The following code is more or less copied from
    # skimage.feature.register_feature, to gain access to the maximum value:
    if sub_pixel_factor != 1:
        # Initial shift estimate in upsampled grid
        shifts = np.round(shifts * sub_pixel_factor) / sub_pixel_factor
        upsampled_region_size = np.ceil(sub_pixel_factor * 1.5)
        # Center of output array at dftshift + 1
        dftshift = np.fix(upsampled_region_size / 2.0)
        sub_pixel_factor = np.array(sub_pixel_factor, dtype=np.float64)
        normalization = (image_product.size * sub_pixel_factor ** 2)
        # Matrix multiply DFT around the current shift estimate
        sample_region_offset = dftshift - shifts*sub_pixel_factor
        cross_correlation = _upsampled_dft(image_product.conj(),
                                           upsampled_region_size,
                                           sub_pixel_factor,
                                           sample_region_offset).conj()
        cross_correlation /= normalization
        # Locate maximum and map back to original pixel grid
        maxima = np.array(np.unravel_index(
                              np.argmax(np.abs(cross_correlation)),
                              cross_correlation.shape),
                          dtype=np.float64)
        maxima -= dftshift
        shifts = shifts + maxima / sub_pixel_factor
        max_val = cross_correlation.max()

    # Plot on demand
    if plot is True or isinstance(plot, plt.Figure):
        if isinstance(plot, plt.Figure):
            f = plot
            axarr = plot.axes
            if len(axarr) < 3:
                for i in xrange(3):
                    f.add_subplot(3, 1, i)
                axarr = plot.axes
        else:
            f, axarr = plt.subplots(1, 3)
        full_plot = len(axarr[0].images) == 0
        if full_plot:
            axarr[0].set_title('Reference')
            axarr[1].set_title('Image')
            axarr[2].set_title('Phase correlation')
            axarr[0].imshow(ref)
            axarr[1].imshow(image)
            d = (np.array(phase_correlation.shape) - 1) // 2
            extent = [-d[1], d[1], -d[0], d[0]]
            axarr[2].imshow(np.fft.fftshift(phase_correlation),
                            cmap=plt.cm.jet, extent=extent)
            plt.show()
        else:
            axarr[0].images[0].set_data(ref)
            axarr[1].images[0].set_data(image)
            axarr[2].images[0].set_data(np.fft.fftshift(phase_correlation))
            # TODO: Renormalize images
            f.canvas.draw()
    # Liberate the memory. It is specially necessary if it is a
    # memory map
    del ref
    del image

    return -shifts, max_val


def contrast_stretching(data, saturated_pixels):
    """Calculate bounds that leaves out a given percentage of the data.

    Parameters
    ----------
    data: numpy array
    saturated_pixels: scalar
        The percentage of pixels that are left out of the bounds.  For example,
        the low and high bounds of a value of 1 are the 0.5% and 99.5%
        percentiles. It must be in the [0, 100] range.

    Returns
    -------
    vmin, vmax: scalar
        The low and high bounds

    Raises
    ------
    ValueError if the value of `saturated_pixels` is out of the valid range.

    """
    # Sanity check
    if not 0 <= saturated_pixels <= 100:
        raise ValueError(
            "saturated_pixels must be a scalar in the range[0, 100]")
    nans = np.isnan(data)
    if nans.any():
        data = data[~nans]
    vmin = np.percentile(data, saturated_pixels / 2.)
    vmax = np.percentile(data, 100 - saturated_pixels / 2.)
    return vmin, vmax

MPL_DIVERGING_COLORMAPS = [
    "BrBG",
    "bwr",
    "coolwarm",
    "PiYG",
    "PRGn",
    "PuOr",
    "RdBu",
    "RdGy",
    "RdYIBu",
    "RdYIGn",
    "seismic",
    "Spectral", ]
# Add reversed colormaps
MPL_DIVERGING_COLORMAPS += [cmap + "_r" for cmap in MPL_DIVERGING_COLORMAPS]


def centre_colormap_values(vmin, vmax):
    """Calculate vmin and vmax to set the colormap midpoint to zero.

    Parameters
    ----------
    vmin, vmax : scalar
        The range of data to display.

    Returns
    -------
    cvmin, cvmax : scalar
        The values to obtain a centre colormap.

    """

    absmax = max(abs(vmin), abs(vmax))
    return -absmax, absmax
