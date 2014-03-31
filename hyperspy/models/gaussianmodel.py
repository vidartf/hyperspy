# -*- coding: utf-8 -*-
# Copyright 2007-2011 The Hyperspy developers
#
# This file is part of  Hyperspy.
#
#  Hyperspy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
#  Hyperspy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with  Hyperspy.  If not, see <http://www.gnu.org/licenses/>.



import math
import numpy as np
from sklearn import mixture
from hyperspy import components
from scipy import ndimage
from hyperspy.signal import Signal1DTools
from hyperspy import hspy
from hyperspy.model import Model


sqrt2pi = math.sqrt(2 * math.pi)
sigma2fwhm = 2 * math.sqrt(2 * math.log(2))     
fwhm2sigma = 1 / (2 * math.sqrt(math.log(2) * 2))

class GaussianModel(Model):

    def __init__(self, spectrum, n_gauss=None, *args, **kwargs):
        Model.__init__(self, spectrum, *args, **kwargs)
        self.gaussians = []
        self.coarse_fit(n_gauss)
        self.fit()

    def coarse_fit(self, n_gauss):

        f = self.spectrum.data

        # Use a small gaussian kernel to do take the derivatives
        df = ndimage.gaussian_filter1d(f, sigma=1, order=1, mode='wrap')
        df2 = ndimage.gaussian_filter1d(df, sigma=1, order=1, mode='wrap')
    
        # Find peak positions
        signs = np.sign(df)
        imax = np.where((np.diff(signs) != 0) & (signs[:-1] > 0))[0] + 1
        self.centres = self.spectrum.axes_manager[0].index2value(imax)

        if n_gauss is not None and n_gauss < imax.size:
            # Select strongest n_gauss
            (sigmas, amps) = self._calc_sigma_amp()
            imax.sort(key=lambda im: amps[im] * sigmas[im]) # Maybe use negative df2?
            imax = imax[:n_gauss]
            self.centres = self.spectrum.axes_manager[0].index2value(imax)

        # Add Gaussian components
        self._reset_gaussians()

    def _calc_sigma_amp(self):
         # Setup nearest neighbour distances
        ds = np.diff(self.centres)
        ds = np.append(ds, ds[-1])
        ds[1:-1] = np.minimum(ds[:-2], ds[1:-1])

        # Estimate sigmas and amplitudes
        sigmas = ds*fwhm2sigma
        ic = self.spectrum.axes_manager[0].value2index(self.centres)
        amps = self.spectrum.data[ic]*sqrt2pi*sigmas

        return (sigmas, amps)


    def _reset_gaussians(self):
        # Remove components from Model
        for g in self.gaussians:
            self.remove(g, False)
        self.gaussians = []

        # Get sigmas and amplitudes
        (sigmas, amps) = self._calc_sigma_amp()
        
        # Create and add Gaussian components
        for i in xrange(self.centres.size):
            g = components.Gaussian(amps[i], sigmas[i], self.centres[i])
            self.gaussians.append(g)
            self.append(g)


    def _update_gaussians(self):
        (sigmas, amps) = self._calc_sigma_amp()
        # Update sigmas and amps for all
        for i, g in enumerate(self.gaussians):
            g.A.value = amps[i]
            g.sigma.value = sigmas[i]
            g.centre.value = self.centres[i]


    def add_at(self, value):
        if isInstance(value, int):
            value = self.spectrum.axes_manager[0].index2value(value)

        # Add new centre in sorted order
        self.centres = np.append(self.centres, value)
        idx = np.argsort(self.centres)
        self.centres = self.centres[idx]

        # Add new Gaussian
        new_idx = np.argmax(idx)
        g = components.Gaussian(1, 1, self.centres[new_idx])
        self.gaussians.insert(new_idx, g)   # Keep gaussians in same order as centres. Could possibly be avoided by using dictionaries
        self.append(g)

        # Update all gaussians parameters
        self._update_gaussians()

        # Do final fitting
        self.fit()

    def remove_at(self, value):
        if isInstance(value, int):
            value = self.spectrum.axes_manager[0].index2value(value)
        idx = (np.abs(self.centres-value)).argmin()

        # Remove centre and Gaussian
        np.delete(self.centres, idx)
        g = self.gaussians[idx]
        self.gaussian.remove(g)
        self.remove(g)
        del g
        
        # Update all gaussians parameters
        self._update_gaussians()

        # Do final fitting
        self.fit()
