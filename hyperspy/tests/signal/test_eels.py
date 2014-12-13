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


import numpy as np
import nose.tools

import hyperspy.hspy as hp


class Test_Estimate_Elastic_Scattering_Threshold:

    def setUp(self):
        # Create an empty spectrum
        s = hp.signals.EELSSpectrumSimulation(np.zeros((3, 2, 1024)))
        energy_axis = s.axes_manager.signal_axes[0]
        energy_axis.scale = 0.02
        energy_axis.offset = -5

        gauss = hp.components.Gaussian()
        gauss.centre.value = 0
        gauss.A.value = 5000
        gauss.sigma.value = 0.5
        gauss2 = hp.components.Gaussian()
        gauss2.sigma.value = 0.5
        # Inflexion point 1.5
        gauss2.A.value = 5000
        gauss2.centre.value = 5
        s.data[:] = (gauss.function(energy_axis.axis) +
                     gauss2.function(energy_axis.axis))
        self.signal = s

    def test_min_in_window_with_smoothing(self):
        s = self.signal
        thr = s.estimate_elastic_scattering_threshold(
            window=5,
            window_length=5,
            tol=0.00001)
        nose.tools.assert_true(np.allclose(thr.data, 2.5))

    def test_min_in_window_without_smoothing_single_spectrum(self):
        s = self.signal[0, 0]
        thr = s.estimate_elastic_scattering_threshold(
            window=5,
            window_length=0,
            tol=0.001)
        nose.tools.assert_true(np.allclose(thr.data, 2.49))

    def test_min_in_window_without_smoothing(self):
        s = self.signal
        thr = s.estimate_elastic_scattering_threshold(
            window=5,
            window_length=0,
            tol=0.001)
        nose.tools.assert_true(np.allclose(thr.data, 2.49))

    def test_min_not_in_window(self):
        # If I use a much lower window, this is the value that has to be
        # returned as threshold.
        s = self.signal
        data = s.estimate_elastic_scattering_threshold(window=1.5,
                                                       tol=0.001).data
        nose.tools.assert_true(np.all(np.isnan(data)))


class TestEstimateZLPCentre():

    def setUp(self):
        s = hp.signals.EELSSpectrumSimulation(np.diag(np.arange(1, 11)))
        s.axes_manager[-1].scale = 0.1
        s.axes_manager[-1].offset = 100
        self.spectrum = s

    def test_estimate_zero_loss_peak_centre(self):
        s = self.spectrum
        nose.tools.assert_true(np.allclose(s.estimate_zero_loss_peak_centre().data,
                                           np.arange(100, 101, 0.1)))


class TestAlignZLP():

    def setUp(self):
        s = hp.signals.EELSSpectrumSimulation(np.zeros((10, 100)))
        self.scale = 0.1
        self.offset = -2
        eaxis = s.axes_manager.signal_axes[0]
        eaxis.scale = self.scale
        eaxis.offset = self.offset
        self.izlp = eaxis.value2index(0)
        self.bg = 2
        self.ishifts = np.array([0, 4, 2, -2, 5, -2, -5, -9, -9, -8])
        self.new_offset = self.offset - self.ishifts.min() * self.scale
        s.data[np.arange(10), self.ishifts + self.izlp] = 10
        s.data += self.bg
        s.axes_manager[-1].offset += 100
        self.spectrum = s

    def test_align_zero_loss_peak_calibrate_true(self):
        s = self.spectrum
        s.align_zero_loss_peak(calibrate=True, print_stats=False)
        zlpc = s.estimate_zero_loss_peak_centre()
        nose.tools.assert_true(np.allclose(zlpc.data.mean(), 0))
        nose.tools.assert_true(np.allclose(zlpc.data.std(), 0))

    def test_align_zero_loss_peak_calibrate_false(self):
        s = self.spectrum
        s.align_zero_loss_peak(calibrate=False, print_stats=False)
        zlpc = s.estimate_zero_loss_peak_centre()
        nose.tools.assert_true(np.allclose(zlpc.data.std(), 0))

    def test_also_aligns(self):
        s = self.spectrum
        s2 = s.deepcopy()
        s.align_zero_loss_peak(calibrate=True,
                               print_stats=False,
                               also_align=[s2])
        zlpc = s2.estimate_zero_loss_peak_centre()
        nose.tools.assert_equal(zlpc.data.mean(), 0)
        nose.tools.assert_equal(zlpc.data.std(), 0)


class TestPowerLawExtrapolation:

    def setUp(self):
        s = hp.signals.EELSSpectrum(0.1 * np.arange(50, 250, 0.5) ** -3.)
        s.metadata.Signal.binned = False
        s.axes_manager[-1].offset = 50
        s.axes_manager[-1].scale = 0.5
        self.s = s

    def test_unbinned(self):
        sc = self.s.isig[:300]
        s = sc.power_law_extrapolation(extrapolation_size=100)
        nose.tools.assert_true(np.allclose(s.data, self.s.data))

    def test_binned(self):
        self.s.data *= self.s.axes_manager[-1].scale
        self.s.metadata.Signal.binned = True
        sc = self.s.isig[:300]
        s = sc.power_law_extrapolation(extrapolation_size=100)
        nose.tools.assert_true(np.allclose(s.data, self.s.data))
