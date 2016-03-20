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


import numpy as np
from nose.tools import assert_true, assert_equal, assert_dict_equal

from hyperspy.signals import EDSTEMSpectrum, Simulation
from hyperspy.defaults_parser import preferences
from hyperspy.components import Gaussian
from hyperspy.misc.eds import utils as utils_eds
from hyperspy.misc.test_utils import ignore_warning


class Test_metadata:

    def setUp(self):
        # Create an empty spectrum
        s = EDSTEMSpectrum(np.ones((4, 2, 1024)))
        s.metadata.Acquisition_instrument.TEM.Detector.EDS.live_time = 3.1
        s.metadata.Acquisition_instrument.TEM.beam_energy = 15.0
        self.signal = s

    def test_sum_live_time1(self):
        s = self.signal
        old_metadata = s.metadata.deepcopy()
        sSum = s.sum(0)
        assert_equal(
            sSum.metadata.Acquisition_instrument.TEM.Detector.EDS.live_time,
            3.1 *
            2)
        # Check that metadata is unchanged
        print(old_metadata, s.metadata)      # Capture for comparison on error
        assert_dict_equal(old_metadata.as_dictionary(),
                          s.metadata.as_dictionary(),
                          "Source metadata changed")

    def test_sum_live_time2(self):
        s = self.signal
        old_metadata = s.metadata.deepcopy()
        sSum = s.sum((0, 1))
        assert_equal(
            sSum.metadata.Acquisition_instrument.TEM.Detector.EDS.live_time,
            3.1 *
            2 * 4)
        # Check that metadata is unchanged
        print(old_metadata, s.metadata)      # Capture for comparison on error
        assert_dict_equal(old_metadata.as_dictionary(),
                          s.metadata.as_dictionary(),
                          "Source metadata changed")

    def test_sum_live_time_out_arg(self):
        s = self.signal
        sSum = s.sum(0)
        s.metadata.Acquisition_instrument.TEM.Detector.EDS.live_time = 4.2
        s_resum = s.sum(0)
        r = s.sum(0, out=sSum)
        assert_equal(r, None)
        assert_equal(
            s_resum.metadata.Acquisition_instrument.TEM.Detector.EDS.live_time,
            sSum.metadata.Acquisition_instrument.TEM.Detector.EDS.live_time)
        np.testing.assert_allclose(s_resum.data, sSum.data)

    def test_rebin_live_time(self):
        s = self.signal
        old_metadata = s.metadata.deepcopy()
        dim = s.axes_manager.shape
        s = s.rebin([dim[0] / 2, dim[1] / 2, dim[2]])
        assert_equal(
            s.metadata.Acquisition_instrument.TEM.Detector.EDS.live_time,
            3.1 *
            2 *
            2)
        # Check that metadata is unchanged
        print(old_metadata, self.signal.metadata)    # Captured on error
        assert_dict_equal(old_metadata.as_dictionary(),
                          self.signal.metadata.as_dictionary(),
                          "Source metadata changed")

    def test_add_elements(self):
        s = self.signal
        s.add_elements(['Al', 'Ni'])
        assert_equal(s.metadata.Sample.elements, ['Al', 'Ni'])
        s.add_elements(['Al', 'Ni'])
        assert_equal(s.metadata.Sample.elements, ['Al', 'Ni'])
        s.add_elements(["Fe", ])
        assert_equal(s.metadata.Sample.elements, ['Al', "Fe", 'Ni'])
        s.set_elements(['Al', 'Ni'])
        assert_equal(s.metadata.Sample.elements, ['Al', 'Ni'])

    def test_default_param(self):
        s = self.signal
        mp = s.metadata
        assert_equal(
            mp.Acquisition_instrument.TEM.Detector.EDS.energy_resolution_MnKa,
            preferences.EDS.eds_mn_ka)

    def test_TEM_to_SEM(self):
        s = self.signal.inav[0, 0]
        signal_type = 'EDS_SEM'
        mp = s.metadata
        mp.Acquisition_instrument.TEM.Detector.EDS.energy_resolution_MnKa =\
            125.3
        sSEM = s.deepcopy()
        sSEM.set_signal_type(signal_type)
        mpSEM = sSEM.metadata
        results = [
            mp.Acquisition_instrument.TEM.Detector.EDS.energy_resolution_MnKa,
            signal_type]
        resultsSEM = [
            mpSEM.Acquisition_instrument.SEM.Detector.EDS.energy_resolution_MnKa,
            mpSEM.Signal.signal_type]
        assert_equal(results, resultsSEM)

    def test_get_calibration_from(self):
        s = self.signal
        scalib = EDSTEMSpectrum(np.ones(1024))
        energy_axis = scalib.axes_manager.signal_axes[0]
        energy_axis.scale = 0.01
        energy_axis.offset = -0.10
        s.get_calibration_from(scalib)
        assert_equal(s.axes_manager.signal_axes[0].scale,
                     energy_axis.scale)


class Test_quantification:

    def setUp(self):
        s = EDSTEMSpectrum(np.ones([2, 1024]))
        energy_axis = s.axes_manager.signal_axes[0]
        energy_axis.scale = 1e-2
        energy_axis.units = 'keV'
        energy_axis.name = "Energy"
        s.set_microscope_parameters(beam_energy=200,
                                    live_time=3.1, tilt_stage=0.0,
                                    azimuth_angle=None, elevation_angle=35,
                                    energy_resolution_MnKa=130)
        elements = ['Al', 'Zn']
        xray_lines = ['Al_Ka', 'Zn_Ka']
        intensities = [300, 500]
        for i, xray_line in enumerate(xray_lines):
            gauss = Gaussian()
            line_energy, FWHM = s._get_line_energy(xray_line, FWHM_MnKa='auto')
            gauss.centre.value = line_energy
            gauss.A.value = intensities[i]
            gauss.sigma.value = FWHM
            s.data[:] += gauss.function(energy_axis.axis)

        s.set_elements(elements)
        s.add_lines(xray_lines)
        self.signal = s

    def test_quant_lorimer(self):
        s = self.signal
        kfactors = [1, 2.0009344042484134]
        intensities = s.get_lines_intensity()
        res = s.quantification(intensities, kfactors)
        assert_true(np.allclose(res[0].data, np.array(
                    [22.70779, 22.70779]), atol=1e-3))

    def test_quant_zeros(self):
        intens = np.array([[0.5, 0.5, 0.5],
                           [0., 0.5, 0.5],
                           [0.5, 0.0, 0.5],
                           [0.5, 0.5, 0.0],
                           [0.5, 0.0, 0.0]]).T
        with ignore_warning(message="divide by zero encountered",
                            category=RuntimeWarning):
            quant = utils_eds.quantification_cliff_lorimer(
                intens, [1, 1, 3]).T
        assert_true(np.allclose(
            quant,
            np.array([[0.2, 0.2, 0.6],
                      [0., 0.25, 0.75],
                      [0.25, 0., 0.75],
                      [0.5, 0.5, 0.],
                      [1., 0., 0.]])))


class Test_vacum_mask:

    def setUp(self):
        s = Simulation(np.array([np.linspace(0.001, 0.5, 20)] * 100).T)
        s.add_poissonian_noise()
        s = EDSTEMSpectrum(s.data)
        self.signal = s

    def test_vacuum_mask(self):
        s = self.signal
        assert_equal(s.vacuum_mask().data[0], True)
        assert_equal(s.vacuum_mask().data[-1], False)


class Test_simple_model:

    def setUp(self):
        s = utils_eds.xray_lines_model(elements=['Al', 'Zn'],
                                       weight_percents=[50, 50])
        self.signal = s

    def test_intensity(self):
        s = self.signal
        assert_true(np.allclose(
            [i.data for i in s.get_lines_intensity(
                integration_window_factor=5.0)], [0.5, 0.5], atol=1e-1))


class Test_get_lines_intentisity:

    def test_with_signals_examples(self):
        from hyperspy.misc.example_signals_loading import \
            load_1D_EDS_TEM_spectrum as EDS_TEM_Spectrum
        s = EDS_TEM_Spectrum()
        np.allclose(np.array([res.data for res in s.get_lines_intensity()]),
                    np.array([3710, 15872]))
