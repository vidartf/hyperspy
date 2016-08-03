import numpy as np
import nose.tools as nt

import hyperspy.api as hs


def test_spectrum_binned_default():
    s = hs.signals.Signal1D([0])
    nt.assert_false(s.metadata.Signal.binned)


def test_image_binned_default():
    s = hs.signals.Signal2D(np.zeros((2, 2)))
    nt.assert_false(s.metadata.Signal.binned)


def test_dielectric_function_binned_default():
    s = hs.signals.DielectricFunction([0])
    nt.assert_false(s.metadata.Signal.binned)


def test_signal_binned_default():
    s = hs.signals.BaseSignal([0])
    nt.assert_false(s.metadata.Signal.binned)


def test_eels_spectrum_binned_default():
    s = hs.signals.EELSSpectrum([0])
    nt.assert_true(s.metadata.Signal.binned)


def test_eds_tem_binned_default():
    s = hs.signals.EDSTEMSpectrum([0])
    nt.assert_true(s.metadata.Signal.binned)


def test_eds_sem_binned_default():
    s = hs.signals.EDSSEMSpectrum([0])
    nt.assert_true(s.metadata.Signal.binned)


class TestModelBinned:

    def setUp(self):
        s = hs.signals.Signal1D([1])
        s.axes_manager[0].scale = 0.1
        m = s.create_model()
        m.append(hs.model.components1D.Offset())
        m[0].offset.value = 1
        self.m = m

    def test_unbinned(self):
        self.m.signal.metadata.Signal.binned = False
        nt.assert_equal(self.m(), 1)

    def test_binned(self):
        self.m.signal.metadata.Signal.binned = True
        nt.assert_equal(self.m(), 0.1)
