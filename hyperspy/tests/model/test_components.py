import numpy as np
import nose.tools as nt

import hyperspy.api as hs
from hyperspy.models.model1D import Model1D
from hyperspy.misc.test_utils import ignore_warning


class TestPowerLaw:

    def setUp(self):
        s = hs.signals.Spectrum(np.zeros(1024))
        s.axes_manager[0].offset = 100
        s.axes_manager[0].scale = 0.01
        m = s.create_model()
        m.append(hs.model.components.PowerLaw())
        m[0].A.value = 10
        m[0].r.value = 4
        self.m = m

    def test_estimate_parameters_binned_only_current(self):
        self.m.spectrum.metadata.Signal.binned = True
        s = self.m.as_signal(show_progressbar=None)
        s.metadata.Signal.binned = True
        g = hs.model.components.PowerLaw()
        g.estimate_parameters(s,
                              None,
                              None,
                              only_current=True)
        nt.assert_almost_equal(g.A.value, 10.084913947965161)
        nt.assert_almost_equal(g.r.value, 4.0017676988807409)

    def test_estimate_parameters_unbinned_only_current(self):
        self.m.spectrum.metadata.Signal.binned = False
        s = self.m.as_signal(show_progressbar=None)
        s.metadata.Signal.binned = False
        g = hs.model.components.PowerLaw()
        g.estimate_parameters(s,
                              None,
                              None,
                              only_current=True)
        nt.assert_almost_equal(g.A.value, 10.064378823244837)
        nt.assert_almost_equal(g.r.value, 4.0017522876514304)

    def test_estimate_parameters_binned(self):
        self.m.spectrum.metadata.Signal.binned = True
        s = self.m.as_signal(show_progressbar=None)
        s.metadata.Signal.binned = True
        g = hs.model.components.PowerLaw()
        g._axes_manager = self.m.spectrum.axes_manager
        g.estimate_parameters(s,
                              None,
                              None,
                              only_current=False)
        nt.assert_almost_equal(g.A.value, 10.084913947965161)
        nt.assert_almost_equal(g.r.value, 4.0017676988807409)

    def test_estimate_parameters_unbinned(self):
        self.m.spectrum.metadata.Signal.binned = False
        s = self.m.as_signal(show_progressbar=None)
        s.metadata.Signal.binned = False
        g = hs.model.components.PowerLaw()
        g._axes_manager = self.m.spectrum.axes_manager
        g.estimate_parameters(s,
                              None,
                              None,
                              only_current=False)
        nt.assert_almost_equal(g.A.value, 10.064378823244837)
        nt.assert_almost_equal(g.r.value, 4.0017522876514304)


class TestOffset:

    def setUp(self):
        s = hs.signals.Spectrum(np.zeros(10))
        s.axes_manager[0].scale = 0.01
        m = s.create_model()
        m.append(hs.model.components.Offset())
        m[0].offset.value = 10
        self.m = m

    def test_estimate_parameters_binned(self):
        self.m.spectrum.metadata.Signal.binned = True
        s = self.m.as_signal(show_progressbar=None)
        s.metadata.Signal.binned = True
        g = hs.model.components.Offset()
        g.estimate_parameters(s,
                              None,
                              None,
                              only_current=True)
        nt.assert_almost_equal(g.offset.value, 10)

    def test_estimate_parameters_unbinned(self):
        self.m.spectrum.metadata.Signal.binned = False
        s = self.m.as_signal(show_progressbar=None)
        s.metadata.Signal.binned = False
        g = hs.model.components.Offset()
        g.estimate_parameters(s,
                              None,
                              None,
                              only_current=True)
        nt.assert_almost_equal(g.offset.value, 10)


class TestPolynomial:

    def setUp(self):
        s = hs.signals.Spectrum(np.zeros(1024))
        s.axes_manager[0].offset = -5
        s.axes_manager[0].scale = 0.01
        m = s.create_model()
        m.append(hs.model.components.Polynomial(order=2))
        m[0].coefficients.value = (0.5, 2, 3)
        self.m = m
        s_2d = hs.signals.Spectrum(np.arange(1000).reshape(10, 100))
        self.m_2d = s_2d.create_model()
        self.m_2d.append(m[0])
        s_3d = hs.signals.Spectrum(np.arange(1000).reshape(2, 5, 100))
        self.m_3d = s_3d.create_model()
        self.m_3d.append(m[0])

    def test_estimate_parameters_binned(self):
        self.m.spectrum.metadata.Signal.binned = True
        s = self.m.as_signal(show_progressbar=None)
        s.metadata.Signal.binned = True
        g = hs.model.components.Polynomial(order=2)
        g.estimate_parameters(s,
                              None,
                              None,
                              only_current=True)
        nt.assert_almost_equal(g.coefficients.value[0], 0.5)
        nt.assert_almost_equal(g.coefficients.value[1], 2)
        nt.assert_almost_equal(g.coefficients.value[2], 3)

    def test_estimate_parameters_unbinned(self):
        self.m.spectrum.metadata.Signal.binned = False
        s = self.m.as_signal(show_progressbar=None)
        s.metadata.Signal.binned = False
        g = hs.model.components.Polynomial(order=2)
        g.estimate_parameters(s,
                              None,
                              None,
                              only_current=True)
        nt.assert_almost_equal(g.coefficients.value[0], 0.5)
        nt.assert_almost_equal(g.coefficients.value[1], 2)
        nt.assert_almost_equal(g.coefficients.value[2], 3)

    def test_2d_signal(self):
        # This code should run smoothly, any exceptions should trigger failure
        s = self.m_2d.as_signal(show_progressbar=None)
        model = Model1D(s)
        p = hs.model.components.Polynomial(order=2)
        model.append(p)
        p.estimate_parameters(s, 0, 100, only_current=False)
        np.testing.assert_allclose(p.coefficients.map['values'],
                                   np.tile([0.5, 2, 3], (10, 1)))

    def test_3d_signal(self):
        # This code should run smoothly, any exceptions should trigger failure
        s = self.m_3d.as_signal(show_progressbar=None)
        model = Model1D(s)
        p = hs.model.components.Polynomial(order=2)
        model.append(p)
        p.estimate_parameters(s, 0, 100, only_current=False)
        np.testing.assert_allclose(p.coefficients.map['values'],
                                   np.tile([0.5, 2, 3], (2, 5, 1)))


class TestGaussian:

    def setUp(self):
        s = hs.signals.Spectrum(np.zeros(1024))
        s.axes_manager[0].offset = -5
        s.axes_manager[0].scale = 0.01
        m = s.create_model()
        m.append(hs.model.components.Gaussian())
        m[0].sigma.value = 0.5
        m[0].centre.value = 1
        m[0].A.value = 2
        self.m = m

    def test_estimate_parameters_binned(self):
        self.m.spectrum.metadata.Signal.binned = True
        s = self.m.as_signal(show_progressbar=None)
        s.metadata.Signal.binned = True
        g = hs.model.components.Gaussian()
        g.estimate_parameters(s,
                              None,
                              None,
                              only_current=True)
        nt.assert_almost_equal(g.sigma.value, 0.5)
        nt.assert_almost_equal(g.A.value, 2)
        nt.assert_almost_equal(g.centre.value, 1)

    def test_estimate_parameters_unbinned(self):
        self.m.spectrum.metadata.Signal.binned = False
        s = self.m.as_signal(show_progressbar=None)
        s.metadata.Signal.binned = False
        g = hs.model.components.Gaussian()
        g.estimate_parameters(s,
                              None,
                              None,
                              only_current=True)
        nt.assert_almost_equal(g.sigma.value, 0.5)
        nt.assert_almost_equal(g.A.value, 2)
        nt.assert_almost_equal(g.centre.value, 1)


class TestExpression:

    def setUp(self):
        self.g = hs.model.components.Expression(
            expression="height * exp(-(x - x0) ** 2 * 4 * log(2)/ fwhm ** 2)",
            name="Gaussian",
            position="x0",
            height=1,
            fwhm=1,
            x0=0,
            module="numpy")

    def test_name(self):
        nt.assert_equal(self.g.name, "Gaussian")

    def test_position(self):
        nt.assert_is(self.g._position, self.g.x0)

    def test_f(self):
        nt.assert_equal(self.g.function(0), 1)

    def test_grad_height(self):
        nt.assert_almost_equal(
            self.g.grad_height(2),
            1.5258789062500007e-05)

    def test_grad_x0(self):
        nt.assert_almost_equal(
            self.g.grad_x0(2),
            0.00016922538587889289)

    def test_grad_fwhm(self):
        nt.assert_almost_equal(
            self.g.grad_fwhm(2),
            0.00033845077175778578)


class TestScalableFixedPattern:

    def setUp(self):
        s = hs.signals.Spectrum(np.linspace(0., 100., 10))
        s1 = hs.signals.Spectrum(np.linspace(0., 1., 10))
        s.axes_manager[0].scale = 0.1
        s1.axes_manager[0].scale = 0.1
        self.s = s
        self.pattern = s1

    def test_both_unbinned(self):
        s = self.s
        s1 = self.pattern
        s.metadata.Signal.binned = False
        s1.metadata.Signal.binned = False
        m = s.create_model()
        fp = hs.model.components.ScalableFixedPattern(s1)
        m.append(fp)
        with ignore_warning(message="invalid value encountered in sqrt",
                            category=RuntimeWarning):
            m.fit()
        nt.assert_almost_equal(fp.yscale.value, 100, delta=0.1)

    def test_both_binned(self):
        s = self.s
        s1 = self.pattern
        s.metadata.Signal.binned = True
        s1.metadata.Signal.binned = True
        m = s.create_model()
        fp = hs.model.components.ScalableFixedPattern(s1)
        m.append(fp)
        m.fit()
        nt.assert_almost_equal(fp.yscale.value, 100, delta=0.1)

    def test_pattern_unbinned_signal_binned(self):
        s = self.s
        s1 = self.pattern
        s.metadata.Signal.binned = True
        s1.metadata.Signal.binned = False
        m = s.create_model()
        fp = hs.model.components.ScalableFixedPattern(s1)
        m.append(fp)
        m.fit()
        nt.assert_almost_equal(fp.yscale.value, 1000, delta=1)

    def test_pattern_binned_signal_unbinned(self):
        s = self.s
        s1 = self.pattern
        s.metadata.Signal.binned = False
        s1.metadata.Signal.binned = True
        m = s.create_model()
        fp = hs.model.components.ScalableFixedPattern(s1)
        m.append(fp)
        m.fit()
        nt.assert_almost_equal(fp.yscale.value, 10, delta=.1)
