
Electron Energy Loss Spectroscopy
*********************************

.. _eels_tools-label:

Tools for EELS data analysis
----------------------------

The functions described in this chapter are only available for the
:py:class:`~._signals.eels.EELSSpectrum` class. To transform a
:py:class:`~.signal.Signal` (or subclass) into a
:py:class:`~._signals.eels.EELSSpectrum`:

.. code-block:: python
       
    >>> s.set_signal_type("EELS")

Note these chapter discusses features that are available only for
:py:class:`~._signals.eels.EELSSpectrum` class. However, this class inherits
many useful feature from its parent class that are documented in previous
chapters.


Elemental composition of the sample
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It can be useful to define the elemental composition of the sample for
archiving purposes or to use some feature (e.g. curve fitting) that requieres
this information.  The elemental composition of the sample can be declared
using :py:meth:`~._signals.eels.EELSSpectrum.add_elements`. The information is
stored in the :py:attr:`~.signal.Signal.metadata` attribute (see
:ref:`metadata_structure`). This information is saved to file when saving in
the hdf5 format.

Thickness estimation
^^^^^^^^^^^^^^^^^^^^

The :py:meth:`~._signals.eels.EELSSpectrum.estimate_thickness` can estimate the
thickness from a low-loss EELS spectrum using the Log-Ratio method.

Zero-loss peak centre and alignment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :py:meth:`~._signals.eels.EELSSpectrum.estimate_zero_loss_peak_centre` can be used to estimate the position of the zero-loss peak. The method assumes that the ZLP is the most intense feature in the spectra. For a more general approach see :py:meth:`~.signal.Signal1DTools.find_peaks1D_ohaver`.

The :py:meth:`~._signals.eels.EELSSpectrum.align_zero_loss_peak` can
align the ZLP with subpixel accuracy. It is more robust and easy to use than
:py:meth:`~.signal.Signal1DTools.align1D` for the task. Note that it is possible to apply the same alignment to other spectra using the `also_align` argument. This can be useful e.g. to align core-loss spectra acquired quasi-simultaneously.


Deconvolutions
^^^^^^^^^^^^^^

Three deconvolution methods are currently available:

* :py:meth:`~._signals.eels.EELSSpectrum.fourier_log_deconvolution`
* :py:meth:`~._signals.eels.EELSSpectrum.fourier_ratio_deconvolution`
* :py:meth:`~._signals.eels.EELSSpectrum.richardson_lucy_deconvolution`

Estimate elastic scattering intensity
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The
:py:meth:`~._signals.eels.EELSSpectrum.estimate_elastic_scattering_intensity`
can be used to calculate the integral of the zero loss peak (elastic intensity)
from EELS low-loss spectra containing the zero loss peak using the
(rudimentary) threshold method. The threshold can be global or spectrum-wise.
If no threshold is provided it is automatically calculated using
:py:meth:`~._signals.eels.EELSSpectrum.estimate_elastic_scattering_threshold`
with default values.

:py:meth:`~._signals.eels.EELSSpectrum.estimate_elastic_scattering_threshold`
can be used to  calculate separation point between elastic and inelastic
scattering on EELS low-loss spectra. This algorithm calculates the derivative
of the signal and assigns the inflexion point to the first point below a
certain tolerance.  This tolerance value can be set using the `tol` keyword.
Currently, the method uses smoothing to reduce the impact of the noise in the
measure. The number of points used for the smoothing window can be specified by
the npoints keyword.


.. _eels.kk:

Kramers-Kronig Analysis
^^^^^^^^^^^^^^^^^^^^^^^

.. versionadded:: 0.7

The single-scattering EEL spectrum is approximately related to the complex
permittivity of the sample and can be estimated by Kramers-Kronig analysis.
The :py:meth:`~._signals.eels.EELSSpectrum.kramers_kronig_analysis` method
inplements the Kramers-Kronig FFT method as in [Egerton2011]_ to estimate the
complex dielectric funtion from a low-loss EELS spectrum. In addition, it can
estimate the thickness if the refractive index is known and approximately
correct for surface plasmon excitations in layers.




EELS curve fitting
------------------

HyperSpy makes it really easy to quantify EELS core-loss spectra by curve
fitting as it is shown in the next example of quantification of a boron nitride
EELS spectrum from the `The EELS Data Base
<http://pc-web.cemes.fr/eelsdb/index.php?page=home.php>`_. 

Load the core-loss and low-loss spectra


.. code-block:: python
       
    >>> s = load("BN_(hex)_B_K_Giovanni_Bertoni_100.msa")
    >>> ll = load("BN_(hex)_LowLoss_Giovanni_Bertoni_96.msa")


Set some important experimental information that is missing from the original
core-loss file

.. code-block:: python
       
    >>> s.set_microscope_parameters(beam_energy=100, convergence_angle=0.2, collection_angle=2.55)
    
    
Define the chemical composition of the sample

.. code-block:: python
       
    >>> s.add_elements(('B', 'N'))
    
    
We pass the low-loss spectrum to :py:func:`~.hspy.create_model` to include the
effect of multiple scattering by Fourier-ratio convolution.

.. code-block:: python
       
    >>> m = create_model(s, ll=ll)


HyperSpy has created the model and configured it automatically:

.. code-block:: python
       
    >>> m
    [<background (PowerLaw component)>,
    <N_K (EELSCLEdge component)>,
    <B_K (EELSCLEdge component)>]


Furthermore, the components are available in the user namespace

.. code-block:: python

    >>> N_K
    <N_K (EELSCLEdge component)>
    >>> B_K
    <B_K (EELSCLEdge component)>
    >>> background
    <background (PowerLaw component)>


Conveniently, variables named as the element symbol contain all the eels
core-loss components of the element to facilitate applying some methods to all
of them at once. Although in this example the list contains just one component
this is not generally the case.

.. code-block:: python
       
    >>> N
    [<N_K (EELSCLEdge component)>]


By default the fine structure features are disabled (although the default value
can be configured (see :ref:`configuring-hyperspy-label`). We must enable them
to accurately fit this spectrum.

.. code-block:: python
       
    >>> m.enable_fine_structure()


We use smart_fit instead of standard fit method because smart_fit is optimized
to fit EELS core-loss spectra

.. code-block:: python
       
    >>> m.smart_fit()


This fit can also be applied over the entire signal to fit a whole spectrum image

.. code-block:: python

    >>> m.multifit(kind='smart')


Print the result of the fit 

.. code-block:: python

    >>> m.quantify()
    Absolute quantification:
    Elem.	Intensity
    B	0.045648
    N	0.048061


Visualize the result

.. code-block:: python

    >>> m.plot()
    

.. figure::  images/curve_fitting_BN.png
   :align:   center
   :width:   500    

   Curve fitting quantification of a boron nitride EELS core-loss spectrum from
   `The EELS Data Base
   <http://pc-web.cemes.fr/eelsdb/index.php?page=home.php>`_
   

There are several methods that are only available in
:py:class:`~.models.eelsmodel.EELSModel`:

* :py:meth:`~.models.eelsmodel.EELSModel.smart_fit` is a fit method that is 
  more robust than the standard routine when fitting EELS data.
* :py:meth:`~.models.eelsmodel.EELSModel.quantify` prints the intensity at 
  the current locations of all the EELS ionisation edges in the model.
* :py:meth:`~.models.eelsmodel.EELSModel.remove_fine_structure_data` removes 
  the fine structure spectral data range (as defined by the 
  :py:attr:`~._components.eels_cl_edge.EELSCLEdge.fine_structure_width)` 
  ionisation edge components. It is specially useful when fitting without 
  convolving with a zero-loss peak.

The following methods permit to easily enable/disable background and ionisation
edges components:

* :py:meth:`~.models.eelsmodel.EELSModel.enable_edges`
* :py:meth:`~.models.eelsmodel.EELSModel.enable_background`
* :py:meth:`~.models.eelsmodel.EELSModel.disable_background`
* :py:meth:`~.models.eelsmodel.EELSModel.enable_fine_structure`
* :py:meth:`~.models.eelsmodel.EELSModel.disable_fine_structure`

The following methods permit to easily enable/disable several ionisation 
edge functionalities:

* :py:meth:`~.models.eelsmodel.EELSModel.set_all_edges_intensities_positive`
* :py:meth:`~.models.eelsmodel.EELSModel.unset_all_edges_intensities_positive`
* :py:meth:`~.models.eelsmodel.EELSModel.enable_free_onset_energy`
* :py:meth:`~.models.eelsmodel.EELSModel.disable_free_onset_energy`
* :py:meth:`~.models.eelsmodel.EELSModel.fix_edges`
* :py:meth:`~.models.eelsmodel.EELSModel.free_edges`
* :py:meth:`~.models.eelsmodel.EELSModel.fix_fine_structure`
* :py:meth:`~.models.eelsmodel.EELSModel.free_fine_structure`


When fitting edges with fine structure enabled it is often desirable that the
fine structure region of nearby ionization edges does not overlap. HyperSpy
provides a method,
:py:meth:`~.models.eelsmodel.EELSModel.resolve_fine_structure`, to
automatically adjust the fine structure to prevent fine structure to avoid
overlapping. This method is executed automatically when e.g. components are
added or removed from the model, but sometimes is necessary to call it
manually.

.. versionadded:: 0.7.1

   Sometimes it is desirable to disable the automatic adjustment of the fine
   structure width. It is possible to suspend this feature by calling
   :py:meth:`~.models.eelsmodel.EELSModel.suspend_auto_fine_structure_width`.
   To resume it use
   :py:meth:`~.models.eelsmodel.EELSModel.suspend_auto_fine_structure_width`
