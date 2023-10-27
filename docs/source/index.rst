.. photonlib documentation master file, created by
   sphinx-quickstart on Wed Oct 26 19:52:12 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to photonlib's documentation!
===========================================
This is a software to interface a photon library file used for Liquid Argon Time Projection Chamber (LArTPC) detectors. For the installation, tutorial notebooks, and learning what photon library is, please see the `software repository <https://github.com/CIDeR-ML/photonlib>`_.


Getting started
---------------

You can find a quick guide to get started below.

Install ``photonlib``
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   git clone https://github.com/cider-ml/photonlib
   cd photonlib
   pip install . --user


You can install to your system path by omitting ``--user`` flag. 
If you used ``--user`` flag to install in your personal space, assuming ``$HOME``, you may have to export ``$PATH`` environment variable to find executables.

.. code-block:: bash
   
   export PATH=$HOME/.local/bin:$PATH

Download data file
^^^^^^^^^^^^^^^^^^

This interface software is only useful with a data file. You can download an example data file used for the ICARUS neutrino experiment.

.. code-block:: bash

   download_icarus_plib.sh

Using ``photonlib``
^^^^^^^^^^^^^^^^^^^

To use the interface, we need to create a ``PhotonLib`` class instance. The only parameter is a string value containing the path to the downloaded data file. Below, we assume you have downloaded the data file `plib.h5` in the current working directory.

.. code-block:: python

   from photonlib import PhotonLib
   plib = PhotonLib.load('./pllib.h5')


.


.. toctree::
   :maxdepth: 2
   :caption: Package Reference
   :glob:

   photonlib <photonlib>

.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`