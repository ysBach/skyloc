Installation
============

Requirements
------------

skyloc requires Python ≥ 3.10 and the following core dependencies:

- ``numpy``
- ``numba``
- ``pandas``
- ``astropy``
- ``spiceypy``
- ``requests``
- ``astroquery``
- ``healpy``
- ``pyarrow``
- ``click``
- ``pqfilt``

Install from PyPI
-----------------

.. code-block:: bash

   pip install skyloc

Install from source
-------------------

.. code-block:: bash

   git clone https://github.com/ysBach/skyloc.git
   cd skyloc
   pip install -e ".[dev]"

Optional Extras
---------------

**kete** — n-body orbit propagation (Caltech IPAC version):

.. code-block:: bash

   pip install skyloc[kete]

   # or from source (v1.x required, NOT v2+):
   pip install git+https://github.com/Caltech-IPAC/kete.git

**plotly** — interactive HTML ephemeris plots:

.. code-block:: bash

   pip install plotly

Documentation build
-------------------

To build the documentation locally:

.. code-block:: bash

   pip install sphinx sphinx_rtd_theme sphinx-copybutton numpydoc
   cd docs
   make html

The built pages will be in ``docs/_build/html/``.
