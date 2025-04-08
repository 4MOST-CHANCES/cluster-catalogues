cluster-catalogues
==================

This repository contains the code used for `"CHANCES, the Chilean Cluster Galaxy Evolution Survey: Selection and initial characterisation of clusters and superclusters" <https://ui.adsabs.harvard.edu/abs/2024arXiv241113655S/abstract>`_
by Sifón, Finoguenov, Haines, Jaffé, et al.

Please contact `Cristóbal Sifón <https://github.com/cristobal-sifon>`_ for any questions with this repository or the public catalogues used.

Requirements
------------

Required pip-installable packages are in ``requirements.txt`` and can be installed by running

.. code-block::
    bash

    pip install -r requirements.txt

We recommend doing this inside a dedicated virtual environment (e.g., with ``conda``). Additionally, much of the code requires the `astro <https://github.com/cristobal-sifon/astro>`_ package, which must be installed manually (i.e., downloaded, unzipped, and installed with ``pip``). However,  ``astro`` requires locally-stored public catalogues which are not provided with this code but can be either obtained online or requested from Cristóbal Sifón.

Other data sets
---------------

The velocity reconstruction data used in Figure 4 can be requested from Helene Courtois and Alexandra Dupuy.

The redMaPPer catalogue used in Figures 5-9 can be requested from Johan Comparat.

Catalogue construction
----------------------

The file ``collate_info.py`` compiles the available information for each cluster from the public catalogues (which have to be stored locally where ``astro.clusters.ClusterCatalog`` can see them). Running

.. code-block::
    bash

    python collate_info.py lowz

will generate three files:

.. code-block::

    catalogues/clusters_chances_lowz_${date}.txt
    catalogues/clusters_chances_lowz_${date}.csv
    catalogues/clusters_chances_lowz_${date}_large.csv

The first two contain the same information but display it differently (with ``ascii.fixed_width`` and ``ascii.csv`` in ``astropy`` language). The last file contains all the cross-matched information (e.g., ID, redshift, M200) from all the queried public catalogues.

Figures
-------

**Figure 1:**

.. code-block::
    bash

    python sky_coverage.py 

**Figure 2:**

.. code-block::
    bash

    python compare_masses.py

**Figure 3:**

.. code-block::
    bash

    python plot_samples.py

**Figure 4:**

.. code-block::
    bash

    python plot_vpec.py

**Figures 5:**

.. code-block::
    bash

    python lss.py lowz
    python lss.py evolution

The first line will also produce **Figure 8**.

**Figures 6 and 7:**

.. code-block::
    bash

    python lss.py all

**Figure 9:**

.. code-block::
    bash

    python superclusters_eromapper.py lowz
