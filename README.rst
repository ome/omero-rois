.. image:: https://travis-ci.org/ome/omero-rois.svg?branch=master
    :target: https://travis-ci.org/ome/omero-rois

.. image:: https://badge.fury.io/py/omero-rois.svg
    :target: https://badge.fury.io/py/omero-rois

OMERO ROI utilities
===================

OMERO Python utilities for handling regions of interest (ROIs).

Requirements
============

* OMERO 5.4.0 or newer
* Python 2.7

Installing from PyPI
====================

This section assumes that an OMERO.py is already installed.

Install the command-line tool using `pip <https://pip.pypa.io/en/stable/>`_::

    $ pip install -U omero-rois

Release process
---------------

This repository uses `bump2version <https://pypi.org/project/bump2version/>`_ to manage version numbers.
To tag a release run::

    $ bumpversion [major|minor|patch]

specifying ``major``, ``minor`` or ``patch`` depending on whether this is a `major, minor or patch release <https://semver.org/>`_. This will also remove the ``.dev0`` suffix.

To switch back to a development version (add ``.dev0`` suffix) run::

    $ bumpversion --no-tag dev


License
-------

This project, similar to many Open Microscopy Environment (OME) projects, is
licensed under the terms of the GNU General Public License (GPL) v2 or later.

Copyright
---------

2019, The Open Microscopy Environment
