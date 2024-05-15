Usage
=====

.. _installation:

Installation
------------

To install, download the .whl file from the releases page and install it locally with pip (eg: pip install PATH_TO_FILE/FILENAME). You will want to install to it to a virtual environment. As a heads up, opencb relies on nvidia cuda - if your machine does not have a dedicated gpu, things will run very slowly.

.. _usage:

Usage
-----

Currently (as of05-14-2024) basic usage of the library is pretty simple: Open a python shell or script, import the opencb module, and run the test011() function from routines.iteration. Your code might look like this:

.. py:function::
   from opencb.routines.iteration import test011

   test011()

This will start a test function for automating the development of a working model. It runs on the gpu using cuda functionality. As of this writing there's no fallback mechanism yet (and to be fair, you wouldn't want there to be one, as it runs incredibly slow on the cpu). The test011() function will continue until you tell it to stop or it develops a model that can win the game it's playing. It's goal is to move a dot in an image from one point to another consistently. Only time will tell whether it can actually achieve that goal.
