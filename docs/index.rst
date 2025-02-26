.. WaveSongs documentation master file, created by
   sphinx-quickstart on Wed Feb 12 22:42:12 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. https://pydata-sphinx-theme.readthedocs.io/en/stable/user_guide/styling.html

.. |check| raw:: html

    <input checked=""  type="checkbox">

.. |check_| raw:: html

    <input checked=""  disabled="" type="checkbox">

.. |uncheck| raw:: html

    <input type="checkbox">

.. |uncheck_| raw:: html

    <input disabled="" type="checkbox">


.. raw:: html
   
   <h1 class="h1-title">WaveSongs</h1>
   

.. container:: badges
   :name: badges

   |Version Package| |Python Version| |Open Source Love svg2| |GPLv3 license| |made-with-sphinx-doc| |Documentation Status|

.. |Version Package| image:: https://img.shields.io/badge/Version-0.0.6b-darkgreen.svg
   :target: .

.. |Python Version| image:: https://img.shields.io/badge/Python-≥3.10-blue?logo=python&logoColor=white
   :target: https://www.python.org/

.. |GPLv3 license| image:: https://img.shields.io/badge/License-GPLv3-blue.svg
   :target: http://perso.crans.org/besson/LICENSE.html

.. |Open Source Love svg2| image:: https://badges.frapsoft.com/os/v2/open-source.svg?v=103
   :target: https://github.com/ellerbrock/open-source-badges/

.. |made-with-sphinx-doc| image:: https://img.shields.io/badge/Made%20with-Sphinx-1f425f.svg
   :target: https://www.sphinx-doc.org/

.. |GitHub release| image:: https://img.shields.io/github/release/Naereen/StrapDown.js.svg
   :target: https://GitHub.com/wavesongs/wavesongs/releases/

.. |Documentation Status| image:: https://readthedocs.org/projects/ansicolortags/badge/?version=latest
   :target: http://ansicolortags.readthedocs.io/?badge=latest

.. raw:: html

   <hr style="margin: -2px 0 20px 0;">

**WaveSongs**  is a Python package designed to generate synthetic songs (currently focused on birdsongs) from audio recordings.

The package utilizes the `motor gestures model for birdsong <http://www.lsd.df.uba.ar/papers/simplemotorgestures.pdf>`_ developed by `Gabo Mindlin <https://scholar.google.com.ar/citations?user=gMzZPngAAAAJ&hl=en>`_ to generate synthetic birdsongs through numerical optimization :cite:p:`b-birdsongs_book,a-Amador2013`. By leveraging **fundamental frequency (FF)** and **spectral content index (SCI)** as key parameters. The package solves a minimization problem using `SciPy <https://docs.scipy.org/doc/scipy/tutorial/optimize.html>`_ :cite:p:`s-2020SciPy` and performs audio analysis with `librosa <https://librosa.org/>`_  :cite:p:`s-McFee2015librosa` and `scikit-maad <https://scikit-maad.github.io/>`_ :cite:p:`s-maad`.  This combination of tools enables the precise and realistic synthesis of birdsongs, achieving relative errors in fundamental frequency (FF) of less than 1%. [#f1]_

.. toctree::
   :maxdepth: 1
   :caption: Guides
   :hidden: 

   contents/Installation.md
   contents/DownloadSamples.ipynb
   contents/GettingStarted.ipynb
   contents/SpectrumMeasures.ipynb
   contents/SyntheticSongs.ipynb
   contents/PhysicalModel.md

⚒️ Installation
---------------

There are two ways to install wavesongs: a single line code installation via pypi, or a manual installation to get the latest  developer version. Check the :ref:`️installation` guide for more details.  

Now, let’s dive into the package! Check out the :ref:`getting_started` guide to learn how to analyze recordings and create synthetic syllables. You can download recording samples from the :ref:`download_samples` guide.

🗂️ Documentation
----------------

.. autosummary::
   :caption: API reference
   :toctree: _autosummary
   :recursive:

   wavesongs

.. </div>


🌱 Contribute
-------------

We welcome contributions! See our roadmap:

- |uncheck_| **Add ROIs analysis** using `scikit-maad`. This will allow automatic syllables detection and generation.
- |uncheck_| **Improve FF parametrization** for small motor gestures, chunks.
- |uncheck_| **Create a Machine Learning module** for data augmentation for synthetic songs.
- |uncheck_| **Add more physical models** for songs generation. Explore the sounds production in other animals as fish, frogs, frogs, etc. Animals who uses waves to geenrate song.
- |uncheck_| **Explore music concepts** (notes, acords, etc) and how to generate them with wavesongs.
- |uncheck_| **Smooth songs** at the extremes of the syllables to make the sound more natural.

To report issues or suggest features, open a `GitHub Issue <https://github.com/wavesongs/wavesongs/issues>`_. 

Do you need some other functionality? Let us know!

🔐 License
----------

**WaveSongs** is licensed under the `GNU General Public License v3.0 <https://github.com/wavesongs/wavesongs/blob/main/LICENSE>`_.


📒 Citation
-----------

If this work contributes to your research, please cite:

.. code-block:: bibtex
   
   @software{san_wavesongs_2025,
      author = {Aguilera Novoa, Sebastian},
      title = {WaveSongs: Computational Birdsong Synthesis},
      year = {2025},
      publisher = {GitHub},
      journal = {GitHub Repository},
      url = {https://github.com/wavesongs/wavesongs}
   }

📚 References
-------------

.. rubric:: Articles

.. bibliography:: references/articles.bib
   :keyprefix: a-
   :labelprefix: A


.. rubric:: Books

.. bibliography:: references/references.bib
   :keyprefix: b-
   :labelprefix: B

.. rubric:: Software
   
.. bibliography:: references/software.bib
   :all:
   :keyprefix: s-
   :labelprefix: S


.. rubric:: Footnotes

.. [#f1] The model performance depends on the syllable quaility and type. Complex syllables may have higher errors. The best performance is obtained in simple syllables well defined without noise and not strong harmonics.