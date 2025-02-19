.. WaveSongs documentation master file, created by
   sphinx-quickstart on Wed Feb 12 22:42:12 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. https://pydata-sphinx-theme.readthedocs.io/en/stable/user_guide/styling.html

.. raw:: html
   
   <h1 class="h1-title">WaveSongs</h1>
   

.. container:: badges
   :name: badges

   |Version Package| |Python Version| |Open Source Love svg2| |GPLv3 license| |made-with-sphinx-doc| |Documentation Status|

.. |Version Package| image:: https://img.shields.io/badge/Version-0.0.3b-darkgreen.svg
   :target: .

.. |Python Version| image:: https://img.shields.io/badge/Python->=3.10-blue?logo=python&logoColor=white
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

**WaveSongs** implements the `motor gestures model for birdsong <http://www.lsd.df.uba.ar/papers/simplemotorgestures.pdf>`_ developed by `Gabo Mindlin <https://scholar.google.com.ar/citations?user=gMzZPngAAAAJ&hl=en>`_ to generate synthetic birdsongs through numerical optimization :cite:p:`b-birdsongs_book,a-Amador2013`. By leveraging **fundamental frequency (FF)** and **spectral content index (SCI)** as key parameters. The package solves a minimization problem using `SciPy <https://docs.scipy.org/doc/scipy/tutorial/optimize.html>`_ and performs audio analysis with `librosa <https://librosa.org/>`_. 

.. toctree::
   :maxdepth: 1
   :caption: Guides
   :hidden: 

   contents/Installation.md
   contents/PhysicalModel.md
   contents/DownloadSamples.ipynb
   contents/Introduction.ipynb
   contents/SpectrumMeasures.ipynb
   contents/SyntheticSongs.ipynb

🗂️ Documentation
----------------

.. autosummary::
   :caption: API reference
   :toctree: _autosummary
   :recursive:

   wavesongs

.. </div>

🔐 License
----------

WaveSongs is licensed under the `GNU General Public License v3.0 <https://github.com/wavesongs/wavesongs/blob/main/LICENSE>`_.

📒 Citation
-----------

If this work contributes to your research, please cite:

.. code-block:: python
   
   @software{aguilera_wavesongs_2025,
      author = {Aguilera Novoa, Sebastián},
      title = {WaveSongs: Computational Birdsong Synthesis},
      year = {2025},
      publisher = {GitHub},
      journal = {GitHub Repository},
      url = {https://github.com/wavesongs/wavesongs}
   }


🌱 Contribute
-------------

We welcome contributions! See our roadmap:

- [ ] **Integrate Xeno Canto API** for direct dataset downloads.
- [ ] **Add ROIs analysis** using `scikit-maad`. This will allo automatic syllables detection and gerenration.
- [ ] **Improve FF parametrization** for small motor gestures, chunks.

To report issues or suggest features, open a `GitHub Issue <https://github.com/wavesongs/wavesongs/issues>`_.



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
