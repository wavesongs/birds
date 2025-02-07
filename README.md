
<div class="container" style="display: flex; align-items: center; justify-content: space-between; max-width: 70%; min-width: 400px; margin: 0 auto; padding: 10px 0 15px 0; border-bottom: 1px solid">
    <div class="image-container" style="flex: 1; padding: 0 0 0 10px;">
        <img src="./assets/logo.png" alt="WaveSongs logo" style="max-width: 100%; height: auto; display: block;">
    </div>
    <div class="text-container" style="flex: 2;">
        <h1 style="margin: 0; padding: 0 0 5px 0; border-bottom: 0">WaveSongs</h1>
        <p style="margin: 0; padding: 5px 0 0 0;">A Python package for birdsong synthesis and bioacoustic analysis</p>
    </div>
</div>

<div align='center' style="margin: 20px 0 0px 0">

![version](https://img.shields.io/badge/version-1.0.0-008000)
[![License: GPL v3](https://img.shields.io/badge/license-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
![Python 3.10+](https://img.shields.io/badge/python->=3.10-blue.svg)
![Open Source](https://img.shields.io/badge/open%20source-♡-lightgrey)

[Overview](#overview) •
[Installation](#installation) •
[Quick Start](#quick-start) •
[Contribute](#-contribute) •
[References](#-references)

</div>


## 🔎 Overview

WaveSongs implements the [motor gestures model for birdsong](http://www.lsd.df.uba.ar/papers/simplemotorgestures.pdf) developed by [Gabo Mindlin](https://scholar.google.com.ar/citations?user=gMzZPngAAAAJ&hl=en) to generate synthetic birdsongs through numerical optimization. By leveraging **fundamental frequency (FF)** and **spectral content index (SCI)** as key parameters, the package solves a minimization problem using [SciPy](https://docs.scipy.org/doc/scipy/tutorial/optimize.html) and performs audio analysis with [librosa](https://librosa.org/).

Validated against field recordings of *Zonotrichia Capensis*, *Ocellated Tapaculo*, and *Mimus Gilvus*, the model achieves **<5% relative error in FF reconstruction** compared to empirical data.


## ⚒️ Installation

### Prerequisites
- [Python](https://www.python.org/) ≥ 3.10
- [Git](https://git-scm.com/)

### Steps
1. **Clone the repository**:
   ```bash
   git clone https://github.com/wavesongs/wavesongs
   cd wavesongs
   ```

2. **Set up a virtual environment** (choose one method):

   #### Using `venv`
   ```bash
   python -m venv venv
   # Activate on Linux/macOS
   source venv/bin/activate
   # Activate on Windows
   .\venv\Scripts\activate
   ```

   #### Using Conda
   ```bash
   conda create -n wavesongs python=3.12
   conda activate wavesongs
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install WaveSongs** in editable mode:
   ```bash
   pip install -e .
   ```


## 🚀 Gettint Started

Explore the [Tutorial Notebook](./Tutorial.ipynb) to generate synthetic birdsongs and analyze acoustic features. Here is an example of simple code to import and display an audio.

```python
# select matplotlib backend for notebook, enable interactive plots
%matplotlib ipympl

from wavesongs.utils.paths import ProjDirs       # project files manager
from wavesongs.objects.song import Song          # song objects
from wavesongs.objects.syllable import Syllable  # syllable objects
from wavesongs.utils import plots                # plotter

proj_dirs = ProjDirs(audios="./assets/audio", results="./assets/results")

# define the song and compute its acoustical features
copeton_song = Song(proj_dirs, file_id="574179401")
copeton_song.acoustical_features(umbral_FF=1.4, NN=256)

# display the song
plots.spectrogram_waveform(copeton_song, save=False)
```

![Sample Output](./assets/results/images/574179401%20-%20Zonotrichia%20Capensis-Song.png)

For advanced usage (e.g., custom gestures, parameter tuning), refer to the [Documentation](./docs/).


## 🎶 Data Integration

Pre-processed field recordings from [Xeno Canto](https://xeno-canto.org/) and [eBird](https://ebird.org/home) are included in `./assets/audio`. To use custom recordings place `.wav` or `.mp3` files in `./assets/audio/` or define the audios path with the `ProjDirs` class.


## 📜 License

WaveSongs is licensed under the [GNU General Public License v3.0](./LICENSE).

## 📒 Citation

If this work contributes to your research, please cite:

```bibtex
@software{aguilera_wavesongs_2025,
    author = {Aguilera Novoa, Sebastián},
    title = {WaveSongs: Computational Birdsong Synthesis},
    year = {2025},
    publisher = {GitHub},
    journal = {GitHub Repository},
    url = {https://github.com/wavesongs/wavesongs}
}
```


## 🌱 Contribute

We welcome contributions! See our roadmap:

- [ ] **Integrate Xeno Canto API** for direct dataset downloads
- [ ] **Add ROIs analysis** using `scikit-maad`
- [ ] **Improve FF parametrization** for non-linear gestures

To report issues or suggest features, open a [GitHub Issue](https://github.com/wavesongs/wavesongs/issues).


## 📚 References

### Core Methodology
1. Mindlin, G. B., & Laje, R. (2005). *The Physics of Birdsong*. Springer.  
   [DOI](https://doi.org/10.1007/3-540-28249-1)
2. Amador, A., et al. (2013). Elemental gesture dynamics in song premotor neurons. *Nature*.  
   [DOI](https://doi.org/10.1038/nature11967)

### Software
- [Librosa](https://librosa.org/) • Audio analysis
- [SciPy](https://scipy.org/) • Optimization routines
- [scikit-maad](https://github.com/scikit-maad/scikit-maad) • Soundscape metrics

### Data Sources
- [Xeno Canto](https://xeno-canto.org/) • Field recordings
- [eBird](https://ebird.org/) • Species metadata

