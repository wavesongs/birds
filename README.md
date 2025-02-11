<div align='center' style="margin: 20px 0 0px 0">
   <img src="./assets/logo.png" alt="WaveSongs logo" style="max-width: 100%; height: 200px;">

   <div class="text-container" style="flex: 2;">
      <h1 style="margin: 0; padding: 10px 0 0px 0; border-bottom: 0">WaveSongs</h1>
      <p style="margin: 0; padding: 0px 0 10px 0;">A Python package for birdsong synthesis and bioacoustic analysis</p>
   </div> 
</div>

<div align='center' style="margin: 20px 0 50px 0">

![version](https://img.shields.io/badge/version-1.0.1-008000)
[![License: GPL v3](https://img.shields.io/badge/license-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
![Python 3.10+](https://img.shields.io/badge/python->=3.10-blue.svg)
![Open Source](https://img.shields.io/badge/open%20source-♡-lightgrey)

[Overview](#-overview) •
[Installation](#️-installation) •
[Quick Start](#-gettint-started) •
[Contribute](#-contribute) •
[References](#-references)

</div>

---

## 🔎 Overview

WaveSongs implements the [motor gestures model for birdsong](http://www.lsd.df.uba.ar/papers/simplemotorgestures.pdf) developed by [Gabo Mindlin](https://scholar.google.com.ar/citations?user=gMzZPngAAAAJ&hl=en) to generate synthetic birdsongs through numerical optimization [[1](#1), [2](#2)] 
. By leveraging **fundamental frequency (FF)** and **spectral content index (SCI)** as key parameters, the package solves a minimization problem using [SciPy](https://docs.scipy.org/doc/scipy/tutorial/optimize.html) and performs audio analysis with [librosa](https://librosa.org/).

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

Explore the [Tutorial 1 Notebook](https://github.com/wavesongs/wavesongs/blob/main/Tutorial1_Introduction.ipynb) to generate synthetic birdsongs and explore the model plots. 

Here is an example of simple code to generate and display a sythetic audio. First, start by loading the wavesongs package:

```python
# select matplotlib backend for notebook, enable interactive plots, just works on notebooks
%matplotlib ipympl

from wavesongs.utils.paths import ProjDirs       # Project files manager
from wavesongs.objects.syllable import Syllable  # Syllable objects
from wavesongs.utils import plots                # Display plots
```

Then, create a project directory manager, select a region of interest, and define the song for study. You can display it with the plots functions.
```python
proj_dirs = ProjDirs(audios="./assets/audio", results="./assets/results")

# Region of Interest
tlim = (0.8798, 1.3009)

# Define the syllable
copeton_syllable_0 = Syllable(
   proj_dirs=proj_dirs, file_id="574179401", obj=copeton_syllable,
   tlim=tlim, type="intro-down", no_syllable="0", sr=44100
)
copeton_syllable_0.acoustical_features(
   umbral_FF=1.4, NN=256, ff_method="yin", flim=(1e2, 2e4)
)

# Display the syllable's spectrogram
plots.spectrogram_waveform(copeton_syllable_0, ff_on=True, save=True)
```

<a href="https://github.com/wavesongs/wavesongs/blob/main/assets/results/images/574179401-ZonotrichiaCapensis-0-intro-down.png">
<figure>
    <img src='https://github.com/wavesongs/wavesongs/blob/main/assets/results/images/574179401-ZonotrichiaCapensis-0-intro-down.png' alt='Sample motor gesture output' width=70% style="display: block; margin: 0 auto 0 auto;"/>
    <figcaption style="text-align: center;"><b><a id="figure1" style="color:#318bf8;">Figure 1</a></b>: Waveform and spectrogram of the audio with id 574179401.</figcaption>
</figure>
</a>

```python
copeton_syllable_0.play() # just work on notebooks
```

https://github.com/user-attachments/assets/d15e7433-5f4c-451f-85aa-d4d53525029f

Now, let's find the optimal values to generate a comparable syllable, with errors below 5 % or even 1%.

```python
from wavesongs.model import optimizer

optimal_z = optimizer.optimal_params(
   syllable=copeton_syllable_0, Ns=10, full_output=False
)
print(f"\nOptimal z values:\n\t{optimal_z}")
```
```text
Computing a0*...
	 Optimal values: a_0=0.0010, t=0.51 min

Computing b0*, b1*, and b2*...
	 Optimal values: b_0=-0.2149, b_2=1.2980, t=13.77 min
	 Optimal values: b_1=1.0000, t=5.69 min

Time of execution: 19.97 min

Optimal z values:
	{'a0': 0.00105, 'b0': -0.21491, 'b1': 1.0, 'b2': 1.29796}
```

With the optimal values, define and dislpay the synthetic syllable:
```python
# Define the synthetic syllable
synth_copeton_syllable_0 = copeton_syllable_0.solve(z=optimal_z, method="best")
plots.spectrogram_waveform(synth_copeton_syllable_0, ff_on=True, save=True)
# Display the socre variables
plots.scores(copeton_syllable_0, synth_copeton_syllable_0, save=False)
```
<a href="https://raw.githubusercontent.com/wavesongs/wavesongs/refs/heads/main/assets/results/images/574179401-ZonotrichiaCapensis-0-intro-down-ScoringVariables.png">
<figure>
    <img src='https://raw.githubusercontent.com/wavesongs/wavesongs/refs/heads/main/assets/results/images/574179401-ZonotrichiaCapensis-0-intro-down-ScoringVariables.png' alt='Sample motor gesture output' width=70% style="display: block; margin: 0 auto 0 auto;"/>
    <figcaption style="text-align: center;"><b><a id="figure2" style="color:#318bf8;">Figure 2</a></b>: Scoring variables realtive errores.</figcaption>
</figure>
</a>

```python
plots.motor_gestures(synth_copeton_syllable_0, save=False)
```
<a href="https://raw.githubusercontent.com/wavesongs/wavesongs/refs/heads/main/assets/results/images/synth_574179401-ZonotrichiaCapensis-0-intro-down-mg_params.png">
<figure>
    <img src='https://raw.githubusercontent.com/wavesongs/wavesongs/refs/heads/main/assets/results/images/synth-574179401-ZonotrichiaCapensis-0-intro-down-mg_params.png' alt='Sample motor gesture output' width=70% style="display: block; margin: 0 auto 0 auto;"/>
    <figcaption style="text-align: center;"><b><a id="figure3" style="color:#318bf8;">Figure 3</a></b>: Motor gesture, model parameters curves.</figcaption>
</figure>
</a>

```python
plots.syllables(copeton_syllable_0, synth_copeton_syllable_0, save=False)
```
<a href="https://raw.githubusercontent.com/wavesongs/wavesongs/refs/heads/main/assets/results/images/574179401-ZonotrichiaCapensis-0-intro-down-SoundAndSpectros.png">
<figure>
    <img src='https://raw.githubusercontent.com/wavesongs/wavesongs/refs/heads/main/assets/results/images/574179401-ZonotrichiaCapensis-0-intro-down-SoundAndSpectros.png' alt='Sample motor gesture output' width=70% style="display: block; margin: 0 auto 0 auto;"/>
    <figcaption style="text-align: center;"><b><a id="figure4" style="color:#318bf8;">Figure 4</a></b>: Real and synthetic syllables.</figcaption>
</figure>
</a>

```python
synth_copeton_syllable_0.play() # just work on notebooks
```

https://github.com/user-attachments/assets/66ca1630-0ad0-43fc-bb56-cb397064ecd3

For advanced usage (e.g., custom gestures, parameter tuning, data measures, etc), check the other tutorials: [Spectrum Measures](https://github.com/wavesongs/wavesongs/blob/main/Tutorial2_SpectrumMeasures.ipynb) or [Synthetic Songs](https://github.com/wavesongs/wavesongs/blob/main/Tutorial3_SyntheticSongs.ipynb). More details can be found in the [Documentation](https://wavesongs.github.io/doc).


## 🎶 Data Integration

Pre-processed field recordings from [Xeno Canto](https://xeno-canto.org/) and [eBird](https://ebird.org/home) are included in `./assets/audio`. To use custom recordings place `.wav` or `.mp3` files in `./assets/audio/` or define the audios path with the `ProjDirs` class.


## 🔐 License

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
- [ ] **Improve FF parametrization** for small motor gestures

To report issues or suggest features, open a [GitHub Issue](https://github.com/wavesongs/wavesongs/issues).


## 📚 References

### Core Methodology

<a id="1" style="color:#318bf8;">[1]</a>  Mindlin, G. B., & Laje, R. (2005). *The Physics of Birdsong*. Springer. [DOI](https://doi.org/10.1007/3-540-28249-1)

<a id="1" style="color:#318bf8;">[2]</a>  Amador, A., et al. (2013). Elemental gesture dynamics in song premotor neurons. *Nature*. [DOI](https://doi.org/10.1038/nature11967)


### Software
- [Librosa](https://librosa.org/) • Audio analysis
- [SciPy](https://scipy.org/) • Optimization routines
- [scikit-maad](https://github.com/scikit-maad/scikit-maad) • Soundscape metrics

### Data Sources
- [Xeno-Canto](https://xeno-canto.org/)
- [eBird](https://ebird.org/)
