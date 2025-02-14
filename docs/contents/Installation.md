(️installation)=
# ⚒️ Installation

## Prerequisites

- [Python](https://www.python.org/) 3.10+
- [Git](https://git-scm.com/)

## Steps

### 1. Clone the repository:

   ```bash
   git clone https://github.com/wavesongs/wavesongs
   cd wavesongs
   ```

### 2. Set up a virtual environment (choose one method):

- **Using `venv`**

   ::::{tab-set}
   :sync-group: category

   :::{tab-item} Linux
   :sync: linux

   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

   :::

   :::{tab-item} Windows
   :sync: windows

   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```
   :::

   ::::


- **Using Conda**

   ```bash
   conda create -n wavesongs python=3.12
   conda activate wavesongs
   ```

### 3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

### 4. Install WaveSongs in editable mode:

   ```bash
   pip install -e .
   ```

(gettint-started)=
## 🚀 Gettint Started

Explore the [Tutorial 1 Notebook](https://github.com/wavesongs/wavesongs/blob/main/Tutorial1_Introduction.ipynb) to generate synthetic birdsongs and explore the model plots. 

For advanced usage (e.g., custom gestures, parameter tuning, data measures, etc), check the other tutorials: [Spectrum Measures](https://github.com/wavesongs/wavesongs/blob/main/Tutorial2_SpectrumMeasures.ipynb) or [Synthetic Songs](https://github.com/wavesongs/wavesongs/blob/main/Tutorial3_SyntheticSongs.ipynb). More details can be found in the [Documentation](https://wavesongs.github.io/doc).


## 🎶 Data Integration

Pre-processed field recordings from [Xeno Canto](https://xeno-canto.org/) and [eBird](https://ebird.org/home) are included in `./assets/audio`. To use custom recordings place `.wav` or `.mp3` files in `./assets/audio/` or define the audios path with the `ProjDirs` class.