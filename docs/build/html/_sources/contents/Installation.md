(️installation)=
# ⚒️ Installation

## Prerequisites

- [Python](https://www.python.org/) 3.10+
- [Git](https://git-scm.com/) (optional)
- [Conda](https://anaconda.org/anaconda/conda) (optional)

## Basic Installation

**WaveSongs** is available at [Pypi](https://pypi.org/). To install, run:

```bash
pip install wavesongs
```

:::::{admonition} Tip: create a Python environment
:class: tip

It is possible to use `pip` to install `wavesongs` outside of a virtual environment, but this is not recommended. Virtual environments create an isolated Python environment that does not interfere with your system's existing Python installation. They can be easily removed and contain only the specific package versions your application requires. Additionally, they help avoid a common issue known as "[dependency hell](https://en.wikipedia.org/wiki/Dependency_hell)", where conflicting package versions cause problems and unexpected behaviors.

It is highly recommended that you create a new virtual environment. This can be done in two ways, but you only need to choose one:

- **Python virtual environments**

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

- **Conda environments**

   ```bash
   conda create -n wavesongs-env python=3.12
   conda activate wavesongs-env
   ```

:::::

## Developer Installation 

To install the latest deveopment version from source clone the main repository from GitHub

```bash
git clone https://github.com/wavesongs/wavesongs
cd wavesongs # enter the cloned directory
```

Install the required dependencies

```bash
pip install -r requirements.txt
```

Install WaveSongs in editable mode:

```bash
pip install -e .
```

## OS Support

**WaveSongs** is developed and tested on Linux. It should also work on macOS and Windows. If you encounter a prooblem, please let me know by opening an issue or a pull request.ss
