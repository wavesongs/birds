(️installation)=
# ⚒️ Installation

## Prerequisites

- [Python](https://www.python.org/) ≥ 3.10
- [Git](https://git-scm.com/) (optional)
- [Conda](https://anaconda.org/anaconda/conda) (optional)

## Basic Installation

**WaveSongs** is available at [Pypi](https://pypi.org/project/wavesongs/). To install the latest stable version, run:

```bash
pip install wavesongs
```

(python_env)=
:::::{admonition} Tip: create a dedicated Python environment
:class: tip

It is possible to use `pip` to install `wavesongs` outside of a virtual environment, but this is not recommended. Virtual environments create an isolated Python environment that does not interfere with your system's existing Python installation. They can be easily removed and contain only the specific package versions your application requires. Additionally, they help avoid a common issue known as "[dependency hell](https://en.wikipedia.org/wiki/Dependency_hell)", where conflicting package versions cause problems and unexpected behaviors.

It is highly recommended that you create a new virtual environment. This can be done in two ways, but you only need to choose one:

- **Python virtual environments**

   You can set the name you prefer, however, a good practice is to name the environment as `env` or `.env`.

   ::::{tab-set}
   :sync-group: category

   :::{tab-item} Linux
   :sync: linux

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

   :::

   :::{tab-item} Windows
   :sync: windows

   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate
   ```
   :::

   ::::

- **Conda environments**

   ```bash
   conda create -n wavesongs-env python=3.12 pip
   conda activate wavesongs-env
   ```

:::::

## Developer Installation 

First, create a Python environment as mentioned in [tip](#python_env). Then, to install the latest deveopment version from source clone the main repository from GitHub

```bash
git clone https://github.com/wavesongs/wavesongs
cd wavesongs # enter the cloned directory
```

Install the required dependencies

```bash
pip install -r requirements.txt
```

You can see all the required libraries in the [`requirements.txt`](https://github.com/wavesongs/wavesongs/blob/main/requirements.txt) file or in the [`pyproject.toml`](https://github.com/wavesongs/wavesongs/blob/main/pyproject.toml) file.

Finally, install **WaveSongs** in editable mode

```bash
pip install -e .
```

:::{note}
It is highly recommended to use Jupyter notebooks for an interactive development experience. However, Python and IPython terminals are also supported and can be used effectively.
:::

## OS Support

**WaveSongs** is developed and tested on Linux. It should also work on macOS and Windows. If you encounter a prooblem, please let us know by opening an issue or a pull request.