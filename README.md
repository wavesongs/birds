<br>
<div align='center'>

<a href="https://github.com/saguileran/birdsongs/">
    <img src="" alt="birdsongs logo" title="Birdsongs" width="300px" style="padding-bottom:1em !important;" />
</a>

<br>
<br>

![version](https://img.shields.io/badge/package_version-1.0.0-orange)
![PyPI status](https://img.shields.io/pypi/status/ansicolortags.svg)
![license](https://img.shields.io/github/license/mashape/apistatus.svg)
![Open Source Love](https://img.shields.io/badge/open%20source-♡-lightgrey)
![Python 3.10](https://img.shields.io/badge/python->=3.10%20-blue.svg)
<a href="https://mybinder.org/v2/gh/wavesongs/wavesongs/main?labpath=test.ipynb"><img src="https://mybinder.org/badge_logo.svg"></a>

**WaveSongs** A python package for birdsongs creation and data extraction.

[Overview](#overview) •
[Installation](#installation) •
[Getting Started](#getting-started) •
[References](#references)

</div>


## Overview

A python package for birdsongs creation and data extraction.

## Installation

### Requirments

`wavesong` is implemented in Python 3.10 or higher. The required libraries are listed in the [requirements.txt](./requirements.txt) file.
    
### Environment

As good practice, create an evironment for the project.

#### Python

```bash
sudo apt install python3-venv    # install virtual environments
python3 -m venv ./venv           # create venv
source ./venv/bin/activate       # activate venv
```

If you are working on Windows, activate the venv with ```source /venv/bin/Activate```.

#### Conda

```bash
conda create -n wavesongs python=3.12
conda activate wavesongs
```

#### Docker

In process...

```bash
docker build -t wavesongs .
docker run -p 8888:8888 wavesongs
```

### WaveSongs

First of all, clone the the **wavesongs** repository and enter to it.

```bat
git clone https://github.com/wavesongs/wavesongs
cd wavesongs
```

Next, install the package and the required libraries. There are two options:

1. `pip`:

    ```bash
    pip install -e .
    ``` 

2. [`poetry`](https://python-poetry.org/):


    ```bat
    sudo apt update
    sudo apt install pipx
    pipx ensurepath
    ```

    For more information visit [Installing pipx](https://pipx.pypa.io/stable/installation/#installing-pipx). Then, install poetry with `pipx install poetry`. 
    
    Finally, install **wavesongs** with: 

    ```bat
    poetry install
    ```

If you are using an IDE, reset it. 

That's all!. Now let's create a synthetic song!

## Getting Started

See the [tutorial](./Tutorial.ipynb) for a complete use guide.

## Datasets


## License

The project is licensed under the [License](./LICENSE)-

## Citation

If you use `wavesongs` in your own work, please cite the associated article and/or the repository:

## References

### Literature

- 2013 Mindlin

### Software

- Librosa
- Scipy
- yin