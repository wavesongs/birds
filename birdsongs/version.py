"""Version info"""

import sys, importlib

short_version = "1.1.0"
__version__ = "1.1"


def __get_mod_version(modname):
    try:
        if modname in sys.modules: mod = sys.modules[modname]
        else:                      mod = importlib.import_module(modname)
        try:     return mod.__version__
        except   AttributeError:
            return "installed, no version number available"
    except ImportError: return None


def show_versions():
    """Return the version information for all birdsongs dependencies."""

    core_deps = [   "librosa",
                    "lmfit",
                    "scipy",
                    "sympy",
                    "numpy",
                    "pandas",
                    "matplotlib",
                    #"playsound",
                    "PeakUtils",
                    "scikit_learn",
                    "scikit_maad",
                    "ipython",
                    "pygobject"
                ]


    print("INSTALLED PAKCAGE VERSIONS")
    print("------------------")
    print("Python: {}\n".format(sys.version))
    print("Birdsongs (bs): {}\n".format(version))
    [print("{}: {}".format(dep, __get_mod_version(dep))) for dep in core_deps]