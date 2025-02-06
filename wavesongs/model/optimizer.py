#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" """
# from model.song import Song
# from model.syllable import Syllable

import numpy as np
from time import time
from multiprocessing import Pool
from wavesongs.model.bird import set_params, set_z, _params
from IPython.display import display as Display

from numpy.typing import ArrayLike
from typing import (
    Any,
    AnyStr,
    Dict,
    Tuple,
    Optional,
    List
)

from scipy.optimize import (
    brute,
    basinhopping,
    differential_evolution,
    shgo,
    dual_annealing,
    direct,
    root,
)

__methods__ = [
    "brute",
    "basinhopping",
    "differential_evolution",
    "shgo",
    "dual_annealing",
    "direct",
    "fmin",
]

# trust regions ranges for minimization
_aplha_range = (0, 0.3)
_beta_range = (0, 0.8)
_gm_range = (1e4, 1e5)
_a0_range = (1e-3, 0.3)
_b0_range = (-1, 0.5)
_b1_range = (0, 2)
_b2_range = (0, 2)

# def _y(x, *params):
#   return params[0] + params[1]*x + params[2]**x**2


# def _beta_ranges(synth_syllable, params: Dict):
#     bmax = synth_syllable.beta.max()
#     bmin = synth_syllable.beta.min()


#     # mayor a la raíz encontrada
#     sol1 = root(_y, [-10,10], args=(params["b0"]-_beta_min, params["b1"], params["b2"]),
#                         method='hybr', jac=None, tol=0.0001, callback=None, options=None)
#     beta_min_roots = sol1.x
#     # menor a la raíz encontrada
#     sol2 = root(_y, [-10,10], args=(params["b0"]-_beta_max, params["b1"], params["b2"]),
#                         method='hybr', jac=None, tol=0.0001, callback=None, options=None)
#     beta_max_roots = sol1.x
# return sol1

# ==========================================================================
# --------------------------- Residual Functions ---------------------------
# ==========================================================================
# %%
def residual(z: List[float], *params: Tuple) -> ArrayLike:
    """


    Parameters
    ----------
        z : list [a0,b0,b1,b2]

        paramvs : tuple

    Return
    ------
        SCIFF: ArrayLike
            Fundamental Frequency and Spectral Content Index scores

    Examples
    --------
        >>>
    """
    syllable = params[-1]
    z = set_z(syllable, z)
    params = set_params(syllable, params)

    synth_syllable = syllable.solve(z, params)
    
    return synth_syllable.SCIFF  # + synth_syllable.scoreFF
    # scoreSCI +  syllable_synth.scoreFF

# %%
def residual_sci(z: List[float], *params: Tuple) -> ArrayLike:
    """


    Parameters
    ----------
        z : list [a0,b0,b1,b2]

        paramvs : tuple

    Return
    ------
        SCIFF: ArrayLike
            Fundamental Frequency and Spectral Content Index scores

    Examples
    --------
        >>>
    """
    syllable = params[-1]
    z = set_z(syllable, z)
    params = set_params(syllable, params)
    
    synth_syllable = syllable.solve(z, params)

    return synth_syllable.SCIFF  # scoreSCI +  syllable_synth.scoreFF


# %%
def residual_sci_a0(z: List[float], *params: Tuple) -> ArrayLike:
    """


    Parameters
    ----------
        z : list

        params : tuple

    Return
    ------
        SCIFF: ArrayLike
            Fundamental Frequency and Spectral Content Index scores

    Examples
    --------
        >>>
    """
    syllable = params[-1]
    z = {"a0": float(z[0])}
    params = set_params(syllable, params)

    synth_syllable = syllable.solve(z, params)

    return synth_syllable.scoreSCI  # syllable_synth.scoreFF


# %%
def residual_ff(z: List[float], *params: Tuple) -> ArrayLike:
    """


    Parameters
    ----------
        z : list

        params : tuple

    Return
    ------
        SCIFF: ArrayLike
            Fundamental Frequency and Spectral Content Index scores

    Examples
    --------
        >>>
    """
    syllable = params[-1]
    z = set_z(syllable, z)
    params = set_params(syllable, params)
    
    synth_syllable = syllable.solve(z, params)
    
    return synth_syllable.scoreFF  # + syllable_synth.scoreCentroid


# %%
def residual_ff_b02(z: List[float], *params: Tuple) -> ArrayLike:
    """


    Parameters
    ----------
        z : list

        params : tuple

    Return
    ------
        SCIFF: ArrayLike
            Fundamental Frequency and Spectral Content Index scores

    Examples
    --------
        >>>
    """
    syllable = params[-1]
    z = {"b0": float(z[0]), "b2": float(z[1])}
    params = set_params(syllable, params)
    
    synth_syllable = syllable.solve(z, params)
    
    return synth_syllable.scoreFF  # + syllable_synth.scoreCentroid


# %%
def residual_ff_b1(z: List[float], *params: Tuple) -> ArrayLike:
    """


    Parameters
    ----------
        z : list

        params : tuple

    Return
    ------
        SCIFF: ArrayLike
            Fundamental Frequency and Spectral Content Index scores

    Examples
    --------
        >>>
    """
    syllable = params[-1]
    z = {"b1": float(z[0])}
    params = set_params(syllable, params)

    synth_syllable = syllable.solve(z, params)

    return synth_syllable.scoreFF  # + syllable_synth.scoreCentroid


# %%
def residual_correlation(z: List[float], *params: Tuple) -> ArrayLike:
    """


    Parameters
    ----------
        z : list

        params : tuple

    Return
    ------
        SCIFF: ArrayLike
            Fundamental Frequency and Spectral Content Index scores

    Examples
    --------
        >>>
    """
    syllable = params[-1]
    z = set_z(syllable, z)
    params = set_params(syllable, params)

    synth_syllable = syllable.solve(z, params)

    return synth_syllable.residualCorrelation


# ==========================================================================
# --------------------------- Optimizer Functions --------------------------
# ==========================================================================
# ----------------
# %%
def optimal(
    syllable,
    params: Dict = _params,
    method: AnyStr = "brute",
    Ns: int = 20,
    full_output: bool = True,
    disp: bool = True,
    workers: int = -1,
) -> Dict:
    """


    Parameters
    ----------
        syllable : Syllable

        params : dict

        method : str = "brute"

        Ns : int, optional = 20

        full_output : bool, optional = False

        disp : bool, optional = False

        workers : int, optional = 1


    Return
    ------
        parameters: Dict


    Examples
    --------
        >>>
    """
    args = tuple(params.values()) + (syllable,)
    ranges = (_a0_range, _b0_range, _b1_range, _b2_range)
    start = time()
    if method == "brute":
        x0, fval, grid, Jout = brute(
            residual,
            ranges=ranges,
            args=args,
            Ns=Ns,
            full_output=full_output,
            disp=disp,
            workers=workers,
        )
    else:
        raise Exception(f"The method {method} does not exits.")
    end = time()
    tdiff = (end - start) / 60
    a0, b0, b1, b2 = x0
    print(
        f"\t Optimal values: a_0={a0:.4f}, b_0={b0:.4f}, b_1={b1:.4f},"
        + " b_2={b2:.4f}, t={tdiff:.2f} min"
    )
    syllable.set_z(x0)

    return syllable.z


# %%
def optimal_bs(
    syllable,
    params: Dict = _params,
    method: AnyStr = "brute",
    Ns: int = 20,
    full_output: bool = True,
    disp: bool = True,
    workers: int = -1,
) -> Dict:
    """


    Parameters
    ----------
        syllable : Syllable

        params : dict

        method : str = "brute"

        Ns : int, optional = 20

        full_output : bool, optional = False

        disp : bool, optional = False

        workers : int, optional = 1


    Return
    ------
        params: Dict


    Examples
    --------
        >>>
    """
    args = tuple(params.values()) + (syllable,)
    # ---------------- b0 and b2 --------------------
    ranges02 = (_b0_range, _b2_range)
    start02 = time()
    if method == "brute":
        x0, fval, grid, Jout = brute(
            residual_ff_b02,
            ranges=ranges02,
            args=args,
            Ns=Ns,
            full_output=full_output,
            disp=disp,
            workers=workers,
        )
    end02 = time()
    b0, b2 = x0
    tdiff = (end02 - start02) / 60
    print(
        f"\t Optimal values: b_0={b0:.4f}, b_2={b2:.4f}, t={tdiff:.2f} min"
    )
    syllable.z["b0"] = float(b0)
    syllable.z["b2"] = float(b2)
    # ---------------- b1--------------------
    ranges1 = (_b1_range,)
    start1 = time()
    if method == "brute":
        x0, fval, grid, Jout = brute(
            residual_ff_b1,
            ranges=ranges1,
            args=args,
            Ns=Ns,
            full_output=full_output,
            disp=disp,
            workers=workers,
        )
    else:
        raise Exception(f"The method {method} does not exits.")
    end1 = time()
    b1 = float(x0[0])
    print(
        f"\t Optimal values: b_1={b1:.4f}, t={(end1-start1)/60:.2f} min"
    )
    syllable.z["b1"] = b1

    return syllable.z


# %%
def optimal_a(
    syllable,
    params: Dict = _params,
    method: AnyStr = "brute",
    Ns: int = 20,
    full_output: bool = True,
    disp: bool = True,
    workers: int = -1,
) -> Dict:
    """


    Parameters
    ----------
        syllable : Syllable

        params : dict

        method : str = "brute"

        Ns : int, optional = 20

        full_output : bool, optional = False

        disp : bool, optional = False

        workers : int, optional = 1


    Return
    ------
        params: Dict


    Examples
    --------
        >>>
    """
    args = tuple(params.values()) + (syllable,)
    ranges = (_a0_range,)
    start = time()
    if method == "brute":
        x0, fval, grid, Jout = brute(
            residual_sci_a0,
            ranges=ranges,
            args=args,
            Ns=Ns,
            full_output=full_output,
            disp=disp,
            workers=workers,
        )
    else:
        raise Exception(f"The method {method} does not exits.")
    end = time()
    a0 = float(x0[0])
    print(f"\t Optimal values: a_0={a0:.4f}, t={(end-start)/60:.2f} min")
    syllable.z["a0"] = a0
    return syllable.z


# %%
def optimal_gamma(
    syllable,
    params: Dict = _params,
    method: AnyStr = "brute",
    Ns: int = 20,
    full_output: bool = True,
    disp: bool = True,
    workers: int = -1,
) -> Dict:
    """


    Parameters
    ----------
        syllable : Syllable

        params : dict

        method : str = "brute"

        Ns : int, optional = 20

        full_output : bool, optional = False

        disp : bool, optional = False

        workers : int, optional = 1


    Return
    ------
        parameters: Dict


    Examples
    --------
        >>>
    """
    args = tuple(params.values()) + (syllable,)
    ranges = _gm_range
    start = time()
    if method == "brute":
        x0, fval, grid, Jout = brute(
            residual_sci,
            ranges=ranges,
            args=args,
            Ns=Ns,
            full_output=full_output,
            disp=disp,
            workers=workers,
        )
    else:
        raise Exception(f"The method {method} does not exits.")
    end = time()
    gamma = x0[0]
    print(
        f"         Optimal values: γ* = {gamma:.0f}, t={(end-start)/60:.2f} min"
    )
    syllable.Z["gm"] = gamma

    return syllable.z


# %%
def optimal_params(
    syllable,
    params: Dict = _params,
    method: AnyStr = "brute",
    Ns: int = 20,
    full_output: bool = True,
    disp: bool = True,
    workers: int = -1,
) -> Dict:
    """


    Parameters
    ----------
        syllable : Syllable

        params : dict

        method : str = "brute"

        Ns : int, optional = 20

        full_output : bool, optional = False

        disp : bool, optional = False

        workers : int, optional = 1


    Return
    ------
        parameters: Dict


    Examples
    --------
        >>>
    """
    start = time()

    print("\nComputing a0*...")
    z_opt_a0 = optimal_a(
        syllable,
        params=params,
        method=method,
        Ns=Ns,
        full_output=full_output,
        disp=disp,
        workers=workers,
    )
    syllable.z = z_opt_a0

    print("\nComputing b0*, b1*, and b2*...")
    z_opt_b01 = optimal_bs(
        syllable,
        params=params,
        method=method,
        Ns=Ns,
        full_output=full_output,
        disp=disp,
        workers=workers,
    )

    syllable.z = z_opt_b01
    end = time()
    print(f"\nTime of execution: {(end-start)/60:.2f} min")

    return z_opt_b01


# %%
def optimal_params_general(
    syllable,
    params: Dict = _params,
    method: AnyStr = "brute",
    Ns: int = 20,
    full_output: bool = True,
    disp: bool = True,
    workers: int = -1,
) -> Dict:
    """


    Parameters
    ----------
        syllable : Syllable

        params : dict

        method : str = "brute"

        Ns : int, optional = 20

        full_output : bool, optional = False

        disp : bool, optional = False

        workers : int, optional = 1


    Return
    ------
        parameters: Dict


    Examples
    --------
        >>>
    """
    # args = tuple(params.values())+(syllable,)
    start = time()
    print("Computing optimal variables: a0*, b0*, b1*, and b2*...")
    z_opt_b01 = optimal(
        syllable,
        params=params,
        method=method,
        Ns=Ns,
        full_output=full_output,
        disp=disp,
        workers=workers,
    )
    syllable.z = z_opt_b01
    print("Finished")
    end = time()
    print(f"Time of execution = {(end-start)/60:.4f} min")

    return syllable.z


# %%
def all_optimal_gammas(bird):
    start = time()

    gammas = np.zeros(bird.no_syllables)
    for i in range(1, bird.no_syllables + 1):
        print(f"Syllable {i}/{bird.no_syllables}")
        syllable = bird.Syllable(i)
        gammas[i - 1] = optimal_gamma(syllable)

    syllable.optimal_gamma = np.mean(gammas)
    syllable.Gammas = gammas
    # syllable = syllable0
    # syllable.p["gm"].set(value=syllable.optimal_gamma, vary=False)
    end = time()
    print(f"Time of execution = {(end-start)/60:.4f} min")
    return syllable.optimal_gamma
