"""
Methods to implement the motor gesture model for birdsongs.
"""
import numpy as np
import pandas as pd

from sympy import (
    symbols,
    lambdify,
    solveset
)

from copy import deepcopy
from maad.sound import normalize
from numpy.polynomial import Polynomial
from pathlib import Path

from wavesongs.utils.tools import (
    envelope,
    rk4
)

from numpy.typing import ArrayLike, DTypeLike
from typing import (
    List,
    Dict,
    Any,
    Union,
    Tuple,
    AnyStr,
    Literal,
    TypeVar
)

Syllable = TypeVar('Syllable')


# Defining motor gestures model constants, measured by Gabo Mindlin
_PARAMS = {
    "gm": 4e4,       # time scaling constant
    # -------------------------------- Trachea --------------------------------
    "C": 343,        # speed of sound in media [m/s]
    "L": 0.025,      # trachea length [m]
    "r": 0.65,       # reflection coeficient [adimensionelss]
    # ------------------------- Beak, Glottis and OEC -------------------------
    "Ch": 1.43E-10,  # OEC Compliance [m^3/Pa]
    "MG": 20,        # Beak Inertance [Pa s^2/m^3 = kg/m^4]
    "MB": 1E4,       # Glottis Inertance [Pa s^2/m^3 = kg/m^4]
    "RB": 5E6,       # Beak Resistance [Pa s/m^3 = kg/m^4 s]
    "Rh": 24E3       # OEC Resistence [Pa s/m^3 = kg/m^4 s]
}
r"""dict : Model parameters

.. table:: Birdsongs model parameters :cite:p:`a-Amador2013`.
    :width: 80%
    :widths: 2 6 3 3

    ==============  ========================  =======  ====================
    Constant        Description               Value     Unit     
    ==============  ========================  =======  ====================  
    :math:`\gamma`  Time scaling constant     40000    :math:`dms`
    :math:`C`       Speed of sound in media   343      :math:`m / s`
    :math:`L`       Trachea length            0.025    :math:`m`
    :math:`r`       Reflection coeficient     0.65     :math:`dms`
    :math:`Ch`      OEC Compliance            1.43     :math:`m^3 / Pa`
    :math:`MG`      Beak Inertance            20       :math:`kg / m^4`
    :math:`MB`      Glottis Inertance         10000    :math:`kg / m^4`
    :math:`RB`      Beak Resistance           5000000  :math:`s\; kg / m^4`
    :math:`Rh`      OEC Resistence            24000    :math:`s\;kg / m^4`
    ==============  ========================  =======  ====================

Where :math:`dms` means dimensionless.
"""
# bifurcation saddle nodes and array length
_N = 1000
_mu2_beta = -2.5
_mu1_alpha = 1/3
# General nonlinear equation model of second order
_F1 = "ys"
r"""str : First linear equation.
Where :math:`x` is the labial position and :math:`y` the labial wall velocity. 

.. math::

    \frac{dx}{dt} = y
"""
_F2 = "(-alpha-beta*xs-xs**3+xs**2)*gamma**2 - (xs+1)*gamma*xs*ys"
r"""str: Second linear equation.
Where :math:`x` is the labial position and :math:`y` the labial wall velocity.

.. math::

    \frac{dy}{dt} = \gamma^2(-\alpha-\beta x-x^3+x^2) - \gamma(x+1)x y

This equation is obtained using the Bogdanov–Takens bifurcation :cite:p:`a-Amador2013`. 
"""
_V_MAX_LABIA = -5e6 # model constraint
"""float : Maximum labia walls velocity.
"""
_ovsr = 20          # over sample rate
_prct_noise = 0
# ---------------- physical model constants -----------------
_Z = {
    "a0": 0.11,
    "b0": -0.1,
    "b1": 1,
    "b2": 0,
}
r"""dict : Motor gesture curves, air-sac pressure (:math:`\alpha`)
and labial wall tension (:math:`\beta`). This function has two approaches:

.. math::
    :label: beta

    \begin{equation}
    \begin{aligned}[c]
        & \text{Performance}\\ \\
        & \alpha(t) = a_0 \\
        & \beta(t) = b_0 + b_1 \tilde{FF} + b_2 \tilde{FF}^2
    \end{aligned}
    \qquad\qquad\qquad
    \begin{aligned}[c]
        & \text{Interpretability}\\ \\
        & \alpha(t) = a_0 \\
        & \beta(t) = b_0 + b_1 t + b_2 t^2
    \end{aligned}
    \end{equation}

The best performance, with the lowest relative errors, is obtained when the rescaled 
fundamental frequency is used as input through a quadratic composition, with :math:`\tilde{FF}=FF/10^4`. 
"""
#%%
def bifurcation_ode(f1, f2):
    """

    Parameters
    ----------
        f1 : str

        f2 : str

    Return
    ------
        beta_bif : np.array

        mu1_curves : np.array

        f1 : lambda functions

        f2 : lambda functions


    Example
    -------
        >>>
    """
    beta_bif = np.linspace(_mu2_beta, _mu1_alpha, _N)
    xs, ys, alpha, beta, gamma = symbols('x y alpha beta gamma')
    # ---------------- Labia EDO's Bifurcation -----------------------
    f1 = eval(f1)
    f2 = eval(f2)

    x01 = solveset(f1, ys) + solveset(f1, xs)
    f2_x01 = f2.subs(ys, x01.args[0])

    f = solveset(f2_x01, alpha)
    g = alpha

    df = f.args[0].diff(xs)
    dg = g.diff(xs)

    roots_bif = solveset(df-dg, xs)

    mu1_curves = []
    for ff in roots_bif.args:
        # root evaluatings beta
        mu1 = np.zeros(_N, dtype=float)
        x_root = np.zeros(_N, dtype=float)
        for i in range(_N):
            x_root[i] = ff.subs(beta, beta_bif[i])
            mu1[i] = f.subs([(beta,beta_bif[i]), (xs,x_root[i])]).args[0]
        mu1_curves.append(np.array(mu1, dtype=float))
    mu1_curves = np.array(mu1_curves)

    f1 = lambdify([xs, ys, alpha, beta, gamma], f1)
    f2 = lambdify([xs, ys, alpha, beta, gamma], f2)

    return beta_bif, mu1_curves, f1, f2
#%%
def alpha_beta(
    obj: Any,
    z: Dict = _Z,
    method: Literal["best", "fast"] = "best"
) -> List[np.array]:
    """


    Parameters
    ----------
        obj : Song | Syllable

        z : dict

    Return
    ------
        alpha : np.array([1,2...N])
            Bronchis pressure, also known as air-sac pressure.
        beta : np.array
            Labial tension.

    Example
    -------
        >>>

    """
    obj.z = z
    a = np.array([z["a0"]], dtype=float)
    b = np.array([z["b0"], z["b1"], z["b2"]], dtype=float)

    t = np.linspace(0, obj.T, len(obj.s))
    t_parabole = np.array([np.ones(t.size), t, t**2])
    obj.alpha = np.dot(a[0], t_parabole[0])    # horizontal lines (or parabolas)
    if method=="fast":
        obj.beta = np.dot(b, t_parabole)
    elif method=="best":
        poly = Polynomial.fit(obj.timeFF, obj.FF, deg=10)
        x, y = poly.linspace(np.size(obj.s))
        obj.beta  = b[0] + b[1]*(y/1e4) + b[2]*(y/1e4)**2
    else:
        raise Exception("The method entered is not implemented."
                        + "There are two possible options: fast and best")

    return obj.alpha, obj.beta
#%%
def motor_gestures(
    obj: Any,
    curves: List[np.array],
    params: Dict = _PARAMS
) -> Syllable:
    """


    Parameters
    ----------
        pramams : Dict

    Return
    ------
        synth : Syllable
            Synthethic syllable with same parameters except
            for s and vs

    Example
    -------
        >>>

    """
    # rk4 constans
    tmax = int(obj.s.size)*_ovsr-1   # maximum time
    dt = 1./(_ovsr*obj.sr)           # step
    t = 0                            # initial time
    # trachea pressure pback and pin vectors initialization
    out = np.zeros(int(obj.s.size))  # output pressure
    pi = np.zeros(tmax)              # input pressure
    pb = np.zeros(tmax)              # pressure back
    # initial vector ODEs (v0), it is not too relevant
    v = 1e-4*np.array([1e2, 1e1, 1, 1, 1, 1])
    vs = [v]
    # ------------- MG BIRD MODEL PARAMETERS -----------
    gamma = params["gm"]
    r = params['r']
    L = params['L']
    c = params['C']
    Ch = params['Ch']
    MG = params['MG']
    MB = params['MB']
    RB = params['RB']
    Rh = params['Rh']
    # ----------------------------------------------------
    alpha, beta = curves
    ## ------------- Bogdanov–Takens bifurcation ------------------
    _, _, f1, f2 = bifurcation_ode(_F1, _F2)
    # ------------------------------ Physical Model -----------------------------
    def ODEs(v: np.array) -> np.array:
        dv = np.zeros(6)
        x, y, pout, i1, i2, i3 = v
        # ----------------- direct implementation of the EDOs -----------
        dv[0] = f1(x, y, alpha[t//_ovsr], beta[t//_ovsr], gamma)
        dv[1] = f2(x, y, alpha[t//_ovsr], beta[t//_ovsr], gamma)
        # ------------------------- trachea ------------------------
        pbold = pb[t] # pressure back before
        # Pin(t) = Ay(t)+pback(t-L/C) = Signal_env*v[1]+pb[t-L/C/dt]
        # pi[t] = (0.5*obj.envelope[t//_ovsr])*dv[1] + pb[t-int(L/c/dt)]
        A = 1 #(0.5*obj.envelope[t//_ovsr])
        pi[t] = A*dv[1] + pb[t-int(L/c/dt)]
        pb[t] = -r*pi[t-int(L/c/dt)]    # pressure back: -rPin(t-L/C)
        pout = (1-r)*pi[t-int(L/c/dt)]  # pout
        # ---------------------------------------------------------------
        dv[2] = (pb[t]-pbold)/dt # dpout
        dv[3] = i2
        dv[4] = -(1/Ch/MG)*i1 - Rh*(1/MB+1/MG)*i2 \
                + (1/MG/Ch+Rh*RB/MG/MB)*i3 + (1/MG)*dv[2] \
                + (Rh*RB/MG/MB)*pout
        dv[5] = -(MG/MB)*i2 - (Rh/MB)*i3 + (1/MB)*pout
        return dv
    # ----------------------- Update EDOs Variables ----------------------
    while t < tmax and v[1] > _V_MAX_LABIA:  # NP.ABS()
        v = rk4(ODEs, v, dt)        # RK4 step
        vs.append(v)                # save step
        out[t//_ovsr] = RB*v[-1]    # update output signal (synthetic)
        t += 1
    # ------------------------------------------------------------
    synth = deepcopy(obj)

    synth.params = params
    synth.alpha = obj.alpha
    synth.beta = obj.beta
    synth.z = obj.z

    if not "synth" in synth.file_name:
        synth.file_name = "synth-" + obj.file_name
        synth.id = "synth-" + obj.id

    synth.times_vs = np.linspace(0, len(obj.s)/obj.sr, len(obj.s)*_ovsr)
    synth.vs = np.array(vs)

    synth.s = normalize(out, max_amp=1.0)

    return synth
#%%
def set_z(
    obj,
    z: Union[List[float],Dict] = _Z
) -> Dict:
    """


    Parameters
    ----------
        z : list[float] | dict
            [a0,a1,a2_,b,b1,b2,gamma]

    Return
    ------
        z : dict

    Exmaple
    -------
        >>>
    """
    z0 = {}
    if type(z) is List:
        z0 = {
            "a0": z[0],
            "b0": z[1],
            "b1": z[2],
            "b2": z[3]
        }
    elif type(z) is Dict:
        for k in z.keys():
            z0[k] = z[k]
    obj.z = z0

    return z0
#%%
def set_params(
    obj,
    params: Union[Tuple[float],Dict] = _PARAMS
) -> Dict:
    """


    Parameters
    ----------
        params : list[float] | dict
            [a0,a1,a2_,b,b1,b2,gamma]

    Return
    ------
        params : dict

    Exmaple
    -------
        >>>
    """
    params0 = _PARAMS
    if type(params) in [List, Tuple]:
        for i in range(len(params)):
            params0[params.keys()[i]] = params[i]
    elif type(params) is Dict:
        for k in params.keys():
            params0[k] = params[k]
    obj.params = params0

    return params0
