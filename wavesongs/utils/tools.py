#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" """
import numpy as np

from maad import sound
from matplotlib.axes import Axes
from IPython.display import Audio
from matplotlib.figure import Figure
from mpl_point_clicker import clicker
from scipy.interpolate import interp1d

from sympy import (
    symbols,
    lambdify,
    solveset
)


from mpl_pan_zoom import (
    zoom_factory,
    PanManager,
    MouseButton
)

from numpy.typing import ArrayLike
from typing import (
    Tuple,
    Union,
    List,
    AnyStr,
    Any
)

_LABELS = [
    r"$f_{max/min}$",
    r"$theme_{ini}$",
    r"$theme_{end}$",
    r"$trill_{ini}$",
    r"$trill_{end}$"
]
_COLORS = [
    "cyan", "olivedrab", "darkgreen", "steelblue", "royalblue"
]
_MARKERS = [
    "p", "*", "*", "o", "o"
]
# bifurcation saddle nodes and array length
_N = 1000
_mu2_beta = -2.5
_mu1_alpha = 1/3


#%%
def envelope(s: ArrayLike, sr: int, Nt: int) -> ArrayLike:
    """
    
    Parameters
    ----------
        s : np.array
            Audio amplitude array
        sr : int
            Sample rate
        Nt : int
    
    Return
    ------
        s_env_interpolated : np.array 
    
    Example
    -------
        >>>
    """
    time = np.linspace(0, len(s)/sr, len(s))
    s_env = sound.envelope(s, Nt=Nt) 
    t_env = np.arange(0, len(s_env), 1)*len(s)/sr/len(s_env)
    t_env[-1] = time[-1] 
    fun_s = interp1d(t_env, s_env)
    s_env_interpolated = np.array(fun_s(time))
    return s_env_interpolated

#%%
def klicker_multiple(
    fig: Figure, 
    ax: Axes,
    labels: List[AnyStr] =_LABELS,
    colors: List[AnyStr] =_COLORS,
    markers: List[AnyStr] =_MARKERS
) -> clicker:
    """
    
    Parameters
    ----------
        fig : Figure
            Matplotlib Figure object
        ax : Axes
            Matplotlib Axes objects
        label : list[str]

        colors : list[str]

        markers : list[str]

    
    Return
    ------
        klicker_data : clicker
            Clicker object with position of the data measured
    
    Example
    -------
        >>>
    """
    zoom_factory(ax)
    pm = PanManager(fig, button=MouseButton.MIDDLE)
    klicker_data = clicker(
                    ax, 
                    labels, 
                    markers=markers,
                    colors=colors,
                    legend_bbox=(1.02, 1.0)
                    )

    klicker_data._pm = pm
    return klicker_data
#%%
def klicker_time(fig: Figure, ax: Axes):
    """
    
    Parameters
    ----------
        fig : Figure
            Matplotlib Figure object
        ax : Axes
            Matplotlib Axes objects
    
    Return
    ------
        klicker_data : clicker
            Clicker object with position of the data measured
    
    Example
    -------
        >>>
    """
    zoom_factory(ax)
    pm = PanManager(fig, button=MouseButton.MIDDLE)
    klicker_time = clicker(
                    ax,
                    [r"$t_{ini}$",r"$t_{end}$"],
                    markers=["o","x"],
                    colors=["blue","green"],
                    legend_bbox=(1.125, 0.975),
                    legend_loc='best'
                    )

    klicker_time._pm = pm
    #ax.legend(title="Interval Points", bbox_to_anchor=(1.1, 1.05))
    return klicker_time
    #%%
def get_positions(klicker: clicker) -> List[Tuple[float]]:
    """
    
    Parameters
    ----------
        klicker : clicker
            Clicker object with position of the data measured
    
    Return
    ------
        times : list[tuple[float], tuple[float]]
            Times select from the spectrogram
    Example
    -------
        >>>
    """
    tinis = klicker.get_positions()[r"$t_{ini}$"]
    tends = klicker.get_positions()[r"$t_{end}$"]
    
    if tinis.shape != tends.shape:
        print("Number of points selectas are nod even. Remember you have \
              to select the same number of initial times than end times")
        times = [tuple(tinis), tuple(tends)]
    else:
        no_points = tinis.shape[0]
        if no_points>1:
            times = [[(tinis[i,0],tends[i,0]), (tinis[i,1],tends[i,1])]
                     for i in range(no_points)]
        else:
            times = [[(tinis[0,0],tends[0,0]), (tinis[0,1],tends[0,1])]]
    return times
#%%
def bifurcation_ode(f1, f2):
    """
    
    Parameters
    ----------
        f1 : function

        f2 : function
    
    Return
    ------
        beta_bif : np.array

        mu1_curves : np.array

        f1 :

        f2 :

    
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
def rk4(f, v: ArrayLike, dt: float):
    """
    Implentation of Runge-Kuta 4th order
    
    Parameters
    ----------
        f : function
            differential equations functions y'=f(y)
        v : np.ndarray [x,y,i1,i2,i3]
            array with the differential variables 
        dt : float
            rk4 time step
    
    Return
    -------
        rk4 : np.ndarray [x,y,i1,i2,i3]
            reulst approximation 
    
    Example
    -------
        >>>
    """
    k1 = f(v)    
    k2 = f(v + dt/2.0*k1)
    k3 = f(v + dt/2.0*k2)
    k4 = f(v + dt*k3)

    return v + dt*(2.0*(k2+k3)+k1+k4)/6.0

# def DownloadXenoCanto(data, XC_ROOTDIR="./examples/", XC_DIR="Audios/", filters=['english name', 'scientific name'],
#                         type=None, area=None, cnt=None, loc=None, nr=None, q='">C"', len=None, len_limits=['00:00', '01:00'],
#                         max_nb_files=20, verbose=False, min_quality="B"):
#     """
#     data = [['Rufous-collared Sparrow', 'Zonotrichia capensis'],
#             ['White-backed',        'Dendrocopos leucotos']]
#     len = '"5-60"'
#     len_limits = ['00:00', '01:00']
#     XC_ROOTDIR = './files/'
#     XC_DIR = 'zonotrichia_dataset' 
#     """
    
#     df_species = pd.DataFrame(data,columns=filters)
#     sp, gen = [], []

#     for name in df_species['scientific name']:
#         gen.append(name.rpartition(' ')[0])
#         sp.append(name.rpartition(' ')[2])

#     df_query = pd.DataFrame()
#     df_query['param1'] = gen
#     df_query['param2'] = sp
#     df_query['param3'] = 'q:'+q
#     if type is not None: df_query['param4'] ='type:'+type
#     if area is not None: df_query['param5'] ='area:'+area
#     if cnt is not None:  df_query['param6'] ='cnt:'+cnt
#     if loc is not None:  df_query['param7'] ='loc:'+loc
#     if nr is not None:   df_query['param8'] ='nr:'+nr
#     if len is not None:  df_query['param9'] ='len:'+len

#     # Get recordings metadata corresponding to the query
#     df_dataset= util.xc_multi_query(df_query, 
#                                     format_time = False,
#                                     format_date = False,
#                                     verbose = verbose)
#     if df_dataset.size!=0:
#         df_dataset = util.xc_selection(df_dataset,
#                                         max_nb_files=max_nb_files,
#                                         min_length=len_limits[0],
#                                         max_length=len_limits[1],
#                                         min_quality=min_quality,
#                                         verbose = verbose )
            
#         # download audio files
#         util.xc_download(df_dataset,
#                         rootdir = XC_ROOTDIR,
#                         dataset_name= XC_DIR,
#                         overwrite=True,
#                         save_csv= True,
#                         verbose = verbose)

#         filelist = grab_audio(XC_ROOTDIR+XC_DIR)
#         df = pd.DataFrame()
#         for file in filelist:
#             df = df.append({'fullfilename': file,
#                             'filename': Path(file).parts[-1][:-4],
#                             'species': Path(file).parts[-2]},
#                             ignore_index=True)

#         for i in range(df_species.shape[0]):
#             df = df_dataset["en"].str.contains(df_species["english name"][i], case=False)
#             spec = df_dataset[df]["gen"][0] +" "+ df_dataset[df]["sp"][0] #df_species["scientific name"][i]
#             scientific = df_dataset[df]["en"][0]#df_species["english name"][i]
#             df_dataset[df].to_csv(XC_ROOTDIR+XC_DIR+spec+"_"+scientific+"/spreadsheet-XC.csv")

#         return df_dataset#["en"][df]
#     else:
#         raise ValueError("No sounds were found with your specifications. Try again with other parameters.")


# def DefineWholeSyllable(paths, df, index, flim=(1e2,15e3)):
    
#     file_id = df.iloc[index]["id_XC"]
#     NN = df.iloc[index]["NN"]
#     umbral_FF = df.iloc[index]["umbral_FF"]
#     birdsong = bs.BirdSong(paths, file_id=file_id, umbral_FF=umbral_FF, tlim=(0,60), Nt=1000, NN=NN, flim=flim)

#     return birdsong
# # def DefineSyllable(paths, df, index, flim=(1e2,15e3)): 
    
#     file_id = df.iloc[index]["id_XC"]
#     NN = df.iloc[index]["NN"]
#     umbral_FF = df.iloc[index]["umbral_FF"]
#     time_interval = df.iloc[index][["t_ini","t_end"]].values
#     type = df.iloc[index]["type"]
#     no_syllable = df.iloc[index]["no_syllable"]
#     coef = df.iloc[index][["coef"]].values[0]
    
#     birdsong = bs.BirdSong(paths, file_id=file_id, umbral_FF=umbral_FF, tlim=(0,60), Nt=1000, NN=NN, flim=flim)
    
#     syllable = bs.Syllable(birdsong=birdsong, tlim=time_interval, Nt=10, #NN=NN, #file_name=file_name,
#                             umbral_FF=umbral_FF, ide="syllable", type=type, no_syllable=no_syllable)
#     syllable.Set(coef)
#     synth_syllable = syllable.Solve(syllable.p)

#     bw = np.max(synth_syllable.FF)-np.min(synth_syllable.FF)
#     lenght = synth_syllable.timeFF[-1]
#     bw_rate = {'file_name':synth_syllable.file_name, 'type':synth_syllable.type, 'no_syllable':synth_syllable.no_syllable,
#                 'bw':bw, "lenght":lenght, 'rate':1/lenght}

#     return syllable, synth_syllable, bw_rate


################

# def smoothstep(x, x_min: int = 0, x_max: int = 1, N: int = 1):
#     result = 0
#     x = np.clip((x - x_min) / (x_max - x_min), 0, 1)

#     for n in range(0, N + 1):
#          result += comb(N+n, n)*comb(2*N+1, N-n)*(-x)**n
#     result *= x**(N+1)

#     return result

# #%%
# def grab_audio(path, audio_format: AnyStr = 'wav'):
#     """
    
#     Parameters
#     ----------

    
#     Return
#     ------

    
#     Example
#     -------
#         >>>
#     """
#     filelist = []
#     for root, _, files in os.walk(path, topdown=False):
#         for name in files:
#             if (name[-3:].casefold() == audio_format
#                 and name[:2] != '._'):
#                 filelist.append(os.path.join(root, name))
#     return filelist
