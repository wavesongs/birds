#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" """
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from wavesongs.utils.tools import klicker_time
from wavesongs.model.bird import mu1_curves, beta_bif

from librosa.display import specshow as Specshow

from mpl_point_clicker import clicker
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter, NullFormatter

from typing import Optional, Tuple, Union, List, AnyStr, Dict, Any

# --------------------------
_COLORES = {
    "Argentina": ["Blues", "lightblue", "blue"],
    "Bolivia": ["Purples", "plum", "purple"],
    "Brazil": ["Greys", "lightgray", "black"],
    "Chile": ["Oranges", "bisque", "chocolate"],
    "Colombia": ["Reds", "lightsalmon", "red"],
    "Costa Rica": ["cool", "paleturquoise", "teal"],
    "Ecuador": ["GnBu", "lightsteelblue", "steelblue"],
    "Peru": ["Greens", "darkseagreen", "darkgreen"],
    "Uruguay": ["copper", "peachpuff", "orange"],
    "Venezuela": ["RdPu", "lightpink", "mediumvioletred"],
}
_CMAP = "magma"
_TITLE_FONTSIZE = 18
#%%
def _suptitle(obj) -> AnyStr:
    """

    Parameters
    ----------
        obj : Syllable | Song
            _description_

    Returns
    -------
        title : AnyStr
            Title template
    """    
    format = obj.file_name[-3:]
    file_name = obj.file_name[:-4].replace("synth_","")
    title = f"{file_name}-{obj.no_syllable}-{obj.type}.{format}" \
                if obj.type!="" else f"{file_name}-{obj.no_syllable}.{format}"
    return title
#%%
def _save_name(obj) -> AnyStr:
    file_name = obj.file_name[:-4]
    img_text = f"{file_name}-{obj.no_syllable}-{obj.type}" \
                if obj.type!="" else f"{file_name}-{obj.no_syllable}"
    return img_text
#%%
def alpha_beta(
    obj: Any,  # Union[Syllable,Song],
    xlim: Tuple[float] = (-0.05, 0.2),
    ylim: Tuple[float] = (-0.2, 0.9),
    figsize: Tuple[float] = (11, 6),
    save: bool = True,
    show: bool = True,
) -> None:
    """


    Parameters
    ----------
        obj : Syllabe|Song
            Song or Syllable to be displayed
        xlim : tuple = (-0.05,.2)
            Time range
        ylim : tuple = (-0.2,0.9)
            Frequency range
        figsize : tuple = (10,6)
            Fogure size (width, height)
        save : bool = True
            Enable save plot
        show : bool = True
            Enable display plot 

    Return
    ------
        None

    Example
    -------
        >>>
    """
    if not "synth" in obj.id:
        raise Exception("This  is not a synthetic syllable, remember create"
                        + " a synthetic file using the funcion bs.Solve().")
    
    plt.close()

    if obj.alpha.max() > 0.2: xlim = (-0.05, 1.1 * obj.alpha.max())
    if obj.beta.max() > 0.9: ylim = (-0.2, 1.1 * obj.beta.max())

    viridis = mpl.colormaps["Blues"]
    c = viridis(np.linspace(0.3, 1, np.size(obj.time_s)))

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 2, figure=fig, wspace=0.1, hspace=0.35)

    # fig.tight_layout(pad=3.0)
    gs.update(top=0.85, bottom=0.1, left=0.075, right=0.935)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(obj.time_s, obj.alpha, c=c, label="alfa")
    ax1.set_title("Air-Sac Pressure")
    ax1.set_ylabel("α (a.u.)")
    ax1.set_ylim(xlim)
    ax1.grid()

    ax2 = fig.add_subplot(gs[1:, 0])
    ax2.scatter(obj.time_s, obj.beta, c=c, label="beta")
    ax2.set_title("Labial Tension")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("β (a.u.)")
    ax2.set_ylim(ylim)
    ax2.sharex(ax1)
    ax2.grid()

    # ------------- Bogdanov–Takens bifurcation ------------------
    ax3 = fig.add_subplot(gs[:, 1])
    ax3.plot(-1 / 27, 1 / 3, "ko")  # , label="Cuspid Point");
    ax3.axvline(0, color="red", lw=1)  # , label="Hopf Bifurcation")
    ax3.scatter(obj.alpha, obj.beta, c=c, marker="_", label="Motor Gesture")
    # label="Saddle-Noddle\nBifurcation"
    ax3.plot(mu1_curves[0], beta_bif, "-g", lw=1)
    ax3.plot(mu1_curves[1], beta_bif, "-g", lw=1)
    ax3.fill_between(
        mu1_curves[1],
        beta_bif,
        10,
        where=mu1_curves[1] > 0,
        color="gray",
        alpha=0.2,
    )
    ax3.text(-0.01, 0.6, "Hopf", rotation=90, color="r")
    ax3.text(-0.0425, 0.37, "CP", rotation=0, color="k")
    ax3.text(-0.0275, 0.15, "SN", rotation=0, color="g")
    ax3.text(0.1, 0.005, "SN", rotation=0, color="g")
    ax3.set_ylabel("Tension (a.u.)")
    ax3.set_xlabel("Pressure (a.u.)")
    ax3.set_title("Parameter Space")
    ax3.yaxis.tick_right()
    ax3.yaxis.set_label_position("right")
    ax3.set_xlim(xlim)
    ax3.set_ylim(ylim)
    ax3.sharey(ax2)
    ax3.legend()

    if obj.type!="":
        gs.update(top=0.8)
        suptitle = f"Motor Gesture Curves\n{_suptitle(obj)}"
    else:
        suptitle = f"Motor Gesture Curves: {_suptitle(obj)}"
    # fig.tight_layout()
    plt.suptitle(
        suptitle,
        fontsize=_TITLE_FONTSIZE,
        y=0.99,
        fontweight="bold",
    )
    
    if save:
        save_name = f"{_save_name(obj)}-mg_params.png"
        fig.savefig(
            obj.proj_dirs.IMAGES / save_name,
            transparent=True,
            bbox_inches="tight",
        )
        print(f"Image save at {save_name}")

    if show:
        plt.show()
    else: plt.close()

    # return fig, gs


# %% plot physical variables
def phsyical_variables(
    obj: Any,  # Union[Syllable,Song],
    xlim: Tuple[float] = (),
    figsize: Tuple[float] = (11, 6),
    save: bool = True,
    show: bool = True,
) -> None:
    """


    Parameters
    ----------
        obj : Syllabe|Song
            Song or Syllable to be displayed
        xlim : tuple = (-0.05,.2)
            Time range
        figsize : tuple = (10,6)
            Fogure size (width, height)
        save : bool = True
            Save plot
        show : bool = True
            Display plot

    Return
    ------
        files_names : list
            List with the audios files names

    Example
    -------
        >>>
    """
    if not "synth" in obj.id:
        raise Exception("This  is not a synthetic syllable, remember create"
                        + " a synthetic file using the funcion bs.Solve().")
    
    plt.close()

    if xlim == ():
        xlim = (obj.times_vs[0], obj.times_vs[-1])

    fig, ax = plt.subplots(2, 2, figsize=figsize, sharex=True)

    plt.subplots_adjust(
        hspace=0.25, wspace=0.2, top=0.825, bottom=0.1, left=0.075, right=0.99
    )

    ax[1, 0].ticklabel_format(
        axis="y", style="scientific", scilimits=(-1, 1)
    )
    ax[1, 0].plot(obj.times_vs, obj.vs[:, 1], color="g")
    ax[1, 0].set_ylabel("$p_{in}$ (x $10^3$)")
    ax[1, 0].set_title(r"Trachea Input Pressure")
    ax[1, 0].set_xlabel("Time (s)")
    ax[1, 0].set_xlim(xlim)

    ax[1, 1].plot(obj.times_vs, obj.vs[:, 4], color="b")
    ax[1, 1].set_ylabel("$p_{out}$")
    ax[1, 1].set_title(r"Trachea Output Pressure")
    ax[1, 1].set_xlim(xlim)
    ax[1, 1].set_xlabel("Time (s)")

    ax[0, 0].plot(obj.times_vs, obj.vs[:, 0], color="r")
    ax[0, 0].set_title(r"Labial Walls Displacement")
    ax[0, 0].set_ylabel("$x(t)$")
    ax[0, 0].set_xlim(xlim)

    ax[0, 1].plot(obj.times_vs, obj.vs[:, 0], color="m")
    ax[0, 1].set_ylabel("$y(t)$")
    ax[0, 1].set_title(r"Labial Walls Velocity")
    ax[0, 1].set_xlim(xlim)

    if obj.type!="":
        plt.subplots_adjust(top=0.8)
        suptitle = f"Physical Model Variables\n{_suptitle(obj)}"
    else:
        suptitle = f"Physical Model Variables: {_suptitle(obj)}"

    fig.suptitle(
        suptitle,
        fontsize=_TITLE_FONTSIZE,
        y=0.99,
        fontweight="bold",
    )
    # fig.tight_layout()

    if save:
        image_text = f"{_save_name(obj)}-PhysicalVariables.png"
        fig.savefig(
            obj.proj_dirs.IMAGES / image_text,
            transparent=True,
            bbox_inches="tight",
        )
        print(f"Image save at {image_text}")
    # return fig, ax
    if show:
        plt.show()
    else:
        plt.close()
# %%
def spectrogram_waveform(
    obj: Any,  # Union[Syllable,Song],
    syllable: Any | None = None,  # Optional[Syllable] = None,
    chunck: Any | None = None,  # Optional[Syllable] = None,
    ff_on: bool = False,
    select_time: bool = False,
    tlim: Optional[Tuple[float]] = None,
    figsize: Tuple[float] = (10, 6),
    save: bool = True,
    show: bool = True,
    ms: int = 7,
) -> clicker:
    """


    Parameters
    ----------
        obj : Syllabe|Song
            Song or Syllable to be displayed
        syllable: Syllable|None = None,

        chunck: Any|None = None,

        ff_on: bool =False,

        select_time: bool = False,

        tlim : tuple = (-0.05,.2)
            Time range
        figsize : tuple = (10,6)
            Fogure size (width, height)
        save : bool = True
            Save plot
        show : bool = True
            Display plot
        ms : int = 7
            Marker size

    Return
    ------
        klicker : cliker
            Clicker object with the points selected

    Example
    -------
        >>>
    """
    ticks = FuncFormatter(lambda x, pos: f"{x*1e-3:g}")
    ticks_x = FuncFormatter(lambda x, pos: f"{x+obj.t0_bs:.2f}")

    if tlim is None:
        tlim = (obj.time[0], obj.time[-1])
    else:
        tlim = (tlim[0] - obj.t0_bs, tlim[1] - obj.t0_bs)

    if syllable is None:
        syllable_on = 0
        ratios = [3, 8]
    else:
        syllable_on = 1
        ratios = [1, 2, 1]
        figsize = (10, 7)

    plt.close()
    # ----------------------- song -----------------------
    if "song" in obj.id:
        fig, ax = plt.subplots(
            2 + int(syllable_on),
            1,
            gridspec_kw={"height_ratios": ratios},
            figsize=figsize,
            sharex=True,
        )

        syllables_array = np.ones(obj.time_s.size) * obj.umbral

        ax[0].plot(obj.time_s, obj.s, "k", label="waveform")
        ax[0].plot(obj.time_s, syllables_array, "--", label="umbral")
        ax[0].plot(obj.time_s, obj.envelope, label="envelope")
        ax[0].legend(bbox_to_anchor=(1.01, 0.5))
        ax[0].xaxis.set_major_formatter(ticks_x)
        ax[0].set_ylabel("Amplitude (a.u)")
        ax[0].set_xlabel("")

        img = Specshow(
            obj.Sxx_dB,
            x_axis="s",
            y_axis="linear",
            sr=obj.sr,
            hop_length=obj.hop_length,
            ax=ax[1],
            cmap=_CMAP,
        )
        if ff_on:
            if obj.ff_method in ("yin", "pyin", "mmanual"):
                ax[1].plot(
                    obj.time,
                    obj.FF,
                    "bo",
                    ms=ms + 1,
                    label=rf"FF$_{{obj.ff_method}}$",
                )
            elif obj.ff_method == "both":
                ax[1].plot(
                    obj.time, obj.FF, "co", ms=ms + 1, label=r"FF$_{pyin}$"
                )
                ax[1].plot(
                    obj.time, obj.FF2, "b*", ms=ms + 1, label=r"FF$_{yin}$"
                )
            ax[1].legend(bbox_to_anchor=(1.135, 1.02))

        ax[1].set_ylim(obj.flim)
        ax[1].set_xlim(tlim)
        ax[1].yaxis.set_major_formatter(ticks)
        ax[1].xaxis.set_major_formatter(ticks_x)
        ax[1].set_ylabel("Frequency (kHz)")
        ax[1].set_xlabel("Time (s)")

        # -------------- chuck -------------------------
        if chunck != None:
            ax[1].plot(
                chunck.time + chunck.t0, chunck.FF, "gv", label="Chunck", ms=10
            )
            ax[2].plot(
                chunck.time + chunck.t0 - syllable.t0,
                chunck.FF,
                "gv",
                label="Chunck",
                ms=8,
            )

        if syllable != None:
            ax[1].plot(
                syllable.time + syllable.t0,
                syllable.FF,
                "b+",
                label="Syllable".format(syllable.sr),
                ms=6,
            )

            img = Specshow(
                syllable.Sxx_dB,
                x_axis="s",
                y_axis="linear",
                sr=syllable.sr,
                ax=ax[2],
                cmap=_CMAP,
                hop_length=syllable.hop_length,
            )

            syllable_info = f"{syllable.file_name}-{syllable.no_syllable}"
            ax2_title = f"Single Syllable Spectrum\n{syllable_info}"
            ax[2].plot(
                syllable.time, syllable.FF, "b+", label="Syllable", ms=15
            )
            ax[2].set_ylim(obj.flim)
            ax[2].legend(loc="upper right", title="FF")
            ax[2].set_xlabel("Time (s)")
            ax[2].set_ylabel("f (khz)")
            ax[2].set_title(ax2_title)
            ax[2].yaxis.set_major_formatter(ticks)
            ax[2].xaxis.set_major_formatter(ticks_x)

            ax[1].legend(loc="upper right", title="FF")

            img_text = _save_name(obj) + "SongAndSyllables.png"
            path_save = obj.proj_dirs.IMAGES / img_text

        else:
            path_save = obj.proj_dirs.IMAGES / f"{obj.file_name[:-4]}-Song.png"

        fig.suptitle(
            f"Waveform and Spectrogram: {obj.file_name}",
            fontsize=_TITLE_FONTSIZE,
            y=0.99,
            fontweight="bold",
        )
        plt.subplots_adjust(wspace=0, hspace=0, top=0.9)
    # ----------------------------- syllable -----------------------------
    else:
        fig, ax = plt.subplots(
            2,
            1,
            gridspec_kw={"height_ratios": [3, 8]},
            figsize=figsize,
            sharex=True,
        )

        ax[0].plot(obj.time_s, obj.s, "k", label="waveform")
        ax[0].plot(obj.time_s, obj.envelope, label="envelope")
        ax[0].legend(bbox_to_anchor=(1.01, 0.65))
        ax[0].xaxis.set_major_formatter(ticks_x)
        ax[0].set_ylabel("Amplitude (a.u)")

        img = Specshow(
            obj.Sxx_dB,
            x_axis="s",
            y_axis="linear",
            sr=obj.sr,
            hop_length=obj.hop_length,
            ax=ax[1],
            cmap=_CMAP,
        )
        ax[1].yaxis.set_major_formatter(ticks)
        ax[1].xaxis.set_major_formatter(ticks_x)

        if ff_on:
            if obj.ff_method in ("yin", "pyin", "manual"):
                ax[1].plot(
                    obj.time, obj.FF, "co", ms=ms, label=r"FF$_{yin}$"
                )  # .format(obj.ff_method))
            elif obj.ff_method == "both":
                ax[1].plot(obj.time, obj.FF, "co", ms=ms, label=r"FF$_{pyin}$")
                ax[1].plot(obj.time, obj.FF2, "b*", ms=ms, label=r"FF$_{yin}$")
            if select_time is False:
                ax[1].legend(bbox_to_anchor=(1.135, 1.02))

        ax[1].set_ylim(obj.flim)
        ax[1].set_xlim(tlim)
        ax[1].set_ylabel("Frequency (kHz)")
        ax[1].set_xlabel("Time (s)")

        plt.subplots_adjust(
            wspace=0.1, hspace=0.1, top=0.85, bottom=0.05, left=0.05, right=0.7
        )

        if obj.type!="":
            plt.subplots_adjust(top=0.8)
            suptitle = f"Waveform and Spectrogram\n{_suptitle(obj)}"
        else:
            suptitle = f"Waveform and Spectrogram: {_suptitle(obj)}"
            
        fig.suptitle(
            suptitle,
            fontsize=_TITLE_FONTSIZE,
            y=0.99,
            fontweight="bold",
        )
        path_save = obj.proj_dirs.IMAGES / _save_name(obj)

    fig.tight_layout()

    if save:
        fig.savefig(path_save,
                    transparent=True,
                    bbox_inches="tight")
        print(f"Image save at {path_save}")

    if show: plt.show()
    else: plt.close()

    if select_time:
        klicker = klicker_time(fig, ax[1])
        return klicker

        # return fig, ax# %%
def syllables(
    obj: Any,  # Union[Syllable,Song],
    obj_synth: Any,  # Union[Syllable,Song],
    ff_on: bool = False,
    figsize: Tuple[float] = (11, 6),
    save: bool = True,
    show: bool = True,
) -> None:
    """


    Parameters
    ----------
        obj : Syllable | Song

        obj : Syllable | Song

        ff_on : bool = False
            Falg to enable fundamental frequency visualization
        save : bool = True
            Flag to save plot
        show : bool = True
            Flag to display plot

    Return
    ------
        None

    Example
    -------
        >>>
    """
    plt.close()

    ticks = FuncFormatter(lambda x, pos: f"{x*1e-3:g}")
    fig, ax = plt.subplots(2, 2, figsize=figsize, sharex=True)

    img = Specshow(
        obj.Sxx_dB,
        x_axis="s",
        y_axis="linear",
        sr=obj.sr,
        hop_length=obj.hop_length,
        ax=ax[0, 0],
        cmap=_CMAP,
    )

    ax[0, 0].set_title("Real", fontweight="bold")
    ax[0, 0].yaxis.set_major_formatter(ticks)
    ax[0, 0].set_ylim(obj.flim)
    ax[0, 0].set_ylabel("Frequency (kHz)")
    ax[0, 0].set_xlabel("")

    img = Specshow(
        obj_synth.Sxx_dB,
        x_axis="s",
        y_axis="linear",
        sr=obj_synth.sr,
        hop_length=obj_synth.hop_length,
        ax=ax[0, 1],
        cmap=_CMAP,
    )

    cbar_ax = fig.add_axes([0.95, 0.47, 0.015, 0.36])
    clb = fig.colorbar(img, cax=cbar_ax, format="%+2.f")
    clb.set_label("Power\n(dB)", labelpad=-25, y=1.2, rotation=0)

    ax[0, 1].yaxis.set_major_formatter(ticks)
    ax[0, 1].set_title("Synthetic", fontweight="bold")
    ax[0, 1].set_ylim(obj.flim)
    ax[0, 1].set_ylabel("")
    ax[0, 1].set_xlabel("")

    if ff_on:
        ax[0, 0].plot(obj.time, obj.FF, "bo-", lw=2, label="FF")
        ax[0, 0].legend(bbox_to_anchor=(0.975, 0.975))

        ax[0, 1].plot(obj_synth.time, obj_synth.FF, "go-", lw=2, label="FF")
        ax[0, 1].legend(bbox_to_anchor=(0.975, 0.975))

    t_end = obj.time[-1] + obj.time[-1] - obj.time[-2]
    ax[1, 0].plot(obj.time_s, obj.s, label="waveform", c="b")
    ax[1, 0].set_xlim((obj.time_s[0], t_end))
    ax[1, 0].plot(obj.time_s, obj.envelope, label="envelope", c="darkblue")
    ax[1, 0].set_ylabel("Amplitud (a.u.)")
    ax[1, 0].set_xlabel("Time (s)")
    ax[1, 0].legend()

    ax[1, 1].plot(obj_synth.time_s, obj_synth.s, label="waveform", c="g")
    ax[1, 1].plot(
        obj_synth.time_s, obj_synth.envelope, label="envelope", c="darkgreen"
    )
    ax[1, 1].set_xlabel("Time (s)")
    ax[1, 1].set_ylabel("")
    ax[1, 1].legend()

    # removing y ticks labels
    ax[1, 1].yaxis.set_major_formatter(NullFormatter())
    ax[0, 1].yaxis.set_major_formatter(NullFormatter())

    plt.subplots_adjust(
        wspace=0.05, hspace=0.075, left=0.07, top=0.85, right=0.93, bottom=0.07
    )
    if obj.type!="":
        plt.subplots_adjust(top=0.8)
        suptitle = f"Comparing Syllables\n{_suptitle(obj)}"
    else:
        suptitle = f"Comparing Syllables: {_suptitle(obj)}"

    fig.suptitle(
        suptitle,
        y=0.99,
        fontsize=_TITLE_FONTSIZE,
        fontweight="bold",
    )

    if save:
        img_name = f"{_save_name(obj)}-SoundAndSpectros.png"
        fig.savefig(
            obj.proj_dirs.IMAGES / img_name,
            transparent=True,
            bbox_inches="tight",
        )
        print(f"Image save at {img_name}")
    # return fig, ax
    if show: 
        plt.show()
    else:
        plt.close()
# %%
def scores(
    obj: Any,  # Union[Syllable,Song],
    obj_synth: Any,  # Union[Syllable,Song],
    figsize: Tuple[float] = (11, 8),
    ylim: Tuple[float] = (),
    save: bool = True,
    show: bool = True,
) -> None:
    """


    Parameters
    ----------
        obj : Syllable | Song

        obj_synth : Syllable | Song

        figsize : tuple = (10,10)
            Size of the figure (width, height)
        ylim : tuple = ()
            Frequnecy range
        save : bool = True
            Flag to save plot
        show : bool = True
            Flag to display plot

    Return
    ------
        None

    Example
    -------
        >>>
    """
    plt.close()

    ticks = FuncFormatter(lambda x, pos: f"{x*1e-3:g}")
    ticks_x = FuncFormatter(lambda x, pos: f"{x:.2g}")

    fig = plt.figure(constrained_layout=False, figsize=figsize)
    gs = fig.add_gridspec(
        nrows=7,
        ncols=5,
        wspace=0.1,
        hspace=0.9,
        left=0.06,
        top=0.925,
        right=0.78,
        bottom=0.075,
    )
    # vmin, vmax = obj.Sxx_dB.min(), obj.Sxx_dB.max()
    # --------------- scores: FF and SCI ---------------------------------
    labelFF = r"FF, $\overline{FF}$="
    labelSCI = r"SCI, $\overline{SCI}$="
    max_error = 100 * max(obj_synth.deltaFF.max(), obj_synth.deltaSCI.max())

    ax1 = fig.add_subplot(gs[:2, :])
    ax1.plot(
        obj_synth.time,
        100 * obj_synth.deltaFF,
        "*-",
        color="k",
        ms=5,
        lw=1,
        alpha=0.8,
        label=labelFF + str(round(100 * obj_synth.deltaFF_mean, 2)),
    )
    ax1.plot(
        obj_synth.time,
        100 * obj_synth.deltaSCI,
        "*-",
        color="purple",
        ms=5,
        lw=1,
        alpha=0.8,
        label=labelSCI + str(round(100 * obj_synth.deltaSCI_mean, 2)),
    )
    ax1.legend(bbox_to_anchor=(1.235, 1.05), borderpad=0.6, labelspacing=0.7)
    ax1.xaxis.set_major_formatter(ticks_x)
    ax1.set_xlim((obj_synth.time[0], obj_synth.time[-1]))
    ax1.set_ylabel("Relative Error (%)")
    ax1.set_xlabel("")
    ax1.set_ylim((0, 1.25 * max_error))
    # ------------------ spectrum ---------------
    ax2 = fig.add_subplot(gs[2:5, :], sharex=ax1)

    img = Specshow(
        obj.Sxx_dB,
        x_axis="s",
        y_axis="linear",
        sr=obj.sr,
        hop_length=obj.hop_length,
        ax=ax2,
        cmap=_CMAP,
    )

    ax2.plot(obj.time, obj.FF, "b*-", label=r"real", ms=7)
    ax2.plot(obj_synth.time, obj_synth.FF, "go-", label=r"synth", ms=3)
    ax2.legend(borderpad=0.6, labelspacing=0.7)
    ax2.yaxis.set_major_formatter(ticks)
    ax2.xaxis.set_major_formatter(ticks_x)
    ax2.set_xlim((obj.time[0], obj.time[-1]))
    ax2.set_ylim(obj.flim)
    ax2.set_ylabel("Frequency (kHz)")
    ax2.set_xlabel("")
    # ------------------ SCI -------------------------
    ax31 = fig.add_subplot(gs[5:7, :], sharex=ax2)
    ax31.set_ylabel(r"Similarity (dl)")
    ax31.set_xlabel("Time (s)")
    # --------------- acousitcal features ----------------------
    ax32 = ax31.twinx()
    label_lskl = r"SKL, $\overline{SKL}$="
    label_lr = r"$SCI_{real}$, $\overline{SCI}$="
    label_ls = r"$SCI_{synth}$, $\overline{SCI}$="
    label_lh = r"DF, $\overline{DF}$="
    label_lc = r"cor, $\overline{corr}$="
    lr = ax32.plot(
        obj.time,
        obj.SCI,
        "b*-",
        ms=7,
        label=label_lr + str(round(obj.SCI.mean(), 2)),
    )
    ls = ax32.plot(
        obj.time,
        obj_synth.SCI,
        "go-",
        ms=5,
        alpha=0.8,
        label=label_ls + str(round(obj_synth.SCI.mean(), 2)),
    )
    lh = ax32.plot(
        obj.time,
        obj_synth.Df,
        "H",
        ms=3,
        label=label_lh + str(round(obj_synth.Df.mean(), 2)),
    )
    lskl = ax32.plot(
        obj.time,
        obj_synth.SKL,
        "s",
        color="purple",
        ms=3,
        label=label_lskl + str(round(obj_synth.SKL.mean(), 2)),
    )
    correlation_synth = obj_synth.correlation.mean()
    lc = ax32.plot(
        obj.time,
        obj_synth.correlation,
        "p",
        ms=3,
        label=label_lc + str(round(correlation_synth, 2)),
    )
    lns = lr + ls + lh + lskl + lc
    labs = [l.get_label() for l in lns]
    ax32.legend(
        lns,
        labs,
        bbox_to_anchor=(1.075, 1.1),
        title="Acoustical Features",
        title_fontproperties={"weight": "bold"},
    )
    ax32.legend(bbox_to_anchor=(1.3, 1))
    ax32.set_ylabel("SCI (dl)")
    ax32.set_ylim((0, 5))

    cbar_ax = fig.add_axes([0.8, 0.332, 0.02, 0.315])
    clb = fig.colorbar(img, cax=cbar_ax)
    clb.set_label("Power (dB)", labelpad=12, y=0.5, rotation=90)

    if obj.type!="":
        plt.subplots_adjust(top=0.775)
        suptitle = f"Scoring Variables\n{_suptitle(obj)}"
    else:
        suptitle = f"Scoring Variables: {_suptitle(obj)}"

    fig.suptitle(
        suptitle,
        fontsize=_TITLE_FONTSIZE,
        y=0.99,
        fontweight="bold",
    )

    if save:
        img_name = f"{_save_name(obj)}-ScoringVariables.png"
        fig.savefig(
            obj.proj_dirs.IMAGES / img_name,
            transparent=True,
            bbox_inches="tight",
        )
        print(f"Image save at {img_name}")

    if show:
        plt.show()
    else:
        plt.close()
    # return fig, gs
# %%
def spectrum_comparison(
    obj: Any,  # Union[Syllable,Song],
    obj_synth: Any,  # Union[Syllable,Song],
    cmap: Union[AnyStr] = "afmhot_r",
    figsize: Tuple[float] = (11, 6),
    sharey: bool = True,
    save: bool = True,
    show: bool = True,
) -> None:
    """


    Parameters
    ----------
        obj : Syllable | Song

        obj_synth : Syllable | Song

        figsize : tuple = (10,10)
            Size of the figure (width, height)
        sharey: bool = True,
            Enable share y axis
        save : bool = True
            Flag to save plot
        show : bool = True
            Flag to display plot

    Return
    ------
        None

    Example
    -------
        >>>
    """
    if cmap is not None:
        _CMAP = cmap
    labelrotation = 90 if obj.time[-1] < 1 else 0
    _fontproperties = {"size": 12, "weight": "bold"}

    plt.close()
    
    ticks = FuncFormatter(lambda x, pos: f"{x*1e-3:g}")

    fig = plt.figure(figsize=figsize)

    gs = fig.add_gridspec(
        nrows=2,
        ncols=3,
        hspace=0.35,
        wspace=0.2,
        top=0.825,
        bottom=0.15,
        left=0.05,
        right=0.95,
    )
    vmin = obj.Sxx_dB.min()
    vmax = obj.Sxx_dB.max()

    # ------------------ spectrogams ----------------------------
    ax1 = fig.add_subplot(gs[0, 0])

    img = Specshow(
        obj.Sxx_dB,
        x_axis="s",
        y_axis="linear",
        sr=obj.sr,
        hop_length=obj.hop_length,
        ax=ax1,
        cmap=_CMAP,
    )

    clb = fig.colorbar(img, ax=ax1)
    clb.set_label("Power\n(dB)", labelpad=-16, y=1.25, rotation=0)

    ax1.tick_params(axis="x", which="both", labelrotation=labelrotation)
    ax1.set_title("Real", fontproperties=_fontproperties)
    ax1.set_xlim((obj.time[0], obj.time[-1]))
    ax1.yaxis.set_major_formatter(ticks)
    ax1.set_ylim(obj.flim)
    ax1.set_ylabel("")
    ax1.set_xlabel("")

    if sharey:
        ax2 = fig.add_subplot(gs[0, 1], sharex=ax1, sharey=ax1)
    else:
        ax2 = fig.add_subplot(gs[0, 1], sharex=ax1)

    img = Specshow(
        obj_synth.Sxx_dB,
        x_axis="s",
        y_axis="linear",
        sr=obj_synth.sr,
        hop_length=obj_synth.hop_length,
        ax=ax2,
        cmap=_CMAP,
    )

    clb = fig.colorbar(img, ax=ax2)
    clb.set_label("Power\n(dB)", labelpad=-16, y=1.25, rotation=0)

    ax2.tick_params(axis="x", which="both", labelrotation=labelrotation)
    ax2.set_title("Synthetic", fontproperties=_fontproperties)
    ax2.yaxis.set_major_formatter(ticks)
    ax2.set_ylim(obj.flim)
    ax2.set_ylabel("")
    ax2.set_xlabel("")

    # ------------------ Mel spectgrograms ------------------
    if sharey:
        ax3 = fig.add_subplot(gs[1, 0], sharex=ax1, sharey=ax1)
    else:
        ax3 = fig.add_subplot(gs[1, 0], sharex=ax1)

    img = Specshow(
        obj.FF_coef,
        x_axis="s",
        y_axis="linear",
        sr=obj.sr,
        hop_length=obj.hop_length,
        ax=ax3,
        cmap=_CMAP,
        vmin=0,
        vmax=100,
    )

    clb = fig.colorbar(img, ax=ax3)

    ax3.tick_params(axis="x", which="both", labelrotation=labelrotation)
    ax3.yaxis.set_major_formatter(ticks)
    ax3.set_ylabel(" " * 65 + "Frequency (kHz)", loc="center", labelpad=10)
    ax3.set_xlabel("")
    ax3.set_ylim(obj.flim)

    if sharey:
        ax4 = fig.add_subplot(gs[1, 1], sharex=ax1, sharey=ax1)
    else:
        ax4 = fig.add_subplot(gs[1, 1], sharex=ax1)

    img = Specshow(
        obj_synth.FF_coef,
        x_axis="s",
        y_axis="linear",
        sr=obj.sr,
        cmap=_CMAP,
        vmin=0,
        vmax=100,
        hop_length=obj_synth.hop_length,
        ax=ax4,
    )

    clb = fig.colorbar(img, ax=ax4)

    ax4.tick_params(axis="x", which="both", labelrotation=labelrotation)
    ax4.yaxis.set_major_formatter(ticks)
    ax4.set_xlabel("Time (s)", labelpad=10)
    ax4.set_ylim(obj.flim)
    ax4.set_ylabel("")

    # ------------------ Delta Sxx - Mel ------------------------
    if sharey:
        ax5 = fig.add_subplot(gs[0, 2], sharex=ax1, sharey=ax1)
    else:
        ax5 = fig.add_subplot(gs[0, 2], sharex=ax1)

    img = Specshow(
        obj_synth.deltaSxx,
        x_axis="s",
        y_axis="linear",
        sr=obj.sr,
        vmin=0,
        vmax=1,
        ax=ax5,
        cmap=_CMAP,
        hop_length=obj_synth.hop_length,
    )

    ax5.set_title(r"Difference ($\Delta$)", fontproperties=_fontproperties)
    ax5.tick_params(axis="x", which="both", labelrotation=labelrotation)
    ax5.yaxis.set_major_formatter(ticks)
    ax5.set_ylim(obj.flim)
    ax5.set_ylabel("")
    ax5.set_xlabel("")

    clb = fig.colorbar(img, ax=ax5)
    clb.set_label("Power\n(dB)", labelpad=-16, y=1.25, rotation=0)

    if sharey:
        ax6 = fig.add_subplot(gs[1, 2], sharex=ax1, sharey=ax1)
    else:
        ax6 = fig.add_subplot(gs[1, 2], sharex=ax1)

    img = Specshow(
        obj_synth.deltaMel,
        x_axis="s",
        y_axis="linear",
        sr=obj.sr,
        hop_length=obj_synth.hop_length,
        ax=ax6,
        cmap=_CMAP,
    )

    ax6.tick_params(axis="x", which="both", labelrotation=labelrotation)
    ax6.yaxis.set_major_formatter(ticks)
    ax6.set_ylim(obj.flim)
    ax6.set_ylabel("")
    ax6.set_xlabel("")
    # ax6.yaxis.set_major_formatter(NullFormatter())

    plt.text(0.04, 6e3, "MEL", rotation=90, fontproperties=_fontproperties)
    plt.text(
        0.04, 3.5e4, "Linear", rotation=90, fontproperties=_fontproperties
    )

    fig.colorbar(img, ax=ax6)

    if obj.type!="":
        plt.subplots_adjust(top=0.8)
        suptitle = f"Comparing Spectral Content\n{_suptitle(obj)}"
    else:
        suptitle = f"Comparing Spectral Content: {_suptitle(obj)}"

    fig.suptitle(
        suptitle,
        fontsize=_TITLE_FONTSIZE,
        y=0.99,
        fontweight="bold",
    )
    if save:
        img_name = f"{_save_name(obj)}-ComparingSpectros.png"
        fig.savefig(
            obj.proj_dirs.IMAGES / img_name,
            transparent=True,
            bbox_inches="tight",
        )
        print(f"Image save at {img_name}")
    # return fig, gs
    if show: 
        plt.show()
    else:
        plt.close()