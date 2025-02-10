#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" """
import json

import numpy as np
import pandas as pd

from numpy.linalg import norm
from IPython.display import Audio
from maad.sound import normalize, write
from scipy.interpolate import interp1d

from wavesongs.utils.paths import ProjDirs
from wavesongs.utils.tools import envelope
from wavesongs.model.bird import (
    _params,
    _z,
    alpha_beta,
    motor_gestures
)

from os.path import (
    basename,
    normpath
)

from librosa import (
    reassigned_spectrogram,
    amplitude_to_db,
    fft_frequencies,
    times_like,
    feature,
    stft,
    pyin,
    yin,
    load
)

from librosa.feature import (
    spectral_centroid,
    mfcc,
    rms,
    melspectrogram
)

from numpy.typing import ArrayLike
from typing import (
    Optional,
    Tuple,
    List,
    AnyStr,
    Dict,
    Any,
    Union,
    Literal,
    TypeVar
)

Syllable = TypeVar('Syllable')
Song = TypeVar('Song')
DataFrame = TypeVar('pandas.core.frame.DataFrame')

#%%
def read_MG(
    file_name: AnyStr,
    no_syllable: Union[int, AnyStr],
    results: AnyStr = "../results",
    type: AnyStr = ""
) -> Syllable:
    """
    
    
    Parameters
    ----------
        
    
    Return
    -------
        
    
    Example
    -------
        >>>
    """
    folder = f"{results}/MG_param"
    file_name = f"{folder}/{file_name}-{no_syllable}-MG.csv" \
                    if type=="" \
                    else f"{folder}/{file_name}-{no_syllable}-{type}-MG.csv"
    df = pd.read_csv(file_name, index_col=0)

    data = df.to_dict()["value"]
    tlim = [float(data["t_ini"]), float(data["t_end"])]
    flim = [float(data["f_ini"]), float(data["f_end"])]
    
    z_json = data["z"].replace("'", "\"")
    z = json.loads(z_json)

    metadata_json = data["metadata"].replace("'", "\"")
    metadata = json.loads(metadata_json)
    root_folder = data["root_folder"] \
                    if data["root_folder"]!=".." \
                    else data["root_folder"]+"/"
    audios_folder = data["audios_folder"].replace(root_folder, "")

    proj_dirs = ProjDirs(root=root_folder, audios=audios_folder)
    syllable = Syllable(
                file_id=data["file_name"][:-4],
                proj_dirs=proj_dirs,
                tlim=tlim,
                no_syllable=int(data["no_syllable"]), 
                id=data["id"],
                sr=int(data["sr"]),
                metadata=metadata,
                type=data["type"]
            )
    syllable.acoustical_features(
        flim=flim,
        umbral_FF=float(data["umbral_FF"]),
        NN=int(data["NN"]),
        ff_method=data["ff_method"]
    )
    syllable.z = z

    return syllable


#%%
class Syllable:
    #%%
    """
    Define a syllable and its properties
    
    Parameters
    ----------
        proj_dirs : ProjDirs | None 
            Object to manage project directories
        song : Syllable | Song | None
            Object
        params : dict | None
            Diccionary with all or some constat of the physical
            model motor gestures
        tlim : tuple
            Time range
        flim : tuple
            Frequency range
        sr : int
            Sample rate
        no_syllable : int 
            Sylalble number in song
        id : str
            Type of the object, "syllable" or "synth-syllable"
        info : dict
            Audio metadata about the audio.
        type : str
            A short description about the part, theme or trill, and the behaviour of the
            fundamental frequency: plane, up, down, up-down, down-up, and complex. 
            Template: "{part}-{behaviour}". Example: theme-up 
            
    Attributes
    ----------


    Examples
    --------
        >>> 
    """ 
    #%%
    def __init__(
        self,
        file_id: Optional[AnyStr] = None,
        proj_dirs: Optional[ProjDirs] = None,
        obj: Any = None,
        tlim: Tuple[float] = (0, 60),
        sr: int = 44100,
        no_syllable: int = 0,
        id: AnyStr = "syllable",
        metadata: Dict = {},
        type: AnyStr = "",
        duration: Optional[int] = None
    ):
        self.no_syllable = no_syllable
        self.proj_dirs = proj_dirs
        self.metadata = metadata
        self.file_id = file_id
        self.type = type
        self.sr = sr
        self.tlim = tlim
        self.id = id
        # defining syllable by songs or file_id with proj_dirs object 
        if (obj is not None) and (file_id is None) and (proj_dirs is None):
            # self.__dict__.update(obj.__dict__)
            self.proj_dirs = obj.proj_dirs
            self.file_name = obj.file_name
            self.t0_bs = obj.t0_bs + tlim[0]
            self.info = obj.info
            self.sr = obj.sr
            s = obj.s

            s = s[int(self.tlim[0]*self.sr):int(self.tlim[1]*self.sr)]
            self.t0 = self.tlim[0]
            
            self.s = normalize(s, max_amp=1.0)
            
            self.acoustical_features(
                NN=obj.NN,
                ff_method=obj.ff_method,
                umbral_FF=obj.umbral_FF,
                Nt=obj.Nt,
                center=obj.center,
                overlap=obj.overlap,
                flim=obj.flim,
                llambda=obj.llambda,
                n_mfcc=obj.n_mfcc,
                n_mels=obj.n_mels,
                stft_window=obj.stft_window
            )

        elif (file_id is not None) and (proj_dirs is not None):
            self.file_path = proj_dirs.find_audio(file_id)
            self.file_name = basename(normpath(self.file_path))
        
            s, sr = load(self.file_path, sr=self.sr, mono=True)
            self.t0_bs = 0
            self.sr = sr

            s = s[int(self.tlim[0]*self.sr):int(self.tlim[1]*self.sr)]
            self.t0 = self.tlim[0]
            
            self.s = normalize(s, max_amp=1.0)
        elif (obj is None) and (file_id is None) and (duration is not None):
            self.file_name = "synthetic"
            self.id = "synth_" +id
            self.no_syllable = no_syllable
            self.proj_dirs = proj_dirs
            self.metadata = metadata
        
            self.sr = sr
            self.type = type
            self.T = duration
            self.s = np.ones(self.T*sr)
            self.t0 = self.t0_bs = 0            
        else:
            raise Exception("You have to enter a file_id with a"
                            + " project object or a song or syllable object")    
    #%%
    def acoustical_features(
        self,
        NN: int = 512,
        ff_method: Literal["yin", "pyin"] = "yin",
        umbral_FF: int = 1,
        flim: Tuple[float] = (1e3, 2e4),
        Nt: int = 10,
        center: bool = False,
        overlap: float = 0.5,
        llambda: float = 1.5,
        n_mfcc: int = 4,
        n_mels: int = 4,
        stft_window: AnyStr = "hann"
    ) -> None:
        """
        Coputing acoustical tempo-spectral variables
        
        Parameters
        ----------
            NN : int
            llambda : float

            overlap : float

            center : bool = False

            umbral_FF : int

            ff_method : str

            Nt : int

            n_mfcc : int

            n_mels : int

            stft_window : str

        Return
        ------
            None

        Examples
        --------
            >>>
        """
        self.stft_window = stft_window
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.flim = flim
        self.Nt = Nt
        self.NN = NN
        
        self.ff_method = ff_method
        self.umbral_FF = umbral_FF
        self.llambda = llambda
        self.overlap = overlap
        self.center = center
        self.envelope = envelope(self.s, self.sr, self.Nt)

        self.time0 = np.linspace(0, len(self.s)/self.sr, len(self.s))
        self.time_s = np.linspace(0, len(self.s)/self.sr, len(self.s))
        self.T = self.s.size / self.sr
        
        self.t_interval = np.array([self.time_s[0], self.time_s[-1]])
        self.t_interval += self.t0_bs

        self.win_length = self.NN//2
        self.hop_length = self.NN//4
        self.no_overlap = int(overlap*self.NN)
        # ------------- ACOUSTIC FEATURES -------------------------------
        self.stft = stft(y=self.s,
                         n_fft=self.NN,
                         hop_length=self.hop_length,
                         win_length=self.NN,
                         window=self.stft_window,
                         center=self.center,
                         dtype=float,
                         pad_mode='constant')
        freqs, times, mags = reassigned_spectrogram(
                                self.s,
                                sr=self.sr,
                                S=self.stft,
                                n_fft=self.NN,
                                hop_length=self.hop_length,
                                win_length=self.win_length,
                                window=self.stft_window, 
                                center=self.center,
                                reassign_frequencies=True,
                                reassign_times=True,
                                ref_power=1e-06,
                                fill_nan=True,
                                clip=True,
                                dtype=float,
                                pad_mode='constant'
                             )
        self.Sxx_dB  = amplitude_to_db(mags, ref=np.max)
        self.freqs = freqs  
        self.times = times 
        self.Sxx = mags 
        
        self.FF_coef = np.abs(self.stft)
        self.freq = fft_frequencies(sr=self.sr, n_fft=self.NN) 
        self.time = times_like(X=self.stft,
                               sr=self.sr,
                               hop_length=self.hop_length,
                               n_fft=self.NN) #, axis=-1
        self.time -= self.time[0]
        
        self.f_msf = [norm(self.FF_coef[:,i]*self.freq, 1)
                      / norm(self.FF_coef[:,i], 1)
                      for i in range(self.FF_coef.shape[1])]
        self.f_msf = np.array(self.f_msf)
        
        self.centroid = spectral_centroid(
                            y=self.s,
                            sr=self.sr,
                            S=np.abs(self.stft),
                            n_fft=self.NN,
                            hop_length=self.hop_length,
                            freq=self.freqs,
                            win_length=self.win_length, 
                            window=self.stft_window,
                            center=self.center,
                            pad_mode='constant'
                         )[0]
        self.mfccs = mfcc(
                        y=self.s,
                        sr=self.sr,
                        S=self.stft,
                        n_mfcc=self.n_mfcc,
                        dct_type=2,
                        norm='ortho',
                        lifter=0
                    )
        self.rms = rms(
                        y=self.s,
                        S=self.stft,
                        frame_length=self.NN,
                        hop_length=self.hop_length,
                        center=self.center,
                        pad_mode='constant'
                    )[0]
        self.s_mel = melspectrogram(
                        y=self.sr,
                        sr=self.sr,
                        S=self.stft,
                        n_fft=self.NN,
                        hop_length=self.hop_length,
                        win_length=self.win_length,
                        window=self.stft_window,
                        center=self.center,
                        pad_mode='constant',
                        power=2.0,
                        n_mels=self.n_mels,
                        fmin=self.flim[0],
                        fmax=self.flim[1]
                    )
        # # ------------- Fundamental Frequency computing --------------
        if self.ff_method=="pyin":
            self.FF,_,_ = pyin(
                            self.s,
                            fmin=self.flim[0],
                            fmax=self.flim[1],
                            sr=self.sr,
                            frame_length=self.NN, 
                            win_length=self.win_length,
                            hop_length=self.hop_length,
                            n_thresholds=100,
                            beta_parameters=(2, 18), 
                            boltzmann_parameter=2,
                            resolution=0.1,
                            max_transition_rate=35.92,
                            switch_prob=0.01, 
                            no_trough_prob=0.01,
                            fill_na=0,
                            center=self.center,
                            pad_mode='constant'
                        )
        elif self.ff_method=="yin":
            self.FF = yin(
                        self.s,
                        fmin=self.flim[0],
                        fmax=self.flim[1],
                        sr=self.sr,
                        frame_length=self.NN, 
                        win_length=self.win_length,
                        hop_length=self.hop_length,
                        center=self.center,
                        trough_threshold=self.umbral_FF,
                        pad_mode='constant'
                    )
        elif self.ff_method=="both":
            self.FF2,_,_ = pyin(
                            self.s,
                            fmin=self.flim[0],
                            fmax=self.flim[1],
                            sr=self.sr,
                            frame_length=self.NN, 
                            win_length=self.win_length,
                            hop_length=self.hop_length,
                            n_thresholds=100,
                            beta_parameters=(2, 18), 
                            boltzmann_parameter=2,
                            resolution=0.1,
                            max_transition_rate=35.92,
                            switch_prob=0.01, 
                            no_trough_prob=0.01,
                            fill_na=0,
                            center=self.center,
                            pad_mode='constant'
                        )
            self.FF = yin(
                        self.s,
                        fmin=self.flim[0],
                        fmax=self.flim[1],
                        sr=self.sr,
                        frame_length=self.NN, 
                        win_length=self.win_length,
                        hop_length=self.hop_length,
                        center=self.center,
                        trough_threshold=self.umbral_FF,
                        pad_mode='constant'
                    )
        elif self.ff_method=="manual":
            print("Not implemented yet.")
        
        self.timeFF = np.linspace(0,self.time[-1],self.FF.size)
        self.FF_fun = interp1d(self.timeFF, self.FF)
        self.SCI = self.f_msf / self.FF_fun(self.time)
    #%%
    def solve(
        self,
        z: List[ArrayLike] = _z,
        params: Dict = _params,
        order: int = 2,
        method: Literal["best", "fast"] = "best"
    ) -> Syllable :
        """
        
        
        Parameters
        ----------
            z: List[ArrayLike]

            params : dict

            order : int

        Return
        ------
            synth : Syllable


        Examples
        --------
            >>>
        """
        self.params = _params
        self.z = _z
        # update parameters if given
        # if self.params!=_params:
        for k in params.keys():
            self.params[k] = params[k]
        # if self.z!=_z:
        for k in z.keys():
            self.z[k] = z[k]
        # define alpha and beta parameters
        curves = alpha_beta(self, self.z, method)
        # generate the synthetic syllable
        synth = motor_gestures(self, curves, self.params)
        synth = self.synth_scores(synth, order=order)
        
        return synth
    #%%
    def synth_scores(
        self,
        synth: Syllable,
        order: int = 2
    ) -> Syllable:
        """
        
        
        Parameters
        ----------
            synth : Sylllable

            order : int
        
        Return
        ------

                
        Example
        -------
            synth : Syllable

        """
        synth.envelope = envelope(synth.s, synth.sr, synth.Nt)
        synth.acoustical_features(
            stft_window = synth.stft_window,
            umbral_FF = synth.umbral_FF,
            ff_method = synth.ff_method,
            overlap = synth.overlap,
            llambda = synth.llambda,
            center = synth.center,
            n_mfcc = synth.n_mfcc,
            n_mels = synth.n_mels,
            NN = synth.NN,
            Nt = synth.Nt
        )
        # residual difference between real and synthetic samples
        synth.deltaCentroid = np.abs(synth.centroid - self.centroid)
        synth.deltaMfccs = np.abs(synth.mfccs - self.mfccs)
        synth.deltaFmsf = np.abs(synth.f_msf - self.f_msf)
        synth.deltaEnv = np.abs(synth.envelope - self.envelope)
        synth.deltaSCI = np.abs(synth.SCI - self.SCI)
        synth.deltaRMS = np.abs(synth.rms - self.rms)
        synth.deltaSxx = np.abs(synth.Sxx_dB - self.Sxx_dB)
        synth.deltaMel = np.abs(synth.FF_coef - self.FF_coef)
        synth.deltaFF = np.abs(synth.FF - self.FF)
        ## --------- normalizing ----------------------
        synth.deltaCentroid /= np.max(synth.centroid)
        synth.deltaMfccs /= np.max(synth.deltaMfccs)
        synth.deltaFmsf /= synth.f_msf
        synth.deltaSCI /= synth.SCI
        synth.deltaEnv /= synth.envelope
        synth.deltaRMS /= synth.rms
        synth.deltaSxx /= np.max(synth.deltaSxx)
        synth.deltaMel /= np.max(synth.deltaMel)
        synth.deltaFF /= synth.FF
        # --------------- scoring variables --------------------
        synth.scoreCentroid = norm(synth.deltaCentroid, ord=order)
        synth.scoreFmsf = norm(synth.deltaFmsf, ord=order)
        synth.scoreMfccs = norm(synth.deltaMfccs, ord=np.inf)
        synth.scoreSCI = norm(synth.deltaSCI, ord=order)
        synth.scoreEnv = norm(synth.deltaEnv, ord=order)
        synth.scoreRMS = norm(synth.deltaRMS, ord=order)
        synth.scoreSxx = norm(synth.deltaSxx, ord=np.inf)
        synth.scoreMel = norm(synth.deltaMel, ord=np.inf)
        synth.scoreFF = norm(synth.deltaFF, ord=order)
        # ------------------- removing size dependency -------------------
        synth.scoreCentroid /= synth.deltaCentroid.size
        synth.scoreMfccs /= synth.deltaMfccs.size
        synth.scoreFmsf /= synth.deltaFmsf.size
        synth.scoreSCI /= synth.deltaSCI.size
        synth.scoreEnv /= synth.deltaEnv.size
        synth.scoreRMS /= synth.deltaRMS.size
        synth.scoreSxx /= synth.deltaSxx.size
        synth.scoreMel /= synth.deltaSxx.size
        synth.scoreFF /= synth.deltaFF.size
        # -------------------- variables mean -------------------------
        # synth.scoreNoHarm = deltaNOP*10**(deltaNOP-2)
        synth.scoreCentroid_mean = synth.scoreCentroid.mean()
        synth.scoreFmsf_mean = synth.deltaFmsf.mean()
        synth.deltaSCI_mean = synth.deltaSCI.mean()
        synth.scoreRMS_mean = synth.scoreRMS.mean()
        synth.deltaEnv_mean = synth.deltaEnv.mean()
        synth.deltaFF_mean = synth.deltaFF.mean()        
        # ------------- acoustic dissimilarity indexes (adi) ---------------
        synth.correlation = np.zeros_like(synth.time)
        synth.SKL = np.zeros_like(synth.time)
        synth.Df = np.zeros_like(synth.time)
        for i in range(synth.mfccs.shape[1]):
            x = self.mfccs[:,i]
            y = synth.mfccs[:,i]
            r = norm(x*y,ord=1) / (norm(x,ord=2)*norm(y,ord=2))
            
            Df = x*np.log2(np.abs(x/y)) + y*np.log2(np.abs(y/x))
            synth.correlation[i] = np.sqrt(1-r)
            synth.SKL[i] = 0.5*norm(np.abs(x-y), ord=1)
            synth.Df[i] = 0.5*norm(Df, ord=1)
            #synth.Df[np.argwhere(np.isnan(synth.Df))]=-10
        # ------------- normalizing adi -----------------
        # synth.correlation /= synth.correlation.max()
        synth.SKL /= synth.SKL.max()
        synth.Df /= synth.Df.max()
        # computing adi scores
        synth.scoreCorrelation = norm(synth.correlation, ord=order)
        synth.scoreSKL = norm(synth.SKL, ord=order)
        synth.scoreDF = norm(synth.Df, ord=order)
        # normalizing
        synth.scoreCorrelation /= synth.correlation.size
        synth.scoreSKL /= synth.SKL.size
        synth.scoreDF /= synth.Df.size
        # mean scores
        mean_scores = np.mean(synth.correlation+synth.Df+synth.scoreSKL)
        synth.residualCorrelation = synth.scoreFF - mean_scores
        synth.SCIFF = synth.scoreSCI + synth.scoreFF

        return synth
    #%%
    def export_mg(self, dataframe: bool=False) -> DataFrame|None:
        """
        
        
        Parameters
        ----------
            

        Return
        ------
            synth : Syllable


        Examples
        --------
            >>>
        """
        if "synth" not in self.id:
            raise Exception("You only can export motor gestures"
                            + " parameters from synthetic objects")
        # ------------ export p values and alpha-beta arrays ------------
        file_name = self.file_name.replace("synth_","")
        type = self.type if type!="" else ""
        info = {
            "t_ini": round(self.t_interval[0], 4),
            "t_end": round(self.t_interval[1], 4),
            "f_ini": self.flim[0],
            "f_end": self.flim[1],
            "id": self.id,
            "no_syllable": self.no_syllable,
            "sr": self.sr,
            "NN": self.NN,
            "umbral_FF": self.umbral_FF,
            "ff_method": self.ff_method,
            "type": type,
            "metadata": str(self.metadata),
            "file_name": file_name,
            "root_folder": self.proj_dirs.ROOT,
            "audios_folder": self.proj_dirs.AUDIOS,
            "z": str(self.z)
        }

        name = f"{file_name[:-4]}-{self.no_syllable}-MG.csv"\
                if type!="" \
                else f"{file_name[:-4]}-{self.no_syllable}-{self.type}-MG.csv"
        path = self.proj_dirs.MG_param / name
        df_mg = pd.DataFrame.from_dict(info, orient="index", columns=["value"])
        df_mg.to_csv(path, index=True)
        print(f"Motor gesture parameters saved at {path}.")

        if dataframe:
            return df_mg
    #%%
    def play(self) -> Audio:
        """


        Parameters
        ----------

        Return
        ------

        Examples
        --------
            >>>
        """
        return Audio(data=self.s, rate=self.sr)
    #%%    
    def write_audio(self, bit_depth: int = 16) -> None:
        """
        
        
        Parameters
        ----------

        Return
        ------

        Examples
        --------
            >>>
        """
        audio_name = f'{self.file_name[:-4]}-{self.id}-{self.no_syllable}.wav'
        path_name = self.proj_dirs.examples / audio_name
        write(filename=path_name, fs=self.sr, data=self.s, bit_depth=bit_depth)
        print(f"Audio saved at {path_name}.")