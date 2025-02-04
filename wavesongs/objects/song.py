#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" """
from wavesongs.utils.paths import ProjDirs
from wavesongs.utils.tools import envelope

import numpy as np
import pandas as pd

from scipy.signal import find_peaks
from maad.rois import create_mask
from maad.sound import write
from IPython.display import Audio

from os.path import (
    basename,
    normpath
)

from librosa import (
    stft,
    reassigned_spectrogram,
    amplitude_to_db,
    fft_frequencies,
    times_like,
    yin,
    load
)

from maad.sound import (
    normalize,
    wave2frames
)

from typing import (
    Tuple,
    AnyStr,
    Dict,
    Optional
)


class Song:
    """
    Store a song and its properties in a class 
    
    Parameters
    ----------
        proj_dirs : ProjDirs

        file_id : str
            Name or id of the audio sample
        tlim : tuple
            Time range
        flim : tuple
            Frequency range
        sr : int
            Sample rate
        info : dict
            Audio metadata.
        id : str = "song"

    
    Attributes
    ----------

    Example
    -------
        >>>
    """
    def __init__(
        self,
        proj_dirs: ProjDirs,
        file_id: AnyStr,
        tlim: Tuple[float] = (0, 60),
        flim: Tuple[float] = (1e3, 2e4),
        sr: int = 44100,
        info: Dict = {},
        id: AnyStr = "song"
    ):  
        self.proj_dirs = proj_dirs
        self.file_id = file_id
        self.tlim = tlim
        self.info = info
        self.flim = flim
        
        self.file_path = proj_dirs.find_audio(file_id)
        self.file_name = basename(normpath(self.file_path))
        
        if self.proj_dirs.CATALOG:
            self.info = proj_dirs.data.iloc[0].to_dict()
        
        s, sr = load(self.file_path, sr=sr, mono=True)
        self.id = id
        self.sr = sr
        # croping the audio in the range tlim
        s = s[int(self.tlim[0]*self.sr):int(self.tlim[1]*self.sr)]
        self.t0 = self.tlim[0]
        self.t0_bs = self.tlim[0]
        
        self.s = normalize(s, max_amp=1.0)
        
    def acoustical_features(
        self,
        llambda: float = 1.,
        NN: int = 1024,
        overlap: float = 0.5,
        center: bool = False,
        umbral_FF: float = 1.05,
        ff_method: AnyStr = 'yin',
        Nt: int = 100,
        n_mfcc: int = 4,
        n_mels: int = 4,
        stft_window: AnyStr = "hann",
        tlim: Optional[Tuple[float]] = None,
        flim: Optional[Tuple[float]] = None
    ) -> None:
        """
        Coputing acoustical tempo-spectral variables
        
        Parameters
        ----------
            llambda : float

            NN : int

            overlap : float

            center : bool = False

            umbral_FF : int

            ff_method : str

            Nt : int

            n_mfcc : int

            stft_window : str

        Return
        ------
            None

        Examples
        --------
            >>>
        """
        if tlim is not None: self.tlim = tlim
        if flim is not None: self.flim = flim
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.stft_window = stft_window

        self.llambda = llambda
        self.center = center
        
        self.umbral = 0.05
        self.umbral_FF = umbral_FF
        self.ff_method = ff_method
        
        self.NN = NN
        self.Nt = Nt
        self.center = center
        self.overlap = overlap
        self.win_length = self.NN//2
        self.hop_length = self.NN//4
        self.no_overlap = int(overlap*self.NN)
        
        self.time_s = np.linspace(0, len(self.s)/self.sr, len(self.s))
        self.envelope = envelope(self.s, self.sr, Nt=self.Nt)

        # Short-time Fourier transform (STFT)
        self.stft = stft(y=self.s,
                         n_fft=self.NN,
                         hop_length=self.hop_length,
                         win_length=self.NN,
                         window=self.stft_window,
                         center=self.center,
                         dtype=None,
                         pad_mode='constant')
        
        # Time-frequency reassigned spectrogram
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
                                dtype=None,
                                pad_mode ='constant'
                            )
        self.freqs = freqs  
        self.times = times 
        self.Sxx = mags 
        self.Sxx_dB = amplitude_to_db(mags, ref=np.max)
        self.freq = fft_frequencies(sr=self.sr, n_fft=self.NN) 
        self.time = times_like(X=self.stft,
                               sr=self.sr,
                               hop_length=self.hop_length,
                               n_fft=self.NN) #, axis=-1
        # put in origin the time
        self.time -= self.time[0]
        
        # method to calculate fundamental frequency
        self.FF = yin(self.s, 
                      fmin=self.flim[0],
                      fmax=self.flim[1],
                      sr=self.sr,
                      frame_length=self.NN, 
                      win_length=self.win_length,
                      hop_length=self.hop_length,
                      trough_threshold=self.umbral_FF,
                      center=self.center,
                      pad_mode='constant')

    def play(self) -> Audio:
        """
        
        
        Parameters
        ----------

        Return
        ------

        Example
        -------
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
        audio_name = f'{self.file_name[:-4]}-{self.id}.wav'
        path_name = self.proj_dirs.examples / audio_name
        write(filename=path_name, fs=self.sr, data=self.s, bit_depth=bit_depth)
        print(f"Audio saved at {path_name}.")
    #%%
    # def Syllable(self, no_syllable, NN=1024):
    #     self.no_syllable   = no_syllable
    #     ss                 = self.syllables[self.no_syllable-1]  # syllable indexes 
    #     self.syll_complet  = self.s[ss]       # audios syllable
    #     self.time_syllable = self.time_s[ss]
    #     self.t0            = self.time_syllable[0]
        
    #     self.syllable      = Syllable(self, tlim=(self.time_syllable[0], self.time_syllable[-1]), flim=self.flim, NN=NN, file_name=self.file_name+"synth")
            
    #     self.syllable.no_syllable  = self.no_syllable
    #     self.syllable.file_name    = self.file_name
    #     self.syllable.state        = self.state
    #     self.syllable.country      = self.country
    #     self.syllable.no_file      = self.no_file
    #     self.syllable.proj_dirs        = self.proj_dirs
    #     self.syllable.id           = "syllable"
        
    #     self.SylInd.append([[no_syllable], [ss]])
        
    #     fraction = self.syll_complet.size/1024
    #     Nt_new = int(((fraction%1)/fraction+1)*1024)
    #     self.chuncks    = wave2frames(self.syll_complet,  Nt=Nt_new)
    #     self.times_chun = wave2frames(self.time_syllable, Nt=Nt_new)
    #     self.no_chuncks = len(self.chuncks)
        
    #     return self.syllable
    
    # #%%
    # def SyntheticSyllable(self):
    #     self.s_synth = np.empty_like(self.s)
    #     for i in range(self.syllables.size):
    #         self.s_synth[self.SylInd[i][1]] = self.syllables[i]

    #%%
    
    # #%%
    # def Set(self, p_array):
    #     self.p["a0"].set(value=p_array[0])
    #     self.p["a1"].set(value=p_array[1])
    #     self.p["a2"].set(value=p_array[2])
    #     self.p["b0"].set(value=p_array[3])
    #     self.p["b1"].set(value=p_array[4])
    #     self.p["b2"].set(value=p_array[5])

# #%%
#     def Syllables(self, method="freq"):
#         if method=="amplitud":
#             supra      = np.where(self.envelope > self.umbral)[0]
#             candidates = np.split(supra, np.where(np.diff(supra) != 1)[0]+1)
            
#             return [x for x in candidates if len(x) > 2*self.NN] 
#         elif method=="freq":
#             # ss = np.where((self.FF < self.flim[1]) & (self.FF>self.flim[0])) # filter frequency
#             # ff_t   = self.time[ss]                        # cleaning timeFF
#             # FF_new = self.FF[ss]                            # cleaning FF
#             # FF_dif = np.abs(np.diff(FF_new))                # find where is it cutted
#             # # alternative form with pandas
#             df = pd.DataFrame(data={"FF":self.FF, "time":self.time})
#             q  = df["FF"].quantile(0.99)
#             df[df["FF"] < q]
#             q_low, q_hi = df["FF"].quantile(0.1), df["FF"].quantile(0.99)
#             df_filtered = df[(df["FF"] < q_hi) & (df["FF"] > q_low)]
            
#             ff_t   = self.time[df_filtered["FF"].index]
#             FF_new = self.FF[df_filtered["FF"].index]
#             FF_dif = np.abs(np.diff(FF_new))
#             # plt.plot(self.FF, 'o');  plt.plot(df_filtered["FF"], 'o')
            
#             peaks, _ = find_peaks(FF_dif, distance=10, height=500) # FF_dif
#             syl = [np.arange(peaks[i]+1,peaks[i+1]) for i in range(len(peaks)-1)]
#             syl = [np.arange(0,peaks[0])]+syl+[np.arange(peaks[-1]+1,len(ff_t))]

#             syl_intervals = np.array([[ff_t[s][0], ff_t[s][-1]] for s in syl])
#             indexes = np.int64(self.sr*syl_intervals)
#             indexes = [np.arange(ind[0],ind[1],1) for ind in indexes]
            
#             return [x for x in indexes if len(x) > 2*self.NN]
        
#         elif "maad":
#             im_bin = create_mask(self.Sxx_dB, bin_std=1.5, bin_per=0.5, mode='relative')
        