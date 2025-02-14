#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Create project directory trees, assign path variables, and get paths for
different types of output files
"""
import numpy as np
import pandas as pd

from os import makedirs
from librosa import load
from os.path import isdir, basename
from pathlib import Path, PosixPath
from typing import Union, List, AnyStr

from wavesongs.models.bird import alpha_beta, motor_gestures
import wavesongs as ws

_CATALOG_LABEL = "ML Catalog Number"
_AUDIO_FORMATS = (".mp3", ".wav")

#%%

class ProjDirs:
    """

    Parameters
    ----------
        audios : str ='./assets/audio'
            Folder path where the audio samples are saved.
        results: str = "./assets/results"
            Folder path for the generated files and data.
        metadata: str = "spreadsheet.csv"
            Name of the csv file with the metadata of the audios. 
            Usually given by the data provider.

    Attributes
    ----------

    Example
    -------
        >>> proj_dirs = ProjDirs(
        >>>     "./assets/audio", "./assets/results", "spreadsheet.csv"
        >>> )
    """
    # %%
    def __init__(
        self,
        audios: AnyStr = "./assets/audio",
        results: AnyStr = "./assets/results",
        metadata: AnyStr = "spreadsheet.csv",
        catalog: bool = False
    ):
        """Constructor"""
        self.AUDIOS = Path(audios)
        self.RESULTS = Path(results)

        self.mg_param = self.RESULTS / "mg_params"
        self.IMAGES = self.RESULTS / "images"
        self.examples = self.RESULTS / "audios"
        
        self.SPREADSHEET = self.AUDIOS / metadata
        self.CATALOG = catalog

        # create folder in case they do not exist
        Path(self.RESULTS).mkdir(parents=True, exist_ok=True)
        Path(self.mg_param).mkdir(parents=True, exist_ok=True)
        Path(self.IMAGES).mkdir(parents=True, exist_ok=True)
        Path(self.examples).mkdir(parents=True, exist_ok=True)
        
        # Check if there is a metadata spreadsheet file inside audios folder
        spreadsheet_file = list(Path(self.AUDIOS).glob("*" + metadata))
        if len(spreadsheet_file) > 0 and self.CATALOG==True:
            self.CATALOG = True 
            self.CATALOG_LABEL = _CATALOG_LABEL

        self.find_audios()
    # %%
    def find_audios(self) -> Union[List, pd.DataFrame]:
        """
        Search for all audios, mp3 and wav type, in the audios folder. 
        
        Parameters
        ----------
            None

        Return
        ------
            files_names : list
                List with the audios files names

        Notes
        -----
            If the audios folder contains a metadata file, spreadsheet.csv,
            the method will return a dataframe. However, the attribute
            files_names always is present.

        Example
        -------
            >>>
        """
        all_filles = Path(self.AUDIOS).glob("**/*")
        self.files = [a for a in all_filles if a.suffix in _AUDIO_FORMATS]
        self.files_names = [basename(f) for f in self.files]
        self.no_files = len(self.files_names)

        if self.CATALOG is True:
            self.data = pd.read_csv(self.SPREADSHEET, encoding_errors="ignore")
            self.data.dropna(axis=0, how="all", inplace=True)
            # self.data = data.convert_dtypes()
            self.data = self.data.astype({self.CATALOG_LABEL: "str"})
            found_files = [
                (
                    str(self.AUDIOS) + f"/{file}.mp3"
                    if file + ".mp3" in self.files_names
                    else str(self.AUDIOS) + f"/{file}.wav"
                )
                for file in self.data[self.CATALOG_LABEL]
            ]
            self.data["File Path"] = found_files
            self.no_files = len(self.data)

            return self.data

        return self.files_names
    # %%
    def audios_info(self) -> None:
        """
        Display information about the audios folder: audios path and 
        number of audios in the folder.

        Parameters
        ----------
            None

        Return
        ------
            None

        Example
        -------
            >>>
        """
        print(f"Audios path: {self.AUDIOS}\n")
        print("The folder has {} audio samples:".format(self.no_files))

        if self.CATALOG:
            print(self.data)
        else:
            for file in self.files_names:
                print("  - " + file)

    def find_audio(self, id: AnyStr) -> PosixPath:
        """
        Find an audio in the audios folder by the id or filename

        Parameters
        ----------
            id : str
                Whole filename of a part of it. Usually, the catalog number. 

        Return
        ------
            path : PosixPath
                Path location of the audio. 

        Example
        -------
            >>>
        """
        if self.CATALOG:
            id_df = self.data.loc[self.data[self.CATALOG_LABEL] == id]
            path = PosixPath(id_df["File Path"].values[0])
        else:
            path = [
                self.files[i]
                for i in range(len(self.files))
                if id in self.files_names[i]
            ][0]
        return path
    
    # %%
    def import_mg(self, id, no_syllable=0):
        all_filles = Path(self.mg_param).glob("**/*")
        path_mg = [a for a in all_filles
                if f"-{no_syllable}-" in str(a) and id in str(a) and "mg." in str(a)][0]
        
        mg_df = pd.read_csv(path_mg, index_col=0)
        mg_df = mg_df.to_dict()["value"]

        t0 = float(mg_df["t_ini"])
        sr = int(mg_df["sr"])
        duration = float(mg_df["duration"])
        self.AUDIOS = mg_df["audios_folder"]
        params = eval(mg_df["params"])
        
        #self = ProjDirs(audios=audios_folder, results=self.RESULTS)
        synth = ws.obj.Syllable(proj_dirs=self, duration=duration, sr=sr)
        synth.id = mg_df["id"]
        synth.type = mg_df["type"]
        synth.no_syllable = int(mg_df["no_syllable"])
        synth.metadata = mg_df["metadata"]
        synth.file_name = mg_df["file_name"]
        synth.z = eval(mg_df["z"])
        synth.t0_bs = t0
        
        if "curves_csv" in mg_df.keys():
            curves_df = pd.read_csv(mg_df["curves_csv"], index_col=0)
            time_s = curves_df["time"].array
            alpha = curves_df["alpha"].array
            beta = curves_df["beta"].array
            duration = time_s[-1]
            curves = [alpha, beta]
            synth.alpha = alpha
            synth.beta = beta
        else:
            curves = alpha_beta(synth, synth.z, "fast")

        
        synth = motor_gestures(synth, curves, params)
        synth.acoustical_features(
            NN = int(mg_df["NN"]),
            ff_method = mg_df["ff_method"],
            umbral_FF = float(mg_df["umbral_FF"]),
            flim = [float(mg_df["f_ini"]), float(mg_df["f_end"])],
            Nt = int(mg_df["Nt"]),
            center = mg_df["center"],
            overlap = float(mg_df["overlap"]),
            llambda = float(mg_df["llambda"]),
            n_mfcc = int(mg_df["n_mfcc"]),
            n_mels = int(mg_df["n_mels"]),
            stft_window = mg_df["stft_window"]
        )
        
        return synth