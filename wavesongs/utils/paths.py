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

        self.MG_param = self.RESULTS / "mg_params"
        self.IMAGES = self.RESULTS / "images"
        self.examples = self.RESULTS / "audios"
        
        self.SPREADSHEET = self.AUDIOS / metadata
        self.CATALOG = catalog

        # create folder in case they do not exist
        Path(self.RESULTS).mkdir(parents=True, exist_ok=True)
        Path(self.MG_param).mkdir(parents=True, exist_ok=True)
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

    # #%%
    # def ImportParameters1(self, XC=None, no_file=None, no_syllable=None, name=None):
    #     self.data_param = self.MG_Files()

    #     if name is not None and XC is None and no_file is None :
    #         df = self.data_param["name"].str.contains(name,case=False)

    #     if XC is not None and no_file is None and name is None:
    #         df = self.data_param[self.data_param['id_XC'] == XC]
    #     if no_file is not None and XC is None and name is None:
    #         df = self.data_param.iloc[no_file]
    #     if no_file is None and XC is None:
    #         df = self.data_param
    #     if no_file is None and XC is None and name is None:
    #         df = self.data_param

    #     if no_syllable is not None:
    #         df = df[df['no_syllable'] == str(no_syllable)]
    #         coef = pd.read_csv(df["coef_path"].values[0]).rename(columns={"Unnamed: 0":"parameter"})
    #         tlim = pd.Series({"t_ini":coef.iloc[-2].value, "t_end":coef.iloc[-1].value})
    #         df = pd.concat([df, tlim]).reset_index()

    #         return df#, coef#, motor_gesture
    #     else:                               # if syllables is Nonex
    #         coefs, type, out, tlim, NN, umbral_FF, country, state = [], [], [], [], [], [], [], []
    #         for i in df.index:
    #             coef = pd.read_csv(self.data_param.iloc[i]["coef_path"], index_col="Unnamed: 0", engine='python')#, encoding = "utf-8") #cp1252
    #             tlim.append([float(coef.iloc[7].value), float(coef.iloc[8].value)])
    #             NN.append(int(coef.iloc[9].value))
    #             umbral_FF.append(float(coef.iloc[10].value))
    #             type.append(coef.iloc[11].value)
    #             country.append(coef.iloc[12].value)
    #             state.append(coef.iloc[13].value)
    #             coefs.append(coef.iloc[:7].astype('float64'))
    #         tlim = np.array(tlim)

    #         df = pd.DataFrame({'id_XC':df['id_XC'], 'no_syllable':df['no_syllable'],
    #         'id':df['id'], 'name':df['name'], 'coef_path':df['coef_path'], 'param_path':df['param_path'],
    #         'audio_path':df['audio_path'], 's':df['s'], 'fs':df['fs'], 'file_name':df['file_name'],
    #         't_ini':tlim[:,0], 't_end':tlim[:,1], 'NN':NN, 'umbral_FF':umbral_FF, 'coef':coefs, 'type':type, 'country':country, 'state':state},
    #         index=df.index)

    #         out = [df.iloc[i] for i in range(len(df.index))]
    #         self.df = df.reset_index(drop=True, inplace=False)

    #         print("{} files were found.".format(len(df.index)))
    #         return out, df
    # #%%
    # def ImportParameters(self, no_syllable=None, country_filter=None):
    #     df = self.MG_Files()
    #     coefs, type, out, tlim, NN, umbral_FF, country, state = [], [], [], [], [], [], [], []
    #     for i in df.index:
    #         coef = pd.read_csv(self.data_param.iloc[i]["coef_path"], index_col="Unnamed: 0", engine='python').T#, encoding = "utf-8") #cp1252
    #         tlim.append([float(coef["t_ini"].value), float(coef["t_end"].value)])
    #         NN.append(int(coef["NN"].value))
    #         umbral_FF.append(float(coef["umbral_FF"].value))
    #         type.append(coef["type"].value)
    #         country.append(coef["country"].value)
    #         state.append(coef["state"].value)
    #         coefs.append(coef[["a0","a1","a2","b0","b1","b2","gm"]].values[0]) # coef.iloc[:7].astype('float64')
    #     tlim = np.array(tlim)

    #     df = pd.DataFrame({'id_XC':df['id_XC'], 'no_syllable':df['no_syllable'],
    #     'id':df['id'], 'name':df['name'], 'coef_path':df['coef_path'],
    #     # 'param_path':df['param_path'], 's':df['s'], 'fs':df['fs'],
    #     'audio_path':df['audio_path'], 'file_name':df['file_name'],
    #     't_ini':tlim[:,0], 't_end':tlim[:,1], 'NN':NN, 'umbral_FF':umbral_FF, 'coef':coefs, 'type':type, 'country':country, 'state':state},
    #     index=df.index)

    #     self.df = df.reset_index(drop=True, inplace=False)
    #     print("{} files were found.".format(len(self.df.index)))

    #     if country_filter is not None:
    #         self.df = self.df[self.df["country"]==country_filter]
    #     if no_syllable is not None:
    #         self.df = self.df[self.df["no_syllable"]==str(no_syllable)]

    #     return self.df
    # #%%
    # def MG_Files(self):
    #     self.MG_coef = list(self.MG_param.glob("*MG.csv"))
    #     MG_coef_splited  = [relpath(MG, self.MG_param) for MG in self.MG_coef]
    #     MG_coef_splited  = [relpath(MG, self.MG_param).replace(" ","").split("-") for MG in self.MG_coef]

    #     id_XC = [x[0] for x in MG_coef_splited]
    #     no_syllables = [x[-2] for x in MG_coef_splited]
    #     id = [x[-3] for x in MG_coef_splited]
    #     name = [x[1]+"-"+x[2] for x in MG_coef_splited]
    #     audios = [list(self.AUDIOS.glob(id+"*"))[0]  for id  in id_XC]
    #     file_name = [relpath(audio, self.AUDIOS) for audio in audios]

    #     self.data_param = pd.DataFrame({'id_XC':id_XC,
    #                                     'no_syllable': no_syllables,
    #                                     'id': id,
    #                                     'name':name,
    #                                     "coef_path":self.MG_coef,
    #                                     "audio_path":audios,
    #                                     "file_name":file_name})
    #                                             #"coef_path":self.MG_coef, "s":ss, "fs":fss,

    #     return self.data_param
