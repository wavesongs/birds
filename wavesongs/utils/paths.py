#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Create and manage project directory trees: input audios
and results (images, parameters files, and audios)
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path


from os import makedirs
from librosa import load
from os.path import isdir, basename
from pathlib import Path, PosixPath
from typing import Union, List, AnyStr, TypeVar

from wavesongs.models.bird import alpha_beta, motor_gestures
import wavesongs as ws

# prefix components:
space =  '    '
branch = '│   '
# pointers:
tee =    '├── '
last =   '└── '

_CATALOG_LABEL = "ML Catalog Number"
_AUDIO_FORMATS = (".mp3", ".wav")

Syllable = TypeVar('Syllable')
#%%
class ProjDirs:
    """
    Creates a ProjDirs class,  which is used to store a project's 
    file structure. This is required when constructing
    a :class:`~wavesongs.obj.Syllable` or a :class:`~wavesongs.obj.Song` 
    objects and generally useful to keep paths tidy and in the same
    location.

    Parameters
    ----------
        audios : str ='./assets/audio'
            Folder path where the audio records samples are saved.
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
        self.IMAGES = self.RESULTS / "figures"
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
        Search for all audios, mp3 and wav types, in the audios folder. 
        
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
            the method will return a dataframe. However, the parameter
            `files_names` always is present.

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
        Get information about the audios folder: audios paths and 
        number of audios.

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
                Aduio path location.

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
        synth = ws.objs.syllable.Syllable(proj_dirs=self, duration=duration, sr=sr)
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
    
    def read_mg(
        self,
        file_name: AnyStr,
        no_syllable: Union[int, AnyStr],
        type: AnyStr = ""
    ) -> Syllable:
        """
        Read motor gesture parameters from csv file
        
        Parameters
        ----------
            proj_dirs : ProjDirs

            file_name: AnyStr

            no_syllable: Union[int, AnyStr]

            type: AnyStr = ""

            
        Return
        -------
            synth: Syllable
        
        Example
        -------
            >>>
        """
        folder = self.mg_param # f"{results}/mg_param"
        file_name = f"{folder}/{file_name}-{no_syllable}-mg.csv" \
                        if type=="" \
                        else f"{folder}/{file_name}-{no_syllable}-{type}-mg.csv"
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


    def tree_list(self, prefix: str=''):
        """A recursive generator, given a directory Path object
        will yield a visual tree structure line by line
        with each line prefixed by the same characters
        """
        dir_path = Path("./")
        contents = list(dir_path.iterdir())
        # contents each get pointers that are ├── with a final └── :
        pointers = [tee] * (len(contents) - 1) + [last]
        for pointer, path in zip(pointers, contents):
            yield prefix + pointer + path.name
            if path.is_dir(): # extend the prefix and recurse:
                extension = branch if pointer == tee else space 
                # i.e. space because last, └── , above so no more |
                yield from self.tree(prefix=prefix+extension)
        
    def tree(self, prefix: str='') -> str:
        tree_str = ""
        possibles = ["assets", "results", "audios", "figures", "mg_params"]
        for line in self.tree_list(prefix):
            count = 0
            for pos in possibles:
                if pos in line:
                    count += 1
            if count>=1:
                tree_str += line+"\n"
        print(tree_str)

        return tree_str
    
    def __str__(self):
        return f"""
    Audios: {self.AUDIOS}
    Results: {self.RESULTS}

        
    """