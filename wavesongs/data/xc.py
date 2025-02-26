"""Query and download data from Xeno Canto"""

import os
import shutil
import pandas as pd
from maad import util
from pathlib import Path
from typing import List, Optional, Union

# %%
def download_audios(
    df_dataset: pd.DataFrame,
    rootdir: str = "./assets/audio", 
    dataset_name: str = "",
    overwrite: bool = True,
    save_csv: bool = True,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Download audios from `Xeno Canto <https://xeno-canto.org/>`_  with 
    `maad.utils.xc_download <https://scikit-maad.github.io/util.html#xeno-canto>`_.

    Args:
        df_dataset (pd.DataFrame):
            Data Frame with the information to download.
        country (str, optional):
            _description_. Defaults to "Colombia".
        rootdir (str, optional):
            _description_. Defaults to "./assets/audio".
        dataset_name (str, optional):
            _description_. Defaults to ''.
        overwrite (bool, optional):
            _description_. Defaults to False.
        save_csv (bool, optional):
            _description_. Defaults to True.
        verbose (bool, optional):
            _description_. Defaults to True.

    Returns:
        df_audios (pd.DataFrame) : Data Frame
    """
    df_audios = util.xc_download(
      df=df_dataset, 
      rootdir=rootdir,
      dataset_name=dataset_name,
      overwrite=overwrite,
      save_csv=save_csv,
      verbose=verbose,
    )

    gen, sp, en = df_dataset.iloc[0][["gen", "sp", "en"]].values

    downloaded_folder = f"{gen} {sp}_{en}"
    
    # can be improved
    dataset_path = f"{rootdir}/{downloaded_folder}" if dataset_name=="" \
                    else f"{rootdir}/{dataset_name}/{downloaded_folder}"
    all_filles = Path(dataset_path).glob("**/*")
    if dataset_name=="": 
        dataset_name = f"{gen}_{sp}".lower()
    Path(f"{rootdir}/{dataset_name}").mkdir(parents=True, exist_ok=True)
    for file in all_filles:
        file_name = str(file).split("/")[-1]
        new_name = f"{rootdir}/{dataset_name}/{file_name}" if dataset_name!="" \
                    else f"{rootdir}/{file_name}"
        Path(file).rename(new_name)
        print(f"Audio saved at {new_name}.")
    shutil.rmtree(dataset_path)

    return df_audios

# %%
def query_audios(
    specie_names: Union[List[str], List[List[str]]],
    max_nb_files: Optional[int] = None,
    random_seed: int = 2025,
    info: dict = {},
    format_time=True,
    format_date=True,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Query me from `Xeno Canto <https://xeno-canto.org/>`_ 
    with `maad.utils.xc_multi_query <https://scikit-maad.github.io/util.html#xeno-canto>`_.
    
    Args:
        specie_names (list[str], list[list[str]]): 
            List with english and scientific specie names.
        max_nb_files (int, optional):
            Maximum number of files to download. Defaults to None.
        random_seed (int, optional):
            Random seed. Defaults to 2025.
        info (dict, optional):
            Dictionary with information to query. Defaults to {}.
        format_time (bool, optional):
            Format time. Defaults to True.
        format_date (bool, optional):
            Format date. Defaults to True.
        verbose (bool, optional):
            Verbose. Defaults to True.

    Returns:
        df_query (pd.DataFrame) : Data Frame
    """
    if type(specie_names[0])==str:
        specie_names = [specie_names]
    df_species = pd.DataFrame(
        data=specie_names,
        columns=['english name', "scientific name"]
        )

    gen = []
    sp = []
    for name in df_species['scientific name']:
        gen.append(name.rpartition(' ')[0])
        sp.append(name.rpartition(' ')[2])

    df_query = pd.DataFrame()
    df_query['gen'] = gen
    df_query['sp'] = sp

    for key in info.keys():
        df_query[key] = f'{key}:{info[key]}'

    df_dataset = util.xc_multi_query(
        df_query=df_query,
        max_nb_files=max_nb_files,
        random_seed=random_seed,
        format_time=format_time,
        format_date=format_date,
        verbose=verbose
    )

    return df_dataset