import os
from pathlib import Path
from typing import Dict


def get_current_file_path(file_name: str) -> Path: 
  """
    Get the root catalog full path. 

    Arguments:
      file_name: The current python file name, where the function 
                 is called from.

    Returns:
      Full path.
  """
  return Path(os.path.dirname(os.path.realpath(file_name))) 


def fetch_paths(root_catalog: Path) -> Dict: 
  """
    Fetch all paths in the project. If there is no file with saved config, then
    the function will create the config.

    Arguments:
      root_catalog: Root catalog path. 

    Returns:
      Dictionary with the structure: { PATH_NAME : PATH }.
  """
  RESOURCES_CATALOG_PATH: Path = root_catalog.joinpath("resources/")
  PATH_CONFIG_FILE: Path = RESOURCES_CATALOG_PATH.joinpath("paths.json")
  WEIGHTS_CATALOG_PATH: Path = RESOURCES_CATALOG_PATH.joinpath("weights/")
  IMAGES_CATALOG_PATH: Path = RESOURCES_CATALOG_PATH.joinpath("images/")
  SPACY_CATALOG_PATH: Path = RESOURCES_CATALOG_PATH.joinpath("spacy/")
  UNET_WEIGHTS_CATALOG_PATH: Path = WEIGHTS_CATALOG_PATH.joinpath("unet")
  DDPM_WEIGHTS_CATALOG_PATH: Path = WEIGHTS_CATALOG_PATH.joinpath("ddpm")
  TRANSFORMER_WEIGHTS_CATALOG_PATH: Path = WEIGHTS_CATALOG_PATH.joinpath(
    "transformer/") 

  return {
    "ROOT" : RESOURCES_CATALOG_PATH,
    "PATH_CONFIG" : PATH_CONFIG_FILE,
    "WEIGHTS_CATALOG": WEIGHTS_CATALOG_PATH,
    "IMAGES_CATALOG": IMAGES_CATALOG_PATH,
    "SPACY_CATALOG": SPACY_CATALOG_PATH,
    "UNET_WEIGHTS_CATALOG": UNET_WEIGHTS_CATALOG_PATH,
    "DDPM_WEIGHTS_CATALOG": DDPM_WEIGHTS_CATALOG_PATH,
    "TRANSFORMER_WEIGHTS_CATALOG": TRANSFORMER_WEIGHTS_CATALOG_PATH
  }


