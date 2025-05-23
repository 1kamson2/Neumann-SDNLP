from dataset.dataset import *
from utils.config.parse_config import get_model_configs 
import argparse
from pathlib import Path
from utils.path_managing import fetch_paths, get_current_file_path
from typing import Dict

parser = argparse.ArgumentParser(prog="NeumannAI")
parser.add_argument(
  "--model",
  help="Specify what model you want to use. Choose between: nlp, ddpm",
  default="nlp",
  type=str,
)

parser.add_argument(
  "--batchsize",
  help="Specify batch size, that will be used in the training.",
  default=64,
  type=int,
)

parser.add_argument(
  "--N",
  help="Specify the number of layers, only for NLP Model.",
  default=6,
  type=int,
)

parser.add_argument(
  "--dmodel",
  help="Specify the dimensions of your model, only for NLP Model.",
  default=512,
  type=int,
)

parser.add_argument(
  "--dffn",
  help="Specify the dimensions of feed forward network, only for NLP Model.",
  default=2048,
  type=int,
)

parser.add_argument(
  "--h",
  help="Specify the hidden layers for your model, only for NLP Model.",
  default=8,
  type=int,
)

parser.add_argument(
  "--dropout",
  help="Specify the hidden layers for your model, only for NLP Model.",
  default=0.1,
  type=float,
)

parser.add_argument(
  "--imgchannels",
  help="Specify the number of channels of your photo, only for DDPM Model.",
  default=3,
  type=int,
)

parser.add_argument(
  "--imgsize",
  help="Specify the size of your photo (N x N), only for DDPM Model.",
  default=32,
  type=int,
)

parser.add_argument(
  "--nchannels",
  help="Specify the number of channels, only for NLP Model.",
  default=64,
  type=int,
)

parser.add_argument(
  "--nsamples",
  help="Specify the number of samples, only for NLP Model.",
  default=16,
  type=int,
)

parser.add_argument(
  "--nsteps",
  help="Specify the number of steps in sampling, only for NLP Model.",
  default=1000,
  type=int,
)

parser.add_argument(
  "--lr",
  help="Specify the learning rate for your model.",
  default=2e-5,
  type=float,
)

parser.add_argument(
  "--verbose", "-v", help="Get more info about your config", action="store_true"
)

parser.add_argument(
  "--prompt", help="Specify your prompt.", default="", type=str
)

# Construct root catalog path. 
ROOT_CATALOG_PATH: Path = get_current_file_path(__file__) 

# Construct the rest of the important paths.
PROJECT_PATHS: Dict = fetch_paths(ROOT_CATALOG_PATH)

args = parser.parse_args()
config = get_model_configs(args) 
config["paths"] = PROJECT_PATHS


def main():
  # Get the parent path of this file.
  # print(get_info(config))
  # nlp = NLP(**config)
  # nlp.run_training()
  print(get_current_file_path(__file__))
  # TODO: Make easier configs

  # dif = DDPMApp()
  # dif.training()
  # dif.evaluate()


if __name__ == "__main__":
  main()
