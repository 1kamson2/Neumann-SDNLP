from dset import *
from beautifiers.info_wrappers import config_info
from config import Config
import argparse
from gpt_wrapper import *

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


args = parser.parse_args()
config_setup = Config(args)
config_model = config_setup.get_config()


# **config_model might raise errors, but you can ignore it.
@config_info(args.verbose, **config_model)
def main():
  """
  WHAT TO FIX?
  todo: NLP
  todo: DDPM
  todo: improve site
  todo: make it read tokens
  """
  # nlp = NLP(**config_model)
  # nlp.run_epoch()

  # dif = DDPMApp()
  # dif.training()
  # dif.evaluate()
  """
    [WARNING]: FOR NOW USE ONLY CHAT GPT API, BECAUSE THERE ARE PROBLEMS WITH
    THE ONES THAT ARE IN THIS PROJECT. NOTE THAT THE GPT'S API WILL BE USED IF
    SOME OF FUNCTIONALITIES WILL FAIL
    """

  answer = gpt_client_run(args.prompt)
  gpt_save_to_file(answer)


if __name__ == "__main__":
  main()
