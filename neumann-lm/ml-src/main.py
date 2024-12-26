from dset import *
from nlp_training import NLP
from ddpm_training import DDPMTraining


def main():
    """
    todo: add logging
    todo: make it read tokens
    """
    # nlp = NLP()
    # nlp.run_epoch()
    dif = DDPMTraining() 
    dif.training()
    
if __name__ == "__main__":
    main()
