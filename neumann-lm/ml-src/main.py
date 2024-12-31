from dset import *
from nlp_model.nlp_training import NLP
from ddpm_model.ddpm_training import DDPMApp


def main():
    """
    todo: add logging
    todo: make it read tokens
    todo: make wrappers for better output 
    """
    # nlp = NLP()
    # nlp.run_epoch()

    dif = DDPMApp() 
    dif.training()
    #dif.evaluate()
    
if __name__ == "__main__":
    main()
