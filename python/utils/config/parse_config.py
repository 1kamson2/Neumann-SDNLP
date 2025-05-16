from typing import Dict, Tuple 
from argparse import Namespace
from utils.config.data_validation import user_config_validation, TRAINING_NLPM_TOKENS, TRAINING_DDPM_TOKENS 

def get_specific_toks(model: str) -> Tuple:
    """
        Get tokens.

        Parameters:
            model: specified in the argparse.

        Returns:
            Specific tokens according to the model name, if fails then empty
            tuple is returned.
    """
    match model:
        case "ddpm":
            return TRAINING_DDPM_TOKENS
        case "nlp":
            return TRAINING_NLPM_TOKENS
        case "all":
            return tuple(list(TRAINING_DDPM_TOKENS) + list(TRAINING_NLPM_TOKENS)) 
        case _:
            return (None,) 



def get_config(args: Namespace | None) -> Dict: 
    """
        Get config for NLP or DDP model.

        Arguments:
            args: All arguments passed through argparse.

        Returns:
            Model parameters in the dictionary.
    """
    assert args is not None, f"[ERROR] Incorrect arguments. \nGot: {args}" 
    model = args.model
    assert model in ("ddpm", "nlp", "all"), "[ERROR] Incorrect model in function."
    toks: Tuple = get_specific_toks(model)
    full_cfg: Dict = vars(args)
    cfg: Dict =  {
        k : v for k, v in full_cfg.items() if k in toks 
    }
    assert user_config_validation(config=cfg), "[ERROR] Incorrect config." 
    return cfg
