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


def get_model_configs(args: Namespace | None) -> Dict: 
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

def get_info(config: Dict) -> str:
    """
        Return formatted info.

        Arguments:
            config: The config, whose information will be displayed. 

        Returns:
            Formatted string info.
    """
    info: str = ""
    max_key_length: int = min(1 << 5, max([len(str(key)) for key in config.keys()]))
    max_val_length: int = min(1 << 5, max([len(str(val)) for val in config.values()])) 

    def trim_key_and_val(k: str, v: str) -> Tuple[str, str]:  
        """
            Inner function to handle dictionaries as values.

            Arguments:
                k: Key as str.
                v: Value as str.

            Returns:
                Formatted key and value.
        """
        k = (k[:(max_key_length - 3)] + "..." if len(k) > max_key_length
        else k) 
        v = (v[:(max_val_length - 3)] + "..." if len(v) > max_val_length
        else v) 
        return (k, v)

    def lex_and_parse_dict(inner: Dict) -> str:
        """
            Inner lexing and parsing function, handles dictionaries.

            Arguments:
                inner: Dictionary as value.

            Returns:
                Formatted dict.
        """
        info_inner: str = "" 
        for k_, v_ in inner.items():
            k_, v_ = trim_key_and_val(str(k_), str(v_))
            info_inner += f"{k_:<{max_key_length}}  {v_:>{max_val_length}}\n"
        return info_inner

    for k, v in config.items():
        if isinstance(v, Dict):
            info += lex_and_parse_dict(v)
            continue
        k, v = trim_key_and_val(str(k), str(v))
        info += f"{k:<{max_key_length}}  {v:>{max_val_length}}\n"
    return info

