from typing import Dict, List, Tuple

IMPLEMENTED_MODELS: Tuple = ("nlp", "ddpm", "all")
TRAINING_NLPM_TOKENS: Tuple = ("model", "imgchannels", "imgsize", 
                               "nchannels", "nsteps")
TRAINING_DDPM_TOKENS: Tuple = ("model", "dmodel", "N", "dffn", "h",
                               "prompt", "dropout", "nsamples")


def user_config_validation(*, config: Dict) -> bool: 
    """
        Validate user's config. This function is a callback, that must be 
        present in the specific config function.

        Parameters:
            args: Arguments to pass to the model.
            model_runs: If true, the model runs (we don't train it). 

        Returns:
            True if the config is valid, otherwise false.
    """
    config_keys: List = list(config.keys())
    model: str | None = config.get("model", None)
    if model not in IMPLEMENTED_MODELS:
        return False

    match model:
        case "ddpm":
            return len(TRAINING_DDPM_TOKENS) <= len(config_keys) 
        case "nlp":
            return len(TRAINING_NLPM_TOKENS) <= len(config_keys) 
        case "all":
            return (len(TRAINING_DDPM_TOKENS) + len(TRAINING_NLPM_TOKENS) ==
                    len(config_keys))
        case _:
            return False
    



