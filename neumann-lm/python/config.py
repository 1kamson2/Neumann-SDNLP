from argparse import Namespace


class Config:
    def __init__(self, args: Namespace | None = None):
        assert args is not None, f"Failed to parse arguments, received: {args}"
        self.full_config = vars(args)
        self.__MINIMUM_PROMPT_LENGTH = 3
        self.__IMPLEMENTED_MODELS = ["nlp", "ddpm"]
        # --- Tokens that MUSTN'T be ignored. --- #
        self.__NLP_IGNORE = ["imgchannels", "imgsize", "nchannels", "nsteps", "nsteps"]
        self.__DDPM_IGNORE = ["N", "dmodel", "dffn", "h", "prompt"]
        self.__DONT_IGNORE = ["model", "batchsize", "lr", "verbose", "h"]
        assert len(list(self.full_config.keys())) == len(
            self.__NLP_IGNORE + self.__DDPM_IGNORE + self.__DONT_IGNORE
        ), (
            f"Config error. Some tokens are not implemented.\n"
            f"Length of config is {len(list(self.full_config.keys()))}\n"
            f"Length of NLP ignore list is {len(self.__NLP_IGNORE)}\n"
            f"Length of DDPM ignore list is {len(self.__DDPM_IGNORE)}\n"
            f"Length of don't ignore list is {len(self.__DONT_IGNORE)}\n"
        )
        assert self.full_config.get("model", None) in self.__IMPLEMENTED_MODELS, (
            "You have given incorrect model."
        )
        if args.model == "nlp":
            assert len(args.prompt.split()) >= self.__MINIMUM_PROMPT_LENGTH, (
                "Invalid prompt."
            )

    def get_config(self):
        match self.full_config.get("model", None):
            case "nlp":
                return {
                    k: v
                    for k, v in self.full_config.items()
                    if k not in self.__NLP_IGNORE
                }
            case "ddpm":
                return {
                    k: v
                    for k, v in self.full_config.items()
                    if k not in self.__DDPM_IGNORE
                }
