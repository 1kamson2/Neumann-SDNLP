from functools import wraps


def training_info(*iargs, **ikwargs):
    """
    This function takes all arguments, that user provided to print out.
    Used only for epoch info printing.
    Parameters:
        > *iargs, **ikwargs, both used to handle user's information input.
        > *args, **kwargs, both used to handle function parameters.
    """

    def decorate(fn):
        @wraps(fn)
        def call(*args, **kwargs):
            if len(iargs) > 0:
                for mess in iargs:
                    print(mess)
            if len(ikwargs) > 0:
                for mess in ikwargs:
                    print(mess)
            return fn(*args, **kwargs)

        return call

    return decorate


def config_info(is_verbose, *iargs, **ikwargs):
    """
    This function takes all arguments, that user provided to print out.
    Used only for config info printing.
    Parameters:
        > *iargs, **ikwargs, both used to handle user's information input.
        > *args, **kwargs, both used to handle function parameters.
    """

    def decorate(fn):
        @wraps(fn)
        def call(*args, **kwargs):
            if len(ikwargs) > 0 and is_verbose:
                ljust_min = max(len(el) for el in ikwargs.keys() if isinstance(el, str))
                rjust_min = max(
                    len(el) for el in ikwargs.values() if isinstance(el, str)
                )

                for k, v in ikwargs.items():
                    print(f"{str(k).ljust(ljust_min)} | {str(v).rjust(rjust_min)}")
            return fn(*args, **kwargs)

        return call

    return decorate


# def outer_dec(*args, **kwargs):
# ...     def wrapper(fn):
# ...         def decorator(*args, **kwargs):
# ...             return fn(*args, **kwargs)
# ...         return decorator
# ...     return wrapper
