from functools import wraps
from typing import List

def epoch_info(pre: List[str], post: List[str]):
    def decorate(fn):
        @wraps(fn)
        def call(*args, **kwargs):
            for mess in pre:
                print(mess)
            fn(*args)
            for mess in post:
                print(mess)
            return call 
    return decorate
