import sys


def get_func_name() -> str:
    return sys._getframe(1).f_code.co_name
