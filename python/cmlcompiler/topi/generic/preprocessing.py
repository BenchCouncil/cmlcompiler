"""
Schedule for preprocessing algorithms, generic
"""
from tvm import te
import cmlcompiler

def schedule_binarizer(outs):
    """
    Schedule for binarizer
    """
    return cmlcompiler.topi.x86.preprocessing.schedule_binarizer(outs)
