"""variance threshold"""
import tvm
from tvm import te, topi, tir
from cmlcompiler.topi.math import variance

def variance_threshold(X, threshold=0):
    """
    X: sample vectors (n_samples, n_features)
    Return (n_samples,n_class)
    """

    # TODO: support tensor slicing
    I, J = X.shape
    x_var = variance(X)
    indices = te.compute(J, lambda j: te.if_then_else(x_var[j] > threshold, 1 , 0))
    """
    indices_remain = []
    indices_len = indices.shape[0]
    for i in range(indices_len):
        if(indices[i] > 0):
            indices_remain.append(i)
    print(indices_remain)
    """
    return indices
