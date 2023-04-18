"""scaler"""
import tvm
from tvm import te, topi

def robust_scaler(x, center_x, scale_x):
    return te.compute(x.shape, lambda i,j: te.div((x[i,j] - center_x[j]), scale_x[j]))

"""
def standard_scaler(x):
    # fit_transform
    K,J = x.shape
    k1 = te.reduce_axis((0, K), name = "k1")
    k2 = te.reduce_axis((0, K), name = "k2")
    sum_x = te.compute(J, lambda j: te.sum(x[k1,j], axis=k1))
    mean_x = te.compute(J, lambda j: te.div(sum_x[j], K))
    power_sum = te.compute(J, lambda j: te.sum(te.power((x[k2,j] - mean_x[j]), 2), axis=k2))
    std_x = te.compute(J, lambda j: te.power(te.div(power_sum[j], K), 0.5))
    return te.compute(x.shape, lambda i,j: te.div((x[i,j] - mean_x[j]), std_x[j]))
"""

def standard_scaler(x, mean_x, std_x):
    return te.compute(x.shape, lambda i,j: te.div((x[i,j] - mean_x[j]), std_x[j]))

"""
def max_abs_scaler(x):
    # fit_transform
    K, J = x.shape
    k = te.reduce_axis((0, K), name = "k")
    max_abs = te.compute(J, lambda j: te.max(te.abs(x[k,j]), axis=k))
    return te.compute(x.shape, lambda i,j: te.div(x[i,j], max_abs[j]))
"""

def max_abs_scaler(x, scale_x):
    return te.compute(x.shape, lambda i,j: te.div(x[i,j], scale_x[j]))

"""
def min_max_scaler(x):
    # fit_transform
    K, J = x.shape
    k1 = te.reduce_axis((0, K), name = "k")
    k2 = te.reduce_axis((0, K), name = "k")
    max_local = te.compute(J, lambda j: te.max(x[k1,j], axis=k1))
    min_local = te.compute(J, lambda j: te.min(x[k2,j], axis=k2))
    return te.compute(x.shape, lambda i,j: 
            te.div((x[i,j] - min_local[j]), (max_local[j] - min_local[j])))
"""

def min_max_scaler(x, min_x, scale_x):
    return te.compute(x.shape, lambda i,j: x[i,j] * scale_x[j] + min_x[j])
