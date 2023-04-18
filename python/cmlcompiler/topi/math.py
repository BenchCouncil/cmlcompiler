import tvm 
from tvm import te,topi

def variance(x):
    K,J = x.shape
    k1 = te.reduce_axis((0, K), name = "k1")
    k2 = te.reduce_axis((0, K), name = "k2")
    sum_x = te.compute(J, lambda j: te.sum(x[k1,j], axis=k1))
    mean_x = te.compute(J, lambda j: te.div(sum_x[j], K))
    power_sum = te.compute(J, lambda j: te.sum(te.power((x[k2,j] - mean_x[j]), 2), axis=k2))
    return te.compute(J, lambda j: te.div(power_sum[j], K))

def percentile(x, q, interpolation='linear'):
    
    return x
