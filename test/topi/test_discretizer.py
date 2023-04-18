"""test scaler"""
import tvm
from tvm import te, topi
from tvm.topi.utils import get_const_tuple
from sklearn.preprocessing import KBinsDiscretizer
import numpy as np

def kbins_discretizer(x):
    
    return 

def standard_scaler(x):
    K,J = x.shape
    k1 = te.reduce_axis((0, K), name = "k1")
    k2 = te.reduce_axis((0, K), name = "k2")
    sum_x = te.compute(J, lambda j: te.sum(x[k1,j], axis=k1))
    mean_x = te.compute(J, lambda j: te.div(sum_x[j], K))
    power_sum = te.compute(J, lambda j: te.sum(te.power((x[k2,j] - mean_x[j]), 2), axis=k2))
    std_x = te.compute(J, lambda j: te.power(te.div(power_sum[j], K), 0.5))
    return te.compute(x.shape, lambda i,j: te.div((x[i,j] - mean_x[j]), std_x[j]))

def test_kbins_discretizer(*shape, scaler_type, dtype="float32", target="llvm"):
    # sklearn implements
    a_np = np.random.randn(*shape).astype(dtype=dtype)
    transformer = StandardScaler().fit(a_np)
    b_np = transformer.transform(a_np)
    # tvm implements
    A = te.placeholder((shape), name="A", dtype=dtype)
    B = standard_scaler(A)
    s = te.create_schedule(B.op)
    ctx = tvm.context(target, 0)
    a = tvm.nd.array(a_np, ctx)
    b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=B.dtype), ctx)
    func = tvm.build(s, [A, B], target, name = "max_abs_scaler")
    func(a, b)
    # check
    tvm.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-5)

for i in range(10, 100, 10):
    for j in range(10, 100, 10):
        for scaler_type in ["max_abs", "min_max", "standard"]:
            test_scaler(i, j, scaler_type=scaler_type)
