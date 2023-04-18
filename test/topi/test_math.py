"""test math"""
import tvm
from tvm import te
from tvm.topi.utils import get_const_tuple
import numpy as np
from cmlcompiler.topi.math import percentile,variance

def test_variance(*shape, dtype="float32", target="llvm"):
    # numpy implements
    a_np = np.random.randn(*shape).astype(dtype=dtype)
    b_np = np.var(a_np,axis=0)
    print(a_np)
    print(b_np)
    # tvm implements
    A = te.placeholder((shape), name="A", dtype=dtype)
    B = variance(A)
    s = te.create_schedule(B.op)
    ctx = tvm.context(target, 0)
    a = tvm.nd.array(a_np, ctx)
    b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=B.dtype), ctx)
    func = tvm.build(s, [A, B], target, name = "variance")
    func(a, b)
    print(b)
    # check
    tvm.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-5)

def test_percentile(*shape, dtype="float32", target="llvm"):
    # numpy implements
    a_np = np.random.randn(*shape).astype(dtype=dtype)
    b_np = np.percentile(a_np,[25,50,75],axis=-1)
    print(a_np)
    print(b_np)
    # tvm implements
    A = te.placeholder((shape), name="A", dtype=dtype)
    B = percentile(A, [25,50,75])
    s = te.create_schedule(B.op)
    ctx = tvm.context(target, 0)
    a = tvm.nd.array(a_np, ctx)
    b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=B.dtype), ctx)
    func = tvm.build(s, [A, B], target, name = "percentile")
    func(a, b)
    print(b)
    # check
    tvm.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-5)

#test_variance(2,10)
#test_percentile(2,10)
for i in range(10, 100, 10):
    for j in range(10, 100, 10):
        test_variance(i, j)    
        #test_norm(i, j, norm=norm)
