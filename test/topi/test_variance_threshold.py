"""test variance threshold"""
import tvm
from tvm import te, topi
from tvm.topi.utils import get_const_tuple
import numpy as np
from sklearn.datasets import load_iris,load_digits
from sklearn.feature_selection import VarianceThreshold
from cmlcompiler.topi.variance_threshold import variance_threshold

def test_variance_threshold(target="llvm", dtype="float64"):
    # load datasets
    X, y = load_digits(return_X_y=True)
    # sklearn implements
    Y = VarianceThreshold().fit_transform(X)
    # tvm implements
    B = te.placeholder(X.shape, name="B", dtype=dtype)
    C = variance_threshold(B)
    s = te.create_schedule(C.op)
    ctx = tvm.context(target, 0)
    b = tvm.nd.array(X, ctx)
    c = tvm.nd.array(np.zeros(get_const_tuple(C.shape), dtype=C.dtype), ctx)
    func = tvm.build(s, [B, C], target, name = "variance_threshold")
    func(b, c)
    
    # Nowadays implement does not support list and tensor indexing, use numpy to complete it
    # TODO: add tensor indexing
    c = c.asnumpy()
    indexs = []
    for i in range(len(c)):
        if(c[i] != 0):
            indexs.append(i)
    Y_tvm = X[:,indexs]
    print(Y)
    print(Y.shape)
    print(Y_tvm)
    print(Y_tvm.shape)
    # check
    #tvm.testing.assert_allclose(c.asnumpy(), Y, rtol=1e-5)
    tvm.testing.assert_allclose(Y_tvm, Y, rtol=1e-5)

test_variance_threshold()
