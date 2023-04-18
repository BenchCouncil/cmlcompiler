"""test polynomial features"""
import tvm
from tvm import te, topi
from tvm.topi.utils import get_const_tuple
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from cmlcompiler.topi.polynomial_features import polynomial_features

def test_polynomial_features(*shape, degree, interaction_only, dtype="float32", target="llvm"):
    # sklearn implements
    a_np = np.random.randn(*shape).astype(dtype=dtype)
    transformer = PolynomialFeatures(degree=degree, interaction_only=interaction_only).fit(a_np)
    b_np = transformer.transform(a_np)
    
    # tvm implements
    A = te.placeholder((shape), name="A", dtype=dtype)
    B = polynomial_features(A, degree=degree, interaction_only=interaction_only)
    s = te.create_schedule(B.op)
    ctx = tvm.context(target, 0)
    a = tvm.nd.array(a_np, ctx)
    b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=B.dtype), ctx)
    func = tvm.build(s, [A, B], target, name = "polynomial_features")
    func(a, b)
    # check
    tvm.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-5)

#test_polynomial_features(10, 2, degree=2, interaction_only=True)
#test_polynomial_features(10, 2, degree=2, interaction_only=False)

for i in range(1,10):
    test_polynomial_features(100, i, degree=2, interaction_only=True)
    test_polynomial_features(100, i, degree=2, interaction_only=False)
