"""test one hot encoder"""
import tvm
from tvm import te, topi
from tvm.topi.utils import get_const_tuple
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from cmlcompiler.topi.one_hot_encoder import one_hot_encoder

def test_one_hot_encoder(*shape, dtype="float32", target="llvm"):
    # sklearn implements
    a_np = np.random.randn(*shape).astype(dtype=dtype)
    transformer = OneHotEncoder().fit(a_np)
    b_np = transformer.transform(a_np).toarray()
    print(a_np)
    print(b_np)
    # tvm implements
    A = te.placeholder((shape), name="A", dtype=dtype)
    B = one_hot_encoder(A, dtype=dtype)
    s = te.create_schedule(B.op)
    ctx = tvm.context(target, 0)
    a = tvm.nd.array(a_np, ctx)
    b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=B.dtype), ctx)
    func = tvm.build(s, [A, B], target, name = "binarizer")
    print(b)
    func(a, b)
    # check
    tvm.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-5)

test_one_hot_encoder(2,2)
#for i in range(10, 100, 10):
#    test_one_hot_encoder(i)

