"""test preprocessing"""
import tvm
from tvm import te, topi
from tvm.topi.utils import get_const_tuple
from sklearn.preprocessing import Binarizer,LabelBinarizer,Normalizer,LabelEncoder
import numpy as np
from cmlcompiler.topi.preprocessing import binarizer,label_binarizer,normalizer,label_encoder
from cmlcompiler.topi.x86.preprocessing import schedule_binarizer

def test_binarizer(*shape, threshold, dtype="float32", target="llvm"):
    # sklearn implements
    a_np = np.random.randn(*shape).astype(dtype=dtype)
    transformer = Binarizer(threshold=threshold).fit(a_np)
    b_np = transformer.transform(a_np)
    
    # tvm implements
    A = te.placeholder((shape), name="A", dtype=dtype)
    B = binarizer(A, threshold, dtype)
    s = schedule_binarizer(B)
    ctx = tvm.context(target, 0)
    a = tvm.nd.array(a_np, ctx)
    b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=B.dtype), ctx)
    
    func = tvm.build(s, [A, B], target, name = "binarizer")
    func(a, b)
    print(tvm.lower(s, (A, B), simple_mode=True))
    # check
    try:
        tvm.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-5)
        print("pass")
    except Exception as e:
        print("error")
        print(e)

def test_label_binarizer(shape, target="llvm", dtype="int64"):
    # sklearn implements
    a_np = np.random.randn(shape).astype(dtype=dtype)
    transformer = LabelBinarizer().fit(a_np)
    classes = transformer.classes_
    b_np = np.random.randn(shape).astype(dtype=dtype)
    y_np = transformer.transform(b_np)
    # tvm implements
    A = te.placeholder((shape,), dtype=dtype)
    C = te.placeholder((len(classes),), dtype=dtype) 
    Y = label_binarizer(A, C)
    s = te.create_schedule(Y.op)
    ctx = tvm.context(target, 0)
    a = tvm.nd.array(b_np, ctx)
    c = tvm.nd.array(classes, ctx)
    y = tvm.nd.array(np.zeros(get_const_tuple(Y.shape), dtype=Y.dtype), ctx)
    func = tvm.build(s, [A, C, Y], target, name = "label_binarizer")
    func(a, c, y)
    # check
    try:
        tvm.testing.assert_allclose(y.asnumpy(), y_np, rtol=1e-5)
        print("pass")
    except Exception as e:
        print("error")
        print(e)

def test_norm(*shape, norm, dtype="float32", target="llvm"):
    # sklearn implements
    a_np = np.random.randn(*shape).astype(dtype=dtype)
    transformer = Normalizer(norm=norm).fit(a_np)
    b_np = transformer.transform(a_np)
    
    # tvm implements
    A = te.placeholder((shape), name="A", dtype=dtype)
    B = normalizer(A, norm=norm)
    s = te.create_schedule(B.op)
    ctx = tvm.context(target, 0)
    a = tvm.nd.array(a_np, ctx)
    b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=B.dtype), ctx)
    func = tvm.build(s, [A, B], target, name = "normalizer")
    func(a, b)
    # check
    try:
        tvm.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-5)
        print("pass")
    except Exception as e:
        print("error")
        print(e)
"""
norm_list = ["l1", "l2", "max"]
for norm in norm_list:
    test_norm(100, 100, norm=norm)
for k in [-1, 0, 1]:
    test_binarizer(100, 100, threshold=k)
test_label_binarizer(100)
"""
def test_label_encoder(shape, target="llvm", dtype="int64"):
    # sklearn implements
    a_np = np.random.randn(shape).astype(dtype=dtype)
    transformer = LabelEncoder().fit(a_np)
    classes = transformer.classes_
    #b_np = np.random.randn(shape).astype(dtype=dtype)
    b_np = a_np
    y_np = transformer.transform(b_np)
    # tvm implements
    A = te.placeholder((shape,), dtype=dtype)
    C = te.placeholder((len(classes),), dtype=dtype) 
    Y = label_encoder(A, C)
    s = te.create_schedule(Y.op)
    ctx = tvm.context(target, 0)
    a = tvm.nd.array(b_np, ctx)
    c = tvm.nd.array(classes, ctx)
    y = tvm.nd.array(np.zeros(get_const_tuple(Y.shape), dtype=Y.dtype), ctx)
    func = tvm.build(s, [A, C, Y], target, name = "label_binarizer")
    func(a, c, y)
    # check
    try:
        tvm.testing.assert_allclose(y.asnumpy(), y_np, rtol=1e-5)
        print("pass")
    except Exception as e:
        print("error")
        print(e)

#test_label_encoder(100)
test_binarizer(100, 100, threshold=0, target="llvm -mcpu=core-avx2")
