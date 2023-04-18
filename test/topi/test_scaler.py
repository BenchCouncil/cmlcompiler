"""test scaler"""
import tvm
from tvm import te, topi
from tvm.topi.utils import get_const_tuple
from sklearn.preprocessing import MaxAbsScaler,MinMaxScaler,StandardScaler,RobustScaler
import numpy as np
from cmlcompiler.topi.scaler import standard_scaler,max_abs_scaler,min_max_scaler,robust_scaler

def test_scaler(*shape, scaler_type, dtype="float32", target="llvm"):
    # sklearn implements
    a_np = np.random.randn(*shape).astype(dtype=dtype)
    if scaler_type == "max_abs":
        transformer = MaxAbsScaler().fit(a_np)
        s_x = transformer.scale_.astype(dtype)
    elif scaler_type == "min_max":
        transformer = MinMaxScaler().fit(a_np)
        m_x, s_x = transformer.min_.astype(dtype), transformer.scale_.astype(dtype)
    elif scaler_type == "standard":
        transformer = StandardScaler().fit(a_np)
        m_x, s_x = transformer.mean_.astype(dtype), transformer.scale_.astype(dtype)
    elif scaler_type == "robust":
        transformer = RobustScaler().fit(a_np)
        m_x, s_x = transformer.center_.astype(dtype), transformer.scale_.astype(dtype)
    b_np = np.random.randn(*shape).astype(dtype=dtype)
    y_np = transformer.transform(b_np)
    
    # tvm implements
    A = te.placeholder((shape), name="A", dtype=dtype)
    M = te.placeholder((shape[1],), dtype=dtype)
    S = te.placeholder((shape[1],), dtype=dtype)
    if scaler_type == "max_abs":
        Y = max_abs_scaler(A, S)
    elif scaler_type == "min_max":
        Y = min_max_scaler(A, M, S)
    elif scaler_type == "standard":
        Y = standard_scaler(A, M, S)
    elif scaler_type == "robust":
        Y = robust_scaler(A, M, S)
    s = te.create_schedule(Y.op)
    ctx = tvm.context(target, 0)
    x = tvm.nd.array(b_np, ctx)
    s_x = tvm.nd.array(s_x, ctx)
    y = tvm.nd.array(np.zeros(get_const_tuple(Y.shape), dtype=Y.dtype), ctx)
    if scaler_type in ["min_max", "standard", "robust"]:
        m_x = tvm.nd.array(m_x, ctx)
        func = tvm.build(s, [A, M, S, Y], target, name = "scaler")
        func(x, m_x, s_x, y)
    else:
        func = tvm.build(s, [A, S, Y], target, name = "scaler")
        func(x, s_x, y)
        
    # check
    try:
        tvm.testing.assert_allclose(y.asnumpy(), y_np, rtol=1e-4)
        print("pass")
    except Exception as e:
        print("error")
        print(e)

test_scaler(100,100,scaler_type="min_max")
test_scaler(100,100,scaler_type="standard")
test_scaler(100,100,scaler_type="max_abs")
test_scaler(100,100,scaler_type="robust")

"""
for i in range(10, 100, 10):
    for j in range(10, 100, 10):
        for scaler_type in ["max_abs", "min_max", "standard", "robust"]:
            test_scaler(i, j, scaler_type=scaler_type)
"""
