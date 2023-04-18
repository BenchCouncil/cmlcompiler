"""
Schedule for linear models, based on x86
"""
from __future__ import absolute_import as _abs
import tvm
from tvm import te,tir,topi
from tvm import autotvm
from tvm.autotvm.task.space import SplitEntity
from tvm.topi.x86.utils import get_fp32_len
from tvm.topi import generic, tag
from tvm.topi.utils import get_const_tuple

def base_classification_nopack(x, coef, bias, dtype):
    """
    Base Classifier
    x [n_samples, n_features]
    coef [n_features, n_classes]
    bias [n_classes, ]
    Output [n_samples,]
    Matmul without array packing, fitting for small n_features
    """
    y = dense_nopack(x, coef, bias, dtype)
    #y = topi.x86.dense_pack(x, coef, bias, dtype)  
    return topi.argmax(y, axis=1)

@autotvm.register_topi_schedule("linear_nopack.x86")
def schedule_classification_nopack(cfg, outs):
    """
    C = W*x
    B = C + bias
    O = argmax(B, axis=-1)
    I = int(O)
    """
    s = te.create_schedule([x.op for x in outs])
    O = outs[0].op.input_tensors[0]
    B = O.op.input_tensors[0]
    #s[B].vectorize(xi)
    C = B.op.input_tensors[0]

    y, x = s[C].op.axis
    (kk,) = s[C].op.reduce_axis
    yo, yi = cfg["tile_y"].apply(s, C, y)
    xo, xi = cfg["tile_x"].apply(s, C, x)
    s[C].reorder(yo, xo, yi, xi)
    xyo = s[C].fuse(yo, xo)
    #s[C].parallel(xyo)
    s[C].unroll(kk)

    (CC,) = s[C].op.input_tensors
    s[CC].compute_at(s[C], xyo)
    z, y, x = s[CC].op.axis
    (k,) = s[CC].op.reduce_axis
    yz = s[CC].fuse(z, y)
    s[CC].reorder(k, yz, x)
    s[CC].unroll(yz)
    s[CC].vectorize(x)
    
    y = s[O].op.axis[0]
    yo, yi = cfg["tile_y"].apply(s, O, y)
    #s[O].vectorize(xi)
    s[O].parallel(yo)
    s[B].compute_at(s[O], yo)
    #s[B].compute_inline()
    s[C].compute_at(s[O], yo)
    return s

def _default_dense_nopack_config(cfg, M, N, K):
    # Generate default schedule for dynamic shape.
    if isinstance(M, (tvm.tir.Var, tvm.tir.Any)):
        M = 16
    if isinstance(N, (tvm.tir.Var, tvm.tir.Any)):
        N = 16
    if isinstance(K, (tvm.tir.Var, tvm.tir.Any)):
        K = 16

    vec_width = get_fp32_len()
    tilek_bn = 1
    for bn in range(vec_width * 2, 0, -1):
        if K % bn == 0:
            tilek_bn = bn
            break
    cfg["tile_k"] = SplitEntity([K // tilek_bn, tilek_bn])
    cfg["tile_x"] = SplitEntity([N, 1])
    cfg["tile_y"] = SplitEntity([1, M])

@autotvm.register_topi_compute("linear_nopack.x86")
def dense_nopack(cfg, data, weight, bias=None, out_dtype=None):
    """Compute dense without packing"""
    if out_dtype is None:
        out_dtype = data.dtype
    M, K = get_const_tuple(data.shape)
    N, _ = get_const_tuple(weight.shape)
    # create tuning space
    cfg.define_split(
        "tile_y", 32 if isinstance(M, (tvm.tir.Var, tvm.tir.Any)) else M, num_outputs=2
    )
    cfg.define_split(
        "tile_x", 32 if isinstance(N, (tvm.tir.Var, tvm.tir.Any)) else N, num_outputs=2
    )
    cfg.define_split(
        "tile_k", 32 if isinstance(K, (tvm.tir.Var, tvm.tir.Any)) else K, num_outputs=2
    )
    if cfg.is_fallback:
        _default_dense_nopack_config(cfg, M, N, K)

    vec = cfg["tile_k"].size[-1]
    k = te.reduce_axis((0, K // vec), "k")
    CC = te.compute(
        (M, N, vec),
        lambda z, y, x: te.sum(
            data[z, k * vec + x].astype(out_dtype) * weight[y, k * vec + x].astype(out_dtype),
            axis=k,
        ),
    )

    kk = te.reduce_axis((0, vec), "kk")
    C = te.compute((M, N), lambda y, x: te.sum(CC[y, x, kk], axis=kk), tag="dense_nopack")
    if bias is not None:
        C = te.compute((M, N), lambda i, j: C[i, j] + bias[j].astype(out_dtype), tag=tag.BROADCAST)
    return C

