"""x86 dense operators"""
from __future__ import absolute_import as _abs
import tvm
from tvm import te
from tvm import autotvm
from tvm.autotvm.task.space import SplitEntity
from tvm.topi.x86.utils import get_fp32_len
from tvm.topi.x86.injective import schedule_injective_from_existing
from tvm.topi import generic, tag
from tvm.topi.utils import get_const_tuple

@autotvm.register_topi_schedule("linear_classification.x86")
def schedule_classification(cfg, outs):
    """
    C = W*x
    B = C + bias
    O = argmax(B, axis=-1)
    I = int(O)
    """
    s = te.create_schedule([x.op for x in outs])
    I = outs[0]
    O = I.op.input_tensors[0]
    B = O.op.input_tensors[0]
    #s[B].vectorize(xi)
    C = B.op.input_tensors[0]
    A, packedB = s[C].op.input_tensors
    CC = s.cache_write(C, "global")
    y, x = s[C].op.axis
    (k,) = s[CC].op.reduce_axis
    yt, yo, yi = cfg["tile_y"].apply(s, C, y)
    xt, xo, xi = cfg["tile_x"].apply(s, C, x)
    s[C].reorder(xt, yt, yo, xo, yi, xi)
    #xyt = s[C].fuse(xt, yt)
    #s[C].parallel(xyt)
    xyo = s[C].fuse(yo, xo)
    s[C].unroll(yi)
    s[C].vectorize(xi)
    s[CC].compute_at(s[C], xyo)
    y, x = s[CC].op.axis
    ko, ki = cfg["tile_k"].apply(s, CC, k)
    s[CC].reorder(ko, ki, y, x)
    #s[CC].vectorize(x)

    tile_inner = cfg["tile_inner"].size[-1]
    if tile_inner > 1:
        yo, yi = s[CC].split(y, tile_inner)
        s[CC].reorder(ko, yo, ki, yi, x)
        s[CC].unroll(yo)
        s[CC].unroll(ki)
        s[CC].unroll(yi)
    else:
        s[CC].unroll(ki)
        s[CC].unroll(y)
    
    y = s[I].op.axis[0]
    xt, xo, xi = cfg["tile_y"].apply(s, I, y)
    #s[I].vectorize(xi)
    s[I].parallel(xt)
    s[O].compute_at(s[I], xt)
    s[B].compute_at(s[I], xt)
    #s[B].compute_inline()
    s[C].compute_at(s[I], xt)
    return s

def _default_dense_pack_config(cfg, M, N, K):
    # Generate default schedule for dynamic shape.
    if isinstance(M, (tvm.tir.Var, tvm.tir.Any)):
        M = 16
    if isinstance(N, (tvm.tir.Var, tvm.tir.Any)):
        N = 16
    if isinstance(K, (tvm.tir.Var, tvm.tir.Any)):
        K = 16

    vec_width = get_fp32_len()
    tilex_ii = 1
    for bn in range(vec_width * 2, 0, -1):
        if N % bn == 0:
            tilex_ii = bn
            break
    NN = N // tilex_ii
    tilex_oi = 1
    while NN // tilex_oi > 4:
        if (NN // tilex_oi) % 2 == 1:
            break
        tilex_oi *= 2

    tiley_ii = 8
    while M % tiley_ii != 0:
        tiley_ii //= 2
    MM = M // tiley_ii
    tiley_oi = 1
    while MM // tiley_oi > 4:
        if (MM // tiley_oi) % 2 == 1:
            break
        tiley_oi *= 2

    cfg["tile_y"] = SplitEntity([MM // tiley_oi, tiley_oi, tiley_ii])
    cfg["tile_x"] = SplitEntity([NN // tilex_oi, tilex_oi, tilex_ii])
    cfg["tile_k"] = SplitEntity([K, 1])
    cfg["tile_inner"] = SplitEntity([M // tiley_ii, tiley_ii])
    print(MM // tiley_oi, tiley_oi, tiley_ii)
    print(NN // tilex_oi, tilex_oi, tilex_ii)
    print(M // tiley_ii, tiley_ii)
@autotvm.register_topi_compute("dense.x86")
def dense_pack(cfg, data, weight, bias=None, out_dtype=None):
    """Compute dense with transformed weight."""
    if out_dtype is None:
        out_dtype = data.dtype
    M, K = get_const_tuple(data.shape)  # batch, in_dim
    if len(weight.shape) == 3:
        N, _, packw_bn = get_const_tuple(weight.shape)  # out_dim
        N = N * packw_bn
    else:
        N, _ = get_const_tuple(weight.shape)  # out_dim
    # create tuning space
    cfg.define_split(
        "tile_y", 32 if isinstance(M, (tvm.tir.Var, tvm.tir.Any)) else M, num_outputs=3
    )
    cfg.define_split(
        "tile_x", 32 if isinstance(N, (tvm.tir.Var, tvm.tir.Any)) else N, num_outputs=3
    )
    cfg.define_split(
        "tile_k", 32 if isinstance(K, (tvm.tir.Var, tvm.tir.Any)) else K, num_outputs=2
    )
    cfg.define_split(
        "tile_inner",
        32 if isinstance(M, (tvm.tir.Var, tvm.tir.Any)) else M,
        num_outputs=2,
        filter=lambda y: y.size[-1] <= 16,
    )
    if cfg.is_fallback:
        _default_dense_pack_config(cfg, M, N, K)

    if len(weight.shape) == 2:
        packw_bn = cfg["tile_x"].size[-1]
        packw_shape = (N // packw_bn, K, packw_bn)
        if autotvm.GLOBAL_SCOPE.in_tuning:
            # Directly use modified data layout placeholder.
            packw = tvm.te.placeholder(packw_shape, weight.dtype, name="packed_weight")
        else:
            packw = te.compute(
                packw_shape, lambda z, y, x: weight[z * packw_bn + x, y], name="packed_weight"
            )
    else:
        packw = weight

    idxdiv = tvm.tir.indexdiv
    idxmod = tvm.tir.indexmod
    k = te.reduce_axis((0, K), name="k")
    C = te.compute(
        (M, N),
        lambda y, x: te.sum(
            data[y, k].astype(out_dtype)
            * packw[idxdiv(x, packw_bn), k, idxmod(x, packw_bn)].astype(out_dtype),
            axis=k,
        ),
        tag="dense_pack",
    )
    if bias is not None:
        C = te.compute((M, N), lambda i, j: C[i, j] + bias[j].astype(out_dtype), tag=tag.BROADCAST)
    
    return C


