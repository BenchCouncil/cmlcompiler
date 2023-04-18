"""
Schedule for preprocessing algorithms, based on x86
"""
from tvm import te

def schedule_binarizer(outs):
    """
    Schedule for binarizer
    """
    sch = te.create_schedule(outs.op)
    if((len(outs.op.axis)) >= 3):
        fused = sch[outs].fuse(outs.op.axis[0], outs.op.axis[1])
        sch[outs].parallel(fused)
    if((len(outs.op.axis)) == 2):
        sch[outs].parallel(outs.op.axis[0])
    lo, li = sch[outs].split(outs.op.axis[-1], factor=16)
    sch[outs].vectorize(li)
    return sch

def schedule_normalizer(outs):
    """
    Schedule for normalizer
    """
    sch = te.create_schedule(outs.op)
    op_tag = outs.op.tag
    if op_tag == "normalizer_l1_output":
        norm = outs.op.input_tensors[1]
        pow_sum = None
    elif op_tag == "normalizer_l2_output":
        norm = outs.op.input_tensors[1]
        pow_sum = sch[norm].op.input_tensors[0]
    elif op_tag == "normalizer_max_output":
        norm = outs.op.input_tensors[1]
        pow_sum = None
    outmost_loop = outs.op.axis[0]
    sch[outs].parallel(outmost_loop)
    if pow_sum is not None:
        sch[pow_sum].compute_at(sch[outs], outmost_loop)
        #sch[pow_sum].vectorize(pow_sum.op.axis[1])
    sch[norm].compute_at(sch[outs], outmost_loop)
    return sch


