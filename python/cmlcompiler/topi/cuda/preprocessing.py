"""
Schedule for preprocessing algorithms, based on cuda
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


