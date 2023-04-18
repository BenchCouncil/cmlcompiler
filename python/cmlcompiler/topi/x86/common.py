"""
Default schedule
"""
from tvm import te
"""
def schedule_fuse_parallel(outs):
    fuse all axes then parallel
    sch = te.create_schedule(outs.op)
    fused = sch[outs].fuse(*outs.op.axis)
    sch[outs].parallel(fused)
    return sch
"""

def schedule_fuse_parallel(outs):
    """
    fuse all axes then parallel
    """
    sch = te.create_schedule(outs.op)
    #te.schedule.AutoInlineInjective(sch)
    #fused = sch[outs].fuse(*outs.op.axis)
    out_loop = sch[outs].op.axis[0]
    sch[outs].parallel(out_loop)
    return sch

def schedule_parallel_vectorize(outs):
    """
    fuse all axes except innermost
    vectorize innermost
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


