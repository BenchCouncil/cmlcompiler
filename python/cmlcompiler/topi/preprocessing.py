"""
Basic implement for preprocessing algorithms
"""
import tvm
from tvm import te, topi, tir

def binarizer(x, threshold, dtype):
    """
    Binarizer
    Values greater than threshold map to 1, otherwise map to 0
    """
    return topi.greater(x, threshold)

def label_binarizer(y, classes):
    """
    Label Binarizer
    y: target vector (n_samples,)
    Return (n_samples,n_class)
    """
    n_samples = y.shape[0]
    n_classes = classes.shape[0]
    return te.compute((n_samples, n_classes), lambda i,j: 
            te.if_then_else(y[i]==classes[j], 1, 0))

@tvm.te.tag_scope(tag="normalizer_l1_output")
def normalizer_l1(x):
    """
    L1 normalizer
    """
    I, K = x.shape
    k = te.reduce_axis((0, K), name = "k")
    norm = te.compute(I, lambda i :te.sum(te.abs(x[i, k]), axis=k))
    return te.compute(x.shape, lambda i,j: te.div(x[i, j], norm[i]))

@tvm.te.tag_scope(tag="normalizer_l2_output")
def normalizer_l2(x):
    """
    L2 normalizer
    """
    I, K = x.shape
    k = te.reduce_axis((0, K), name = "k")
    pow_sum = te.compute(I, lambda i :te.sum(te.power(x[i, k], 2), axis=k))
    norm = te.compute(I, lambda i :te.power(pow_sum[i], 0.5))
    return te.compute(x.shape, lambda i,j: te.div(x[i, j], norm[i]))

@tvm.te.tag_scope(tag="normalizer_max_output")
def normalizer_max(x):
    """
    Max normalizer
    """
    I, K = x.shape
    k = te.reduce_axis((0, K), name = "k")
    norm = te.compute(I, lambda i :te.max(te.abs(x[i, k]), axis=k))
    return te.compute(x.shape, lambda i,j: te.div(x[i, j], norm[i]))


def normalizer(x, norm):
    """
    Normalizer
    """
    if norm == "l1" :
        return normalizer_l1(x)
    if norm == "l2" :
        return normalizer_l2(x)
    if norm == "max" :
        return normalizer_max(x)

def label_encoder(y, classes):
    """
    udo python -m easy_install --upgrade pyOpenSSL: target vector (n_samples,)
    Return (n_samples,)
    TODO: change fit_transform to transform
    TODO: add string as input
    """
    n_samples = y.shape[0]
    y_class, indices, num_unique = topi.unique(y)
    return indices

