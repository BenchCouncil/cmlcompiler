"""ensemble models based on gemm"""
import tvm
from tvm import te, topi, tir

def ensemble_gemm_dense(A, B, C, D, E, x):
    """
    ensemble implement based on dense gemm
    x [batch_size, n_feature]
    A [n_estimator, n_feature, internal_node]
    B [n_estimator, internal_node, 1]
    C [n_estimator, internal_node, leaf_node]
    D [n_estimator, leaf_node, 1]
    E [n_estimator, leaf_node, label]
    """
    batch_size = x.shape[0]
    n_estimator, n_feature, n_internal = A.shape
    n_leaf, n_label = E.shape[1], E.shape[2]
    k1 = te.reduce_axis((0, n_feature), name="k1")
    y = te.compute((batch_size, n_estimator, n_internal), lambda i, j, l:
            te.sum(x[i, k1] *  A[j, k1, l], axis=k1))
    y = topi.less(y, B)
    k2 = te.reduce_axis((0, n_internal), name="k2")
    y = te.compute((batch_size, n_estimator, n_leaf), lambda i, j, l:
            te.sum(y[i, j, k2] *  C[j, k2, l], axis=k2))
    y = topi.equal(y, D)
    k3 = te.reduce_axis((0, n_leaf), name="k2")
    y = te.compute((batch_size, n_estimator, n_label), lambda i, j, l:
            te.sum(y[i, j, k3] *  E[j, k3, l], axis=k3))
    return y 

def ensemble_gemm_sparse(A, B, C, D, E, x):
    """
    tree implement based on sparse gemm
    
    """
    y = topi.sparse.csrmv(A, x)
    y = topi.less(y, B)
    y = topi.sparse.csrmv(C, y)
    y = topi.equal(y, D)
    y = topi.sparse.csrmv(E, y)
    return y

def random_forest_classifier(A, B, C, D, E, x):
    """
    random forest classifier
    """
    y = ensemble_gemm_dense(A, B, C, D, E, x)
    y = topi.sum(y, axis=1)
    y = topi.argmax(y, axis=-1)
    return y

def random_forest_regressor(A, B, C, D, E, x):
    """
    random forest regressor
    """
    y = ensemble_gemm_dense(A, B, C, D, E, x)
    batch_size, n_estimator = y.shape[0], y.shape[1]
    y = topi.sum(y, axis=1)
    mean_y = te.compute(batch_size, lambda i: te.div(y[i][0], n_estimator))
    return mean_y

def extra_trees_classifier(A, B, C, D, E, x):
    """
    extra trees classifier
    """
    y = ensemble_gemm_dense(A, B, C, D, E, x)
    y = topi.sum(y, axis=1)
    y = topi.argmax(y, axis=-1)
    return y

def extra_trees_regressor(A, B, C, D, E, x):
    """
    extra trees regressor
    """
    y = ensemble_gemm_dense(A, B, C, D, E, x)
    batch_size, n_estimator = y.shape[0], y.shape[1]
    y = topi.sum(y, axis=1)
    mean_y = te.compute(batch_size, lambda i: te.div(y[i][0], n_estimator))
    return mean_y


