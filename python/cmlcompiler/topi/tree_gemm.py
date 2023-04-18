"""tree models based on gemm"""
import tvm
from tvm import te, topi, tir

def tree_gemm_dense(A, B, C, D, E, x):
    """
    tree implement based on dense gemm
    x [batch_size, n_feature]
    A [n_feature, internal_node]
    B [internal_node, 1]
    C [internal_node, leaf_node]
    D [leaf_node, 1]
    E [leaf_node, label]
    """
    y = topi.matmul(x, A)
    # [batch_size, internal_node]
    y = topi.less(y, B)
	# [batch_size, internal_node]
    y = topi.matmul(y, C)
	# [batch_size, leaf_node]

    y = topi.equal(y, D)
	# [batch_size, leaf_node]
    y = topi.matmul(y, E)
	# [batch_size, label]
    return y 

def tree_gemm_sparse(A, B, C, D, E, x):
    """
    tree implement based on sparse gemm
    
    """
    y = topi.sparse.csrmv(A, x)
    y = topi.less(y, B)
    y = topi.sparse.csrmv(C, y)
    y = topi.equal(y, D)
    y = topi.sparse.csrmv(E, y)
    return y

def decision_tree_classifier(A, B, C, D, E, x):
    """
    decision tree classifier
    """
    y = tree_gemm_dense(A, B, C, D, E, x)
    y = topi.argmax(y, axis=-1)
    # [batch_size, 1]
    return y

def decision_tree_regressor(A, B, C, D, E, x):
    """
    decision tree regressor
    """
    y = tree_gemm_dense(A, B, C, D, E, x)
    return y

def extra_tree_classifier(A, B, C, D, E, x):
    """
    extra tree classifier
    """
    y = tree_gemm_dense(A, B, C, D, E, x)
    y = topi.argmax(y, axis=-1)
    # [batch_size, 1]
    return y

def extra_tree_regressor(A, B, C, D, E, x):
    """
    extra tree regressor
    """
    y = tree_gemm_dense(A, B, C, D, E, x)
    return y
