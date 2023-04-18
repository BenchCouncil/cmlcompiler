import tvm
import numpy as np
"""
Tree common for topi function
"""

def parse_tree(X_shape, clf, classification, dtype="float32"):
    """
    Using Level Order Traversal
    Input: sklearn tree and input data shape
    Output: parameters of gemm tree implement in tvm
    A [internal_node, X_shape]
    B [internal_node, 1]
    C [leaf_node, internal_node]
    D [leaf_node, 1]
    E [label, leaf_node]
    A, B decides the direction in internal node 
    C, D decides the leaf node finally reaches
    E is the relationship between leaf node index with class number
    """
    #get parameters from sklearn tree
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold
    value = clf.tree_.value
    
    # Get gemm parameters
    B = []
    leaf_index = []
    internal_index = []
    n_node = len(children_left)
    n_internal = 0 
    for i in range(n_node):
        if (feature[i] < 0):
            # leaf node
            leaf_index.append(i)
        else:
            # internal node
            B.append(threshold[i])
            internal_index.append(i)
    n_internal = len(internal_index)
    n_leaf = n_node - n_internal
    A = np.zeros((X_shape, n_internal), dtype=dtype)
    C = np.zeros((n_internal, n_leaf), dtype=dtype)
    D = np.zeros(n_leaf, dtype=dtype)
    tree_path = np.zeros((n_node, n_node), dtype=dtype)
    A_i = 0 
    C_j = 0
    for i in range(n_node):
        if (feature[i] < 0):
            # leaf node
            C_i = 0
            for i_index in internal_index:
                C[C_i][C_j] = tree_path[i_index][i]
                C_i = C_i + 1
            C_j = C_j + 1
        else:
            # internal node
            A[feature[i]][A_i] = 1
            A_i = A_i + 1
            tree_path[i][children_left[i]] = 1
            tree_path[i][children_right[i]] = -1
            for j in range(n_node):
                if (tree_path[j][i] == 1):
                    tree_path[j][children_left[i]] = 1
                    tree_path[j][children_right[i]] = 1
                elif (tree_path[j][i] == -1):
                    tree_path[j][children_left[i]] = -1
                    tree_path[j][children_right[i]] = -1
    for i in range(n_leaf):
        for index in internal_index:
            if (tree_path[index][leaf_index[i]] == 1):
                D[i] = D[i] + 1 
    E = []
    for index in leaf_index:
        E.append(value[index][0])
    B = np.array(B)
    B = B.astype(dtype)
    E = np.array(E)
    # If classification, normalize E
    if(classification == True):
        for i in range(E.shape[0]):
            E[i] = E[i] / np.sum(E[i])
    E = E.astype(dtype)
    return A, B, C, D, E

def convert_decision_tree(X_shape, clf, classification, dtype="float32", target="llvm"):
    """
    Convert sklearn decision tree to tvm gemm
    Fit for extra tree as well
    """
    A, B, C, D, E = parse_tree(X_shape, clf, classification, dtype=dtype)
    ctx = tvm.context(target, 0)
    A = tvm.nd.array(A, ctx)
    B = tvm.nd.array(B, ctx)
    C = tvm.nd.array(C, ctx)
    D = tvm.nd.array(D, ctx)
    E = tvm.nd.array(E, ctx)
    return A, B, C, D, E


def count_node(clf):
    """
    Return the leaf node and internal node number of all trees
    Note that the gemm formal of those trees only differ in leaf node number and internal node number
    """
    n_leaf_nodes = []
    n_internal_nodes = []
    for tree in clf.estimators_:
        n_leaf = 0
        n_internal = 0
        for i in tree.tree_.children_left:
            if(i < 0):
                n_leaf = n_leaf + 1
            else:
                n_internal = n_internal + 1
        n_leaf_nodes.append(n_leaf)
        n_internal_nodes.append(n_internal)
    return n_leaf_nodes, n_internal_nodes

def expand_matrix(A, B, C, D, E, max_nleaf, max_ninternal):
    """
    Expand parameter matrices
    """

    if(A.shape[1] < max_ninternal):
        pad_size = max_ninternal - A.shape[1]
        pad_array = np.zeros((A.shape[0], pad_size), dtype=A.dtype)
        A = np.concatenate((A, pad_array), axis=1)
    if(B.shape[0] < max_ninternal):
        pad_size = max_ninternal - B.shape[0]
        pad_array = np.zeros(pad_size, dtype=B.dtype)
        B = np.concatenate((B, pad_array), axis=0)
    if(C.shape[0] < max_ninternal):
        pad_size = max_ninternal - C.shape[0]
        pad_array = np.zeros((pad_size, C.shape[1]), dtype=C.dtype)
        C = np.concatenate((C, pad_array), axis=0)
    if(C.shape[1] < max_nleaf):
        pad_size = max_nleaf - C.shape[1]
        pad_array = np.zeros((C.shape[0], pad_size), dtype=C.dtype)
        C = np.concatenate((C, pad_array), axis=1)
    if(D.shape[0] < max_nleaf):
        pad_size = max_nleaf - D.shape[0]
        pad_array = np.zeros(pad_size, dtype=D.dtype)
        D = np.concatenate((D, pad_array), axis=0)
    if(E.shape[0] < max_nleaf):
        pad_size = max_nleaf - E.shape[0]
        pad_array = np.zeros((pad_size, E.shape[1]), dtype=E.dtype)
        E = np.concatenate((E, pad_array), axis=0)
    return A, B, C, D, E

def convert_random_forest(X_shape, clf, clf_flag=True, dtype="float32", target="llvm"):
    """
    Convert sklearn random forest to tvm gemm
    Fit for extra trees as well
    """
    n_tree = 0
    
    n_leaf_nodes, n_internal_nodes = count_node(clf)
    max_nleaf = max(n_leaf_nodes)
    max_ninternal = max(n_internal_nodes)

    for tree in clf.estimators_:
        # A, B, C, D, E is the parameters of single classifier
        A, B, C, D, E = parse_tree(X_shape, tree, clf_flag, dtype=dtype)
        # Expand all matrices to the same shape
        # TODO: Add support for non-expanding method
        A, B, C, D, E = expand_matrix(A, B, C, D, E, max_nleaf, max_ninternal)
        if(n_tree == 0):
            # AE, BE, CE, DE, EE is the parameters of all estimators
            AE = A
            BE = B
            CE = C
            DE = D
            EE = E
            n_tree = n_tree + 1
        else:
            AE = np.concatenate((AE, A), axis=0)            
            BE = np.concatenate((BE, B), axis=0)
            CE = np.concatenate((CE, C), axis=0)
            DE = np.concatenate((DE, D), axis=0)
            EE = np.concatenate((EE, E), axis=0)
            n_tree = n_tree + 1
    AE = AE.reshape((n_tree, A.shape[0], A.shape[1]))
    BE = BE.reshape((n_tree, B.shape[0]))
    CE = CE.reshape((n_tree, C.shape[0], C.shape[1]))
    DE = DE.reshape((n_tree, D.shape[0]))
    EE = EE.reshape((n_tree, E.shape[0], E.shape[1]))
    ctx = tvm.context(target, 0)
    AE = tvm.nd.array(AE, ctx)
    BE = tvm.nd.array(BE, ctx)
    CE = tvm.nd.array(CE, ctx)
    DE = tvm.nd.array(DE, ctx)
    EE = tvm.nd.array(EE, ctx)
    return AE, BE, CE, DE, EE 
