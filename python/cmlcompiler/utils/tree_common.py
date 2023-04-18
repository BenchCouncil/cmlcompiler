import tvm
from tvm import te
import numpy as np
from scipy.sparse import csr_matrix,bsr_matrix
from collections import namedtuple

def parse_tree(X_shape, clf, clf_flag, dtype):
    """
    Convert decision trees into tensor computation, proposed by UCML
    Internal Nodes are ordered in Level Order Traversal
    Leaf Nodes are ordered in Mid-Order Traversal
    Input: sklearn tree and input data shape
    Output: parameters of gemm tree implement in tvm
    S [internal_node, X_shape] The relationship between internal node and feature
    T [internal_node, 1] Threshold for each internal node
    B [leaf_node, internal_node] The relationship between leaf node and internal node
    L [leaf_node,] Label for each leaf node
    """
    #get parameters from sklearn tree
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold
    value = clf.tree_.value
    # Get gemm parameters
    T = []
    n_node = len(children_left)
    # internal node and leaf node need to be renumbered
    level_order_traversal = [0]
    internal_index = []
    def level_order_traverse(i):
        # Use level order traverse to number internal node 
        while len(level_order_traversal) > 0:
            node_i = level_order_traversal.pop(0)
            if(feature[node_i] >= 0):
                internal_index.append(node_i)
            if(children_left[node_i] > 0):
                level_order_traversal.append(children_left[node_i])
            if(children_right[node_i] > 0):
                level_order_traversal.append(children_right[node_i])
    
    level_order_traverse(0)
    n_internal = len(internal_index)
    dic_internal = {internal_index[i] : i for i in range(n_internal)}
    #print(internal_index)
    for i in range(n_internal):
        T.append(threshold[internal_index[i]])
    T = np.array(T)
    S = np.zeros((n_internal, X_shape), dtype=dtype)
    for i in range(n_internal):
        S[i][feature[internal_index[i]]] = 1
    #print(S)
    S = np.array(S)
    
    n_leaf = n_node - n_internal
    leaf_index = []
    mid_order_traversal = []
    
    def mid_order_traverse(i):
        # Use mid order traverse to number leaf node
        if(children_left[i] > 0):
            mid_order_traverse(children_left[i])
        mid_order_traversal.append(i)
        if(feature[i] < 0):
            leaf_index.append(i)
        if(children_right[i] > 0):
            mid_order_traverse(children_right[i])
    mid_order_traverse(0)
    dic_leaf = {leaf_index[i] : i for i in range(n_leaf)}
    
    #print(mid_order_traversal)
    #print(leaf_index)
    
    tree_path = np.ones((n_node, n_node), dtype=dtype)
    for i in range(n_node):
        if (feature[i] >= 0):
            # internal node
            tree_path[i][children_left[i]] = 0
            for j in range(n_node):
                if (tree_path[i][j] == 0 and feature[j] >=0):
                    tree_path[i][children_left[j]] = 0
                    tree_path[i][children_right[j]] = 0
    #print(tree_path)
    B = np.ones((n_leaf, n_internal), dtype=dtype)
    for i in range(n_node):
        for j in range(n_node):
            if(tree_path[i][j] == 0 and j in leaf_index):
                B[dic_leaf[j]][dic_internal[i]] = 0
    #print(B)
    L = []
    for i in range(n_leaf):
        L.append(value[leaf_index[i]][0])
    L = np.array(L)
    #print(L)
    # Note that not converting for forest
    #print(L)
    if(clf_flag == "tree_clf"):
        for i in range(L.shape[0]):
            L[i] = L[i] / np.sum(L[i])
        L = np.argmax(L, axis=1)
        L = L.astype(dtype)
    
    elif(clf_flag == "forest_clf"):
        for i in range(L.shape[0]):
            L[i] = L[i] / np.sum(L[i])
    return S, T, B, L

def dense_to_sparse(x, dtype, sparse_type):
    """
    Convert dense data to sparse data in csr format
    """
    if(sparse_type == "csr"):
        x = csr_matrix(x)
    elif(sparse_type == "bsr"):
        x = bsr_matrix(x)
    else:
        print("Unsupported sparse type")
    data = x.data.astype(dtype)
    indices = x.indices.astype("int32")
    indptr = x.indptr.astype("int32")
    data = tvm.nd.array(data)
    indices = tvm.nd.array(indices)
    indptr = tvm.nd.array(indptr)
    return data, indices, indptr

def convert_decision_tree(X_shape, clf, clf_flag, dtype, target, sparse, type_convert):
    """
    Convert sklearn decision tree to tvm gemm
    Fit for extra tree as well
    """
    S, T, B, L = parse_tree(X_shape, clf, clf_flag, dtype)
    ctx = tvm.device(target, 0)
    S = S.astype(dtype)
    T = T.astype(dtype)
    B = B.astype(dtype)
    L = L.astype(dtype)
    if(clf_flag == "tree_clf"):
        classes = clf.classes_
        L = L.astype("int32")
        L = np.take(classes, L)
        L = L.astype("float32")
    if(type_convert == True):
        B = B.astype("int8")
    T = tvm.nd.array(T)
    B = tvm.nd.array(B)
    L = tvm.nd.array(L)
    if(sparse == True):
        S_data, S_indices, S_indptr = dense_to_sparse(S, "float32", "csr")
        return S_data, S_indices, S_indptr, T, B, L
    else:
        S = tvm.nd.array(S)
        return S, T, B, L

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

def expand_matrix(S, T, B, L, max_nleaf, max_ninternal):
    """
    Expand parameter matrices
    """
    if(S.shape[0] < max_ninternal):
        pad_size = max_ninternal - S.shape[0]
        pad_array = np.zeros((pad_size, S.shape[1]), dtype=S.dtype)
        S = np.concatenate((S, pad_array), axis=0)
    if(T.shape[0] < max_ninternal):
        pad_size = max_ninternal - T.shape[0]
        pad_array = np.zeros(pad_size, dtype=T.dtype)
        T = np.concatenate((T, pad_array), axis=0)
    if(B.shape[0] < max_nleaf):
        pad_size = max_nleaf - B.shape[0]
        pad_array = np.zeros((pad_size, B.shape[1]), dtype=B.dtype)
        B = np.concatenate((B, pad_array), axis=0)
    if(B.shape[1] < max_ninternal):
        pad_size = max_ninternal - B.shape[1]
        pad_array = np.zeros((B.shape[0], pad_size), dtype=B.dtype)
        B = np.concatenate((B, pad_array), axis=1)
    if(L.shape[0] < max_nleaf):
        pad_size = max_nleaf - L.shape[0]
        pad_array = np.zeros((pad_size, L.shape[1]), dtype=L.dtype)
        L = np.concatenate((L, pad_array), axis=0)
    if(L.shape[1] < max_nleaf):
        pad_size = max_nleaf - L.shape[1]
        pad_array = np.zeros((L.shape[0], pad_size), dtype=L.dtype)
        L = np.concatenate((L, pad_array), axis=1)
    return S, T, B, L

def convert_random_forest(X_shape, clf, clf_flag, dtype, target, dtype_converting, sparse_replacing):
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
        #clf_flag = "tree_clf"
        S, T, B, L = parse_tree(X_shape, tree, clf_flag=clf_flag, dtype=dtype)
        # Expand all matrices to the same shape
        # TODO: Add support for non-expanding method
        S, T, B, L = expand_matrix(S, T, B, L, max_nleaf, max_ninternal)
        if(n_tree == 0):
            # SE, TE, BE, DE, TE is the parameters of all estimators
            SE = S
            TE = T
            BE = B
            LE = L
            n_tree = n_tree + 1
            #with np.printoptions(threshold=np.inf):
                #print(E.T)
        else:
            SE = np.concatenate((SE, S), axis=0)            
            TE = np.concatenate((TE, T), axis=0)
            BE = np.concatenate((BE, B), axis=0)
            LE = np.concatenate((LE, L), axis=0)
            n_tree = n_tree + 1
    SE = SE.astype(dtype)
    TE = TE.astype(dtype)
    BE = BE.astype(dtype)
    LE = LE.astype(dtype)
    BE = BE.reshape([n_tree, max_nleaf, max_ninternal])
    step = np.arange(0, max_nleaf * n_tree, max_nleaf)
    step = step.astype("int32")
    if(dtype_converting == True):
        BE = BE.astype("int8")
    if(clf_flag == "forest_clf"):
        classes = np.array(clf.classes_)
        classes = classes.astype("int32")
        classes = tvm.nd.array(classes)
    #with np.printoptions(threshold=np.inf):
    #    print(TE)
    TE = tvm.nd.array(TE)
    BE = tvm.nd.array(BE)
    step = tvm.nd.array(step)
    LE = tvm.nd.array(LE)
    print("*"*50)
    if(clf_flag == "forest_clf"):
        if(sparse_replacing == True):
            SE_data, SE_indices, SE_indptr = dense_to_sparse(SE, "float32", "bsr")
            return SE_data, SE_indices, SE_indptr, TE, BE, step, LE, classes
        else:
            SE = tvm.nd.array(SE)
            return SE, TE, BE, step, LE, classes
    else:
        if(sparse_replacing == True):
            SE_data, SE_indices, SE_indptr = dense_to_sparse(SE, "float32", "bsr")
            return SE_data, SE_indices, SE_indptr, TE, BE, step, LE
        else:
            SE = tvm.nd.array(SE)
            return SE, TE, BE, step, LE

def convert_gbdt_feature(X_shape, sklearn_model, flag_clf, dtype, target, dtype_converting, sparse_replacing):
    
    gbdt_model = sklearn_model.gbm
    onehot_encoder = sklearn_model.encoder
    # Parse GBDT model
    tmp = []
    for tree in gbdt_model.estimators_:
        if(type(tree) is np.ndarray):
            # GradientBoosting models wrap tree in np.ndarray    
            tmp.append(tree[0])
    gbdt_model.estimators_ = tmp
    A, B, C, D, E, classes = convert_random_forest(X_shape=X_shape, clf=gbdt_model, clf_flag="forest_clf", dtype="float32", target=target, dtype_converting=False, sparse_replacing=False)
    # Parse OneHot Encoder
    len_categories = [0] + [len(i) for i in onehot_encoder.categories_]
    len_cumsum = np.cumsum(np.array(len_categories))[0:-1].astype("int32")
    len_cumsum = tvm.nd.array(len_cumsum)
    # The leaf sample reaches is marked as 1, other as 0
    return A, B, C, len_cumsum

