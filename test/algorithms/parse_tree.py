import numpy as np

def parse_tree(X_shape, clf, clf_flag, dtype):
    """
    Internal Nodes are ordered in Level Order Traversal
    Leaf Nodes are ordered in Mid-Order Traversal
    Input: sklearn tree and input data shape
    Output: parameters of gemm tree implement in tvm
    S [internal_node, X_shape] The relationship between internal node and feature
    T [internal_node, 1] Threshold for each internal node
    B [leaf_node, internal_node] The relationship between lead node and internal node
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
    #print(T)
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
        L.append(value[leaf_index[i]][0][0])
    L = np.array(L)
    #print(L)
    # If clf_flag, normalize L and convert it to 0-1 matrix
    # Note that not converting for forest
    if(clf_flag == "tree_clf"):
        for i in range(L.shape[0]):
            L[i] = L[i] / np.sum(L[i])
        L = L.astype(dtype)
        L_tmp = np.argmax(L, axis=1)
        L = np.zeros_like(L)
        L[np.arange(len(L_tmp)), L_tmp] = 1
    elif(clf_flag == "forest_clf"):
        for i in range(E.shape[0]):
            E[i] = E[i] / np.sum(E[i])
    return S, T, B, L

