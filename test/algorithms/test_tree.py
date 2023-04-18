"""test tree models based on gemm, including Decision Tree and Extra Tree"""
import tvm
from tvm import te, topi
from tvm.topi.utils import get_const_tuple
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor,ExtraTreeClassifier,ExtraTreeRegressor
from sklearn.datasets import make_classification,make_regression
import numpy as np
from sklearn.model_selection import train_test_split
from hummingbird.ml import convert
from cmlcompiler.model import build_model
import os
import tvm.testing
from cmlcompiler.utils.tree import parse_tree

# Add nvcc path
os.environ["PATH"] = os.environ["PATH"]+":/usr/local/cuda/bin/"
os.environ["TVM_BACKTRACE"] = "1"

def new_tree(x, S, T, B, L):
    y = np.matmul(x, S.T)
    y = np.greater(y, T)
    y = np.matmul(y, B.T)
    y = np.argmax(y, axis=-1)
    y = np.take(L, y)
    return y  

def test_tree_gemm(sklearn_func, max_leaf_nodes, n_samples, n_features, n_classes, dtype="float32", target="llvm"):
    """
    Testing tree based on gemm
    Input: sklearn function and corresponding tvm function
    Output: The result equals or not 
    """
    if(sklearn_func in [DecisionTreeRegressor, ExtraTreeRegressor]):
        classification = False
        X, y = make_regression(n_samples=n_samples, n_features=n_features)
        out_dtype = "float32"
    else:
        classification = True
        n_informative = int(n_features / 2)
        X, y = make_classification(n_samples=n_samples, n_features=n_features, n_classes=n_classes, n_informative=n_informative)
        out_dtype = "int8"
    
    # load dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    
    # sklearn implements
    clf = sklearn_func(max_leaf_nodes=max_leaf_nodes, random_state=0)
    clf.fit(X_train, y_train)
    #print(type(clf.classes_))
    Y_test = clf.predict(X_test)
    model = build_model(clf, X_test.shape, out_dtype=out_dtype, target="llvm", sparse_replacing=False, dtype_converting=False)
    y_tvm = model.run(X_test)
    S, T, B, L = parse_tree(X_test.shape[1], clf, "233", "float32")
    # check
    y_new = new_tree(X_test, S, T, B, L)
    y_np = np.array(Y_test) 
    #y_index = np.array(y_tvm).astype("int32")
    #y_tvm = np.take(L, y_index)
    y_tvm = np.array(y_tvm)
    print(y_tvm)
    """
    try:
        tvm.testing.assert_allclose(y_tvm, y_np, rtol=1e-5)
        print("pass")
    except Exception as e:
        print("error")
        print(e)
    """
    try:
        tvm.testing.assert_allclose(y_new, y_tvm, rtol=1e-5)
        print("pass")
    except Exception as e:
        print("error")
        print(e)

#test_tree_gemm(DecisionTreeRegressor, max_leaf_nodes=5, n_samples=100, n_features=10, n_classes=2)
#test_tree_gemm(ExtraTreeRegressor, max_leaf_nodes=10, n_samples=1010, n_features=10, n_classes=2)
test_tree_gemm(DecisionTreeClassifier, max_leaf_nodes=10, n_samples=1010, n_features=10, n_classes=4)
#test_tree_gemm(ExtraTreeClassifier, max_leaf_nodes=10, n_samples=1000, n_features=10, n_classes=2)

