"""test tree models based on gemm, including Decision Tree and Extra Tree"""
import tvm
from tvm import te, topi
from tvm.topi.utils import get_const_tuple
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor,ExtraTreeClassifier,ExtraTreeRegressor
from sklearn.datasets import make_classification,make_regression
import numpy as np
from sklearn.model_selection import train_test_split
from cmlcompiler.topi.tree_gemm import decision_tree_classifier,decision_tree_regressor,extra_tree_classifier,extra_tree_regressor
from cmlcompiler.model import build_model
from tree_common import convert_decision_tree

def test_tree_gemm(sklearn_func, tvm_func, max_leaf_nodes, n_samples, n_features, n_classes, dtype="float32", target="llvm"):
    """
    Testing tree based on gemm
    Input: sklearn function and corresponding tvm function
    Output: The result equals or not 
    """
    if(sklearn_func in [DecisionTreeRegressor, ExtraTreeRegressor]):
        classification = False
        X, y = make_regression(n_samples=n_samples, n_features=n_features)
    else:
        classification = True
        n_informative = int(n_features / 2)
        X, y = make_classification(n_samples=n_samples, n_features=n_features, n_classes=n_classes, n_informative=n_informative)
    
    # load dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    
    # sklearn implements
    clf = sklearn_func(max_leaf_nodes=max_leaf_nodes, random_state=0)
    clf.fit(X_train, y_train)
    Y_test = clf.predict(X_test)
    X_shape = X_train.shape[1]
    
    # convert sklearn tree to tvm gemm
    a, b, c, d, e = convert_decision_tree(X_shape, clf, classification=classification, dtype=dtype, target=target)
    # tvm implements
    X_tvm = te.placeholder((X_test.shape), name="X", dtype=dtype)
    A = te.placeholder((a.shape), name="X", dtype=dtype)
    B = te.placeholder((b.shape), name="B", dtype=dtype)
    C = te.placeholder((c.shape), name="C", dtype=dtype)
    D = te.placeholder((d.shape), name="D", dtype=dtype)
    E = te.placeholder((e.shape), name="E", dtype=dtype)
    Y_tvm = tvm_func(A, B, C, D, E, X_tvm)
    s = te.create_schedule(Y_tvm.op)
    ctx = tvm.context(target, 0)
    X_test = X_test.astype(np.float32)
    x_tvm = tvm.nd.array(X_test, ctx)
    y_tvm = tvm.nd.array(np.zeros(get_const_tuple(Y_tvm.shape), dtype=Y_tvm.dtype), ctx)
    func = tvm.build(s, [A, B, C, D, E, X_tvm, Y_tvm], target, name = "tree_gemm_dense")
    func(a, b, c, d, e, x_tvm, y_tvm)
    # check
    print(e)
    y_np = np.array(Y_test)
    out_dtype = "float32"
    model = build_model(clf, X_test.shape, out_dtype=out_dtype)
    y_relay = model.run(X_test)
    if(sklearn_func in [DecisionTreeRegressor, ExtraTreeRegressor]):
        y_tvm = y_tvm.asnumpy().flatten()
        y_relay = y_relay.asnumpy().flatten()
    else:
        y_tvm = y_tvm.asnumpy()
        y_relay = y_relay.asnumpy()
    try:
        tvm.testing.assert_allclose(y_tvm, y_relay, rtol=1e-5)
        print("pass")
    except Exception as e:
        print("error")
        print(e)

for n in range(1):
    for n_features in range(10,20,10):
        n_classes = int(n_features / 2)
        test_tree_gemm(DecisionTreeClassifier, decision_tree_classifier, max_leaf_nodes=n_features, n_samples=1000, n_features=n_features, n_classes=n_classes)
        #test_tree_gemm(DecisionTreeRegressor, decision_tree_regressor, max_leaf_nodes=n_features, n_samples=1000, n_features=n_features, n_classes=n_classes)
        #test_tree_gemm(ExtraTreeRegressor, extra_tree_regressor, max_leaf_nodes=n_features, n_samples=1000, n_features=n_features, n_classes=n_classes)
        #test_tree_gemm(ExtraTreeRegressor, extra_tree_regressor, max_leaf_nodes=n_features, n_samples=1000, n_features=n_features, n_classes=n_classes)
 
