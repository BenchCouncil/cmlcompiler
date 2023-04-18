"""Ensemble models based on gemm, including Random Forests, Extra Trees and ..."""
import tvm
from tvm import te, topi
from tvm.topi.utils import get_const_tuple
from sklearn.datasets import make_classification,make_regression
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor,ExtraTreesClassifier,ExtraTreesRegressor
import numpy as np
from sklearn.model_selection import train_test_split
from cmlcompiler.topi.ensemble_gemm import random_forest_classifier,random_forest_regressor,extra_trees_classifier,extra_trees_regressor
from hummingbird.ml import convert
from tree_common import convert_random_forest
from cmlcompiler.model import build_model
import os

os.environ["TVM_BACKTRACE"] = "1"

def test_ensemble_gemm(sklearn_func, tvm_func, max_leaf_nodes, n_estimators, n_samples=100, n_features=4, n_classes=2, dtype="float32", target="llvm"):
    """
    Testing tree based on gemm
    Input: sklearn function and corresponding tvm function
    Output: The result equals or not 
    """
    # load dataset
    if(sklearn_func in [RandomForestRegressor, ExtraTreesRegressor]):
        classification = False
        X, y = make_regression(n_samples=n_samples, n_features=n_features)
    else:
        classification = True
        n_informative = int(n_features / 2)
        X, y = make_classification(n_samples=n_samples, n_features=n_features, n_classes=n_classes, n_informative=n_informative)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    
    # sklearn implements
    clf = sklearn_func(max_leaf_nodes=max_leaf_nodes, n_estimators=n_estimators, random_state=0)
    clf.fit(X_train, y_train)
    Y_test = clf.predict(X_test)
    X_shape = X_train.shape[1]
    #print(Y_test)
    # convert sklearn tree to tvm gemm
    a, b, c, d, e = convert_random_forest(X_shape, clf, classification, dtype=dtype, target=target)
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
    out_dtype = "float32"
    model = build_model(clf, X_test.shape, out_dtype=out_dtype)
    y_relay = model.run(X_test)
    # check
    if(sklearn_func in [RandomForestRegressor, ExtraTreesRegressor]):
        y_tvm = y_tvm.asnumpy().flatten()
        y_relay = y_relay.asnumpy().flatten()
    else:
        y_tvm = y_tvm.asnumpy()
        y_relay = y_relay.asnumpy()
    try:
        tvm.testing.assert_allclose(y_tvm, y_relay, rtol=1e-4)
        print("pass")
    except Exception as e:
        print("error")
        print(e)

test_ensemble_gemm(RandomForestClassifier, random_forest_classifier, max_leaf_nodes=20, n_estimators=100, n_samples=40000, n_features=100, n_classes=10)
#test_ensemble_gemm(RandomForestRegressor, random_forest_regressor, max_leaf_nodes=n_features, n_estimators=100, n_samples=1000, n_features=n_features)
#test_ensemble_gemm(ExtraTreesClassifier, extra_trees_classifier, max_leaf_nodes=n_features, n_estimators=100, n_samples=1000, n_features=n_features, n_classes=n_classes)
#test_ensemble_gemm(ExtraTreesRegressor, extra_trees_regressor, max_leaf_nodes=n_features, n_estimators=100, n_samples=1000, n_features=n_features)

