"""test tree models based on gemm, including Decision Tree and Extra Tree"""
import tvm
from tvm import te, topi
from tvm.topi.utils import get_const_tuple
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor,ExtraTreesClassifier,ExtraTreesRegressor
from sklearn.datasets import make_classification,make_regression
import numpy as np
from sklearn.model_selection import train_test_split
from hummingbird.ml import convert
from cmlcompiler.model import build_model
import os
from cmlcompiler.topi.ensemble_gemm import random_forest_classifier,random_forest_regressor,extra_trees_classifier,extra_trees_regressor
from cmlcompiler.utils.supported_ops import ensemble_clf,ensemble_reg
import tvm.testing

os.environ["TVM_BACKTRACE"] = "1"

def test_ensemble_gemm(sklearn_func, max_leaf_nodes, n_estimators, n_samples, n_features, n_classes, dtype="float32", target="llvm"):
    """
    Testing tree based on gemm
    Input: sklearn function and corresponding tvm function
    Output: The result equals or not 
    """
    if(sklearn_func in ensemble_reg):
        classification = False
        X, y = make_regression(n_samples=n_samples, n_features=n_features)
        out_dtype = "float32"
    else:
        classification = True
        n_informative = int(n_features / 2)
        X, y = make_classification(n_samples=n_samples, n_features=n_features, n_classes=n_classes, n_informative=n_informative)
        out_dtype = "int"
    
    # load dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    
    # sklearn implements
    clf = sklearn_func(max_leaf_nodes=max_leaf_nodes, n_estimators=n_estimators, random_state=0)
    clf.fit(X_train, y_train)
    Y_test = clf.predict(X_test)
    X_test = tvm.nd.array(X_test)
    model = build_model(clf, X_test.shape, out_dtype=out_dtype)
    y_tvm = model.run(X_test)
    # check
    y_np = np.array(Y_test)

    try:
        tvm.testing.assert_allclose(y_tvm, y_np, rtol=1e-5)
        print("pass")
    except Exception as e:
        print("error")
        print(e)

#test_ensemble_gemm(RandomForestClassifier, max_leaf_nodes=10, n_estimators=100, n_samples=1010, n_features=20, n_classes=2)
test_ensemble_gemm(ExtraTreesClassifier, max_leaf_nodes=10, n_estimators=100, n_samples=10000, n_features=20, n_classes=2)
#test_ensemble_gemm(RandomForestRegressor, max_leaf_nodes=10, n_estimators=100, n_samples=1010, n_features=20, n_classes=2)
#test_ensemble_gemm(ExtraTreesRegressor, max_leaf_nodes=10, n_estimators=100, n_samples=10000, n_features=20, n_classes=2)

