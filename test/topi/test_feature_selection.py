"""test feature selection"""
import tvm
from tvm import te
from tvm.topi.utils import get_const_tuple
from sklearn import feature_selection
import numpy as np
from cmlcompiler.topi.feature_selection import chi2,pchi2,select_kbest,select_percentile,select_fpr
from sklearn.datasets import load_iris,load_digits
from scipy import special

def test_score_func(target="llvm"):
    # load datasets
    X, y = load_digits(return_X_y=True)
    # sklearn implements
    scores, pval = feature_selection.chi2(X, y)
    # tvm implements
    A = te.placeholder((X.shape), name="A", dtype="float64")
    B = te.placeholder((y.shape), name="B", dtype="int64")
    C = chi2(A, B)
    s = te.create_schedule(C.op)
    ctx = tvm.context(target, 0)
    a = tvm.nd.array(X, ctx)
    b = tvm.nd.array(y, ctx)
    c = tvm.nd.array(np.zeros(get_const_tuple(C.shape), dtype=C.dtype), ctx)
    func = tvm.build(s, [A, B, C], target, name = "chi2")
    func(a, b, c)
    print(scores)
    print(pval)
    k = 10
    p = special.chdtrc(k - 1, scores)
    print(p)
    # check
    tvm.testing.assert_allclose(c.asnumpy(), scores, rtol=1e-5)

def test_select_fpr(alpha, target="llvm"):
    # load datasets
    X, y = load_digits(return_X_y=True)   
    # sklearn implements
    X_new = feature_selection.SelectFpr(feature_selection.chi2, alpha=alpha).fit_transform(X, y)
    # tvm implements
    A = te.placeholder((X.shape), name="A", dtype="float64")
    B = te.placeholder((y.shape), name="B", dtype="int64")
    C = select_fpr(A, B, k, chi2)
    s = te.create_schedule(C.op)
    ctx = tvm.context(target, 0)
    a = tvm.nd.array(X, ctx)
    b = tvm.nd.array(y, ctx)
    c = tvm.nd.array(np.zeros(get_const_tuple(C.shape), dtype=C.dtype), ctx)
    func = tvm.build(s, [A, B, C], target, name = "feature selection")
    func(a, b, c)
    print(X_new)
    print("X_new shape", X_new.shape)
    print(c)
    print("c shape", c.shape)
    # check
    tvm.testing.assert_allclose(c.asnumpy(), X_new, rtol=1e-5)


def test_select_kbest(k, target="llvm"):
    # load datasets
    X, y = load_digits(return_X_y=True)   
    # sklearn implements
    X_new = feature_selection.SelectKBest(feature_selection.chi2, k=k).fit_transform(X, y)
    # tvm implements
    A = te.placeholder((X.shape), name="A", dtype="float64")
    B = te.placeholder((y.shape), name="B", dtype="int64")
    C = select_kbest(A, B, k, chi2)
    s = te.create_schedule(C.op)
    ctx = tvm.context(target, 0)
    a = tvm.nd.array(X, ctx)
    b = tvm.nd.array(y, ctx)
    c = tvm.nd.array(np.zeros(get_const_tuple(C.shape), dtype=C.dtype), ctx)
    func = tvm.build(s, [A, B, C], target, name = "feature selection")
    func(a, b, c)
    print(X_new)
    print("X_new shape", X_new.shape)
    print(c)
    print("c shape", c.shape)
    # check
    tvm.testing.assert_allclose(c.asnumpy(), X_new, rtol=1e-5)

def test_select_percentile(p, target="llvm"):
    # load datasets
    X, y = load_digits(return_X_y=True)   
    # sklearn implements
    fselect = feature_selection.SelectPercentile(feature_selection.chi2, percentile=p)
    X_new = fselect.fit_transform(X, y)
    """
    scores = fselect.scores_
    indices = fselect.get_support(indices=True)
    print(indices)
    print(scores[indices])
    print(min(scores[indices]))
    scores[np.isnan(scores)] = np.finfo(scores.dtype).min
    n = int(64*k/100)
    print(np.sort(scores)[-n-1])
    print(np.sort(scores)[-n-2])
    print(np.percentile(scores,100-k))
    """
    # tvm implements
    A = te.placeholder((X.shape), name="A", dtype="float64")
    B = te.placeholder((y.shape), name="B", dtype="int64")
    C = select_percentile(A, B, p, chi2)
    s = te.create_schedule(C.op)
    ctx = tvm.context(target, 0)
    a = tvm.nd.array(X, ctx)
    b = tvm.nd.array(y, ctx)
    c = tvm.nd.array(np.zeros(get_const_tuple(C.shape), dtype=C.dtype), ctx)
    func = tvm.build(s, [A, B, C], target, name = "feature selection")
    func(a, b, c)
    print(X_new)
    print("X_new shape", X_new.shape)
    print(c)
    print("c shape", c.shape)
    # check
    tvm.testing.assert_allclose(c.asnumpy(), X_new, rtol=1e-5)

#test_score_func()
"""
for k in range(10,100,10):
    print(k)
    print(64*k/100)
    try:
        test_select(k, "SelectPercentile")
    except Exception as e:
        print(e)
        pass
    continue
"""
"""
for k in range(64):
    print(k)
    test_select(20, "SelectKBest")
"""
#test_select_kbest(20)
test_select_percentile(0.2)
