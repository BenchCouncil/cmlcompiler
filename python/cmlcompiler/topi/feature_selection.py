"""feature selection"""
import tvm
from tvm import te, topi
from cmlcompiler.topi.label_binarizer import label_binarizer

def chi2(X, y):
    """
    X: sample vectors (n_samples, n_features)
    y: target vector (n_samples,)
    Return chi2 statistics of each feature (n_features,)
    
    Y (n_samples, n_classes)
    observed = Y.T * X (n_features, n_classes)
    feature_count (n_features,)
    class_prob (n_classes,)
    excepted = feature_count.T * class_prob (n_features, n_classes)
    return chisquare of observed and excepted (n_features,)
    """
    n_samples, n_features = X.shape
    k_observed = te.reduce_axis((0, n_samples), name = "k_observed")
    k_feature = te.reduce_axis((0, n_samples), name = "k_feature")
    k_class = te.reduce_axis((0, n_samples), name = "k_class")
    Y = label_binarizer(y)
    _, n_classes = Y.shape
    observed = te.compute((n_features, n_classes), lambda i,j: 
            te.sum(X[k_observed, i] * Y[k_observed, j], axis=k_observed))
    feature_count = te.compute((n_features,), lambda i: te.sum(X[k_feature, i], axis=k_feature))
    class_sum = te.compute((n_classes,), lambda i: te.sum(Y[k_class, i], axis=k_class)).astype("float64")
    class_prob = te.compute((n_classes,), lambda i:te.div(class_sum[i], n_samples))
    excepted = te.compute((n_features, n_classes), lambda i,j: feature_count[i] * class_prob[j])
    k = te.reduce_axis((0, n_classes), name = "k")
    return te.compute((n_features,), lambda i: 
        te.sum(te.div(te.power((observed[i,k] - excepted[i,k]), 2), excepted[i,k]), axis=k))    

def pval(x):
    """
    Input chi2 statistics
    return p value
    """
    k = x.shape[0]
    v = k - 1 
    return gammainc(v/2, x/2)

def select_kbest(X, y, k, score_func):
    """
    Select K best features
    X : sample vectors (n_samples, n_features)
    y : target vector (n_samples, ) 
    k : number of top features to select
    score_func : function to score X and y including chi2, 
    Return (n_samples, k)
    """
    n_samples, _ = X.shape
    scores = score_func(X, y)
    # replace nan with 0
    scores_new = te.compute(scores.shape, lambda i: 
            te.if_then_else(te.isnan(scores[i]), 0, scores[i])) 
    index_remain = topi.sort(topi.topk(scores_new, k=k, ret_type="indices"))
    return te.compute((n_samples, k), lambda i, j: X[i, index_remain[j]])

def select_percentile(X, y, percentile, score_func):
    """
    Select features according to the percentile of the highest scores
    X : sample vectors (n_samples, n_features)
    y : target vector (n_samples, ) 
    percentile : the percentile to select features
    score_func : function to score X and y including chi2, 
    k = 
    Return (n_samples, )
    """
    n_samples, _ = X.shape
    scores = score_func(X, y)
    #replace nan with 0
    scores_new = te.compute(scores.shape, lambda i: 
            te.if_then_else(te.isnan(scores[i]), 0, scores[i])) 
    #k = int(scores.shape[0]*percentile/100)
    k = int(te.div(scores.shape[0] * percentile , 100)) + 1
    index_remain = topi.sort(topi.topk(scores_new, k=k, ret_type="indices"))
    return te.compute((n_samples, k), lambda i, j: X[i, index_remain[j]])
