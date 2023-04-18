import numpy as np
import tvm
import pickle
from tvm import te
from tvm.contrib import graph_executor
from tvm import relay, auto_scheduler
from tvm.topi.sparse.utils import convert_model_dense_to_sparse
from cmlcompiler.algorithms.preprocessing import binarizer,normalizer,label_encoder
from cmlcompiler.algorithms.scaler import min_max_scaler,max_abs_scaler,standard_scaler,robust_scaler
from cmlcompiler.algorithms.linear import linear_binary_classification
from cmlcompiler.algorithms.svm import linear_kernel_svr,sigmoid_kernel_svr,poly_kernel_svr,rbf_kernel_svr,linear_kernel_svc,sigmoid_kernel_svc,poly_kernel_svc,rbf_kernel_svc
from cmlcompiler.utils.supported_ops import preprocessing_op,linear_clf,linear_reg,tree_clf,tree_reg,ensemble_clf,ensemble_reg,svm_clf,svm_reg
from cmlcompiler.utils.name_parser import func_name_parser
#from cmlcompiler.utils.tree_common import convert_decision_tree,convert_random_forest
from cmlcompiler.utils.tree_common import convert_decision_tree,convert_random_forest,convert_gbdt_feature
import tarfile
from tvm.topi.sparse.utils import convert_model_dense_to_sparse
from tvm.contrib.debugger import debug_executor
import time
from cmlcompiler.algorithms.tree import decision_tree_classifier,decision_tree_regressor
from cmlcompiler.algorithms.forest import random_forest_classifier,random_forest_regressor,forest_feature_gemm_dense

def init_model(sklearn_model, data_shape, target, dtype, out_dtype, batch_size, sparse_replacing, dtype_converting, elimination, auto_tuning):
    """
    Init model class
    """
    if type(sklearn_model) in linear_clf:
        out_dtype = "int"
        model = LinearModel(sklearn_model, data_shape, target, dtype, out_dtype, "linear_clf", batch_size, sparse_replacing, dtype_converting, elimination, auto_tuning)
    elif type(sklearn_model) in linear_reg:
        model = LinearModel(sklearn_model, data_shape, target, dtype, out_dtype, "linear_reg", batch_size, sparse_replacing, dtype_converting, elimination, auto_tuning)
    elif type(sklearn_model) in tree_clf:
        #out_dtype = "int"
        model = TreeModel(sklearn_model, data_shape, target, dtype, out_dtype, "tree_clf", batch_size, sparse_replacing, dtype_converting, elimination, auto_tuning)
    elif type(sklearn_model) in tree_reg:
        model = TreeModel(sklearn_model, data_shape, target, dtype, out_dtype, "tree_reg", batch_size, sparse_replacing, dtype_converting, elimination, auto_tuning)
    elif type(sklearn_model) in ensemble_clf:
        #out_dtype = "int"
        model = ForestModel(sklearn_model, data_shape, target, dtype, out_dtype, "forest_clf", batch_size, sparse_replacing, dtype_converting, elimination, auto_tuning)
    elif type(sklearn_model) in ensemble_reg:
        model = ForestModel(sklearn_model, data_shape, target, dtype, out_dtype, "forest_reg", batch_size, sparse_replacing, dtype_converting, elimination, auto_tuning)
    elif type(sklearn_model) in svm_clf:
        model = SVMModel(sklearn_model, data_shape, target, dtype, out_dtype, "svm_clf", batch_size, sparse_replacing, dtype_converting, elimination, auto_tuning)
    elif type(sklearn_model) in svm_reg:
        model = SVMModel(sklearn_model, data_shape, target, dtype, out_dtype, "svm_reg", batch_size, sparse_replacing, dtype_converting, elimination, auto_tuning)
    elif type(sklearn_model) in preprocessing_op:
        model = PreprocessingModel(sklearn_model, data_shape, target, dtype, out_dtype, "preprocessing_op", batch_size, sparse_replacing, dtype_converting, elimination, auto_tuning)
    else:
        print("Unkown ops")
    return model

def build_model(sklearn_model, data_shape, target="llvm", dtype="float32", out_dtype="float32", batch_size=None, sparse_replacing=False, dtype_converting=False, elimination=False, auto_tuning=False):
    """
    Input sklearn model
    Return tvm model
    """
    if(batch_size == None):
        batch_size = data_shape[0]
    model = init_model(sklearn_model, data_shape, target, dtype, out_dtype, batch_size, sparse_replacing, dtype_converting, elimination, auto_tuning)
    model.build()
    return model

def tune_log_name(sklearn_model, target, sparse_replacing, dtype_converting):
    """
    Set tune log name according to sklearn_model, target, sparse_replacing and dtype_converting
    """
    func_name = str(sklearn_model).split("(")[0]
    if((sparse_replacing == True) and (dtype_converting == True)):
        log_file = func_name + "_" + target + "_sparse_dtype"+ ".json"
    if((sparse_replacing == True) and (dtype_converting == False)):
        log_file = func_name + "_" + target + "_sparse"+ ".json"
    if((sparse_replacing == False) and (dtype_converting == True)):
        log_file = func_name + "_" + target + "_dtype"+ ".json"
    else:
        log_file = func_name + "_" + target + ".json"
    return log_file

def tune_model(sklearn_model, data_shape, log_file=None, n_trials=2000, n_repeat=10, target="llvm", dtype="float32", out_dtype="float32", batch_size=None, sparse_replacing=False, dtype_converting=True, elimination=True, auto_tuning=True):
    """
    Input sklearn model
    Return best tvm model by auto tuning
    """
    if(batch_size == None):
        batch_size = data_shape[0]
    model = init_model(sklearn_model, data_shape, target, dtype, out_dtype, batch_size, sparse_replacing, dtype_converting, elimination, auto_tuning)
    if(log_file == None):
        log_file = tune_log_name(sklearn_model, target, sparse_replacing, dtype_converting)
    model.tune(n_trials, n_repeat, log_file)
    model.load_tune(log_file)
    return model

def load_tune(sklearn_model, data_shape, log_file=None, target="llvm", dtype="float32", out_dtype="float32", batch_size=None, sparse_replacing=False, dtype_converting=True, elimination=True, auto_tuning=True):
    """
    load tuning log and return the best model
    """
    if(batch_size == None):
        batch_size = data_shape[0]
    model = init_model(sklearn_model, data_shape, target, dtype, out_dtype, batch_size, sparse_replacing, dtype_converting, elimination, auto_tuning)
    if(log_file == None):
        log_file = tune_log_name(sklearn_model, target, sparse_replacing, dtype_converting)
    model.load_tune(log_file)
    return model

def load_model(filename):
    """
    Input filename
    Return tvm model
    """
    with open(filename + ".sav", "rb") as f:
        init_params = pickle.load(f)
    model = init_model(*init_params)
    lib = tvm.runtime.load_module(filename + ".so")
    model.model = graph_executor.GraphModule(lib["default"](model.dev))
    return model

def save_model(model, filename):
    """
    save model init_params into filename.sav
    save model.lib into filename.so
    """
    # TODO: Change the save and load logic to simplify it
    # can't pickle the whole model, using tvm export_library to save tvm lib, then pickle init_params
    model.lib.export_library(filename + ".so")
    init_params = []
    init_params.append(model.sklearn_model)
    init_params.append(model.data_shape)
    init_params.append(model.target)
    init_params.append(model.dev)
    init_params.append(model.dtype)
    init_params.append(model.out_dtype)
    init_params.append(model.batch_size)
    with open(filename + ".sav", "wb") as f:
        pickle.dump(init_params, f)
    """
    with tarfile.open(filename + ".tar.gz", "w:gz") as tar:
        tar.add(filename + ".sav")
        tar.add(filename + ".so")
    """

class BaseModel:
    """
    Base model
    """
    def __init__(self, sklearn_model, data_shape, target, dtype, out_dtype, flag_clf, batch_size, sparse_replacing, dtype_converting, auto_tuning):
        self.sklearn_model = sklearn_model
        self.data_shape = data_shape
        self.target = target
        self.dev = tvm.device(str(target), 0)
        self.dtype = dtype
        self.out_dtype = out_dtype
        self.flag_clf = flag_clf
        self.batch_size = batch_size
        try:
            self.batch_shape = (batch_size, self.data_shape[1])
        except:
            self.batch_shape = (batch_size, )
        self.sparse_replacing = sparse_replacing
        self.dtype_converting = dtype_converting
        self.auto_tuning = auto_tuning

    def get_mod(self):
        args = relay.analysis.free_vars(self.algo)
        net = relay.Function(args, self.algo)
        mod = tvm.IRModule.from_expr(net)
        #mod = relay.transform.InferType()(mod)
        name_list = [v.name_hint for v in mod["main"].params]
        params = {}
        i = 0
        for n in name_list:
            if n == "data":
                pass
            else:
                params[n] = self.params[i]
                i = i + 1
        #print(params)
        return mod, params

    def build(self):
        """
        Build model

        Input sklearn model
        param_list : [Ndarray]
            list of parameters for model, parsing from sklearn
        Returns
        
        model : 
            Executable model to run 
        """
        mod, params = self.get_mod()
        #mod, params = convert_model_dense_to_sparse(mod, params, bs_r=1)
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(mod, self.target, params=params)
        self.lib = lib
        if(self.target in ["llvm", "llvm -mcpu=core-avx2", "cuda"]):
        #if(tvm.testing.device_enabled(self.target)): 
            self.model = graph_executor.GraphModule(lib["default"](self.dev))
        #self.model = debug_executor.create(lib.get_graph_json(), lib.lib, self.dev, dump_root="/tmp")

    def run(self, data, breakdown=False):
        if((self.flag_clf in ["tree_clf", "tree_reg", "forest_clf", "forest_reg"]) and self.target != "cuda"):
            data = data.asnumpy()
            out = np.empty([self.data_shape[0]], dtype=self.out_dtype)
            n_batch = self.data_shape[0] // self.batch_size
            load_time = 0
            exec_time = 0 
            store_time = 0
            for i in range(n_batch):
                start = i * self.batch_size
                end = (i + 1) * self.batch_size
                a = time.perf_counter()
                input_data = tvm.nd.array(data[start:end])
                #self.model.set_input("data", data[start:end])
                self.model.set_input("data", input_data)
                b = time.perf_counter()
                self.model.run()
                c = time.perf_counter()
                out[start:end] = self.model.get_output(0).asnumpy().flatten()
                d = time.perf_counter()
                load_time = load_time + b - a
                exec_time = exec_time + c - b
                store_time = store_time + d - c
        else:
            a = time.perf_counter()
            self.model.set_input("data", data)
            b = time.perf_counter()
            self.model.run()
            c = time.perf_counter()
            out = self.model.get_output(0)
            d = time.perf_counter()
            load_time = b - a
            exec_time = c - b
            store_time = d - c
        if(breakdown == True):
            take_time = 0
            return load_time, exec_time, store_time, take_time, out
        else:
            return out

    def tune(self, n_trials, n_repeat, log_file):
        mod, params = self.get_mod()
        tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, self.target)
        for idx, task in enumerate(tasks):
            print("========== Task %d  (workload key: %s) ==========" % (idx, task.workload_key))
            print(task.compute_dag)
        tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
        tune_option = auto_scheduler.TuningOptions(
                num_measure_trials=n_trials,
                runner=auto_scheduler.LocalRunner(repeat=n_repeat, enable_cpu_cache_flush=True),
                measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
                )
        tuner.tune(tune_option)
    
    def load_tune(self, log_file):
        mod, params = self.get_mod()
        with auto_scheduler.ApplyHistoryBest(log_file):
            with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
                lib = relay.build(mod, target=self.target, params=params)
        self.model = graph_executor.GraphModule(lib["default"](self.dev))

    def save_model(self, filename):
        self.lib.export_library(filename + ".tar")

class PreprocessingModel(BaseModel):
    """
    Preprocessing model
    """
    def __init__(self, sklearn_model, data_shape, target, dtype, out_dtype, flag_clf, batch_size, sparse_replacing, dtype_converting, elimination, auto_tuning):
        super(PreprocessingModel, self).__init__(sklearn_model, data_shape, target, dtype, out_dtype, flag_clf, batch_size, sparse_replacing, dtype_converting, auto_tuning)
        self.params = self._parse_params()    
        self.elimination = elimination
        self._get_algo()

    def _parse_params(self):
        """
        """
        func_name = str(self.sklearn_model).split("(")[0]
        if(func_name == "Binarizer"):
            threshold = np.array(self.sklearn_model.threshold)
            threshold = threshold.astype(self.dtype)
            threshold = tvm.nd.array(threshold)
            return [threshold]
        if(func_name == "MinMaxScaler"):
            scale = self.sklearn_model.scale_.reshape(-1,1).T
            scale = scale.astype(self.dtype)
            scale = tvm.nd.array(scale)
            min_x = self.sklearn_model.min_
            min_x = min_x.astype(self.dtype)
            min_x = tvm.nd.array(min_x)
            return scale, min_x
        if(func_name == "MaxAbsScaler"):
            scale = self.sklearn_model.scale_
            scale = scale.astype(self.dtype)
            scale = tvm.nd.array(scale)
            return [scale]
        if(func_name == "StandardScaler"):
            mean = self.sklearn_model.mean_
            mean = mean.astype(self.dtype)
            mean = tvm.nd.array(mean)
            scale = self.sklearn_model.scale_
            scale = scale.astype(self.dtype)
            scale = tvm.nd.array(scale)
            return mean, scale
        if(func_name == "RobustScaler"):
            center = self.sklearn_model.center_
            center = center.astype(self.dtype)
            center = tvm.nd.array(center)
            scale = self.sklearn_model.scale_
            scale = scale.astype(self.dtype)
            scale = tvm.nd.array(scale)
            return center, scale

    def _get_algo(self):
        func_name = str(self.sklearn_model).split("(")[0]
        if(func_name == "Binarizer"):
            func = binarizer
            self.algo = func(self.batch_shape, self.dtype)
        if(func_name == "LabelEncoder"):
            func = label_encoder
            self.algo = func(self.batch_shape, self.dtype)
        if(func_name == "Normalizer"):
            norm = self.sklearn_model.norm
            func = normalizer
            self.algo = func(self.batch_shape, self.dtype, norm)
        if(func_name == "MinMaxScaler"):
            func = min_max_scaler
            self.algo = func(self.batch_shape, self.batch_shape[1], self.dtype)
        if(func_name == "MaxAbsScaler"):
            func = max_abs_scaler
            self.algo = func(self.batch_shape, self.batch_shape[1], self.dtype)
        if(func_name == "StandardScaler"):
            func = standard_scaler
            self.algo = func(self.batch_shape, self.batch_shape[1], self.dtype)
        if(func_name == "RobustScaler"):
            func = robust_scaler
            self.algo = func(self.batch_shape, self.batch_shape[1], self.dtype)

class LinearModel(BaseModel):
    """
    Linear model
    """
    def __init__(self, sklearn_model, data_shape, target, dtype, out_dtype, flag_clf, batch_size, sparse_replacing, dtype_converting, elimination, auto_tuning):
        super(LinearModel, self).__init__(sklearn_model, data_shape, target, dtype, out_dtype, flag_clf, batch_size, sparse_replacing, dtype_converting, auto_tuning)
        self.params = self._parse_params()    
        self.elimination = elimination
        self._get_algo()

    def _parse_params(self):
        """
        Parsing differs between Linear Classification and Regression
        """
        coef = self.sklearn_model.coef_
        bias = self.sklearn_model.intercept_
        if self.flag_clf == "linear_reg":
            coef = coef.reshape(-1, 1).T
            if not isinstance(bias, np.ndarray):
                bias = np.array([bias])
        coef = coef.astype(self.dtype)
        bias = bias.astype(self.dtype)
        coef = tvm.nd.array(coef)
        bias = tvm.nd.array(bias)
        self.n_class = coef.shape[0]
        if (self.flag_clf == "linear_reg") or (self.flag_clf == "linear_clf" and self.n_class == 1):
            return coef, bias 
        else:   
            classes = np.array(self.sklearn_model.classes_)
            classes = classes.astype(self.dtype)
            classes = tvm.nd.array(classes)
            # Note that relay.take(classes, y) influence the order of params in relay.build
            return classes, coef, bias

    def _get_algo(self):
        # For linear binary classification, using one var to represent 0-1 class, differs from multi classification 
        if (type(self.sklearn_model) in linear_clf) and (self.n_class == 1):
            func = linear_binary_classification
        else:
            func = func_name_parser(self.sklearn_model)
        self.algo = func(self.batch_shape, self.n_class, self.elimination)
    
class TreeModel(BaseModel):
    """
    Tree model
    """
    def __init__(self, sklearn_model, data_shape, target, dtype, out_dtype, flag_clf, batch_size, sparse_replacing, dtype_converting, elimination, auto_tuning):
        super(TreeModel, self).__init__(sklearn_model, data_shape, target, dtype, out_dtype, flag_clf, batch_size, sparse_replacing, dtype_converting, auto_tuning)
        self.params = self._parse_params()    
        self._get_algo()
        #print(self.params)
   
    def _parse_params(self):
        if(self.sparse_replacing == True):
            S_data, S_indices, S_indptr, T, B, L = convert_decision_tree(self.batch_shape[1], self.sklearn_model, self.flag_clf, self.dtype, self.target, self.sparse_replacing, self.dtype_converting)
            self.L = L.asnumpy()
            self.internal_node = T.shape[0]
            self.leaf_node = B.shape[0]
            return L, S_data, S_indices, S_indptr, T, B
        else:
            S, T, B, L = convert_decision_tree(self.batch_shape[1], self.sklearn_model, self.flag_clf, self.dtype, self.target, self.sparse_replacing, self.dtype_converting)
            self.internal_node = S.shape[0]
            self.leaf_node = B.shape[0]
            return L, S, T, B
   
    def _get_algo(self):
        if(self.flag_clf == "tree_clf"):
            func = decision_tree_classifier
        else:
            func = decision_tree_regressor
        #func = func_name_parser(self.sklearn_model)
        print(self.batch_shape)
        self.algo = func(self.batch_shape, self.internal_node, self.leaf_node, self.dtype, self.sparse_replacing, self.dtype_converting)

class ForestModel(BaseModel):
    """
    Forest model
    More general ensemble learning models to be considered 
    """
    def __init__(self, sklearn_model, data_shape, target, dtype, out_dtype, flag_clf, batch_size, sparse_replacing, dtype_converting, elimination, auto_tuning):
        #dtype_converting = True
        super(ForestModel, self).__init__(sklearn_model, data_shape, target, dtype, out_dtype, flag_clf, batch_size, sparse_replacing, dtype_converting, auto_tuning)
        self.params = self._parse_params()    
        self._get_algo()

    def _parse_params(self):
        if(self.flag_clf == "forest_clf"):
            if(self.sparse_replacing == True):
                S_data, S_indices, S_indptr, T, B, step, L, classes = convert_random_forest(self.batch_shape[1], self.sklearn_model, self.flag_clf, self.dtype, self.target, self.dtype_converting, self.sparse_replacing)
            else:
                S, T, B, step, L, classes = convert_random_forest(self.batch_shape[1], self.sklearn_model, self.flag_clf, self.dtype, self.target, self.dtype_converting, self.sparse_replacing)
        else:
            if(self.sparse_replacing == True):
                S_data, S_indices, S_indptr, T, B, step, L = convert_random_forest(self.batch_shape[1], self.sklearn_model, self.flag_clf, self.dtype, self.target, self.dtype_converting, self.sparse_replacing)
            else:
                S, T, B, step, L = convert_random_forest(self.batch_shape[1], self.sklearn_model, self.flag_clf, self.dtype, self.target, self.dtype_converting, self.sparse_replacing)
        self.n_estimator_x_internal_node = T.shape[0]
        # n_estimator * internal_node
        self.n_estimator = B.shape[0]
        self.leaf_node = B.shape[1]
        self.internal_node = B.shape[2]
        self.n_estimator_x_leaf_node = self.n_estimator * self.leaf_node
        # n_estimator * leaf_node
        self.label = L.shape[0]
        if(self.flag_clf == "forest_clf"):
            if(self.sparse_replacing == True):
                return classes, L, S_data, S_indices, S_indptr, T, B, step
            else:
                return classes, L, S, T, B, step
        else:
            if(self.sparse_replacing == True):
                return L, S_data, S_indices, S_indptr, T, B, step
            else:
                return L, S, T, B, step

    def _get_algo(self):
        #func = func_name_parser(self.sklearn_model)
        if(self.flag_clf == "forest_clf"):
            func = random_forest_classifier
        else:
            func = random_forest_regressor
        self.algo = func(
                self.batch_shape, 
                self.n_estimator_x_internal_node,
                self.n_estimator,
                self.batch_size,
                self.internal_node,
                self.leaf_node,
                self.n_estimator_x_leaf_node,
                self.label,
                self.dtype_converting,
                self.sparse_replacing,
                self.dtype
                )


class SVMModel(BaseModel):
    """
    SVM model
    """
    def __init__(self, sklearn_model, data_shape, target, dtype, out_dtype, flag_clf, batch_size, sparse_replacing, dtype_converting, auto_tuning):
        super(SVMModel, self).__init__(sklearn_model, data_shape, target, dtype, out_dtype, flag_clf, batch_size, sparse_replacing, dtype_converting, auto_tuning)
        self.params = self._parse_params()    
        self._get_algo()
    
    def _parse_params(self):
        self.kernel = self.sklearn_model.kernel
        support_vectors = self.sklearn_model.support_vectors_
        support_vectors = support_vectors.astype(self.dtype)
        dual_coef = self.sklearn_model.dual_coef_
        dual_coef = dual_coef.astype(self.dtype)
        dual_coef = tvm.nd.array(dual_coef)
        bias = self.sklearn_model.intercept_
        bias = bias.astype(self.dtype)
        bias = tvm.nd.array(bias)
        n_support = self.sklearn_model.n_support_
        n_support = n_support.astype(self.dtype)
        n_support = tvm.nd.array(n_support)
        ctx = tvm.device(self.target, 0)
        gamma = self.sklearn_model._gamma
        gamma = gamma.astype(self.dtype)
        coef0 = self.sklearn_model.coef0
        coef0 = np.array([coef0])
        coef0 = coef0.astype(self.dtype)
        coef0 = tvm.nd.array(coef0)
        self.n_sv = support_vectors.shape[0]
        if(self.kernel == "linear"):
            support_vectors = tvm.nd.array(support_vectors)
            return support_vectors, dual_coef, bias
        elif(self.kernel == "sigmoid"):
            # kernel function is tanh(gamma * <x, x'> + coef0), calculate gamma * x' here as support_vectors
            support_vectors = support_vectors * gamma
            support_vectors = tvm.nd.array(support_vectors)
            return support_vectors, coef0, dual_coef, bias
        elif(self.kernel == "poly"):
            # kernel function is (gamma * <x, x'> + coef0) ^ degree, calculate gamma * x' here as support_vectors
            support_vectors = support_vectors * gamma
            support_vectors = tvm.nd.array(support_vectors)
            degree = self.sklearn_model.degree
            degree = np.array([degree])
            degree = degree.astype(self.dtype)
            degree = tvm.nd.array(degree)
            return support_vectors, coef0, degree, dual_coef, bias
        else:
            # "rbf" as default
            # kernel function is exp(-gamma * ||x - x'|| ^ 2) = exp(-gamma * x^2 - gamma * x'^2 + 2 * gamma * <x, x'>)
            # calculate - gamma * x'^2 here as sv_norm, calculate 2 * gamma * x' here as supported_vectors
            sv_norm = np.power(support_vectors, 2)
            sv_norm = np.sum(sv_norm, axis=1)
            sv_norm = sv_norm * gamma * (-1)
            sv_norm = sv_norm.astype(self.dtype)
            sv_norm = sv_norm.reshape(1,-1)
            sv_norm = tvm.nd.array(sv_norm)
            support_vectors = support_vectors * gamma * 2
            support_vectors = tvm.nd.array(support_vectors)
            gamma = -gamma
            gamma = tvm.nd.array(gamma)
            return support_vectors, gamma, sv_norm, dual_coef, bias

    def _get_algo(self):
        if(self.flag_clf == "svm_reg"):
            if(self.kernel == "linear"):
                self.algo = linear_kernel_svr(self.batch_shape, self.n_sv, self.dtype)
            elif(self.kernel == "sigmoid"):
                self.algo = sigmoid_kernel_svr(self.batch_shape, self.n_sv, self.dtype)
            elif(self.kernel == "poly"):
                self.algo = poly_kernel_svr(self.batch_shape, self.n_sv, self.dtype)
            else:
                self.algo = rbf_kernel_svr(self.batch_shape, self.n_sv, self.dtype)
        elif(self.flag_clf == "svm_clf"):
            if(self.kernel == "linear"):
                self.algo = linear_kernel_svc(self.batch_shape, self.n_sv, self.dtype)
            elif(self.kernel == "sigmoid"):
                self.algo = sigmoid_kernel_svc(self.batch_shape, self.n_sv, self.dtype)
            elif(self.kernel == "poly"):
                self.algo = poly_kernel_svc(self.batch_shape, self.n_sv, self.dtype)
            else:
                self.algo = rbf_kernel_svc(self.batch_shape, self.n_sv, self.dtype)
    
class GBDTFeature(BaseModel):
    """
    GBDT model to get feature embedding
    """
    def __init__(self, sklearn_model, data_shape, target, dtype, out_dtype, flag_clf, batch_size, sparse_replacing, dtype_converting, elimination, auto_tuning):
        #dtype_converting = True
        super(GBDTFeature, self).__init__(sklearn_model, data_shape, target, dtype, out_dtype, flag_clf, batch_size, sparse_replacing, dtype_converting, auto_tuning)
        self.params = self._parse_params()    
        self._get_algo()

    def _parse_params(self):
        self.batch_shape = (self.batch_size, self.data_shape)
        S, T, B, len_cumsum = convert_gbdt_feature(self.batch_shape[1], self.sklearn_model, self.flag_clf, self.dtype, self.target, self.dtype_converting, self.sparse_replacing)
        self.n_estimator_x_internal_node = T.shape[0]
        # n_estimator * internal_node
        self.n_estimator = B.shape[0]
        self.leaf_node = B.shape[1]
        self.internal_node = B.shape[2]
        self.n_estimator_x_leaf_node = self.n_estimator * self.leaf_node
        self.label = 0
        # n_estimator * leaf_node
        return S, T, B, len_cumsum
    
    def _get_algo(self):
        self.algo = forest_feature_gemm_dense(
                self.batch_shape, 
                self.n_estimator_x_internal_node,
                self.n_estimator,
                self.batch_size,
                self.internal_node,
                self.leaf_node,
                self.n_estimator_x_leaf_node,
                self.label,
                self.dtype_converting,
                self.sparse_replacing,
                self.dtype
                )


