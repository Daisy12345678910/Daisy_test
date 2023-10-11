# coding=utf-8
# Scripts for AI-based cooling load prediction

import os
import xlrd
import xlsxwriter
from xgboost.sklearn import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error as mse
import numpy as np
from itertools import combinations
import warnings


def _flatten(l):
    for ele in l:
        if not isinstance(ele, (list, tuple)):
            yield ele
        else:
            yield from _flatten(ele)


class _CombOut(object):
    def __init__(self, keys):
        self.combs = []
        for i in range(len(keys)):
            iter_comb = combinations(keys, i+1)
            for obj in iter_comb:
                self.combs.append(obj)
        self.nt = len(self.combs)
        self.n = -1

    def gen(self):
        self.n += 1
        if self.n < self.nt:
            return self.combs[self.n]
        else:
            return None


class _DatLoad(object):
    def __init__(self, input_data, n_per_day=24, trainday=30, keys=None):
        self.n_per_day = n_per_day
        self.trainday = trainday

        self.timestamp = input_data["timestamp"]
        self.loads = input_data["loads"]
        self.vars = input_data["in_vars"]

        self.ndays = len(self.timestamp)/n_per_day
        self.tvars = []
        if not keys:
            keys = self.vars.keys()
        self.key_comb = _CombOut(keys)

    def GetParams(self):
        r = self.key_comb.gen()
        # Check task complete signal
        if r is None:
            return None
        # Update vars
        t_keys = r
        print("Current input parameters: %s" % str(r))
        keys = _flatten(t_keys)
        vars_temp = []
        key_tup = tuple(keys)
        for key in key_tup:
            vars_temp.append(self.vars[key])
        self.tvars = np.column_stack(vars_temp)
        return key_tup

    def GetData(self, start_day, pred_days):
        if start_day + pred_days > self.ndays:
            return None
        start_ind = int(round(start_day * self.n_per_day, 0))
        end_ind = int(round(start_ind + pred_days * self.n_per_day, 0))

        test_vars = self.tvars[start_ind: end_ind]
        test_loads = self.loads[start_ind: end_ind]
        test_time = self.timestamp[start_ind: end_ind]

        if start_day < self.trainday:
            train_ind = int((start_day + self.ndays - self.trainday) * self.n_per_day)
            train_loads = np.hstack((self.loads[train_ind:], self.loads[:start_ind]))
            train_vars = np.vstack((self.tvars[train_ind:], self.tvars[: start_ind]))
        else:
            train_ind = (start_day - self.trainday) * self.n_per_day
            train_loads = self.loads[train_ind: start_ind]
            train_vars = self.tvars[train_ind: start_ind]

        return train_vars, train_loads, test_vars, test_loads, test_time


# For writing results to specific location
class _DataLog(object):
    def __init__(self, save_dir):
        self.wb = xlsxwriter.Workbook(save_dir)
        self.st = self.wb.add_worksheet()
        self.st.write(0, 0, "TimeStamp")
        self.st.write(0, 1, "Real Load")
        self.st.write(0, 2, "Predicted Load")

    def logData(self, timestamp, real, pred):
        start_ind = 1
        for i in range(len(timestamp)):
            self.st.write(start_ind+i, 0, timestamp[i])
            self.st.write(start_ind+i, 1, real[i])
            self.st.write(start_ind+i, 2, pred[i])

    def saveFile(self):
        self.wb.close()


# Main class for AI-based prediction
class Model(object):
    def __init__(self, n_per_day, method="XGB",  train_days=30, update_days=1):
        self.n_per_day = n_per_day
        self.method = method
        self.train_days = train_days
        self.update_days = update_days
        self.para_comb = None
        self._raw_data = None
        self._preded_days = 0
        self.predictor = None

    # return models based on selections
    @staticmethod
    def _model_select(method):
        if method == "XGB":
            model = XGBRegressor()
            tuned_parameters = {"max_depth": [3, 4, 5, 6, 7],
                                "min_child_weight": [0.01, 0.1, 1, 10],
                                "reg_alpha": [0, 0.01, 1, 10],
                                "reg_lambda": [0, 0.01, 1, 10],
                                "objective": ['reg:squarederror', 'reg:gamma', 'reg:tweedie']}
        elif method == "RF":
            model = RandomForestRegressor()
            tuned_parameters = {"max_depth": [6, 8, 10, 12, 15],
                                "n_estimators": [50, 80, 100, 150, 200],
                                "min_samples_split": [5, 10, 20, 50],
                                "min_samples_leaf": [2, 5, 10]}
        elif method == "MLP":
            model = MLPRegressor()
            tuned_parameters = {"hidden_layer_sizes": [(50,)*1, (100,)*1, (200,)*1, (500,)*1,
                                                       (50,)*2, (100,)*2, (200,)*2, (500,)*2,
                                                       (50,)*3, (100,)*3, (200,)*3, (500,)*3,
                                                       (50,)*4, (100,)*4, (200,)*4, (500,)*4, ]}
        else:
            raise Exception("Method not supported")
        return model, tuned_parameters

    @staticmethod
    def _get_vars_by_keys(in_vars, keys):
        keys = _flatten(keys)
        vars_temp = []
        for key in keys:
            vars_temp.append(in_vars[key])
        vars_array = np.column_stack(vars_temp)
        return vars_array

    # Base function for prediction
    def _predict(self, dLoader, parallel=1):
        n_day = 0
        pred_result = []
        real_result = []
        time_stamps = []
        # Predict by day
        while True:
            data = dLoader.GetData(n_day + self.train_days, self.update_days)
            if not data:
                break

            train_vars, train_loads, test_vars, test_loads, test_time = data
            AI_model, AI_para = self._model_select(self.method)
            clf = GridSearchCV(AI_model, AI_para, scoring="neg_mean_squared_error", n_jobs=parallel, cv=5)
            clf.fit(train_vars, train_loads)
            pred_loads = clf.predict(test_vars)

            pred_result += pred_loads.tolist()
            real_result += test_loads.tolist()
            time_stamps += test_time
            n_day += self.update_days

        err_rmse = np.sqrt(mse(real_result, pred_result))
        return err_rmse, pred_result, real_result, time_stamps

    # Base function to get error
    def _predict_get_error(self, dLoader, parallel=1):
        err, _, __, ___ = self._predict(dLoader, parallel=parallel)
        return err

    # Based function to get prediction results
    def _predict_get_result(self, dLoader, parallel=1):
        _, pred_result, real_result, time_stamp = self._predict(dLoader, parallel=parallel)
        return time_stamp, real_result, pred_result

    # Load raw_data from an excel template file (.xlsx)
    def load_from_file(self, fpath):
        if os.path.exists(fpath):
            wb = xlrd.open_workbook(fpath)
            st = wb.sheet_by_index(0)
            timestamp = st.col_values(0, 1)
            loads = np.array(st.col_values(1, 1))
            in_vars = {}
            for i in range(st.ncols-2):
                in_vars[st.cell_value(0, i+2)] = np.array(st.col_values(i+2, 1))
            self._raw_data = {
                "timestamp": timestamp,
                "loads": loads,
                "in_vars": in_vars
            }
            if self.para_comb == None:
                self.para_comb = list(in_vars.keys())
        else:
            raise Exception("File not exists")

    # Direct input as raw data
    def load_data(self, input_data):
        self._raw_data = input_data
        self.para_comb = list(input_data["in_vars"].keys())

    # Optimization of input parameters
    def para_optimize(self, keys=None, parallel=1):
        dLoader = _DatLoad(
            input_data=self._raw_data,
            n_per_day=self.n_per_day,
            trainday=self.train_days,
            keys=keys,
        )
        err_min = None
        keys_opt = None
        # Optimize through different parameter combinations
        while True:
            keys = dLoader.GetParams()
            if not keys:
                break

            err_rmse = self._predict_get_error(dLoader, parallel=parallel)
            if not err_min:
                err_min = err_rmse
                keys_opt = keys
            elif err_rmse < err_min:
                err_min = err_rmse
                keys_opt = keys

        self.para_comb = tuple(keys_opt)

    # Optimization of methods
    def method_optimize(self, parallel=1):
        dLoader = _DatLoad(
            input_data=self._raw_data,
            n_per_day=self.n_per_day,
            trainday=self.train_days,
            keys=(self.para_comb,),
        )
        dLoader.GetParams()

        err_min = None
        method_opt = None
        methods = ["XGB", "MLP", "RF"]
        # Optimize through different predicting methods
        for method in methods:
            self.method = method

            err_rmse = self._predict_get_error(dLoader, parallel=parallel)
            if not err_min:
                err_min = err_rmse
                method_opt = method
            elif err_rmse < err_min:
                err_min = err_rmse
                method_opt = method

        self.method = method_opt

    # Optimization of training days
    def traindays_optimize(self, train_days=None, parallel=1):
        if not train_days:
            train_days = (7, 14, 30, 60)
        dLoader = _DatLoad(
            input_data=self._raw_data,
            n_per_day=self.n_per_day,
            trainday=self.train_days,
            keys=(self.para_comb,),
        )
        dLoader.GetParams()

        err_min = None
        trainday_opt = None

        # Optimize different training data size (by day)
        for train_day in train_days:
            dLoader.trainday = train_day
            self.train_days = train_day

            err_rmse = self._predict_get_error(dLoader, parallel=parallel)
            if not err_min:
                err_min = err_rmse
                trainday_opt = train_day
            elif err_rmse < err_min:
                err_min = err_rmse
                trainday_opt = train_day

        self.train_days = trainday_opt

    # Optimization of update days
    def updatedays_optimize(self, update_days=None, parallel=1):
        if not update_days:
            update_days = (1/24, 1, 7)
        dLoader = _DatLoad(
            input_data=self._raw_data,
            n_per_day=self.n_per_day,
            trainday=self.train_days,
            keys=(self.para_comb,),
        )
        dLoader.GetParams()

        err_min = None
        updateday_opt = None

        # Optimize different update day intervals (by day)
        for update_day in update_days:
            self.update_days = update_day

            err_rmse = self._predict_get_error(dLoader, parallel=parallel)
            if not err_min:
                err_min = err_rmse
                updateday_opt = update_day
            elif err_rmse < err_min:
                err_min = err_rmse
                updateday_opt = update_day

        self.update_days = updateday_opt

    def eval_get_error(self, parallel=1):
        dLoader = _DatLoad(
            input_data=self._raw_data,
            n_per_day=self.n_per_day,
            trainday=self.train_days,
            keys=(self.para_comb,),
        )
        dLoader.GetParams()

        err_rmse = self._predict_get_error(dLoader, parallel)
        return err_rmse

    def eval_get_result(self, to_file=None, parallel=1):
        dLoader = _DatLoad(
            input_data=self._raw_data,
            n_per_day=self.n_per_day,
            trainday=self.train_days,
            keys=(self.para_comb,),
        )
        dLoader.GetParams()
        timestamp, real_result, pred_result = self._predict_get_result(dLoader, parallel=parallel)
        if to_file:
            dLogger = _DataLog(save_dir=to_file)
            dLogger.logData(timestamp=timestamp, real=real_result, pred=pred_result)
            dLogger.saveFile()
        else:
            return timestamp, pred_result
    
    def train(self, in_vars, loads, parallel=1):
        vars_array = self._get_vars_by_keys(in_vars, self.para_comb)

        if len(vars_array) != self.n_per_day * self.train_days:
            raise Exception("Dimension not match: train_days=%f, expect %f samples, get %f samples."
                            % (self.train_days, self.n_per_day*self.train_days, len(vars_array)))

        AI_model, AI_para = self._model_select(self.method)
        clf = GridSearchCV(AI_model, AI_para, scoring="neg_mean_squared_error", n_jobs=parallel, cv=5)
        clf.fit(vars_array, loads)
        self.predictor = clf
        self._preded_days = 0
    
    def predict(self, pred_vars):
        vars_array = self._get_vars_by_keys(pred_vars, self.para_comb)

        self._preded_days += len(vars_array)
        if self._preded_days > self.n_per_day * self.update_days:
            warnings.warn("Predicted day exceeds update_day limit. Use train() to update model.", DeprecationWarning)

        pred_loads = self.predictor.predict(vars_array)
        return pred_loads
