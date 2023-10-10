# ComAIPred
AI-based cooling load prediction module for cooling load prediction in commercial buildings

## Copyright
This software is developed under Tsinghua University - Swire Properties Joint Research Center for Building Energy Efficiency and Sustainability. Neither the software, source code or name of the software may be used to endorse or promote commercialized activities derived from this software without specific prior written permission jointly by Tsinghua University and Swire Properties.

## Requirements
ComAIPred requires the installation of the following Python libraries:

- python == 3.7.4
- xlrd == 1.2.0
- XlsxWriter == 1.2.8
- numpy == 1.18.1
- xgboost == 1.0.2
- sklearn == 0.23.2

ComAIPred may still work with some other versions of these libraries, but we have tested our modules with those.

## APIs
```
class AIPred.Model(method="XGB", n_per_day=24, train_days=30, update_days=1):
```

**Parameters:**

|Parameters|Description|
|:----|:----|
|*n_per_day*|*Type: int*. Number of entries for a day. For hourly data, n_per_day is 24. For 15-min data, n_per_day is 96. |
|*method*|*Type: str*. The pre-defined predicting method for the model. Could be one of ("XGB", "RF", "MLP"). Default to "XGB".|
|*train_days*|*Type: int*. Use *train_days* of data as training set for step-ahead prediction. Equivalent concept as "historical data size". Default to 30.|
|*update_days*|*Type: int*. The model updates/trains again after predicting *update_days* of data. Equivalent concept as "predicting window". Default to 1.|

**Attributes:**

|Attributes|Description|
|:----|:----|
|*n_per_day*|*Type: int*. Same as the parameter *n_per_day*|
|*method*|*Type: str*. Same as the parameter *method*|
|*train_days*|*Type: int*. Same as the parameter *train_days*|
|*update_days*|*Type: int*. Same as the parameter *update_days*|
|*para_comb*|*Type: tuple*. Set of parameter names (in *"str"* type) used for predicting cooling loads. Available after calling `load_from_file()` function or `load_data()` function. Updated after calling `para_optimize()` function.|
|*predictor*|*Type: class\<Estimator\>*. The well-tuned and well-trained estimator. Available after calling `train()` function.|

**Methods:**
```
load_from_file(self, fpath)
```
*Parameters:*

> **fpath** (*str*): the path of the input Excel file (.xls or .xlsx) for model optimization. 
>    
> The input file will only have one sheet with headers. The first column is the timestamp, the second column is the real cooling load. All the input parameter data should be arrayed by column from the third column. See template input file.

*Returns:*

> **self**: The method returns self.  
>
> The method will read the data from Excel file and store a reference to the data in the instance.



```
load_data(self, input_data)
```
*Parameters:*

> **input_data** (*dict*): A dictionary that contains data for model optimization. 
>    
> The dictionary must include the following keys:
> 
> *"Timestamp"*: a *list* with a series of timestamps
> 
> *"loads"*: a *list* or *ndarray* object with a series of the real cooling load
> 
> *"in_vars"*: a *dict* with keys as parameter names and values as series of the attribute data

*Returns:*

> **self**: The method returns self.  
>
> The method will get the data and store a reference in the instance.



```
para_optimize(self, keys=None, parallel=1)
```
*Parameters:*

> **keys** (*list* or *tuple*): a list of the name of parameters. The combinations of parameters will be optimized upon the list. If None, use all parameters separately from the raw data. Default to None. 
> 
> If the element in the list is a tuple or list of parameters, these parameters will be considered as a whole. Example:`keys=["Temp", "RH", ("Weekdays", "Hour")]`, parameter "Weekdays" and "Hour" will be considered as a whole and will not appear alone in the optimized parameter combinations.
> 
> The parameter names in the list must correspond with the input data (the keys of `input_data["in_vars"]` or the headers of the Excel file).  
>
> 
> **parallel** (*int*): number of parallel process for the optimization. If -1, the number of parallel process is set to the number of cores. Default to 1.

*Returns:*

> **self**: The method returns self.  
>
> The method iterate all combinations of parameters and select the combinations with the lowest error. This method will override the model attribute `self.para_comb` with the optimal parameter combination.



```
method_optimize(self, parallel=1)
```
*Parameters:*

> **parallel** (*int*): number of parallel process for the optimization. If -1, the number of parallel process is set to the number of cores. Default to 1.

*Returns:*

> **self**: The method returns self.  
>
> The method iterate the following three methods: ["XGB", "RF", "MLP"] and select the optimal predicting method with the lowest error. This method will override the model attribute `self.method` with the optimal predicting method.



```
traindays_optimize(self, train_days=None, parallel=1)
```
*Parameters:*

> **train_days** (*list* or *tuple*):list of train_day. If None, train_days is [7, 14, 30, 60]. Default to None.
> 
> **parallel** (*int*): number of parallel process for the optimization. If -1, the number of parallel process is set to the number of cores.

*Returns:*

> **self**: The method returns self.  
>
> The method iterate the list of integers and select the optimal train_day with the lowest error. This method will override the model attribute `self.train_days` with the optimal train_day.



```
updatedays_optimize(self, update_days=None, parallel=1)
```
*Parameters:*

> **update_days** (*list* or *tuple*):list of update_day. If None, update_days is [1/24, 1, 7]. Default to None.
> 
> **parallel** (*int*): number of parallel process for the optimization. If -1, the number of parallel process is set to the number of cores. Default to 1.

*Returns:*

> **self**: The method returns self.  
>
> The method iterate the list of integers and select the optimal update_day with the lowest error. This method will override the model attribute `self.update_days` with the optimal update_day.



```
eval_get_error(self, parallel=1)
```
*Parameters:*

> **parallel** (*int*): number of parallel process for the optimization. If -1, the number of parallel process is set to the number of cores. Default to 1.

*Returns:*

> **err_rmse**: The Root Mean Squared Error (RMSE) of the prediction with the raw dataset under the current model configurations.
>
> The method evaluates the predicting error under the current model configurations.



```
eval_get_result(self, to_file=None, parallel=1)
```
*Parameters:*

> **to_file** (*str*): the path of the output file to save the predicting results. If None, the method returns the result. Default to None.
> 
> **parallel** (*int*): number of parallel process for the optimization. If -1, the number of parallel process is set to the number of cores. Default to 1.

*Returns:*

> **timestamp**: the series of the timestamps of the predicted results.
> 
> **pred_result**: the series of the predicted cooling load.
> 
> The method predict the cooling load of the raw dataset under the current model configurations. The result can either be exported to an Excel file, or be returned as a tuple `(timestamp, pred_result)`



```
eval_get_result(self, to_file=None, parallel=1)
```
*Parameters:*

> **to_file** (*str*): the path of the output file to save the predicting results. If None, the method returns the result. Default to None.
> 
> **parallel** (*int*): number of parallel process for the optimization. If -1, the number of parallel process is set to the number of cores. Default to 1.

*Returns:*

> **timestamp**: the series of the timestamps of the predicted results.
> 
> **pred_result**: the series of the predicted cooling load.
> 
> The method predict the cooling load of the raw dataset under the current model configurations. The result can either be exported to an Excel file, or be returned as a tuple `(timestamp, pred_result)`



```
train(self, in_vars, loads, parallel=1)
```
*Parameters:*

> **in_vars** (*dict*): a dictionary containing the input parameter data for training, with keys as parameter names and values as data series.
> 
> **loads** (*list* or *tuple*): a list containing the load data.
> 
> **parallel** (*int*): number of parallel process for the optimization. If -1, the number of parallel process is set to the number of cores. Default to 1.

*Returns:*

> **self**: The method returns self.
> 
> The method trains a predictor based on current model configuration and the input training data from this method. The trained predictor will be stored in `self.predictor` as a reference.



```
predict(self, pred_vars)
```
*Parameters:*

> **pred_vars** (*dict*): a dictionary containing the input parameter data for prediction, with keys as parameter names and values as data series.

*Returns:*

> **pred_loads** (*list*): a list containing the predicted cooling load data.
> 
> The method uses the predictor from `self.predictor` and the input variables from `pred_vars` to predict the cooling load.

## Examples
```python
from Scripts.AIpred import Model

# Create an instance of the model
Case1Model = Model(n_per_day=24)
# Load raw data from the file
Case1Model.load_from_file("input_template.xlsx")
# Optimize the parameter combinations
Case1Model.para_optimize(
    keys=["Temp", "RH", "Cloud Cover", "Wind Speed", ("Weekdays", "Hour")],
    parallel=4
)
# Optimize the methods
Case1Model.method_optimize(parallel=4)
# Optimize train days
Case1Model.traindays_optimize(train_days=[7, 14, 30, 60], parallel=4)
# Optimize update days
Case1Model.updatedays_optimize(update_days=[1/24, 1, 7, 30], parallel=4)
# Get error of the training dataset
error = Case1Model.eval_get_error(parallel=4)
# Get the predicting result of the raw dataset and save to file
Case1Model.eval_get_result(to_file="result.xlsx", parallel=4)
# Train a predictor based on optimized model configurations
train_vars = {
    "Temp": temp,  # temp is a list of temperature series
    "RH": rh,  # rh is a list of rh series
    "Cloud Cover": cc,  # cc is a list of cloud cover series
    "Weekdays": wd,  # wd is a list of weekdays series
    "Hour": hour,  # hour is a list of hour series
}
Case1Model.train(train_vars, train_loads, parallel=4)
# Predict the cooling load with the trained predictor
pred_vars = {
    "Temp": temp_pred,  # temp_pred is a list of temperature series for prediction
    "RH": rh_pred,  # rh_pred is a list of rh series for prediction
    "Cloud Cover": cc_pred,  # cc_pred is a list of cloud cover series for prediction
    "Weekdays": wd_pred,  # wd_pred is a list of weekdays series for prediction
    "Hour": hour_pred,  # hour_pred is a list of hour series for prediction
}
pred_loads = Case1Model.predict(pred_vars)
```

## Save models
The trained model can be saved with python's built-in `pickle` module. See the following example:
```python
import pickle

model_save_path = "Case1Model.mdl"
with open(model_save_path, "wb") as sfile:
    pickle.dump(Case1Model, sfile)

```

The saved model can be loaded similarly with `pickle` model. See example below:
```python
import pickle

model_path = "Case1Model.mdl"
with open(model_path, "rb") as mfile:
    Case1Model = pickle.load(mfile)
```