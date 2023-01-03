# DL_2022_final_project

# Configrations
In ./configs.py you will see:
``` python 
class DataConfigs:
    dataset = {
            "train" : {
                        "data" : [],
                        "label" : []
                        },
            "test" : {
                        "data" : [],
                        "label" : []
                        }
            }
```
Here is in order to store dataset, you don't need to config it. </br>

In TrainConfigs you can see current model_type only have XGB now. </br>
So if you want to try more model please check model.py and config "model_type" here.
``` python 
class TrainConfigs:
    model_type = ['XGB']
```
In TestConfigs you can see current model_type only have XGB now. </br>
So if you want to try ensemble prediction, please config 
"model_type" and "model_weights" here. </br>

``` python 
class TestConfigs:
    model_type = ['XGB']
    model_weights = ['./checkpoints/XGBRegressor.json']
    prediction_dir = './result'
```
Notice ... the position of model_type and model_weights need to relative !! </br>
And the final result will store in ./result/submission.csv

# For training 
``` python 
python main.py

```
The model weights will save in ./checkpoints and don't need to worry about the save name. </br> 
It will auto depart by each model if you use multi-model to train it.

# For testing
``` python
python test.py

```
result will store in ./result/submission.csv </br>

# Current ensemble XGB and LGBM, But feature engineering not enough...
TODO:
hisGBM + lagdata
