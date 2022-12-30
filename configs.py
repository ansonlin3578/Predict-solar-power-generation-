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
class TrainConfigs:
    model_type = ['XGB', 'LGBM']

class TestConfigs:
    model_type = ['XGB', 'LGBM']
    model_weights = ['./checkpoints/XGBRegressor.json', './checkpoints/LGBMRegressor.json']
    prediction_dir = './result'