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
    model_type = ['XGB', 'LGBM', 'histGBM']

class TestConfigs:
    model_type = ['XGB', 'LGBM', 'histGBM']
    model_weights = ['./checkpoints/XGBRegressor.json',
                     './checkpoints/LGBMRegressor.json',
                     './checkpoints/HistGradientBoostingRegressor.json']
    prediction_dir = './result'