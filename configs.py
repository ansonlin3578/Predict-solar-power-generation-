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
    model_type = ['XGB']

class TestConfigs:
    model_type = ['XGB']
    model_weights = ['./checkpoints/XGBRegressor.json']
    prediction_dir = './result'