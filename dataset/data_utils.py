import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class load_data(object):
    def load_train(path, dataset):
        df = pd.read_csv(path)
        dataset["train"]["data"] = df
        dataset["train"]["label"] = df["Generation"]
    def load_test(path, dataset):
        df = pd.read_csv(path)
        dataset["test"]["data"] = df
        dataset["test"]["label"] = df["Generation"]