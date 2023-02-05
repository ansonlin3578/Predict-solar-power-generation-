import matplotlib.pyplot as plt 
from models import Model
import os

def plot_pred_gt(ground_ture, pred, model_name):
    idx_len = len(ground_ture)
    X = [i for i in range(idx_len)]
    plt.plot(X, ground_ture,color='g', label="ground true", lw=0.4)
    plt.plot(X, pred, color='r', label="prediction", lw=0.4)
    plt.xlabel("index length")
    plt.ylabel("Generation")
    plt.title("Comparison of ground_truth and prediction")
    savedir = './Analysis/train_data'
    plt.savefig(os.path.join(savedir, "{}-plot_truth_pred".format(model_name)))
    plt.show()
    return