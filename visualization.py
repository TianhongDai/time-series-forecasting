import numpy as np 
from matplotlib import pyplot as plt 
import argparse
import os
from tqdm import tqdm

"""
This script is used to plot predictions and ground truth
"""

parser = argparse.ArgumentParser()
parser.add_argument("--results_path", type=str, default="checkpoints/ETTm1/itransformer/M_96_96_192/seed_1")
parser.add_argument("--save_dir", type=str, default="saved_plots")
parser.add_argument("--save_interval", type=int, default=100, help="save plots interval")

if __name__ == "__main__":
    # get args
    args = parser.parse_args()
    # create folder to save plots
    save_plot_path = args.results_path.replace(args.results_path.split("/")[0], args.save_dir)
    if not os.path.exists(save_plot_path):
        os.makedirs(save_plot_path)
    # start to plots
    pred = np.load("{}/pred.npy".format(args.results_path))
    real = np.load("{}/real.npy".format(args.results_path)) 
    data_topic = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"]
    model = args.results_path.split("/")[2]
    for i in tqdm(range(pred.shape[0])):
        if i % 100 == 0:
            pred_, real_ = pred[i], real[i]
            time_ = np.arange(pred_.shape[0])
            plt.figure(figsize=(15, 8))
            for data_id in range(len(data_topic)):
                plt.subplot(3, 2, data_id + 1)
                plt.plot(time_, real_[:, data_id], label="Groundtruth")
                plt.plot(time_, pred_[:, data_id], label="Prediction")
                plt.title("{} ({} Index: {})".format(data_topic[data_id], model, i))
                plt.legend(loc="lower right")
                plt.tight_layout()
            plt.savefig("{}/{}_FEAT.pdf".format(save_plot_path, str(i).zfill(5)))
            plt.figure(figsize=(7, 3))
            plt.plot(time_, real_[:, -1], label="Groundtruth")
            plt.plot(time_, pred_[:, -1], label="Prediction")
            plt.title("Oil Temperature ({}, Index: {})".format(model, i))
            plt.legend(loc="lower right")
            plt.tight_layout()
            plt.savefig("{}/{}_OT.pdf".format(save_plot_path, str(i).zfill(5)))