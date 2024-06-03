import numpy as np 
import torch
from torch.utils.data import DataLoader
from utils.dataset_loader import Dataset_ETT

# adjust learning rate
def adjust_lr(optim, epoch, args):
    lr = args.learning_rate * (0.5 ** (epoch - 1))
    for param_group in optim.param_groups:
        param_group['lr'] = lr
    print("Learning rate changes to {}".format(lr))

def create_dataset_loader(args, dataset_type, size):
    time_enc = 0 if args.embed != 'timeF' else 1
    datasets = Dataset_ETT(args.dataset_path, size, dataset_type, args.features, \
                                args.target, True, args.inverse, time_enc, args.freq)
    # create the dataloader
    data_loader = DataLoader(datasets, batch_size=args.batch_size, \
                shuffle=True if dataset_type in ["train", "val"] else False, \
                num_workers=args.num_workers, drop_last=True)
    return datasets, data_loader

# this function is used to generate the outputs for the network
def process_inputs(data, use_gpu, padding, pred_len, label_len, features):
    batch_x, batch_y, batch_x_mark, batch_y_mark = data
    # process inputs
    batch_x = batch_x.float().to("cuda" if use_gpu else "cpu")
    batch_y = batch_y.float().to("cuda" if use_gpu else "cpu")
    batch_x_mark = batch_x_mark.float().to("cuda" if use_gpu else "cpu")
    batch_y_mark = batch_y_mark.float().to("cuda" if use_gpu else "cpu")
    # start to create the inputs for the decoder
    if padding == 0:
        dec_inp = torch.zeros([batch_y.shape[0], pred_len, batch_y.shape[-1]], dtype=torch.float32, device="cuda" if use_gpu else "cpu")
    elif padding == 1:
        dec_inp = torch.ones([batch_y.shape[0], pred_len, batch_y.shape[-1]], dtype=torch.float32, device="cuda" if use_gpu else "cpu")
    dec_inp = torch.cat([batch_y[:,:label_len,:], dec_inp], dim=1)
    # split to get the ground truth
    f_dim = -1 if features == "MS" else 0
    batch_y = batch_y[:, -pred_len:, f_dim:]
    return batch_x, batch_x_mark, batch_y_mark, dec_inp, batch_y