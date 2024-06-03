import os
import torch
import numpy as np 
import random
from utils.arguments import get_args
from utils.utils import adjust_lr, create_dataset_loader, process_inputs
from models.informer import Informer
from models.itransformer import iTransformer
from datetime import datetime

"""
this script is used to train the transformer
"""

# evaluate the model
def evaluate(model, dataset, dataset_loader, args, loss_func, save_results=False, save_dir=None):
    losses, outputs, gts = [], [], []
    for _, data_ in enumerate(dataset_loader):
        batch_x, batch_x_mark, batch_y_mark, dec_inp, gt = process_inputs(data_, args.use_gpu, \
                            args.padding, args.pred_len, args.label_len, args.features)
        # start to go through the model
        with torch.no_grad():
            output = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        # start to calculate the loss - MSE loss
        loss = loss_func(output, gt)
        losses.append(loss.item())
        # append the outputs
        outputs.append(output.cpu().numpy())
        gts.append(gt.cpu().numpy())
    # save the outputs
    outputs, gts = np.array(outputs), np.array(gts)
    outputs = outputs.reshape(-1, outputs.shape[-2], outputs.shape[-1])
    gts = gts.reshape(-1, gts.shape[-2], gts.shape[-1])
    if save_results:
        # save the numpy files
        np.save("{}/pred.npy".format(save_dir), outputs)
        np.save("{}/real.npy".format(save_dir), gts)
    mse = np.mean((outputs - gts)**2)
    mae = np.mean(np.abs(outputs - gts))
    return np.mean(losses), mse, mae

# train the model
def train(args):
    # create the datasets
    if "ETT" in args.dataset_path:
        target = "OT"
        enc_in_dims = 7 if args.features in ["M", "MS"] else 1
        dec_in_dims = 7 if args.features in ["M", "MS"] else 1
        output_dims = 1 if args.features in ["S", "MS"] else 7
    # datset name
    dataset_name = args.dataset_path.split("/")[-1][:-4]
    # create datasets [train, val, test]
    train_data, train_loader = create_dataset_loader(args, dataset_type="train", size=[args.seq_len, args.label_len, args.pred_len])
    val_data, val_loader = create_dataset_loader(args, dataset_type="val", size=[args.seq_len, args.label_len, args.pred_len])
    test_data, test_loader = create_dataset_loader(args, dataset_type="test", size=[args.seq_len, args.label_len, args.pred_len])
    # build up the model
    if args.model == "informer":
        model = Informer(enc_in_dims, dec_in_dims, output_dims, args.seq_len, args.label_len, args.pred_len, args.factor, \
                        args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff, args.dropout, args.attn, args.embed, \
                        args.freq, args.activation, args.output_attention, args.distil, args.mix)
    elif args.model == "itransformer":
        model = iTransformer(args.seq_len, args.pred_len, args.e_layers, args.d_ff, args.d_model)
    else:
        raise NotImplementedError
    # use cuda or not
    if args.use_gpu: 
        model.cuda()
    # define the optimizer
    optim = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    # define the loss function
    criterion = torch.nn.MSELoss()
    # create the folder for saving
    save_dir = "{}/{}/{}/{}_{}_{}_{}/seed_{}".format(args.checkpoints, dataset_name, args.model, args.features, \
                                        args.seq_len, args.label_len if args.model == "informer" else args.seq_len, \
                                                                                args.pred_len, args.seed)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    # best loss
    best_loss = np.inf
    # start to do the training
    for epoch in range(args.train_epochs):
        for i, data_ in enumerate(train_loader):
            # put the inputs into the tensor
            batch_x, batch_x_mark, batch_y_mark, dec_inp, gt = process_inputs(data_, args.use_gpu, \
                                        args.padding, args.pred_len, args.label_len, args.features)
            # start to go through the model
            output = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            if args.inverse:
                output = train_data.inverse_transform(output)
            # start to calculate the loss - MSE loss
            loss = criterion(output, gt)
            # start to update
            optim.zero_grad()
            loss.backward()
            optim.step()
            if i % 100 == 0:
                print("[{}] Epoch: {}, Step: {}, Loss: {:.3f}".format(datetime.now(), epoch, i, loss.item()))
        # validate the datasets
        model.eval()
        valid_loss, _, _ = evaluate(model, val_data, val_loader, args, criterion)
        test_loss, mse, mae = evaluate(model, test_data, test_loader, args, criterion)
        print("[{}] Valid Loss: {:.3f}, Test Loss: {:.3f}, MSE: {:.3f}, MAE: {:.3f}".format(datetime.now(), valid_loss, test_loss, mse, mae))
        # only save the best model - depend on the validation loss
        if valid_loss < best_loss:
            torch.save(model.state_dict(), "{}/model.pt".format(save_dir))
            best_loss = valid_loss
        model.train()
        # adjust the learning rate
        adjust_lr(optim, epoch+1, args)
    # evaluate the model and output the best metrics
    model.load_state_dict(torch.load("{}/model.pt".format(save_dir)))
    model.eval()
    # calculate the best performance and save the results
    _, mse, mae = evaluate(model, test_data, test_loader, args, criterion, save_results=True, save_dir=save_dir)
    print("[{}] Best Metrics -- MSE: {:.3f}, MAE: {:.3f}".format(datetime.now(), mse, mae))

if __name__ == "__main__":
    args = get_args()
    # seed the environment
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    # start the training
    train(args)