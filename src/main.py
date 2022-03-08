"""
Train 3D U-Net network, for prostate MRI scans.

Ideas taken from:
https://github.com/cs230-stanford/cs230-code-examples/tree/master/tensorflow/vision

and

https://github.com/tensorflow/models/blob/master/samples/core/
get_started/custom_estimator.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import pickle
import argparse

from model_fn import model_fn, loss_fn
from input_fn import input_fn
from utils import Params, set_logger

import torch
import numpy as np


def arg_parser(args):
    """
    Define cmd line help for main.
    """
    
    parser_desc = "Train, eval, predict 3D U-Net model."
    parser = argparse.ArgumentParser(description=parser_desc)
    
    parser.add_argument(
        '-model_dir', 
        default='../models/base_model',
        required=True,
        help="Experiment directory containing params.json"
    )

    parser.add_argument(
        '-mode', 
        default='train_eval',
        help="One of train, train_eval, eval, predict."
    )

    parser.add_argument(
        '-weights', 
        default=None,
        help="Base weights to load."
    )

    parser.add_argument(
        '--cuda', 
        action='store_true',
        default=False, 
        help='enables cuda'
    )

    parser.add_argument(
        '-pred_ix',
        nargs='+',
        type=int,
        default=[1],
        help="Space separated list of indices of patients to predict."
    )
    

    # parse input params from cmd line
    try:
        return parser.parse_args(args)
    except:
        parser.print_help()
        sys.exit(0)




def val(net, data_loader, logger):
    """
    Validate model.
    """
    print('Start Validation')

    for p in net.parameters():
        p.requires_grad = False

    net.eval()
    val_iter = iter(data_loader)

    if opt.eval_all:
        max_iter = len(data_loader)
    else:
        max_iter = min(max_iter, len(data_loader))

    for i in range(max_iter):
        data = val_iter.next()
        i += 1
        _, _ = data # TODO progress here

def loadData(v, data):
    with torch.no_grad():
        v.resize_(data.size()).copy_(data)

def main(argv):
    """
    Main driver/runner of 3D U-Net model.
    """
    
    # -------------------------------------------------------------------------
    # setup
    # -------------------------------------------------------------------------

    if not os.path.isdir("./checkpoints"):
        os.mkdir("./checkpoints")

    # load the parameters from model's json file as a dict
    args = arg_parser(argv)
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path).dict

    in_cuda = False
    if args.cuda:
        in_cuda = True
    
    # create logger, add loss and IOU to logging
    logger = set_logger()


    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #logger.info("Device: {}".format(device))

    # check mode
    modes = ['train_eval', 'eval', 'predict']
    assert args.mode in modes, "mode has to be one of %s" % ','.join(modes) 
    
    # -------------------------------------------------------------------------
    # data loader
    # -------------------------------------------------------------------------
    
    train_loader, test_loader = input_fn(params)

    train_data_loader = torch.utils.data.DataLoader(
        train_loader, shuffle=True, batch_size=params["batch_size"], num_workers=int(params["num_workers"]))
    test_data_loader = torch.utils.data.DataLoader(
       test_loader, shuffle=True, batch_size=params["batch_size"], num_workers=int(params["num_workers"]))

    # -------------------------------------------------------------------------
    # model and optimizer
    # -------------------------------------------------------------------------
    
    if in_cuda:
        if not torch.cuda.is_available():
            in_cuda = False

    unet, optimizer = model_fn(params, logger, in_cuda)

    if in_cuda:
        cross_entropy_loss = torch.nn.CrossEntropyLoss().cuda()
    else:
        cross_entropy_loss = torch.nn.CrossEntropyLoss()

    # -------------------------------------------------------------------------
    # variables
    # -------------------------------------------------------------------------

    input_data = torch.FloatTensor(params["batch_size"], 1, 128, 128, 32)
    output_data = torch.FloatTensor(params["batch_size"], 3, 128, 128, 32)

    # -------------------------------------------------------------------------
    # training
    # -------------------------------------------------------------------------
    
    if args.mode == 'train_eval':

        if args.weights is not None:
            logger.info("Loading weights from {}".format(args.weights))
            unet.load_state_dict(torch.load(args.weights))

        training_loss = []
        validation_loss = []

        count = 0
        while(count < params['max_train_steps']):
            for p in unet.parameters():
                p.requires_grad = True
            train_iter = iter(train_data_loader)

            unet.train()

            train_losses = []  # accumulate the losses here

            for cpu_input_data, cpu_output_data in train_iter:
                loadData(input_data, cpu_input_data)
                loadData(output_data, cpu_output_data)

                if in_cuda:
                    input_data = input_data.cuda()
                    output_data = output_data.cuda()

                # compute gradients and update parameters
                optimizer.zero_grad()
                
                preds = unet(input_data)
                loss = loss_fn(cross_entropy_loss, preds, output_data)

                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
                count += 1 

            logger.info("Training loss: {}".format(np.mean(train_losses)))
            training_loss.append(np.mean(train_losses))

            # Testing after every epoch

            unet.eval()  # evaluation mode
            valid_losses = []  # accumulate the losses here

            test_iter = iter(test_data_loader)
            for cpu_input_data, cpu_output_data in test_iter:
                loadData(input_data, cpu_input_data)
                loadData(output_data, cpu_output_data)

                if in_cuda:
                    input_data = input_data.cuda()
                    output_data = output_data.cuda()

                with torch.no_grad():
                    out = unet(input_data)
                    loss = loss_fn(cross_entropy_loss, out, output_data)
                    loss_value = loss.item()
                    valid_losses.append(loss_value)


            logger.info("Val loss: {}".format(np.mean(valid_losses)))
            
            validation_loss.append(np.mean(valid_losses))

            save_name = os.path.join('./checkpoints','model_{}.pth'.format(count))
            print("Reached Checkpoint. Saving Weights ...")
            torch.save(unet.state_dict(), save_name)
            print("Saved Weights : {}".format(save_name))
                
            
    if args.mode == 'eval':
        if args.weights is not None:
            logger.info("Loading weights from {}".format(args.weights))
            unet.load_state_dict(torch.load(args.weights))

            # Testing after every epoch

            unet.eval()  # evaluation mode
            valid_losses = []  # accumulate the losses here

            test_iter = iter(test_data_loader)
            for cpu_input_data, cpu_output_data in test_iter:
                loadData(input_data, cpu_input_data)
                loadData(output_data, cpu_output_data)

                if in_cuda:
                    input_data = input_data.cuda()
                    output_data = output_data.cuda()

                with torch.no_grad():
                    out = unet(input_data)
                    loss = loss_fn(cross_entropy_loss, out, output_data)
                    loss_value = loss.item()
                    valid_losses.append(loss_value)


            logger.info("Val loss: {}".format(np.mean(valid_losses)))
        else:
            logger.info("No weights specified. exiting the evalution.")


    if args.mode == 'predict':
        if args.weights is not None:
            logger.info("Loading weights from {}".format(args.weights))
            unet.load_state_dict(torch.load(args.weights))

            # Testing after every epoch

            softmax = torch.nn.Softmax(dim=1)

            unet.eval()  # evaluation mode

            # extract predictions, only save predicted classes not probs
            to_save = dict()
            test_iter = iter(test_data_loader)
            for i, (cpu_input_data, cpu_output_data) in enumerate(test_iter):
                loadData(input_data, cpu_input_data)
                loadData(output_data, cpu_output_data)

                if in_cuda:
                    input_data = input_data.cuda()
                    output_data = output_data.cuda()

                if i in args.pred_ix:
                    with torch.no_grad():
                        out = unet(input_data)
                        out = softmax(out)
                        out = out.cpu().numpy().squeeze()
                        truth = output_data.cpu().numpy().squeeze()
                        to_save[i] = {
                            'classes': out,
                            'truth': truth
                        }
            
            # save them with pickle to model dir
            pred_file = os.path.join(args.model_dir, 'preds.npy')
            pickle.dump(to_save, open(pred_file,"wb"))
            logger.info('Predictions saved to: %s.' % pred_file)
        else:
            logger.info("No weights specified. exiting the evalution.")

        

   

if __name__ == '__main__':
    main(sys.argv[1:])
