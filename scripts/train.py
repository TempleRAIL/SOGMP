#!/usr/bin/env python
#
# file: $ISIP_EXP/SOGMP/scripts/train.py
#
# revision history: xzt
#  20220824 (TE): first version
#
# usage:
#  python train.py mdir trian_data val_data
#
# arguments:
#  mdir: the directory where the output model is stored
#  trian_data: the directory of training data
#  val_data: the directory of valiation data
#
# This script trains a SOGMP++ model
#------------------------------------------------------------------------------

# import pytorch modules
#
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

# visualize:
from tensorboardX import SummaryWriter
import numpy as np

# import the model and all of its variables/functions
#
from model import *
from local_occ_grid_map import LocalMap

# import modules
#
import sys
import os


#-----------------------------------------------------------------------------
#
# global variables are listed here
#
#-----------------------------------------------------------------------------

# general global values
#
model_dir = './model/model.pth'  # the path of model storage 
NUM_ARGS = 3
NUM_EPOCHS = 50 #100
BATCH_SIZE = 128 #512 #64
LEARNING_RATE = "lr"
BETAS = "betas"
EPS = "eps"
WEIGHT_DECAY = "weight_decay"

# Constants
NUM_INPUT_CHANNELS = 1
NUM_LATENT_DIM = 512 # 16*16*2 
NUM_OUTPUT_CHANNELS = 1
BETA = 0.01

# Init map parameters
P_prior = 0.5	# Prior occupancy probability
P_occ = 0.7	    # Probability that cell is occupied with total confidence
P_free = 0.3	# Probability that cell is free with total confidence 
MAP_X_LIMIT = [0, 6.4]      # Map limits on the x-axis
MAP_Y_LIMIT = [-3.2, 3.2]   # Map limits on the y-axis
RESOLUTION = 0.1        # Grid resolution in [m]'
TRESHOLD_P_OCC = 0.8    # Occupancy threshold

# for reproducibility, we seed the rng
#
set_seed(SEED1)       

# adjust_learning_rate
#　
def adjust_learning_rate(optimizer, epoch):
    lr = 1e-4
    if epoch > 30000:
        lr = 3e-4
    if epoch > 50000:
        lr = 2e-5
    if epoch > 48000:
       # lr = 5e-8
       lr = lr * (0.1 ** (epoch // 110000))
    #  if epoch > 8300:
    #      lr = 1e-9
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# train function:
def train(model, dataloader, dataset, device, optimizer, criterion, epoch, epochs):
    # set model to training mode:
    model.train()
    # for each batch in increments of batch size:
    running_loss = 0.0
    # kl_divergence:
    kl_avg_loss = 0.0
    # CE loss:
    ce_avg_loss = 0.0

    counter = 0
    # get the number of batches (ceiling of train_data/batch_size):
    num_batches = int(len(dataset)/dataloader.batch_size)
    for i, batch in tqdm(enumerate(dataloader), total=num_batches):
    #for i, batch in enumerate(dataloader, 0):
        counter += 1
        # collect the samples as a batch:
        scans = batch['scan']
        scans = scans.to(device)
        positions = batch['position']
        positions = positions.to(device)
        velocities = batch['velocity']
        velocities = velocities.to(device)

        # create occupancy maps:
        batch_size = scans.size(0)

        # Create mask grid maps:
        mask_gridMap = LocalMap(X_lim = MAP_X_LIMIT, 
                        Y_lim = MAP_Y_LIMIT, 
                        resolution = RESOLUTION, 
                        p = P_prior,
                        size=[batch_size, SEQ_LEN],
                        device = device)
        # robot positions:
        x_odom = torch.zeros(batch_size, SEQ_LEN).to(device)
        y_odom = torch.zeros(batch_size, SEQ_LEN).to(device)
        theta_odom = torch.zeros(batch_size, SEQ_LEN).to(device)
        # Lidar measurements:
        distances = scans[:,SEQ_LEN:]
        # the angles of lidar scan: -135 ~ 135 degree
        angles = torch.linspace(-(135*np.pi/180), 135*np.pi/180, distances.shape[-1]).to(device)
        # Lidar measurements in X-Y plane: transform to the predicted robot reference frame
        distances_x, distances_y = mask_gridMap.lidar_scan_xy(distances, angles, x_odom, y_odom, theta_odom)
        # discretize to binary maps:
        mask_binary_maps = mask_gridMap.discretize(distances_x, distances_y)
        
        # Create input grid maps: 
        input_gridMap = LocalMap(X_lim = MAP_X_LIMIT, 
                    Y_lim = MAP_Y_LIMIT, 
                    resolution = RESOLUTION, 
                    p = P_prior,
                    size=[batch_size, SEQ_LEN],
                    device = device)
        # current position and velocities: 
        obs_pos_N = positions[:, SEQ_LEN-1]
        vel_N = velocities[:, SEQ_LEN-1]
        # Predict the future origin pose of the robot:
        T = 1 
        noise_std = [0, 0, 0] #[0.00111, 0.00112, 0.02319]
        pos_origin = input_gridMap.origin_pose_prediction(vel_N, obs_pos_N, T, noise_std)
        # robot positions:
        pos = positions[:,:SEQ_LEN]
        # Transform the robot past poses to the predicted reference frame.
        x_odom, y_odom, theta_odom =  input_gridMap.robot_coordinate_transform(pos, pos_origin)
        # Lidar measurements:
        distances = scans[:,:SEQ_LEN]
        # the angles of lidar scan: -135 ~ 135 degree
        angles = torch.linspace(-(135*np.pi/180), 135*np.pi/180, distances.shape[-1]).to(device)
        # Lidar measurements in X-Y plane: transform to the predicted robot reference frame
        distances_x, distances_y = input_gridMap.lidar_scan_xy(distances, angles, x_odom, y_odom, theta_odom)
        # discretize to binary maps:
        input_binary_maps = input_gridMap.discretize(distances_x, distances_y)
        # occupancy map update:
        input_gridMap.update(x_odom, y_odom, distances_x, distances_y, P_free, P_occ)
        input_occ_grid_map = input_gridMap.to_prob_occ_map(TRESHOLD_P_OCC)

        # add channel dimension:
        input_binary_maps = input_binary_maps.unsqueeze(2)
        mask_binary_maps = mask_binary_maps.unsqueeze(2)

        # set all gradients to 0:
        optimizer.zero_grad()
        # feed the batch to the network:
        prediction, kl_loss = model(input_binary_maps, input_occ_grid_map)
        # calculate the total loss:
        ce_loss = criterion(prediction, mask_binary_maps[:,0]).div(batch_size)
        # beta-vae:
        loss = ce_loss + BETA*kl_loss
        # perform back propagation:
        loss.backward(torch.ones_like(loss))
        optimizer.step()
        # get the loss:
        # multiple GPUs:
        if torch.cuda.device_count() > 1:
            loss = loss.mean()  
            ce_loss = ce_loss.mean()
            kl_loss = kl_loss.mean()

        running_loss += loss.item()
        # kl_divergence:
        kl_avg_loss += kl_loss.item()
        # CE loss:
        ce_avg_loss += ce_loss.item()

        # display informational message:
        if(i % 128 == 0):
            print('Epoch [{}/{}], Step[{}/{}], Loss: {:.4f}, CE_Loss: {:.4f}, KL_Loss: {:.4f}'
                    .format(epoch, epochs, i + 1, num_batches, loss.item(), ce_loss.item(), kl_loss.item()))
    train_loss = running_loss / counter 
    train_kl_loss = kl_avg_loss / counter
    train_ce_loss = ce_avg_loss / counter

    return train_loss, train_kl_loss, train_ce_loss

# validate function:
def validate(model, dataloader, dataset, device, criterion):
    # set model to evaluation mode:
    model.eval()
    # for each batch in increments of batch size:
    running_loss = 0.0
    # kl_divergence:
    kl_avg_loss = 0.0
    # CE loss:
    ce_avg_loss = 0.0

    counter = 0
    # get the number of batches (ceiling of train_data/batch_size):
    num_batches = int(len(dataset)/dataloader.batch_size)
    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataloader), total=num_batches):
        #for i, batch in enumerate(dataloader, 0):
            counter += 1
            # collect the samples as a batch:
            scans = batch['scan']
            scans = scans.to(device)
            positions = batch['position']
            positions = positions.to(device)
            velocities = batch['velocity']
            velocities = velocities.to(device)

            # create occupancy maps:
            batch_size = scans.size(0)

            # Create mask grid maps:
            mask_gridMap = LocalMap(X_lim = MAP_X_LIMIT, 
                            Y_lim = MAP_Y_LIMIT, 
                            resolution = RESOLUTION, 
                            p = P_prior,
                            size=[batch_size, SEQ_LEN],
                            device = device)
            # robot positions:
            x_odom = torch.zeros(batch_size, SEQ_LEN).to(device)
            y_odom = torch.zeros(batch_size, SEQ_LEN).to(device)
            theta_odom = torch.zeros(batch_size, SEQ_LEN).to(device)
            # Lidar measurements:
            distances = scans[:,SEQ_LEN:]
            # the angles of lidar scan: -135 ~ 135 degree
            angles = torch.linspace(-(135*np.pi/180), 135*np.pi/180, distances.shape[-1]).to(device)
            # Lidar measurements in X-Y plane: transform to the predicted robot reference frame
            distances_x, distances_y = mask_gridMap.lidar_scan_xy(distances, angles, x_odom, y_odom, theta_odom)
            # discretize to binary maps:
            mask_binary_maps = mask_gridMap.discretize(distances_x, distances_y)
            
            # Create input grid maps: 
            input_gridMap = LocalMap(X_lim = MAP_X_LIMIT, 
                        Y_lim = MAP_Y_LIMIT, 
                        resolution = RESOLUTION, 
                        p = P_prior,
                        size=[batch_size, SEQ_LEN],
                        device = device)
            # current position and velocities: 
            obs_pos_N = positions[:, SEQ_LEN-1]
            vel_N = velocities[:, SEQ_LEN-1]
            # Predict the future origin pose of the robot: n+1 
            T = 1 
            noise_std = [0, 0, 0] #[0.00111, 0.00112, 0.02319]
            pos_origin = input_gridMap.origin_pose_prediction(vel_N, obs_pos_N, T, noise_std)
            # robot positions:
            pos = positions[:,:SEQ_LEN]
            # Transform the robot past poses to the predicted reference frame.
            x_odom, y_odom, theta_odom =  input_gridMap.robot_coordinate_transform(pos, pos_origin)
            # Lidar measurements:
            distances = scans[:,:SEQ_LEN]
            # the angles of lidar scan: -135 ~ 135 degree
            angles = torch.linspace(-(135*np.pi/180), 135*np.pi/180, distances.shape[-1]).to(device)
            # Lidar measurements in X-Y plane: transform to the predicted robot reference frame
            distances_x, distances_y = input_gridMap.lidar_scan_xy(distances, angles, x_odom, y_odom, theta_odom)
            # discretize to binary maps:
            input_binary_maps = input_gridMap.discretize(distances_x, distances_y)
            # occupancy map update:
            input_gridMap.update(x_odom, y_odom, distances_x, distances_y, P_free, P_occ)
            input_occ_grid_map = input_gridMap.to_prob_occ_map(TRESHOLD_P_OCC)

            # add channel dimension:
            input_binary_maps = input_binary_maps.unsqueeze(2)
            mask_binary_maps = mask_binary_maps.unsqueeze(2)

            # feed the batch to the network:
            prediction, kl_loss= model(input_binary_maps, input_occ_grid_map)
            # calculate the total loss:
            ce_loss = criterion(prediction, mask_binary_maps[:,0]).div(batch_size)
            # beta-vae:
            loss = ce_loss + BETA*kl_loss
            # multiple GPUs:
            if torch.cuda.device_count() > 1:
                loss = loss.mean()
                ce_loss = ce_loss.mean()
                kl_loss = kl_loss.mean()

            # get the loss:
            running_loss += loss.item()
            # kl_divergence:
            kl_avg_loss += kl_loss.item()
            # CE loss:
            ce_avg_loss += ce_loss.item()

    val_loss = running_loss / counter
    val_kl_loss = kl_avg_loss / counter
    val_ce_loss = ce_avg_loss / counter

    return val_loss, val_kl_loss, val_ce_loss

#------------------------------------------------------------------------------
#
# the main program starts here
#
#------------------------------------------------------------------------------

# function: main
#
# arguments: none
#
# return: none
#
# This method is the main function.
#
def main(argv):
    # ensure we have the correct amount of arguments:
    #global cur_batch_win
    if(len(argv) != NUM_ARGS):
        print("usage: python train.py [MDL_PATH] [TRAIN_PATH] [VAL_PATH]")
        exit(-1)

    # define local variables:
    mdl_path = argv[0]
    pTrain = argv[1]
    pDev = argv[2]

    # get the output directory name:
    odir = os.path.dirname(mdl_path)

    # if the odir doesn't exits, we make it:
    if not os.path.exists(odir):
        os.makedirs(odir)

    # set the device to use GPU if available:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('...Start reading data...')
    ### training data ###
    # training set and training data loader
    train_dataset = VaeTestDataset(pTrain, 'train')
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=4, \
                                                   shuffle=True, drop_last=True, pin_memory=True)

    ### validation data ###
    # validation set and validation data loader
    dev_dataset = VaeTestDataset(pDev, 'val')
    dev_dataloader = torch.utils.data.DataLoader(dev_dataset, batch_size=BATCH_SIZE, num_workers=2, \
                                                 shuffle=True, drop_last=True, pin_memory=True)

    # instantiate a model:
    model = RVAEP(input_channels=NUM_INPUT_CHANNELS,
                  latent_dim=NUM_LATENT_DIM,
                  output_channels=NUM_OUTPUT_CHANNELS)
    # moves the model to device (cpu in our case so no change):
    model.to(device)

    # set the adam optimizer parameters:
    opt_params = { LEARNING_RATE: 0.001,
                   BETAS: (.9,0.999),
                   EPS: 1e-08,
                   WEIGHT_DECAY: .001 }
    # set the loss criterion and optimizer:
    criterion = nn.BCELoss(reduction='sum') #, weight=class_weights)
    criterion.to(device)
    # create an optimizer, and pass the model params to it:
    optimizer = Adam(model.parameters(), **opt_params)

    # get the number of epochs to train on:
    epochs = NUM_EPOCHS

    # if there are trained models, continue training:
    if os.path.exists(mdl_path):
        checkpoint = torch.load(mdl_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        print('Load epoch {} success'.format(start_epoch))
    else:
        start_epoch = 0
        print('No trained models, restart training')

    # multiple GPUs:
    if torch.cuda.device_count() > 1:
        print("Let's use 2 of total", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model) #, device_ids=[0, 1])
    # moves the model to device (cpu in our case so no change):
    model.to(device)

    # tensorboard writer:
    writer = SummaryWriter('runs')

    epoch_num = 0
    for epoch in range(start_epoch+1, epochs):
        # adjust learning rate:
        adjust_learning_rate(optimizer, epoch)
        ################################## Train #####################################
        # for each batch in increments of batch size
        #
        train_epoch_loss, train_kl_epoch_loss, train_ce_epoch_loss = train(
            model, train_dataloader, train_dataset, device, optimizer, criterion, epoch, epochs
        )
        valid_epoch_loss, valid_kl_epoch_loss, valid_ce_epoch_loss = validate(
            model, dev_dataloader, dev_dataset, device, criterion
        )
        
        # log the epoch loss
        writer.add_scalar('training loss',
                        train_epoch_loss,
                        epoch)
        writer.add_scalar('training kl loss',
                        train_kl_epoch_loss,
                        epoch)
        writer.add_scalar('training ce loss',
                train_ce_epoch_loss,
                epoch)
        writer.add_scalar('validation loss',
                        valid_epoch_loss,
                        epoch)
        writer.add_scalar('validation kl loss',
                        valid_kl_epoch_loss,
                        epoch)
        writer.add_scalar('validation ce loss',
                        valid_ce_epoch_loss,
                        epoch)

        print('Train set: Average loss: {:.4f}'.format(train_epoch_loss))
        print('Validation set: Average loss: {:.4f}'.format(valid_epoch_loss))
        
        # save the model:
        if(epoch % 10 == 0):
            if torch.cuda.device_count() > 1: # multiple GPUS: 
                state = {'model':model.module.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}
            else:
                state = {'model':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}
            path='./model/model' + str(epoch) +'.pth'
            torch.save(state, path)

        epoch_num = epoch

    # save the final model
    if torch.cuda.device_count() > 1: # multiple GPUS: 
        state = {'model':model.module.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch_num}
    else:
        state = {'model':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch_num}
    torch.save(state, mdl_path)

    # exit gracefully
    #

    return True
#
# end of function


# begin gracefully
#
if __name__ == '__main__':
    main(sys.argv[1:])
#
# end of file
