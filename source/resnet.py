import os
import time
from os import path as osp

import numpy as np
import torch
import json
import quaternion

import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from data_glob_speed import *
from transformations import *
from metric import compute_ate_rte
from model_resnet1d import *
from numpy import linalg as LA

_input_channel, _output_channel = 6, 2
_fc_config = {'fc_dim': 512, 'in_dim': 7, 'dropout': 0.5, 'trans_planes': 128}


def get_model(arch):
    if arch == 'resnet18':
        network = ResNet1D(_input_channel, _output_channel, BasicBlock1D, [2, 2, 2, 2],
                           base_plane=64, output_block=FCOutputModule, kernel_size=3, **_fc_config)
    elif arch == 'resnet50':
        # For 1D network, the Bottleneck structure results in 2x more parameters, therefore we stick to BasicBlock.
        _fc_config['fc_dim'] = 1024
        network = ResNet1D(_input_channel, _output_channel, BasicBlock1D, [3, 4, 6, 3],
                           base_plane=64, output_block=FCOutputModule, kernel_size=3, **_fc_config)
    elif arch == 'resnet101':
        _fc_config['fc_dim'] = 1024
        network = ResNet1D(_input_channel, _output_channel, BasicBlock1D, [3, 4, 23, 3],
                           base_plane=64, output_block=FCOutputModule, **_fc_config)
    else:
        raise ValueError('Invalid architecture: ', args.arch)
    return network

def contrastiveModule(input_arrayy,device):
    # input_array=input_arrayy.cpu()
    input_array=input_arrayy.clone().detach()
    #TODO: add functionalities for contrastive module
    tensor1 = torch.tensor((), dtype=torch.float32,requires_grad=True,device=device)
    tensor1.new_zeros((len(input_array), len(input_array[0])))

    for i in range (len(input_array)):
        w,x,y,z=0.0,input_array[i][0],input_array[i][1],0.0
        interm=[w,x.tolist(),y.tolist(),z]
        intermm=[x.tolist(),y.tolist(),z]
        # print(interm)
        interm_q=quaternion.from_float_array(interm)
        # q1=quaternion.from_float_array([0.369969745324723, 0.629673216173061, 0.363760369952464, -0.578197723777012])
        q1 = Quaternion(axis=[0, 0, 1], angle=3.14159265 / (2*45))
        # print(q1)
        new_f=q1.rotate(intermm)[:-1]
        # new_f = quaternion.as_float_array(q1 * interm_q * q1.conj())[1:3]
        input_array[i]=torch.tensor(new_f,device=device)

        # zz=1
        # if (zz!=2):
        #     print([intermm,input_array[i]])
        #     zz=2

    # input_array.to("cuda:0")
    # print(input_array)

    return input_array

def run_test(network, data_loader, device, eval_mode=True):
    targets_all = []
    preds_all = []
    if eval_mode:
        network.eval()
    for bid, (feat, targ,feat_c,targ_c, _, _) in enumerate(data_loader):
        pred = network(feat.to(device)).cpu().detach().numpy()
        targets_all.append(targ.detach().numpy())
        preds_all.append(pred)
    targets_all = np.concatenate(targets_all, axis=0)
    preds_all = np.concatenate(preds_all, axis=0)
    return targets_all, preds_all


def add_summary(writer, loss, step, mode):
    names = '{0}_loss/loss_x,{0}_loss/loss_y,{0}_loss/loss_z,{0}_loss/loss_sin,{0}_loss/loss_cos'.format(
        mode).split(',')

    for i in range(loss.shape[0]):
        writer.add_scalar(names[i], loss[i], step)
    writer.add_scalar('{}_loss/avg'.format(mode), np.mean(loss), step)


def get_dataset(root_dir, data_list, args, **kwargs):
    mode = kwargs.get('mode', 'train')

    random_shift, shuffle, transforms, grv_only = 0, False, None, False
    if mode == 'train':
        random_shift = args.step_size // 2
        shuffle = True
        transforms = RandomHoriRotate(math.pi * 2)
    elif mode == 'val':
        shuffle = True
    elif mode == 'test':
        shuffle = False
        grv_only = True

    if args.dataset == 'ronin':
        seq_type = GlobSpeedSequence
    dataset = StridedSequenceDataset(
        seq_type, root_dir, data_list, args.cache_path, args.step_size, args.window_size,
        random_shift=random_shift, transform=transforms,
        shuffle=shuffle, grv_only=grv_only, max_ori_error=args.max_ori_error)

    global _input_channel, _output_channel
    _input_channel, _output_channel = dataset.feature_dim, dataset.target_dim
    return dataset


def get_dataset_from_list(root_dir, list_path, args, **kwargs):
    with open(list_path) as f:
        data_list = [s.strip().split(',' or ' ')[0] for s in f.readlines() if len(s) > 0 and s[0] != '#']
    return get_dataset(root_dir, data_list, args, **kwargs)

def write_config(args):
    if args.out_dir:
        with open(osp.join(args.out_dir, 'config.json'), 'w') as f:
            json.dump(vars(args), f)

def train(args, **kwargs):
    # Loading data
    start_t = time.time()
    train_dataset = get_dataset_from_list(args.root_dir, args.train_list, args, mode='train')  # big needed function
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)    #just data loading with iterable access and batch size and sampling technique we can provide

    end_t = time.time()
    print('Training set loaded. Feature size: {}, target size: {}. Time usage: {:.3f}s'.format(
        train_dataset.feature_dim, train_dataset.target_dim, end_t - start_t))
    val_dataset, val_loader = None, None
    if args.val_list is not None:
        val_dataset = get_dataset_from_list(args.root_dir, args.val_list, args, mode='val')
        val_loader = DataLoader(val_dataset, batch_size=512, shuffle=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() and not args.cpu else 'cpu')

    summary_writer = None
    if args.out_dir is not None:
        if not osp.isdir(args.out_dir):
            os.makedirs(args.out_dir)
        write_config(args)
        if not osp.isdir(osp.join(args.out_dir, 'checkpoints')):
            os.makedirs(osp.join(args.out_dir, 'checkpoints'))
        if not osp.isdir(osp.join(args.out_dir, 'logs')):
            os.makedirs(osp.join(args.out_dir, 'logs'))

    global _fc_config
    _fc_config['in_dim'] = args.window_size // 32 + 1  #default window size is 200

    network = get_model(args.arch).to(device)  #default network is resnet18, it will send the architecture to the cuda
    print('Number of train samples: {}'.format(len(train_dataset)))
    if val_dataset:
        print('Number of val samples: {}'.format(len(val_dataset)))
    total_params = network.get_num_params()
    print('Total number of parameters: ', total_params)

    criterion = torch.nn.MSELoss()
    criterion_2=torch.nn.CosineSimilarity(dim=1)
    criterion_3=torch.nn.MSELoss()
    optimizer = torch.optim.Adam(network.parameters(), args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10, verbose=True, eps=1e-12)

    start_epoch = 0
    if args.continue_from is not None and osp.exists(args.continue_from):  #default continue from is none
        checkpoints = torch.load(args.continue_from)
        start_epoch = checkpoints.get('epoch', 0)
        network.load_state_dict(checkpoints.get('model_state_dict'))
        optimizer.load_state_dict(checkpoints.get('optimizer_state_dict'))

    if args.out_dir is not None and osp.exists(osp.join(args.out_dir, 'logs')):
        summary_writer = SummaryWriter(osp.join(args.out_dir, 'logs'))
        summary_writer.add_text('info', 'total_param: {}'.format(total_params))

    step = 0
    best_val_loss = np.inf  #assigning infinity

    print('Start from epoch {}'.format(start_epoch))  #will print Start from epoch 1
    total_epoch = start_epoch
    train_losses_all, val_losses_all = [], []

    # Get the initial loss.
    init_train_targ, init_train_pred = run_test(network, train_loader, device, eval_mode=False)

    init_train_loss = np.mean((init_train_targ - init_train_pred) ** 2, axis=0)
    train_losses_all.append(np.mean(init_train_loss))
    print('-------------------------')
    print('Init: average loss: {}/{:.6f}'.format(init_train_loss, train_losses_all[-1]))
    if summary_writer is not None:
        add_summary(summary_writer, init_train_loss, 0, 'train')

    if val_loader is not None:
        init_val_targ, init_val_pred = run_test(network, val_loader, device)
        init_val_loss = np.mean((init_val_targ - init_val_pred) ** 2, axis=0) #np.mean([[1,2],[3,4]]) = [2,3]
        val_losses_all.append(np.mean(init_val_loss)) #np.mean([2,3]) = 5
        print('Validation loss: {}/{:.6f}'.format(init_val_loss, val_losses_all[-1]))
        if summary_writer is not None:
            add_summary(summary_writer, init_val_loss, 0, 'val')

    try:
        for epoch in range(start_epoch, args.epochs):
            start_t = time.time()
            network.train()  #according to my understanding its just change the mode of the network to train mode
            train_outs, train_targets,v_1_c_all,v_2_all = [], [], [],[]
            for batch_id, (feat, targ,feat_c,targ_c, _, _) in enumerate(train_loader):
                feat, targ,feat_c,targ_c = feat.to(device), targ.to(device),feat_c.to(device), targ_c.to(device)
                # feat_c=contrastiveModule(feat) #format[[glob_ang_vel_c1,glob_acc_c1],[],...]

                # v_1_c=contrastiveModule(targ)
                # v_1_c.to(device)
                #
                # # contrastive_loss=criterion_2(v_1_c,v_2)
                # # contrastive_loss=torch.mean(contrastive_loss)
                #
                optimizer.zero_grad()
                v_1 = network(feat)  #in book this is like y=mx+c
                v_2 = network(feat_c)
                v_1_c = contrastiveModule(v_1,device)
                # v_1_c.to(device)
                train_outs.append(v_1.cpu().detach().numpy())  #.cpu mean move all the parameters and buffer to the cpu, returning  self
                train_targets.append(targ.cpu().detach().numpy())
                # v_1_c_all.append(v_1_c.cpu().detach().numpy())
                # v_2_all.append(v_2.cpu().detach().numpy())
                loss = criterion(v_1, targ)  #MSE Loss = [1,2,3,4]
                # v_2=v_2.cpu()
                # v_1_c=v_1_c.to('cpu')
                # v_2.to(device)
                # print("v_2",v_2)
                # v_1_c.to(device)
                # print("v_1_c", v_1_c)

                # loss_2=criterion_2(v_2,v_1_c,torch.ones(len(v_2),device=device))
                # loss_3=criterion_3(v_2,v_1_c)

                # ol=(criterion_2(v_1_c, v_2))
                # print("shape v_1_c: ",v_1_c.shape,"  shape v_2: ",v_2.shape," shape ol: ",ol.shape)
                # print("length of contrastive loss",len(ol))
                # print(v_1_c[2],v_2[2],ol[2])
                # loss_2 = torch.mean(ol)
                loss_2=0
                for i in range (len(v_1)):
                    if (torch.norm(v_1[i])>0.5):
                        loss_2-=criterion_2(torch.unsqueeze(v_2[i],0),torch.unsqueeze(v_1_c[i],0))
                    else:
                        loss_2-=0


                # loss_2 = 1 - loss_2
                loss = torch.mean(loss) #loss=2.5
                # loss_2=torch.mean(loss_2)
                loss_2=loss_2/len(v_1)
                # loss_3=torch.mean(loss_3)
                total_loss=loss+loss_2
                total_loss.backward()
                optimizer.step()
                step += 1

                # print("--------v_2--------------")
                # print(v_2)
                # print("--------v_1_c--------------")
                # print(loss,loss_2,total_loss)
                # for j in range(len(v_1)):
                #     print([v_1_c[j],v_2[j]])
                #     zz=torch.nn.CosineSimilarity()
                    # print(zz(torch.Tensor(torch.Tensor.tolist([v_2[j]])),torch.Tensor(torch.Tensor.tolist([v_1_c[j]]))))

                # print("--------v_1--------------")
                # print(v_1)
                # print(loss_2)

            train_outs = np.concatenate(train_outs, axis=0) #axis 0 means a=[[1,2],[3,4]] b=[3,4], concatenate a,b in axis 0 mean [[1,2],[3,4],[3,4]]
            train_targets = np.concatenate(train_targets, axis=0)
            train_losses = np.average((train_outs - train_targets) ** 2, axis=0) #already Criterion(MSE loss) calculated why redo this?

            end_t = time.time()
            print('-------------------------')
            print('Epoch {}, time usage: {:.3f}s, average loss: {}/{:.6f}'.format(
                epoch, end_t - start_t, train_losses, np.average(train_losses)))
            train_losses_all.append(np.average(train_losses))

            if summary_writer is not None:
                add_summary(summary_writer, train_losses, epoch + 1, 'train')
                summary_writer.add_scalar('optimizer/lr', optimizer.param_groups[0]['lr'], epoch)

            if val_loader is not None:
                network.eval()  #now network change from train mode to evalutation mode
                val_outs, val_targets = run_test(network, val_loader, device)  #better to see the implementation, it pass the features to the network and get predicted outcomes and return that outcome with target outcome
                val_losses = np.average((val_outs - val_targets) ** 2, axis=0)
                avg_loss = np.average(val_losses)
                print('Validation loss: {}/{:.6f}'.format(val_losses, avg_loss))
                scheduler.step(avg_loss)
                if summary_writer is not None:
                    add_summary(summary_writer, val_losses, epoch + 1, 'val')
                val_losses_all.append(avg_loss)
                if avg_loss < best_val_loss:  #initial best_val_loss is infinity
                    best_val_loss = avg_loss
                    if args.out_dir and osp.isdir(args.out_dir):
                        model_path = osp.join(args.out_dir, 'checkpoints', 'checkpoint_%d.pt' % epoch)
                        torch.save({'model_state_dict': network.state_dict(),
                                    'epoch': epoch,
                                    'optimizer_state_dict': optimizer.state_dict()}, model_path)
                        print('Model saved to ', model_path)
            else:
                if args.out_dir is not None and osp.isdir(args.out_dir):
                    if (epoch%10==0):
                        model_path = osp.join(args.out_dir, 'checkpoints', 'checkpoint_%d.pt' % epoch)
                        torch.save({'model_state_dict': network.state_dict(),
                                    'epoch': epoch,
                                    'optimizer_state_dict': optimizer.state_dict()}, model_path)
                        print('Model saved to ', model_path)

            total_epoch = epoch

    except KeyboardInterrupt:
        print('-' * 60)
        print('Early terminate')

    print('Training complete')
    if args.out_dir:
        model_path = osp.join(args.out_dir, 'checkpoints', 'checkpoint_latest.pt')
        torch.save({'model_state_dict': network.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': total_epoch}, model_path)
        print('Checkpoint saved to ', model_path)

    return train_losses_all, val_losses_all



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_list', type=str)
    parser.add_argument('--val_list', type=str, default=None)
    parser.add_argument('--test_list', type=str, default=None)
    parser.add_argument('--test_path', type=str, default=None)
    parser.add_argument('--root_dir', type=str, default=None, help='Path to data directory')
    parser.add_argument('--cache_path', type=str, default=None, help='Path to cache folder to store processed data')
    parser.add_argument('--dataset', type=str, default='ronin', choices=['ronin'])
    parser.add_argument('--max_ori_error', type=float, default=20.0)
    parser.add_argument('--step_size', type=int, default=10)
    parser.add_argument('--window_size', type=int, default=200)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--lr', type=float, default=1e-04) #learning rate
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--arch', type=str, default='resnet18')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--run_ekf', action='store_true')
    parser.add_argument('--fast_test', action='store_true')
    parser.add_argument('--show_plot', action='store_true')

    parser.add_argument('--continue_from', type=str, default=None)
    parser.add_argument('--out_dir', type=str, default=None)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--feature_sigma', type=float, default=0.00001)
    parser.add_argument('--target_sigma', type=float, default=0.00001)

    args = parser.parse_args()

    np.set_printoptions(formatter={'all': lambda x: '{:.6f}'.format(x)})

    if args.mode == 'train':
        train(args)
    # elif args.mode == 'test':
    #     # test_sequence(args)
    else:
        raise ValueError('Undefined mode')
