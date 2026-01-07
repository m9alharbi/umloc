import json
import os
from os import path as osp
from shutil import copyfile
import argparse
import pdb
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import torch.nn.utils as nn_utils

from pathlib import Path
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from utils import *
from data_utils import *
from losses import *
from models import *

# Set seeds for reproducibility
# random.seed(42)
# np.random.seed(42)
# torch.manual_seed(42)
# torch.cuda.manual_seed(42)
# torch.cuda.manual_seed_all(42)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

def get_losses(args):
    loss_fns = {}
    if args.model_type == 'Q_LSTM':
        quantiles = [0.05, 0.5, 0.95]
        loss_fns['LSTM_quantile_loss'] = QTrajLoss(quantiles, args)
    elif args.model_type == 'GAN':
        loss_fns['GAN_generator_loss'] = gan_g_loss
        loss_fns['GAN_discrimentator_loss'] = gan_d_loss
        loss_fns['GAN_generator_l2_loss'] = TrajLoss(args)
        loss_fns['GAN_generator_feasibility_loss'] = feasibility_aware_loss
        loss_fns['GAN_generator_aux_attn_loss'] = aux_attention_loss
    else:
        quantiles = [0.05, 0.5, 0.95]
        loss_fns['LSTM_quantile_loss'] = QTrajLoss(quantiles, args)
        loss_fns['GAN_generator_loss'] = gan_g_loss
        loss_fns['GAN_discrimentator_loss'] = gan_d_loss
        loss_fns['GAN_generator_l2_loss'] = TrajLoss(args)
        loss_fns['GAN_generator_feasibility_loss'] = feasibility_aware_loss
        loss_fns['GAN_generator_aux_attn_loss'] = aux_attention_loss
    return loss_fns

def trainQmodel(args, train_loader, val_loader, log_file, Qtraining_epochs, loss_fns):
    train_mini_batches = len(train_loader)
    val_mini_batches = len(val_loader)

    # Initialize QLSTM model
    lstm_model = QLSTMModel(input_dim=args.input_dim, output_dim=args.output_dim, args=args).to(DEVICE)

    criterion = loss_fns['LSTM_quantile_loss']
    q_optimizer = torch.optim.Adam(lstm_model.parameters(), args.lr)
    q_scheduler = ReduceLROnPlateau(q_optimizer, 'min', patience=15, factor=0.75, verbose=True, eps=1e-12)

    start_epoch = 0
    best_val_loss = np.inf
    try:
        for epoch in range(start_epoch, Qtraining_epochs):
            log_line = ''
            lstm_model.train()
            train_loss = 0
            for batch in train_loader:
                feat, targ, _, _, _, _ = batch
                feat, targ = feat.to(DEVICE), targ.to(DEVICE)
                q_optimizer.zero_grad()
                predicted_1, predicted_2, _ = lstm_model(feat)
                loss = criterion(predicted_1, predicted_2, targ)
                train_loss += loss.cpu().detach().numpy()

                loss.backward()
                q_optimizer.step()

            train_loss = train_loss / train_mini_batches
            log_line = format_string(log_line, epoch, q_optimizer.param_groups[0]['lr'], train_loss)
            saved_model = False

            if val_loader:
                lstm_model.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch in val_loader:
                        feat, targ, _, _, _, _ = batch
                        feat, targ = feat.to(DEVICE), targ.to(DEVICE)
                        q_optimizer.zero_grad()
                        pred_1, pred_2, _ = lstm_model(feat)
                        v_loss = criterion(pred_1, pred_2, targ)
                        val_loss += v_loss.cpu().detach().numpy()

                val_loss = val_loss / val_mini_batches
                log_line = format_string(log_line, val_loss)

                print(f'Epoch [{epoch}/{Qtraining_epochs}] | Train Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f}')
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    saved_model = True
                    if args.output_directory:
                        model_path = osp.join(args.output_directory, 'checkpoints', 'lstm_checkpoint_%d.pt' % epoch)
                        torch.save({'model_state_dict': lstm_model.state_dict(),
                                    'epoch': epoch,
                                    'loss': train_loss,
                                    'optimizer_state_dict': q_optimizer.state_dict()}, model_path)
                        print('Best Validation Model saved to ', model_path)
                if args.use_scheduler:
                    q_scheduler.step(val_loss)

            if log_file:
                log_line += '\n'
                with open(log_file, 'a') as f:
                    f.write(log_line)
            if np.isnan(train_loss):
                print("Invalid value. Stopping training.")
                break


    except KeyboardInterrupt:
        print('-' * 60)
        print('Early terminate')
    return model_path

def discriminator_step(batch, generator, discriminator, loss_fns, optimizer_d, Q_model, dts, target_type, t):
    losses = {}
    feat, targ, initial_pos, traj_real, map_tensor, map_meta = batch
    feat, targ, initial_pos, traj_real, map_tensor = feat.to(DEVICE), targ.to(DEVICE), initial_pos.to(DEVICE), traj_real.to(DEVICE), map_tensor.to(DEVICE)
    optimizer_d.zero_grad()
    pred_traj_fake_rel, _ = generator(map_tensor, initial_pos, Q_model, feat, map_meta, t)
    pred_traj_fake = relative_to_abs(pred_traj_fake_rel, initial_pos, dts, target_type)

    scores_fake = discriminator(pred_traj_fake_rel.detach())
    scores_real = discriminator(targ)

    data_loss = loss_fns['GAN_discrimentator_loss'](scores_real, scores_fake)
    data_loss.backward()
    # if args.clipping_threshold_g > 0:
    #     nn.utils.clip_grad_norm_(
    #         generator.parameters(), args.clipping_threshold_g
    #     )
        
    optimizer_d.step()
    losses['d_loss'] = data_loss.item()
    return losses

def generator_step(batch, generator, discriminator, loss_fns, optimizer_g, Q_model, dts, target_type, t):

    losses = {}
    # traj_w, aux_w, feas_w = 0.001, 100.0, 0.5
    loss = torch.zeros(1, device=DEVICE)
    feat, targ, initial_pos, _, map_tensor, map_meta = batch
    feat, targ, initial_pos, map_tensor = feat.to(DEVICE), targ.to(DEVICE), initial_pos.to(DEVICE), map_tensor.to(DEVICE)

    optimizer_g.zero_grad()
    min_l2_loss = torch.inf
    #for k in range(1):
    pred_traj_fake_rel, attn_mask = generator(map_tensor, initial_pos, Q_model, feat, map_meta, t)
    pred_traj_fake = relative_to_abs(pred_traj_fake_rel, initial_pos, dts, target_type)
        #obstacle_loss = loss_fns['GAN_generator_feasibility_loss'](pred_traj_fake, map_tensor, map_meta, obstacle_dynamic_w)

    g_l2_loss = loss_fns['GAN_generator_l2_loss'](pred_traj_fake_rel, targ)#, obstacle_loss)
    # loss_aux = loss_fns['GAN_generator_aux_attn_loss'](attn_w, attn_mask)
    # loss_feas = loss_fns['GAN_generator_feasibility_loss'](pred_traj_fake, map_tensor, map_meta)
        # if g_l2_loss < min_l2_loss:
        #     min_l2_loss = g_l2_loss
        #     best_trajetory_rel = pred_traj_fake_rel
    loss += g_l2_loss
    losses['G_l2_loss_rel'] = g_l2_loss.item()
    scores_fake = discriminator(pred_traj_fake_rel)
    discriminator_loss = loss_fns['GAN_generator_loss'](scores_fake)
    loss += discriminator_loss
    losses['G_discriminator_loss'] = discriminator_loss.item()

    # loss = traj_w*g_l2_loss + feas_w*loss_feas + discriminator_loss #+ aux_w*loss_aux
    

    #pred_traj_fake = relative_to_abs(best_trajetory_rel, initial_pos, dts, target_type)
    #obstacle_loss = loss_fns['GAN_generator_feasibility_loss'](pred_traj_fake, map_mask, bounds, obstacle_dynamic_w)
    #losses['G_obstacle_loss'] = obstacle_loss.item()
    #loss += obstacle_loss

    # losses['G_l2_loss_rel'] = traj_w*g_l2_loss.item()

    # loss += loss_aux
    # losses['G_aux_loss'] = aux_w*loss_aux.item()
    # loss += g_l2_loss
    # losses['G_discriminator_loss'] = discriminator_loss.item()
    # losses['G_feasibility_loss'] = feas_w*loss_feas.item()
    # loss += discriminator_loss

    losses['G_total_loss'] = loss.item()

    loss.backward()
    # nn.utils.clip_grad_norm_(generator.parameters(), 5.0)
    optimizer_g.step()

    return losses

def check_accuracy(loader, generator, discriminator, metric_fn, Q_model, dts, target_type, t):
    metrics = {}
    d_losses = []
    g_l2_losses_rel = []
    disp_error = []
    f_disp_error = []
    ds_score = []
    cr_score = []
    total_samples = 0
    generator.eval()
    with torch.no_grad():
        for batch in loader:
            feat, targ, initial_pos, traj_real, map_tensor, map_meta = batch
            feat, targ, initial_pos, traj_real, map_tensor = feat.to(DEVICE), targ.to(DEVICE), initial_pos.to(DEVICE), traj_real.to(DEVICE), map_tensor.to(DEVICE)
    
            pred_traj_fake_rel, _ = generator(map_tensor, initial_pos, Q_model, feat, map_meta, t)
            pred_traj_fake = relative_to_abs(pred_traj_fake_rel, initial_pos, dts, target_type)

            g_l2_loss_rel = metric_fn['ade'](pred_traj_fake_rel, targ)

            ade = metric_fn['ade'](pred_traj_fake, traj_real)
            fde = metric_fn['fde'](pred_traj_fake, traj_real)
            ds = metric_fn['ds'](pred_traj_fake, map_tensor, map_meta)
            cr = metric_fn['cr'](pred_traj_fake, map_tensor, map_meta)
            

            scores_fake = discriminator(pred_traj_fake_rel)
            scores_real = discriminator(targ)
            d_loss = metric_fn['bce'](scores_real, scores_fake)

            total_samples += pred_traj_fake.shape[0]
            d_losses.append(d_loss.item())
            g_l2_losses_rel.append(g_l2_loss_rel.sum().item())
            disp_error.append(ade.sum().item())
            f_disp_error.append(fde.sum().item())
            ds_score.append(ds.sum().item())
            cr_score.append(cr.sum().item())

    metrics['d_loss'] = sum(d_losses) / len(d_losses)
    metrics['g_l2_loss_rel'] = sum(g_l2_losses_rel) / total_samples
    metrics['ade'] = sum(disp_error) / total_samples
    metrics['fde'] = sum(f_disp_error) / total_samples
    metrics['ds'] = sum(ds_score) / total_samples
    metrics['cr'] = sum(cr_score) / total_samples

    generator.train()
    return metrics
    
def trainGAN(args, train_loader, val_loader, Q_model, GANtraining_epochs, loss_fns, dts):
    d_steps = 2
    g_steps = 1
    g_model = Generator(args.input_channel, dts, args)
    g_model = g_model.to(DEVICE)
    d_model = TrajectoryDiscriminator(args)
    d_model = d_model.to(DEVICE)
    g_model.train()
    d_model.train()
    # Define loss function and optimizer
    # gan_g_loss, gan_d_loss  = loss_fns['GAN_generator_loss'], loss_fns['GAN_discrimentator_loss']
    # gan_g_feas_loss = loss_fns['GAN_generator_feasibility_loss']
    # gan_g_aux_loss = loss_fns['GAN_generator_aux_attn_loss']
    g_optimizer = torch.optim.Adam(g_model.parameters(), lr=0.001)
    d_optimizer = torch.optim.Adam(d_model.parameters(), lr=0.001)

    # Training loop
    iterations_per_epoch = len(train_loader) / d_steps
    num_iterations = int(iterations_per_epoch * GANtraining_epochs)
    print('There are {} iterations per epoch'.format(iterations_per_epoch))
    t, epoch = 0, 0
    print_every = 10
    # traj_w = 100.0
    # obstacle_dynamic_w = 10.0
    best_min_ade = np.inf
    reset_ade = True
    metric_fn = {'ade':displacement_error,
                 'fde': final_displacement_error,
                 'bce': gan_d_loss,
                 'ds': get_distance_score,
                 'cr': get_collision_rate}
    history = {
    'iter':        [],
    'd_loss':      [],
    'g_loss':      [],
    'g_traj_loss': [],
    'ade_val':     [],
    'fde_val':     [],
    'ds_val':      [],
    'cr_val':      [],
    'ade_train':     [],
    'fde_train':     [],
    'ds_train':      [],
    'cr_train':      [],
    }
    while t < num_iterations:
        d_steps_left = d_steps
        g_steps_left = g_steps
        #if t > 500:
         #   obstacle_dynamic_w = 1.0
          #  traj_w = 100
            # if reset_ade is True:
            #     best_min_ade = np.inf
            #     reset_ade = False
        epoch += 1
        for batch in train_loader:
            if d_steps_left > 0:
                losses_d = discriminator_step(batch, g_model,
                                              d_model, loss_fns,
                                              d_optimizer, Q_model, dts, args.target_type, t)
                history['iter'].append(t)
                history['d_loss'].append(losses_d['d_loss'])

                d_steps_left -= 1
            elif g_steps_left > 0:
                losses_g = generator_step(batch, g_model,
                                          d_model, loss_fns,
                                          g_optimizer, Q_model, dts, args.target_type, t)
                history['g_loss'].append(losses_g['G_total_loss'])
                history['g_traj_loss'].append(losses_g['G_l2_loss_rel'])
                g_steps_left -= 1

            if d_steps_left > 0 or g_steps_left > 0:
                continue

            if t > 0 and t % print_every == 0:
                np.savez_compressed(os.path.join(args.output_directory, 'train_history.npz'), **history)

                print('t = {} / {}'.format(t + 1, num_iterations))
                for k, v in sorted(losses_d.items()):
                    print('  [D] {}: {:.3f}'.format(k, v))
                for k, v in sorted(losses_g.items()):
                    print('  [G] {}: {:.3f}'.format(k, v))

                print('Checking stats on validation ...')
                metrics_val = check_accuracy(val_loader, g_model, d_model, metric_fn, Q_model, dts, args.target_type, t)
                metrics_train = check_accuracy(train_loader, g_model, d_model, metric_fn, Q_model, dts, args.target_type, t)
                history['ade_val'].append(metrics_val['ade'])
                history['fde_val'].append(metrics_val['fde'])
                history['ds_val'] .append(metrics_val['ds'])
                history['cr_val'] .append(metrics_val['cr'])

                history['ade_train'].append(metrics_train['ade'])
                history['fde_train'].append(metrics_train['fde'])
                history['ds_train'] .append(metrics_train['ds'])
                history['cr_train'] .append(metrics_train['cr'])
                for k, v in sorted(metrics_val.items()):
                    print('  [val] {}: {:.3f}'.format(k, v))

                for k, v in sorted(metrics_train.items()):
                    print('  [val] {}: {:.3f}'.format(k, v))

                if metrics_val['cr'] + metrics_val['ade'] < best_min_ade:
                    best_min_ade = metrics_val['cr'] + metrics_val['ade']
                    if args.output_directory:
                        GAN_model_path = osp.join(args.output_directory, 'checkpoints', 'gan_checkpoint_%d.pt' % epoch)
                        torch.save({'g_best_state': g_model.state_dict(),
                                    'd_best_state': d_model.state_dict(),
                                    'best_t': t,
                                    'g_optim_state': g_optimizer.state_dict(),
                                    'd_optim_state': d_optimizer.state_dict()}, GAN_model_path)
                        print('Best Validation Model saved to ', GAN_model_path)

            t += 1
            d_steps_left = d_steps
            g_steps_left = g_steps
            if t >= num_iterations:
                break
    

  

def train(args):
    train_lstm_flag, train_gan_flag = True, True
    if args.model_type == 'GAN':
        train_lstm_flag = False
        train_gan_flag = True

    if args.model_type == 'Q_LSTM':
        train_lstm_flag = True
        train_gan_flag = False

    if args.output_directory:
        if not osp.isdir(args.output_directory):
            os.makedirs(args.output_directory)
        if not osp.isdir(osp.join(args.output_directory, 'checkpoints')):
            os.makedirs(osp.join(args.output_directory, 'checkpoints'))
        if not osp.isdir(osp.join(args.output_directory, 'logs')):
            os.makedirs(osp.join(args.output_directory, 'logs'))
        copyfile(args.train_list, osp.join(args.output_directory, "train_list"))
        if args.val_list is not None:
            copyfile(args.val_list, osp.join(args.output_directory, "validation_list"))
        write_config(args)

    log_file = None
    if args.output_directory:
        log_file = osp.join(args.output_directory, 'logs', 'log.txt')
        if osp.exists(log_file):
            if args.continue_from is None:
                os.remove(log_file)
            else:
                copyfile(log_file, osp.join(args.output_directory, 'logs', 'log_old.txt'))
    train_dataset = get_data_list(args.dataset_directory, args.train_list, args)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4, shuffle=True, drop_last=True, pin_memory=True)
    if args.dataset == 'our':
        dts = (train_dataset.ts[0][1:] - train_dataset.ts[0][:-1])[:, None].mean()/1000
    else:
        dts = (train_dataset.ts[0][1:] - train_dataset.ts[0][:-1])[:, None].mean()
    
    #if args.use_map:
     #   train_dist_map, boundries = get_map_data(train_dataset.gt_pos, args.sigma, args.map_size)

    val_dataset, val_loader = None, None
    if args.val_list is not None:
        val_dataset =get_data_list(args.dataset_directory, args.train_list, args)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=4, shuffle=True, drop_last=True, pin_memory=True)
        #if args.use_map:
         #   val_dist_map, boundries = get_map_data(val_dataset.gt_pos, args.sigma, args.map_size)
        print('Validation set loaded')

    print('\nNumber of train samples: {}'.format(len(train_dataset)))
    print('Number of val samples: {}'.format(len(val_dataset)))
    loss_fns = get_losses(args)

    Qtraining_epochs = int(3*args.epochs/4)
    if train_lstm_flag:
        model_path = trainQmodel(args, train_loader, val_loader, log_file, Qtraining_epochs, loss_fns)
        with open(osp.join(str(Path(model_path).parents[1]), 'config.json'), 'r') as f:
            model_data = json.load(f)

        if args.device == 'cpu':
            checkpoint = torch.load(model_path, map_location=lambda storage, location: storage)
        else:
            checkpoint = torch.load(model_path, map_location={model_data['device']: args.device})
    else:
        with open(osp.join(str(Path(args.lstm_path).parents[1]), 'config.json'), 'r') as f:
            model_data = json.load(f)
        if args.device == 'cpu':
            checkpoint = torch.load(args.lstm_path, map_location=lambda storage, location: storage)
        else:
            checkpoint = torch.load(args.lstm_path, map_location={model_data['device']: args.device})

    if train_gan_flag:
        lstm_model = QLSTMModel(input_dim=args.input_dim, output_dim=args.output_dim, args=args).to(DEVICE)
        lstm_model.load_state_dict(checkpoint.get('model_state_dict'))
        lstm_model.eval()
        print('Model {} loaded to device {}.'.format(args.lstm_path, args.device))
        for param in lstm_model.parameters():
            param.requires_grad = False  # Freeze Module1 parameters

        GANtraining_epochs = args.epochs - Qtraining_epochs
        #train_dist_map = train_dist_map.unsqueeze(0).unsqueeze(1).repeat(args.batch_size, 1, 1, 1)
        #val_dist_map = val_dist_map.unsqueeze(0).unsqueeze(1).repeat(args.batch_size, 1, 1, 1)
        trainGAN(args, train_loader, val_loader, lstm_model, GANtraining_epochs, loss_fns, dts)


def test_Q_GAN(args):
    if args.test_list is not None:
        data_dir = args.dataset_directory if args.dataset_directory else osp.split(args.test_list)[0]
        with open(args.test_list) as f:
            test_data_list = [s.strip().split(',')[0] for s in f.readlines() if len(s) > 0 and s[0] != '#']
    else:
        raise ValueError('Either test_list must be specified.')

        
    if args.output_directory and not osp.exists(args.output_directory):
        os.makedirs(args.output_directory)

    with open(osp.join(str(Path(args.lstm_path).parents[1]), 'config.json'), 'r') as f:
        model_data = json.load(f)


    lstm_checkpoint = torch.load(args.lstm_path)
     # Initialize QLSTM model
    lstm_model = QLSTMModel(input_dim=args.input_dim, output_dim=args.output_dim, args=args).to(args.device)


    lstm_model.load_state_dict(lstm_checkpoint.get('model_state_dict'))
    lstm_model.eval().to(args.device)

    print('Model {} loaded to device {}.'.format(args.lstm_path, args.device))
    seq_dataset = get_data_list(data_dir, args.test_list, args)
    if args.dataset == 'our':
        dts = (seq_dataset.ts[0][1:] - seq_dataset.ts[0][:-1])[:, None].mean()/1000
    else:
        dts = (seq_dataset.ts[0][1:] - seq_dataset.ts[0][:-1])[:, None].mean()


    gan_checkpoint = torch.load(args.gan_path)
    g_model = Generator(args.input_channel, dts, args)
    g_model.load_state_dict(gan_checkpoint.get('g_best_state'))
    g_model.eval().to(args.device)

    log_file = None
    if args.test_list and args.output_directory:
        log_file = osp.join(args.output_directory, osp.split(args.test_list)[-1].split('.')[0] + '_log.txt')
        with open(log_file, 'w') as f:
            f.write(args.lstm_path + '\n')
            f.write('Seq traj_len velocity ate rte\n')
    if args.dataset == 'our':
        fs = 60
    else:
        fs = 200

    pred_per_min = fs * 60

    metric_fn = {'ade':displacement_error,
               'fde': final_displacement_error,
                'cr': get_collision_rate,
                'ds': get_distance_score}


    #if args.use_map:
     #   test_dist_map, boundries =  get_map_data(seq_dataset.gt_pos, args.sigma, args.map_size)
      #  test_dist_map = test_dist_map.unsqueeze(0).unsqueeze(1).repeat(args.batch_size, 1, 1, 1)

    fde_outer, ade_outer = [], []
    real, predicted = [], []
    ate_all, rte_all = [], []
    cr_all, ds_all = [], []
    with torch.no_grad():
        for idx, data in enumerate(test_data_list):
            assert data == osp.split(seq_dataset.data_path_list[idx])[1]
            gt = seq_dataset.gt_pos[idx]
            feat, _, map_tensor, map_meta = seq_dataset.get_test_seq(idx, gt)
            feat = torch.Tensor(feat).to(args.device)
            map_tensor = torch.Tensor(map_tensor).to(args.device)
            if args.perturb:
                feat = perturb_imu(feat)
            initial_pos = gt[[0], :]

            initial_pos = torch.Tensor(initial_pos).to(args.device)
            gt_traj_real = torch.Tensor(gt).unsqueeze(0).to(args.device)


            min_ade = torch.inf
            for k in range(1):
                #breakpoint()
                pred_traj_fake_rel, _ = g_model(map_tensor, initial_pos, lstm_model, feat, map_meta, 1)
                pred_traj_fake = relative_to_abs(pred_traj_fake_rel, initial_pos, dts, args.target_type)
                ade = metric_fn['ade'](pred_traj_fake, gt_traj_real)
                fde = metric_fn['fde'](pred_traj_fake, gt_traj_real)
                if ade < min_ade:
                    min_ade = ade
                    best_trajetory = pred_traj_fake
                    best_fde = fde

            ds = metric_fn['ds'](pred_traj_fake, map_tensor, map_meta)
            cr = metric_fn['cr'](pred_traj_fake, map_tensor, map_meta)
            X_quantiles, Y_quantiles, _ = lstm_model(feat)

            X_quantiles = relative_to_abs(X_quantiles, initial_pos[:, [0]], dts, args.target_type)
            Y_quantiles = relative_to_abs(Y_quantiles, initial_pos[:, [1]], dts, args.target_type)
            
            best_fde = best_fde.item()
            min_ade = min_ade.item()
            ds = ds.item()
            cr = cr.item()

            gt_traj_real = gt_traj_real.squeeze(0).cpu().detach().numpy()
            pred_traj = best_trajetory.squeeze(0).cpu().detach().numpy()
            X_quantiles = X_quantiles.squeeze(0).cpu().detach().numpy()
            Y_quantiles = Y_quantiles.squeeze(0).cpu().detach().numpy()

                

            ade_outer.append(min_ade)
            fde_outer.append(best_fde)
            ds_all.append(ds)
            cr_all.append(cr)

            ate, rte = compute_ate_rte(pred_traj, gt_traj_real, pred_per_min)
            pos_losses = np.mean((gt_traj_real - pred_traj) ** 2, axis=0)
            pos_cum_error = np.linalg.norm(pred_traj - gt_traj_real, axis=1)
            if args.output_directory is not None and osp.isdir(args.output_directory):
                np.save(osp.join(args.output_directory, data + '_umgloc.npy'), pred_traj)
                np.save(osp.join(args.output_directory, data + '_gt.npy'), gt_traj_real)
                np.save(osp.join(args.output_directory, data + '_quantiles.npy'),
                        np.stack([X_quantiles, Y_quantiles], axis=-1))
                # np.save(osp.join(args.output_directory, data + '_umgloc_error.npy'), pos_cum_error)
                            
            error = np.sort(pos_cum_error)
            cumulative_prob = np.arange(1, len(error) + 1) / len(error)
            log_line = format_string(data, np.mean(pos_losses), ate, rte, min_ade, best_fde, ds, cr)
            if log_file is not None:
                with open(log_file, 'a') as f:
                    log_line += '\n'
                    f.write(log_line)

            ate_all.append(ate)
            rte_all.append(rte)
            print('Sequence {}, ATE: {}, RTE: {}, ADE: {}, FDE: {}, DS: {}, CR: {}'.format(data, ate, rte, min_ade, best_fde, ds, cr))

            # Assuming the first column is 'x' and the second column is 'y' for both preds and gt
            time = np.linspace(0, len(pred_traj)//fs, len(pred_traj))
            plot_result(pred_traj, gt_traj_real, X_quantiles[:, 0], Y_quantiles[:, 0], X_quantiles[:, -1], Y_quantiles[:, -1], error, cumulative_prob, pos_cum_error, data, args)
    ate_all = np.array(ate_all)
    rte_all = np.array(rte_all)
    ade_outer = np.array(ade_outer)
    fde_outer = np.array(fde_outer)
    ds_all = np.array(ds_all)
    cr_all = np.array(cr_all)

    measure = format_string('ATE', 'RTE', 'ADE', 'FDE', 'DS', 'CR', sep='\t')
    values = format_string(np.mean(ate_all), np.mean(rte_all), np.mean(ade_outer), np.mean(fde_outer), np.mean(ds_all), np.mean(cr_all), sep='\t')
    print(measure, '\n', values)

    if log_file is not None:
        with open(log_file, 'a') as f:
            f.write(measure + '\n')
            f.write(values)


# ArgumentParser for both the quantile model and GAN
def get_args():
    parser = argparse.ArgumentParser(description="Arguments for Quantile Model and Generator")

    # General arguments
    parser.add_argument('--input_dim', type=int, default=6, help='Input dimension for IMU data')
    parser.add_argument('--output_dim', type=int, default=2, help='Output dimension for predicted positions')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run the model (cuda or cpu)')
    parser.add_argument('--dataset_directory', default='data/datasets/seq_data', type=str)
    parser.add_argument('--output_directory', default='results/toy_exp_3/', type=str)
    
    # Model argument
    parser.add_argument('--context_feature_dim', type=int, default=16, help='Hidden dimension for LSTM/ConvLSTM')
    parser.add_argument('--imu_hidden_dim', type=int, default=64, help='Hidden dimension for LSTM/ConvLSTM')
    parser.add_argument('--mlp_dim', type=int, default=128, help='Hidden dimension for LSTM/ConvLSTM')
    parser.add_argument('--decoder_hidden_dim', type=int, default=64, help='Hidden dimension for LSTM/ConvLSTM')
    parser.add_argument('--bottleneck_dim', type=int, default=64, help='Hidden dimension for LSTM/ConvLSTM')
    parser.add_argument('--embedding_dim', type=int, default=32, help='Hidden dimension for LSTM/ConvLSTM')
    parser.add_argument('--convlstm_layers', type=int, default=1, help='Number of layers in LSTM/ConvLSTM')
    parser.add_argument('--lstm_layers', type=int, default=2, help='Number of layers in LSTM/ConvLSTM')
    parser.add_argument('--decoder_layers', type=int, default=1, help='Number of layers in LSTM/ConvLSTM')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate for LSTM')
    parser.add_argument('--latent_dim', type=int, default=32, help='Latent dimension for the decoder in generator')
    parser.add_argument('--kernel_size', type=int, default=3, help='Kernel size for ConvLSTM in encoder')
    parser.add_argument('--input_channel', type=int, default=1, help='Kernel size for ConvLSTM in encoder')

    # Data argument
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--step_size', default=60, type=int) #
    parser.add_argument('--window_size', default=10*60, type=int) #
    parser.add_argument('--seq_len', default=10*60, type=int) #
    parser.add_argument('--map_size', type=int, default=256, help='Hidden dimension for LSTM/ConvLSTM')
    parser.add_argument('--sigma', default=2, type=int)
    parser.add_argument('--feat_sigma', default=0.001, type=float)
    parser.add_argument('--targ_sigma', default=0.0, type=float)

    # Design argumnet
    parser.add_argument('--target_type', default='global_vel', type=str, choices=['disp', 'vel', 'global_vel'])
    parser.add_argument('--dataset', type=str, default='our', choices=['neurit', 'ronin', 'our'])
    parser.add_argument('--use_map', action='store_true')
    parser.add_argument('--perturb', action='store_true')
    parser.add_argument('--model_type', type=str, default='Q_LSTM_GAN', choices=['Q_LSTM', 'GAN', 'Q_LSTM_GAN'])


    mode = parser.add_subparsers(title='mode', dest='mode', help='Operation: [train] train model, [test] evaluate model')
    mode.required = True

    # Train argument
    train_cmd = mode.add_parser('train')
    train_cmd.add_argument('--train_list', type=str, default='lists/our/list_train.txt') #
    train_cmd.add_argument('--val_list', type=str, default='lists/our/list_val.txt') #
    train_cmd.add_argument('--continue_from', type=str, default=None)
    train_cmd.add_argument('--epochs', type=int, default=400)
    train_cmd.add_argument('--save_interval', type=int, default=20)
    train_cmd.add_argument('--lr', '--learning_rate', type=float, default=1e-3)
    train_cmd.add_argument('--lstm_path', type=str, default=None)
    parser.add_argument('--use_scheduler', action='store_true')

    # Test argument
    test_cmd = mode.add_parser('test')
    test_cmd.add_argument('--test_list', type=str, default='lists/our/test_list.txt') #
    test_cmd.add_argument('--lstm_path', type=str, default=None)
    test_cmd.add_argument('--gan_path', type=str, default=None)
    test_cmd.add_argument('--show_plot', action='store_true')

    return parser.parse_known_args()

# Parse arguments
args, _ = get_args()
DEVICE = args.device

if __name__ == '__main__':
    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        if not args.lstm_path:
            raise ValueError("Model path required")
        args.batch_size = 1
        test_Q_GAN(args)
#-------------------------------------------------------------------------------------------------------------------------











import torch
from torch import nn
from utils import *
import matplotlib.pyplot as plt
import numpy as np
from main import DEVICE
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

# TODO: 
# 1) try different combination of initialization and input for the convlstm
# 2) Optimize the forward methods for faster computations
# Implemented attention encoder (cross attention last hidden with feasible mask) with quantile as images and last hidden state of imu_features and simple decoder initilized with encoder output
# Implemented attention encoder (cross attention last hidden of quantile features and feasible map)with quantile as sequence and simple decoder initilized with encoder output imu feature as input
# Implemented attention encoder (cross attention last hidden of quantile features and feasible map)with quantile as sequence and simple decoder initilized with encoder output and quantiles as input

def visualize_map_features(model, map_input, sample_idx=10):
    """
    Visualize CNN-extracted map features for a specific sample.
    
    Args:
    - model: TrajectoryGeneratorEncoder instance.
    - map_input: Feasible map input tensor [N, C, H, W].
    - sample_idx: Index of the sample to visualize.

    Returns:
    - Visualization of feature maps.
    """
    model.eval()
    with torch.no_grad():
        feature_maps = model.context_cnn(map_input)  # Extract map features
        feature_maps = feature_maps[sample_idx].cpu().numpy()  # Convert to NumPy
        map_input = map_input[sample_idx].cpu().numpy()  # Convert to NumPy
    # Visualize individual feature maps
    fig, axes = plt.subplots(1, 8, figsize=(15, 5))
    for i in range(8):
        axes[i].imshow(feature_maps[i], cmap='viridis')
        axes[i].set_title(f"Feature Map {i+1}")

    plt.show()

def inspect_feature_dominance(model, imu_features, map_input):
    """
    Compare the magnitudes of IMU features and CNN-extracted map features.
    
    Args:
    - model: TrajectoryGeneratorEncoder instance.
    - imu_features: IMU input tensor [N, L, h].
    - map_input: Feasible map input tensor [N, C, H, W].

    Returns:
    - Average magnitudes of IMU and map features.
    """
    model.eval()
    with torch.no_grad():
        context_encoded = model.context_cnn(map_input)
        imu_features_expanded = imu_features[:, -1, :].unsqueeze(2).unsqueeze(3).expand(-1, -1, context_encoded.size(-2), context_encoded.size(-1))
        imu_magnitude = torch.mean(torch.abs(imu_features_expanded)).item()
        map_magnitude = torch.mean(torch.abs(context_encoded)).item()

    print(f"IMU Feature Magnitude: {imu_magnitude}")
    print(f"Map Feature Magnitude: {map_magnitude}")

def make_mlp(dim_list, norm=None):
    layers = []
    if norm == 'spectral':
        for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
            layers.append(spectral_norm(nn.Linear(dim_in, dim_out)))
            layers.append(nn.ReLU())
    else:
        for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
            layers.append(nn.Linear(dim_in, dim_out))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)

def get_noise(shape):
    return torch.randn(*shape, device=DEVICE)


class LSTMModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim, args):
        super(LSTMModel, self).__init__()
        self.input_dim = input_dim
        self.h_dim = args.imu_hidden_dim
        self.output_dim = output_dim
        self.num_layers = args.lstm_layers
        self.batch_size = args.batch_size

        self.lstm = torch.nn.LSTM(self.input_dim, self.h_dim, self.num_layers, batch_first=True, dropout=args.dropout)
        self.linear1 = torch.nn.Linear(self.h_dim, self.output_dim * 5)
        self.linear2 = torch.nn.Linear(self.output_dim * 5, self.output_dim)


    def init_hidden_cell(self, hidden=None):
        """Initialize hidden and cell states."""
        if hidden == None:
            h = torch.zeros((self.num_layers, self.batch_size, self.h_dim), device=DEVICE)
        else:
            h = hidden
        c = torch.zeros((self.num_layers, self.batch_size, self.h_dim), device=DEVICE)
        return h, c

    def forward(self, input, hidden=None):
        output, self.hidden = self.lstm(input, self.init_hidden_cell())
        q_feature = self.linear1(output)
        traj_output = self.linear2(q_feature)
        return traj_output, output

class QLSTMModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim, args):
        super(QLSTMModel, self).__init__()
        self.input_dim = input_dim
        self.h_dim = args.imu_hidden_dim
        self.output_dim = output_dim
        self.num_layers = args.lstm_layers
        self.batch_size = args.batch_size
        self.num_quantiles = 3

        self.lstm = torch.nn.LSTM(self.input_dim, self.h_dim, self.num_layers, batch_first=True, dropout=args.dropout)
        self.linear1 = torch.nn.Linear(self.h_dim, self.output_dim * 5)
        self.linear2 = torch.nn.Linear(self.output_dim * 5, self.num_quantiles)
        self.linear3 = torch.nn.Linear(self.output_dim * 5, self.num_quantiles)

    def init_hidden_cell(self, hidden=None):
        """Initialize hidden and cell states."""
        if hidden == None:
            h = torch.zeros((self.num_layers, self.batch_size, self.h_dim), device=DEVICE)
        else:
            h = hidden
        c = torch.zeros((self.num_layers, self.batch_size, self.h_dim), device=DEVICE)
        return h, c

    def forward(self, input, hidden=None):
        output, self.hidden = self.lstm(input, self.init_hidden_cell())
        q_feature = self.linear1(output)
        traj_output_1 = self.linear2(q_feature)
        traj_output_2 = self.linear3(q_feature)
        return traj_output_1, traj_output_2, output

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias

        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, hidden):
        """ Initialize hidden and cell states"""
        # Expecting map_feature of shape [batch_size, hidden_dim, height, width]
        return hidden, torch.zeros_like(hidden)


class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, batch_first=True, bias=True):
        super(ConvLSTM, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias

        cell_list = []
        for i in range(self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim
            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim,
                                          kernel_size=self.kernel_size,
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, imu_input_tensor, map_feature, seq_len):
        """
        imu_input_tensor: IMU feature sequence, shape [batch_size, seq_len, input_dim, height, width]
        map_feature: map feature used to initialize hidden state, shape [batch_size, hidden_dim, height, width]
        """
        # Initialize hidden and cell states
        hidden_state = [self.cell_list[i].init_hidden(imu_input_tensor) for i in range(self.num_layers)]
        cur_layer_input = map_feature

        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            #output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input,
                                                 cur_state=[h, c])
                #output_inner.append(h)
            #layer_output = torch.stack(output_inner, dim=1)
            #cur_layer_input = layer_output  # For the next layer
        return h  # Output of the last ConvLSTM layer

class TrajectoryGeneratorEncoder(nn.Module):
    def __init__(self, input_channel, args):
        super(TrajectoryGeneratorEncoder, self).__init__()
        # Define input dimensions and ConvLSTM parameters
        self.imu_hidden_dim = args.imu_hidden_dim
        self.map_size = args.map_size
        self.context_feature_dim = args.context_feature_dim
        self.kernel_size = args.kernel_size
        self.convlstm_layers = args.convlstm_layers

        # ConvLSTM for spatio-temporal encoding
        self.conv_lstm = ConvLSTM(input_dim=self.context_feature_dim,
                                hidden_dim=self.imu_hidden_dim,
                                kernel_size=self.kernel_size,
                                num_layers=self.convlstm_layers,
                                batch_first=True)

        self.context_cnn = nn.Sequential(
            nn.Conv2d(input_channel, 32, kernel_size=self.kernel_size, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=self.kernel_size, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, self.context_feature_dim, kernel_size=self.kernel_size, stride=1, padding=1),
            nn.ReLU())#,
            #nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        #)


    def forward(self, imu_lstm_features, feasible_map):
        """
        :param imu_lstm_output: LSTM output tensor of shape [N, L, h]
        :param feasible_map: Dynamic feasible map of shape [N, L, C, M, M]
        :return: Spatio-temporal encoded features [N, L, hidden_dim, M, M]
        """
        seq_len = imu_lstm_features.shape[1]
        imu_lstm_features = imu_lstm_features[:, -1, :]
        context_encoded = self.context_cnn(feasible_map)

        # Step 2: Expand the IMU LSTM output from [N, h] to [N, h, M, M]
        imu_lstm_features = imu_lstm_features.unsqueeze(2).unsqueeze(3)  # Shape: [N, h, 1, 1]
        imu_lstm_features = imu_lstm_features.expand(-1, -1, context_encoded.shape[-2], context_encoded.shape[-1])  # Shape: [N, h, M, M]
        # Step 5: Forward pass through ConvLSTM
        last_hidden_state = self.conv_lstm(imu_lstm_features, context_encoded, seq_len)

        return last_hidden_state
        
class SimpleEncoder(nn.Module):
    def __init__(self, input_channel, args):
        super(SimpleEncoder, self).__init__()
        # Define input dimensions and ConvLSTM parameters
        self.imu_hidden_dim = args.imu_hidden_dim
        self.map_size = args.map_size
        self.context_feature_dim = args.context_feature_dim
        self.kernel_size = args.kernel_size
        self.convlstm_layers = args.convlstm_layers
        self.decoder_hidden_dim = args.decoder_hidden_dim

        self.context_cnn = nn.Sequential(
            nn.Conv2d(input_channel, 32, kernel_size=self.kernel_size, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=self.kernel_size, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, self.context_feature_dim, kernel_size=self.kernel_size, stride=1, padding=1),
            nn.ReLU())#,
            #nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        #)
        self.mlp = nn.Sequential(
            nn.Linear(self.imu_hidden_dim + self.context_feature_dim * int(self.map_size/4) * int(self.map_size/4), self.decoder_hidden_dim))

    def forward(self, imu_lstm_features, feasible_map):
        """
        :param imu_lstm_output: LSTM output tensor of shape [N, L, h]
        :param feasible_map: Dynamic feasible map of shape [N, L, C, M, M]
        :return: Spatio-temporal encoded features [N, L, hidden_dim, M, M]
        """
        seq_len = imu_lstm_features.shape[1]
        imu_lstm_features = imu_lstm_features.mean(dim=1)
        context_encoded = self.context_cnn(feasible_map)
        """
        # Step 2: Expand the IMU LSTM output from [N, h] to [N, h, M, M]
        imu_lstm_features = imu_lstm_features.unsqueeze(2).unsqueeze(3)  # Shape: [N, h, 1, 1]
        imu_lstm_features = imu_lstm_features.expand(-1, -1, context_encoded.shape[-2], context_encoded.shape[-1])  # Shape: [N, h, M, M]
        # Step 5: Forward pass through ConvLSTM
        last_hidden_state = self.conv_lstm(imu_lstm_features, context_encoded, seq_len)
        """
        map_flat = context_encoded.view(context_encoded.size(0), -1)  # Flatten to [batch_size, feature_dim]
        fused_features = torch.cat((imu_lstm_features, map_flat), dim=-1)
        last_hidden_state = self.mlp(fused_features)

        return last_hidden_state

class AttentionEncoder(nn.Module):
    def __init__(self, input_channel, args):
        super(AttentionEncoder, self).__init__()
        self.imu_hidden_dim = args.imu_hidden_dim
        self.context_feature_dim = args.context_feature_dim
        self.hidden_dim = args.decoder_hidden_dim
        self.kernel_size = args.kernel_size
        self.map_size = args.map_size

        # Linear layers to transform IMU and map features
        self.imu_to_query = nn.Linear(self.imu_hidden_dim, self.hidden_dim)
        self.map_to_key = nn.Linear(self.context_feature_dim*int(self.map_size/4)*int(self.map_size/4), self.hidden_dim)
        self.map_to_value = nn.Linear(self.context_feature_dim*int(self.map_size/4)*int(self.map_size/4), self.hidden_dim)

        # Output projection layer
        self.output_layer = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.context_cnn = nn.Sequential(
            nn.Conv2d(input_channel, 32, kernel_size=self.kernel_size, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=self.kernel_size, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, self.context_feature_dim, kernel_size=self.kernel_size, stride=1, padding=1),
            nn.ReLU())

    def forward(self, imu_lstm_features, feasible_map):
        """
        Args:
        - imu_features: IMU input tensor [batch_size, seq_len, imu_dim].
        - map_features: Map feature tensor [batch_size, map_feature_dim].

        Returns:
        - Encoded representation [batch_size, seq_len, hidden_dim].
        """
        batch_size, seq_len, _ = imu_lstm_features.shape
        context_encoded = self.context_cnn(feasible_map)
        context_encoded = context_encoded.view(context_encoded.size(0), -1)
        # Compute query, key, and value
        query = self.imu_to_query(imu_lstm_features)  # [batch_size, seq_len, hidden_dim]
        key = self.map_to_key(context_encoded).unsqueeze(1).expand(-1, seq_len, -1)  # [batch_size, seq_len, hidden_dim]
        value = self.map_to_value(context_encoded).unsqueeze(1).expand(-1, seq_len, -1)  # [batch_size, seq_len, hidden_dim]

        # Compute attention weights
        attention_weights = torch.bmm(query, key.transpose(1, 2))  # [batch_size, seq_len, seq_len]
        attention_weights = torch.softmax(attention_weights, dim=-1)

        # Compute attention output
        attended_map_features = torch.bmm(attention_weights, value)  # [batch_size, seq_len, hidden_dim]

        # Combine IMU and attended map features
        fused_features = query + attended_map_features
        output = self.output_layer(fused_features)  # [batch_size, seq_len, hidden_dim]

        return output.mean(dim=1)



# def summary(t):
#     return {'mean': t.mean().item(),
#             'std' : t.std().item(),
#             'min' : t.min().item(),
#             'max' : t.max().item()}
# class AttnTracker:
#     """Collects statistics for one forward pass."""
#     def __init__(self):
#         self.q, self.k = None, None
#         self.logits, self.probs = None, None

#     def hook(self, module, inp, out):
#         # soft-maxed attention
#         self.probs = out[1].detach().cpu()                        # (B,H,Lq,Lk)

#         # --- rebuild Q & K exactly as the layer does -------------
#         if module._qkv_same_embed_dim:
#             Wq, Wk, _ = module.in_proj_weight.chunk(3, 0)
#             bq, bk, _ = module.in_proj_bias.chunk(3)
#         else:                 # <-- rarely used path
#             Wq, bq = module.q_proj_weight, module.q_proj_bias
#             Wk, bk = module.k_proj_weight, module.k_proj_bias

#         query, key = inp[0], inp[1]                               # before proj
#         dk = module.head_dim**0.5

#         q = F.linear(query, Wq, bq) / dk                          # (B,Lq,D)
#         k = F.linear(key,   Wk, bk)                               # (B,Lk,D)
#         B, Lq, _ = q.shape
#         _, Lk, _ = k.shape
#         q = q.view(B, module.num_heads, Lq, module.head_dim)
#         k = k.view(B, module.num_heads, Lk, module.head_dim)

#         # save raw tensors (optional – big!)
#         self.q, self.k = q.cpu().detach(), k.cpu().detach()

#         # pre-softmax logits
#         self.logits = torch.einsum('bhtd,bhsd->bhts', q, k).cpu().detach() # (B,H,Lq,Lk)


class CrossAttentionEncoderImuMask(nn.Module):  # Quantiles as Images
    def __init__(self, input_channel, args):
        super(CrossAttentionEncoderImuMask, self).__init__()
        self.context_feature_dim = args.context_feature_dim 
        self.kernel_size = args.kernel_size
        self.map_size = args.map_size
        self.imu_hidden_dim = args.imu_hidden_dim
        self.batch_size = args.batch_size
        self.num_heads = 4
        self.embed_dim = 128 # self.context_feature_dim * int(self.map_size/4) * int(self.map_size/4)
        self.head_dim = self.embed_dim // self.num_heads

        # ---------------------------------------  in __init__  -----------------------
        # self.attn_logits = None        # to store last batch for inspection
        # self.attn_probs  = None

        # Transform IMU features into Query
        self.imu_mlp = nn.Linear(self.imu_hidden_dim, self.embed_dim)
        # self.imu_mlp = nn.Sequential(
        # nn.Linear(self.imu_hidden_dim, self.embed_dim),
        # nn.LayerNorm(self.embed_dim),
        # nn.Dropout(0.1),          # 10 % keep-out
        # nn.ReLU(inplace=True))

        # self.q_scale = nn.Parameter(torch.tensor(1.0))
        # F = (self.map_size // 4) ** 2        # number of spatial tokens
        # self.logit_bias = nn.Parameter(torch.zeros(self.num_heads, 1, F))
        # nn.init.normal_(self.logit_bias, std=0.02)
    
        # self.logit_bias = nn.Parameter(torch.zeros(self.num_heads, 1, 1))  # (H,1,1)
        self.map_mlp = nn.Linear(1, self.embed_dim)
        # CNN for spatial feature extraction (Key, Value)
        self.context_cnn = nn.Sequential(
            nn.Conv2d(input_channel, 32, kernel_size=self.kernel_size, stride=1, padding=1),
            # nn.BatchNorm2d(32),
            nn.ReLU(),#inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=self.kernel_size, stride=1, padding=1),
            # nn.BatchNorm2d(64),
            nn.ReLU(),#inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Conv2d(64, 64, kernel_size=self.kernel_size, padding=1, groups=64),
            # nn.BatchNorm2d(64),
            # nn.ReLU(),#inplace=True),
            nn.Conv2d(64, self.context_feature_dim, stride=1, kernel_size=1),
            nn.ReLU(),#inplace=True))#,
            nn.Flatten())
        # Cross-Attention
        # self.map_mlp = nn.Linear(self.context_feature_dim + 2, self.embed_dim)
        # self.k_scale = nn.Parameter(torch.tensor(1.0))
    
        #self.cross_attention = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=4, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=4, batch_first=True)
        # self.cross_attention.register_forward_pre_hook(
        #     self._inject_bias_as_mask
        # )
        # self.tracker = AttnTracker()
        # self.cross_attention.register_forward_hook(self._save_logits)

    def forward(self, imu_lstm_features, feasible_mask, t):
        B, _, H, W = feasible_mask.shape
        # H4 = W4 = H // 4
        # F_tokens = H4 * W4
        # Extract spatial features (Key, Value)
        spatial_features = self.context_cnn(feasible_mask)  # Shape: [batch_size, flattened_dim]
        spatial_features = spatial_features.unsqueeze(-1)
        attn_mask = (feasible_mask.squeeze() < 0.1)
        attn_mask = attn_mask.view(self.batch_size, -1)
        # spatial_features = spatial_features.permute(0, 2, 3, 1)   # [B, 32, 32, 16]
        # spatial_features = spatial_features.reshape(B, int(self.map_size/4) * int(self.map_size/4), self.context_feature_dim)        # [B, F, C]

        device = spatial_features.device

        # y, x = torch.meshgrid(
        # torch.linspace(-1, 1, H4, device=device),
        # torch.linspace(-1, 1, W4, device=device),
        # indexing='ij')                           # ij = (row,col) order
        # pos = torch.stack((x, y), dim=-1)            # [H4, W4, 2]
        # pos = pos.view(1, F_tokens, 2)                   # [1, F, 2]
        # pos    = pos.expand(B, -1, -1)      # [B, F, 2]
        # spatial_features = torch.cat([spatial_features, pos], dim=-1)  # [B, F, C+2]
        
        spatial_embed = self.map_mlp(spatial_features)
        # spatial_embed = F.layer_norm(spatial_embed, spatial_embed.shape[-1:])
        # spatial_embed = spatial_embed * self.k_scale                  # learnable temperature

        query = self.imu_mlp(imu_lstm_features.mean(dim=1))#.unsqueeze(1)  # Shape: [1, batch_size, feature_dim]
        # query = self.imu_mlp(imu_lstm_features)
        # query = query * self.q_scale
        query = query.unsqueeze(1)  
        
        #spatial_features = spatial_features.unsqueeze(-1)  # Shape: [1, batch_size, feature_dim]
        # attn_mask = (feasible_mask < 0.1).float()
        # attn_mask = F.max_pool2d(attn_mask, 2, 2)             # H/2 × W/2
        # attn_mask = F.max_pool2d(attn_mask, 2, 2)             # H/4 × W/4
        # attn_mask_bin = attn_mask.view(B, -1).bool()              # [B, F]  → bool mask
        # bias = self.logit_bias            # (H,1,F)
        # bias = bias.expand(B, -1, -1, -1) # [B,H,1,F]
        # bias = bias.reshape(B*self.num_heads, 1, F_tokens)  # (B·H,1,F)
        

        #att_mask = att_mask.view(self.batch_size, -1)
        # Transform IMU features into Query
        # query = query * 2.0
        #self.context_feature_dim += 2     # update once in __init__

        

        # key_norm_std = spatial_embed.norm(dim=-1).std(dim=-1).mean().item()
        # key_feat_std = spatial_embed.std(dim=1).mean().item()   # variance across F
        # print(f"std ∥kᵢ∥ across keys ≈ {key_norm_std:.4f}")
        # print(f"mean feature std across keys ≈ {key_feat_std:.4f}")
        # spatial_embed = spatial_embed * 10.0    # try 5–10 first; tweak later
        
        ######spatial_embed = spatial_embed.permute(1, 0, 2)
        # Apply cross-attention
        attn_output, _ = self.cross_attention(query, spatial_embed, spatial_embed, key_padding_mask=attn_mask)#, average_attn_weights=False)
        attn_output = attn_output.squeeze(1)

        # attn_w = attn_w + self.logit_bias.unsqueeze(0)            # broadcast (1,H,1,1)
        # attn_w = attn_w.softmax(dim=-1)
        # var_soft = self.attn_probs.var(dim=-1).mean().item()
        # std_log  = self.attn_logits.std().item()
        # print(f"var(softmax)={var_soft:.4e}   std(logits)={std_log:.3f}")

        # row_std = self.attn_logits.std(dim=-1)        # std per row
        # row_var = self.attn_probs.var(dim=-1)         # var per row
        # print("mean row-std(logits):", row_std.mean().item())
        # print("mean row-var(softmax):", row_var.mean().item())

        # # pick first sample, head 0 (adapt if multi-head debug needed)
        # mask = ~attn_mask_bin[0]                         # shape [F]
        # free_logits = self.attn_logits[0, 0, 0, mask]    # (src_len_free,)
        
        # print("std logits (free cells only):", free_logits.std().item())
        # print("var softmax (free cells only):",
        #       self.attn_probs[0, 0, mask].var().item())

        # examine ONE head, ONE query row, ONLY free cells
        # head  = 1
        # row   = 0
        # mask  = ~attn_mask_bin[0]                       # free cells bool mask  (F,)
        # free_logits = self.attn_logits[0, head, row, mask]   # (N_free,)
        
        # print("min/max logits in free cells:", free_logits.min().item(),
        #                                      free_logits.max().item())
        # print("pairwise max │max-min│:", (free_logits.max() - free_logits.min()).item())
        # if t > 0 and t % 20 == 0:
        #     print("‖q‖ :", summary(self.q.norm(dim=-1)))          # (B,H,Lq)
        #     print("‖k‖ :", summary(self.k.norm(dim=-1)))          # (B,H,Lk)
        #     print("logits:", summary(self.logits))
        #     print("probs :", summary(self.probs))
        #     row_std = self.logits.std(-1).mean().item()       # avg σ over rows
        #     print("row-wise logit σ:", row_std)        # expect ≈ 0.3-0.4 now
    
            # B, H, Lq, Lk = self.logits.shape
            # F_side = int(Lk**0.5)            # 32 when F = 1024
            
            # # 1️⃣ histogram of logits
            # plt.figure(); plt.hist(self.logits.flatten(), bins=100)
            # plt.title("Distribution of dot-product logits"); plt.show()
            
            # # 2️⃣ histogram of probas
            # plt.figure(); plt.hist(self.probs.flatten(), bins=100)
            # plt.title("Distribution of soft-max probabilities"); plt.show()
            
            # # 3️⃣ heat-map of a single head (batch 0, head 0)
            # plt.figure(figsize=(4,4))
            # plt.imshow(self.probs[0,0,0].view(F_side, F_side),
            #            origin='lower', interpolation='nearest')
            # plt.colorbar(); plt.title("Attention map – B0 H0"); plt.show()
            
        return attn_output, attn_mask#, attn_w

    # def _inject_bias_as_mask(self, module, args):
    #     """
    #     Args
    #     ----
    #     module : the MultiheadAttention instance
    #     args   : tuple(query, key, value, attn_mask, key_padding_mask, ...)
    #     Returns
    #     -------
    #     new_args : tuple(...)  with additive bias merged into attn_mask
    #     """
    #     query, key, value, *rest = args
    #     B = query.size(0)                # current batch
    #     F = key.size(1)                  # num. spatial tokens

    #     # (H,1,F) → (B·H, 1, F)   matches the batched-mask shape that
    #     # PyTorch expects when batch_first=True
    #     bias_mask = self.logit_bias      # param, already on device
    #     bias_mask = bias_mask.expand(B, -1, -1, -1)      # [B,H,1,F]
    #     bias_mask = bias_mask.reshape(B * self.num_heads, 1, F)

    #     # existing attn_mask (could be None)
    #     if rest and rest[0] is not None:
    #         attn_mask = rest[0]
    #         # broadcast existing mask if needed then add
    #         attn_mask = attn_mask + bias_mask
    #     else:
    #         attn_mask = bias_mask

    #     # rebuild the *args tuple*
    #     new_args = (query, key, value, attn_mask, *rest[1:])
    #     return new_args
        
    def _save_logits(self, module, inp, out):
        self.probs = out[1].detach()

        # reconstruct logits (pre‑softmax)
        if module._qkv_same_embed_dim:
            Wq, Wk, _ = module.in_proj_weight.chunk(3, dim=0)
            bq, bk, _ = module.in_proj_bias.chunk(3)
        else:
            Wq, bq = module.q_proj_weight, module.q_proj_bias
            Wk, bk = module.k_proj_weight, module.k_proj_bias

        query, key = inp[0], inp[1]  # (B,1,D) & (B,F,D)
        dk = module.head_dim ** 0.5
        q = F.linear(query, Wq, bq) / dk
        k = F.linear(key, Wk, bk)
        B, Lq, _ = q.shape
        self.q = q.view(B, module.num_heads, Lq, module.head_dim)
        self.k = k.view(B, module.num_heads, key.shape[1], module.head_dim)
        logits = torch.einsum('bhtd,bhsd->bhts', self.q, self.k)
        self.logits = logits.detach()

class TrajectoryGeneratorDecoder_2(nn.Module):
    def __init__(self, args):
        """
        :param input_dim: Dimensionality of the input (e.g., 2 for 2D positions).
        :param hidden_dim: Hidden size of the LSTM layer.
        :param latent_dim: Size of the latent vector for stochasticity.
        :param output_dim: Output size (2 for x, y coordinates).
        :param num_layers: Number of LSTM layers.
        """
        super(TrajectoryGeneratorDecoder_2, self).__init__()
        self.decoder_hidden_dim = args.decoder_hidden_dim
        self.output_dim = args.output_dim
        self.decoder_layers = args.decoder_layers
        self.embedding_dim = args.embedding_dim
        self.latent_dim = args.latent_dim
        #self.h_dim = args.imu_hidden_dim # (simple, convlstm, attention) encoder
        self.h_dim = 6 #((CrossAttentionImuMask) Encoder)
        # LSTM cell for step-by-step prediction
        # IMU features as input
        self.embedding = nn.Linear(self.h_dim, self.embedding_dim)
        # self.attn_embed = nn.Sequential(
        # nn.Linear(128, self.embedding_dim),
        # nn.LayerNorm(self.embedding_dim),
        # nn.ReLU(inplace=True),
        # nn.Dropout(p=0.1)
        # )
        # self.embedding = nn.Sequential(
        # nn.Linear(self.h_dim, self.embedding_dim),
        # nn.LayerNorm(self.embedding_dim),
        # nn.ReLU(inplace=True),
        # nn.Dropout(p=0.1)
        # )
        # encoder attention as input
        self.lstm = nn.LSTM(self.embedding_dim, self.decoder_hidden_dim, self.decoder_layers, batch_first=True)
        # Output layer to generate 2D coordinates
        self.output_layer = nn.Linear(self.decoder_hidden_dim, self.output_dim)
        self.relu = torch.nn.ReLU()
    
    def forward(self, feature_vec, state_tuple):
        """
        :param initial_position: Initial position of shape [N, 2] (x, y coordinates).
        :param encoder_hidden: Hidden state features from the ConvLSTM encoder [N, hidden_dim].
        :param latent_vector: Latent vector for initialization [N, latent_dim].
        :param seq_length: The length of the sequence to generate (number of time steps).
        :return: Generated trajectory of shape [N, seq_length, 2].
        """
        # Step 2: Prepare input position and container for generated trajectory
        B, L, _ = feature_vec.shape

        decoder_input = self.embedding(feature_vec)
        decoder_input = self.relu(decoder_input)
        # attn = self.attn_embed(last_hidden_state).unsqueeze(1).expand(B, L, -1)
        output, state_tuple = self.lstm(decoder_input, state_tuple)
        relative_position = self.output_layer(output)
        return relative_position


class Generator(nn.Module):
    def __init__(self, input_channel, dts, args):
        super(Generator, self).__init__()
        self.imu_hidden_dim = args.imu_hidden_dim
        self.map_size = args.map_size
        self.latent_dim = args.latent_dim
        self.decoder_hidden_dim = args.decoder_hidden_dim
        self.batch_size = args.batch_size
        self.mlp_dim = args.mlp_dim
        self.context_feature_dim = args.context_feature_dim
        self.dts = dts
        self.target_type = args.target_type
        self.embed_dim = 128

        # Encoder: ConvLSTM for spatio-temporal feature extraction
        #self.encoder = TrajectoryGeneratorEncoder(input_channel, args)
        #self.encoder  = SimpleEncoder(input_channel, args)
        self.encoder = CrossAttentionEncoderImuMask(input_channel, args)
        #self.encoder = AttentionEncoder(input_channel, args)

        #mlp_decoder_context_dims = [self.context_feature_dim*int(self.map_size/4)*int(self.map_size/4), self.mlp_dim, self.decoder_hidden_dim - self.latent_dim]  # (CrossAttentionImuMask) encoder
        #mlp_decoder_context_dims = [self.decoder_hidden_dim, self.mlp_dim, self.decoder_hidden_dim - self.latent_dim] # (simple, attention) encoder
        #mlp_decoder_context_dims = [self.imu_hidden_dim*int(self.map_size/4)*int(self.map_size/4), self.mlp_dim, self.decoder_hidden_dim - self.latent_dim]  # (convlstm) encoder

        mlp_decoder_context_dims = [self.embed_dim, self.mlp_dim, self.decoder_hidden_dim - self.latent_dim]  # (CrossAttentionImuMask) encoder

        self.mlp_decoder_context = make_mlp(mlp_decoder_context_dims)


        # Decoder full trajectory prediction
        self.decoder = TrajectoryGeneratorDecoder_2(args)

    def get_masks(self, X_quantiles, Y_quantiles, map_mask, initial_position, bounds):
        X_quantiles = relative_to_abs(X_quantiles, initial_position[:, [0]], self.dts, self.target_type)
        Y_quantiles = relative_to_abs(Y_quantiles, initial_position[:, [1]], self.dts, self.target_type)
        lower_x, lower_y = X_quantiles[:, :, 0], Y_quantiles[:, :, 0]
        upper_x, upper_y = X_quantiles[:, :, -1], Y_quantiles[:, :, -1]
        x_min, x_max, y_min, y_max = bounds

        # Method 1 without interpolation
        # Create a grid of all possible indices in the x and y dimensions
        x_full_range = torch.linspace(x_min, x_max, self.map_size, device=DEVICE)
        y_full_range = torch.linspace(y_min, y_max, self.map_size, device=DEVICE)
        x_full_range, y_full_range = torch.meshgrid(x_full_range, y_full_range, indexing='ij')

        # Reshape x_full_range and y_full_range for comparison across batches and sequence length
        x_full_range = x_full_range.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, height, width)
        y_full_range = y_full_range.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, height, width)

        # Expand lower and upper quantiles to match the grid shape for comparison
        lower_x = lower_x.unsqueeze(-1).unsqueeze(-1)  # Shape: (batch_size, L, 1, 1)
        upper_x = upper_x.unsqueeze(-1).unsqueeze(-1)  # Shape: (batch_size, L, 1, 1)
        lower_y = lower_y.unsqueeze(-1).unsqueeze(-1)  # Shape: (batch_size, L, 1, 1)
        upper_y = upper_y.unsqueeze(-1).unsqueeze(-1)  # Shape: (batch_size, L, 1, 1)

        # Use sigmoid for soft boundaries
        temperature = 2  # Adjust to control the smoothness of the boundary
        x_mask = torch.sigmoid(temperature * (x_full_range - lower_x)) * torch.sigmoid(temperature * (upper_x - x_full_range))
        y_mask = torch.sigmoid(temperature * (y_full_range - lower_y)) * torch.sigmoid(temperature * (upper_y - y_full_range))
        mask = x_mask * y_mask
        # Use Gaussian-like soft boundaries
        # sigma = 1.0  # Adjust for smoothness of boundaries
        # x_mask = torch.exp(-((x_full_range - lower_x) ** 2) / (2 * sigma ** 2)) * \
        #          torch.exp(-((x_full_range - upper_x) ** 2) / (2 * sigma ** 2))
        # y_mask = torch.exp(-((y_full_range - lower_y) ** 2) / (2 * sigma ** 2)) * \
        #          torch.exp(-((y_full_range - upper_y) ** 2) / (2 * sigma ** 2))
    
        # # Combine masks
        # mask = x_mask * y_mask  # Combine x and y masks
        # mask = mask.sum(dim=1)  # Smooth summation across sequence length
    
        # # Normalize mask to the range [0, 1]
        # mask = mask / mask.max(dim=1)[0].max(dim=1)[0].unsqueeze(1).unsqueeze(2)  # Ensure values are scaled between 0 and 1
        mask = mask.sum(dim=1)  # Smooth summation across sequence length
        mask = mask / mask.max(dim=1)[0].max(dim=1)[0].unsqueeze(1).unsqueeze(2)
        """
        # For validation
        # Compute precise floating-point grid coordinates
        x_grid_lower = ((lower_x - x_min) / (x_max - x_min)) * (M)
        y_grid_lower = ((lower_y - y_min) / (y_max - y_min)) * (M)

        x_grid_upper = ((upper_x - x_min) / (x_max - x_min)) * (M)
        y_grid_upper = ((upper_y - y_min) / (y_max - y_min)) * (M)
        # Compute nearest integer grid indices for array access
        x_index_lower = x_grid_lower.int().long()
        y_index_lower = y_grid_lower.int().long()

        x_index_upper = x_grid_upper.int().long()
        y_index_upper = y_grid_upper.int().long()

        x_index_lower = torch.clamp(x_index_lower, 0, M - 1)
        y_index_lower = torch.clamp(y_index_lower, 0, M - 1)

        x_index_upper = torch.clamp(x_index_upper, 0, M - 1)
        y_index_upper = torch.clamp(y_index_upper, 0, M - 1)
        """
        mask = mask.unsqueeze(1)  # Shape: [batch_size, 1, height, width]
        #final_mask = (1 - map_mask) * mask
        final_mask = map_mask * mask
        final_mask = final_mask / final_mask.max(dim=2)[0].max(dim=2)[0].unsqueeze(2).unsqueeze(3)
        return final_mask

    def add_noise(self, _input):
        npeds = _input.size(0)
        # seq_len = _input.size(1)
        noise_shape = (self.latent_dim,)
        z_decoder = get_noise(noise_shape)
        vec = z_decoder.view(1, self.latent_dim).repeat(npeds, 1)
        # vec = torch.randn(npeds, self.latent_dim, device=_input.device)
        return torch.cat((_input, vec), dim=-1)

    def forward(self, map_mask, initial_position, Q_model, imu_data, bounds, t):
        # Pass IMU data through the quantile model to get upper and lower quantiles
        X_quantiles, Y_quantiles, feature_vector = Q_model(imu_data)
        # Get the plausible region
        #static_mask = self.get_masks(X_quantiles, Y_quantiles, map_mask, initial_position, bounds)

        # Simple encoder, convlstm encoder
        #last_hidden_state = self.encoder(feature_vector, static_mask) # (Simple, convlstm, attention) encoder
        last_hidden_state, attn_mask = self.encoder(feature_vector, map_mask, t) #(CrossAttentionImuMask Encoder
        # breakpoint()
        #convlstm
        #last_hidden_state = last_hidden_state.view(last_hidden_state.size(0), -1)  # Flatten to [batch_size, hidden_dim * m * m]

        noise_input = self.mlp_decoder_context(last_hidden_state)

        decoder_h = self.add_noise(noise_input)
        decoder_h = torch.unsqueeze(decoder_h, 0)
        decoder_c = torch.zeros((1, self.batch_size, self.decoder_hidden_dim), device=DEVICE)
        state_tuple = (decoder_h, decoder_c)
        
        #IMU features as input
        #predicted_positions = self.decoder(feature_vector, state_tuple) # (simple, convlstm, attention) encoder
        predicted_positions = self.decoder(torch.cat((X_quantiles, Y_quantiles), dim=-1), state_tuple) #(CrossAttentionImuMask) Encoder
    
        return predicted_positions, attn_mask#, attn_w

class DiscriminatorEncoder(nn.Module):
    def __init__(self, traj_hidden_size, traj_embed_size):
        super(DiscriminatorEncoder, self).__init__()

        self.h_dim = traj_hidden_size
        self.embedding_dim = traj_embed_size

        self.encoder = nn.LSTM(self.embedding_dim, self.h_dim, 1, batch_first=True)
        self.spatial_embedding = spectral_norm(nn.Linear(2, self.embedding_dim))

    def init_hidden(self, batch):
        h = torch.zeros((1, batch, self.h_dim), device=DEVICE)
        c = torch.zeros((1, batch, self.h_dim), device=DEVICE)
        return (h, c)

    def forward(self, obs_traj):
        traj_embedding = self.spatial_embedding(obs_traj)
        state = self.init_hidden(obs_traj.shape[0])
        output, state = self.encoder(traj_embedding, state)
        final_h = state[0].squeeze()
        return final_h


class TrajectoryDiscriminator(nn.Module):
    def __init__(self, args):
        super(TrajectoryDiscriminator, self).__init__()
        self.h_dim = args.decoder_hidden_dim
        self.embedding_dim = args.embedding_dim
        self.mlp_dim = args.mlp_dim

        self.encoder = DiscriminatorEncoder(self.h_dim, self.embedding_dim)
        real_classifier_dims = [self.h_dim, self.mlp_dim, 1]
        self.real_classifier = make_mlp(real_classifier_dims)#, norm='spectral')

    def forward(self, traj_rel):
        final_h = self.encoder(traj_rel)
        scores = self.real_classifier(final_h)
        return scores

