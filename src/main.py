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
from copy import deepcopy
# Set seeds for reproducibility
# random.seed(42)
# np.random.seed(42)
# torch.manual_seed(42)
# torch.cuda.manual_seed(42)
# torch.cuda.manual_seed_all(42)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

G_UPDATES_TARGET = 50000
TOTAL_ITERATIONS   = G_UPDATES_TARGET                 # because g_steps = 1


PHASES = dict(
    warmup_iters      = int(TOTAL_ITERATIONS * 0.1),          # Phase-0  – TrajLoss only
    adv_iters = int(TOTAL_ITERATIONS * 0.25),         # Phase-1  – Traj + GAN
    curriculum_iters  = int(TOTAL_ITERATIONS * 0.25),          # Phase-2  – raise feas_w 0→0.5
    finetune_iters    = int(TOTAL_ITERATIONS * 0.4),          # Phase-3  – joint fine-tune
)
PHASES["cum_boundaries"] = np.cumsum(list(PHASES.values()))
SUP_START, SUP_END   = 6.0, 5.0
ADV_TARGET, ADV_FT   = 1.0, 0.30     # keep 10% in Phase 2
TARGET_FEAS_W        = 0.05


def get_losses(args, dts):
    loss_fns = {}
    if args.model_type == 'Q_LSTM':
        quantiles = [0.025, 0.975]
        loss_fns['LSTM_quantile_loss'] = QTrajLoss(quantiles, args, dts)
    elif args.model_type == 'GAN':
        loss_fns['GAN_generator_loss'] = gan_g_loss
        loss_fns['GAN_discrimentator_loss'] = gan_d_loss
        loss_fns['GAN_generator_l2_loss'] = TrajLoss(args)
        loss_fns['GAN_generator_feasibility_loss'] = feasibility_aware_loss
    else:
        quantiles = [0.025, 0.975]
        loss_fns['LSTM_quantile_loss'] = QTrajLoss(quantiles, args, dts)
        loss_fns['GAN_generator_loss'] = gan_g_loss
        loss_fns['GAN_discrimentator_loss'] = gan_d_loss
        loss_fns['GAN_generator_l2_loss'] = TrajLoss(args)
        loss_fns['GAN_generator_feasibility_loss'] = feasibility_aware_loss
    return loss_fns

def current_phase(t):
    w, a, c,f = PHASES["cum_boundaries"]
    if t <  w:        return 0               # warm-up
    elif t < a:       return 1               # adversarial
    elif t < c:       return 2               # curriculum
    else:             return 3               # fine-tune

# def get_loss_weights(t):
#     phase = current_phase(t)
#     # default values
#     traj_w = SUP_START + (SUP_END - SUP_START) * (min(1, (t/PHASES["cum_boundaries"][1])))
#     # traj_w, feas_w, adv = 5.0, 0.0, True
#     if phase == 0:                     # only TrajLoss
#         # adv = False
#         adv_w, feas_w = 0.0, 0.0
#         status = "warm-up"
#     elif phase == 1:                   # Traj + GAN
#         status = "curriculm training"
#         adv_w = ADV_TARGET * ((t - PHASES["cum_boundaries"][0]) / PHASES["curriculum_iters"])
#         feas_w = TARGET_FEAS_W * min(1, ((t - PHASES["cum_boundaries"][0]) / (PHASES["curriculum_iters"] * 0.8)))
#     else: #phase == 2:                   # ramp feasibility
#         status = "fine tunning"
#         adv_w = ADV_TARGET * ADV_FT
#         feas_w = TARGET_FEAS_W 
#         # rel = (t - PHASES["cum_boundaries"][1]) / PHASES["curriculum_iters"]
#         # feas_w = TARGET_FEAS_W * rel   # linear ramp
#     # else:                              # Phase-3: joint fine-tune
#     #     status = "fine_tunning"
#     #     feas_w = TARGET_FEAS_W

#     # print(f' iteration: {t} | Phase [{status}] | feasiblity weight: {feas_w:.4f} | trajectory weight: {traj_w:.4f}')
#     return traj_w, feas_w, adv_w, phase

def get_loss_weights(t):
    phase = current_phase(t)
    # default values
    traj_w = SUP_END #SUP_START + (SUP_END - SUP_START) * (min(1, (t/PHASES["cum_boundaries"][1])))
    feas_w, adv_w = 0.0, 0.0
    if phase == 0:                     # only TrajLoss
        # adv = False
        adv_w, feas_w = 0.0, 0.0
        status = "warm-up"
    elif phase == 1:                   # Traj + GAN
        status = "adversarial training"
        feas_w = 0.0
        adv_w = ADV_TARGET * ((t - PHASES["cum_boundaries"][0]) / PHASES["adv_iters"])
    elif phase == 2:                   # ramp feasibility
        feas_w = TARGET_FEAS_W * min(1, ((t - PHASES["cum_boundaries"][1]) / (PHASES["curriculum_iters"])))
        status = "curriculum"
        adv_w = ADV_TARGET
        # feas_w = TARGET_FEAS_W 
        # rel = (t - PHASES["cum_boundaries"][1]) / PHASES["curriculum_iters"]
        # feas_w = TARGET_FEAS_W * rel   # linear ramp
    else:                              # Phase-3: joint fine-tune
        status = "fine_tunning"
        feas_w = TARGET_FEAS_W
        adv_w = 0.0

    # print(f' iteration: {t} | Phase [{status}] | feasiblity weight: {feas_w:.4f} | trajectory weight: {traj_w:.4f}')
    return traj_w, feas_w, adv_w, phase

def trainQmodel(args, train_loader, val_loader, log_file, Qtraining_epochs, loss_fns):
    train_mini_batches = len(train_loader)
    val_mini_batches = len(val_loader)

    # Initialize QLSTM model
    lstm_model = QLSTMModel(input_dim=args.input_dim, output_dim=args.output_dim, args=args).to(DEVICE)

    criterion = loss_fns['LSTM_quantile_loss']
    q_optimizer = torch.optim.Adam(lstm_model.parameters(), args.lr)
    q_scheduler = ReduceLROnPlateau(q_optimizer, 'min', patience=15, factor=0.75, verbose=True, eps=1e-12)
    val_loader.dataset.transform = None
    # train_loader.dataset.transform = None
    start_epoch = 0
    best_val_loss = np.inf
    try:
        for epoch in range(start_epoch, Qtraining_epochs):
            log_line = ''
            # if (epoch+1)%5 == 0:
            #     breakpoint()
            lstm_model.train()
            train_loss = 0
            for batch in train_loader:
                feat, targ, _, _, _, _, _ = batch
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
                        feat, targ, _, _, _, _, _ = batch
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

# def discriminator_step(batch, generator, discriminator, loss_fns, optimizer_d, Q_model, dts, target_type, t):
#     feat, targ, init_pos, traj_real, map_tensor, map_meta = batch
#     feat, targ, init_pos, traj_real, map_tensor = map(lambda x: x.to(DEVICE),
#                                            (feat, targ, init_pos, traj_real, map_tensor))
#     losses = {}
#     gp_w   = 20.0
#     optimizer_d.zero_grad()
#     # 1. generate fake trajectories
#     def add_noise(x, std=0.02, max_iter=3000, i=0):
#         if i < max_iter:
#             return x + torch.randn_like(x) * std * (1 - i / max_iter)
#         return x

#     with torch.no_grad():
#         pred_traj_fake_rel, _, _ = generator(map_tensor, init_pos, Q_model,
#                                    feat, map_meta, t)

#     # 2. critic forward
#     scores_real = discriminator(add_noise(targ, i=t))
#     scores_fake = discriminator(add_noise(pred_traj_fake_rel.detach(), i=t))
    
#     # 3. Wasserstein loss + gradient penalty
#     d_loss = loss_fns['GAN_discrimentator_loss'](scores_real, scores_fake)
#     gp = gradient_penalty(discriminator, targ, pred_traj_fake_rel.detach())
#     d_total = d_loss + gp_w * gp
#     losses['d_loss'] = d_loss.item()
#     losses['GP'] = gp_w * gp.item()
#     w_dist = scores_real.mean() - scores_fake.mean()
#     losses['w_dist'] = w_dist.item()
#     d_total.backward()
#     optimizer_d.step()

#     if t % 200 == 0:
#         print(f"[{t:05d}]  W={w_dist:+.3f}  GP={losses['GP']:4.2f}  "
#               f"μr={scores_real.mean():+.3f}  μf={scores_fake.mean():+.3f}")

#     if t % 200 == 0:            # every 200 steps
#         with torch.no_grad():
#             # pick first weight tensor that requires_grad
#             for name, p in discriminator.named_parameters():
#                 if p.grad is not None:
#                     print(f"[{t:05d}] D grad ‖{name}‖ = {p.grad.abs().mean():.4e}")
#                     break
    
#     return losses
    
def discriminator_step(batch, generator, discriminator, loss_fns, optimizer_d, Q_model, dts, target_type, t):
    losses = {}
    losses['d_loss'] = 0.0 #torch.tensor(0.0, device=DEVICE)
    if not any(p.requires_grad for p in discriminator.parameters()):
        return losses    # D is frozen in Phase-0 & Phase-3
    feat, targ, initial_pos, traj_real, map_tensor, map_meta, ts = batch
    # dts = (ts[:,1:] - ts[:,:-1]).to(DEVICE)
    feat, targ, initial_pos, traj_real, map_tensor = feat.to(DEVICE), targ.to(DEVICE), initial_pos.to(DEVICE), traj_real.to(DEVICE), map_tensor.to(DEVICE)
    optimizer_d.zero_grad()
    pred_traj_fake_rel, _, _, quantiles, imu_feat = generator(map_tensor, initial_pos, Q_model, feat, map_meta, t)
    pred_traj_fake = relative_to_abs(pred_traj_fake_rel, initial_pos, dts, target_type)

    # def add_noise(x, std=0.02, max_iter=1000, i=0):
    #     if i < max_iter:
    #         return x + torch.randn_like(x) * std * (1 - i / max_iter)
    #     return x
    
    scores_fake = discriminator(pred_traj_fake_rel.detach(), quantiles, map_tensor)# imu_feat)
    scores_real = discriminator(targ, quantiles, map_tensor)#, imu_feat)
    # scores_fake = discriminator(pred_traj_fake_rel.detach())
    # scores_real = discriminator(targ)

    data_loss = loss_fns['GAN_discrimentator_loss'](scores_real, scores_fake)
    data_loss.backward()
    # if args.clipping_threshold_g > 0:
    #     nn.utils.clip_grad_norm_(
    #         generator.parameters(), args.clipping_threshold_g
    #     )

    # if t % 256 == 0:            # every 200 steps
    #     with torch.no_grad():
    #         # pick first weight tensor that requires_grad
    #         for name, p in discriminator.named_parameters():
    #             if p.grad is not None:
    #                 print(f"[{t:05d}] D grad ‖{name}‖ = {p.grad.abs().mean():.4e}")
    #                 break
                
    optimizer_d.step()
    losses['d_loss'] = data_loss.item()

    # if t % 256 == 0:
    #     print(f"[{t:05d}] "
    #           f"D(real)={scores_real.sigmoid().mean():.3f} "
    #           f"D(fake)={scores_fake.sigmoid().mean():.3f} "
    #           f"G_adv={data_loss.item():.3f}")
        # print(scores_fake.min().item(), scores_fake.max().item())   # should show both + and – values
        # print(scores_real.min().item(), scores_real.max().item())   # should show both + and – values


    return losses

def generator_step(batch, generator, discriminator, loss_fns, optimizer_g, Q_model, dts, target_type, t, use_map=True):

    losses = {}
    # traj_w, aux_w, feas_w = 5, 100.0, 0.5
    traj_w, feas_w, adv_w, phase = get_loss_weights(t)
    # traj_w = min(5.0, 5.0 * t / 2000)
    # feas_w = min(0.5, 0.5 * t / 2000)         # reaches 0.5 after 2 k iters

    loss = torch.zeros(1, device=DEVICE)
    feat, targ, initial_pos, traj_real, map_tensor, map_meta, ts = batch
    # dts = (ts[:,1:] - ts[:,:-1]).to(DEVICE)
    feat, targ, initial_pos, traj_real, map_tensor = feat.to(DEVICE), targ.to(DEVICE), initial_pos.to(DEVICE), traj_real.to(DEVICE), map_tensor.to(DEVICE)

    optimizer_g.zero_grad()
    pred_traj_fake_rel, attn_mask, attn_w, quantiles, imu_feat = generator(map_tensor, initial_pos, Q_model, feat, map_meta, t)
    pred_traj_fake = relative_to_abs(pred_traj_fake_rel, initial_pos, dts, target_type)

    g_l2_loss = loss_fns['GAN_generator_l2_loss'](pred_traj_fake_rel, targ)#, obstacle_loss)
    g_l2_loss = g_l2_loss / (pred_traj_fake_rel.shape[1] * pred_traj_fake_rel.shape[2])
    loss = traj_w * g_l2_loss                          # always present
    
    if adv_w > 0:                                        # Phase 1-3
        scores_fake = discriminator(pred_traj_fake_rel, quantiles, map_tensor)#, imu_feat)
        discriminator_loss = loss_fns['GAN_generator_loss'](scores_fake)
        loss += discriminator_loss
    else:
        discriminator_loss = torch.tensor(0.0, device=DEVICE)

    if feas_w > 0: # and args.use_map:                                     # Phase 2-3
        feas_loss = loss_fns['GAN_generator_feasibility_loss'](pred_traj_fake, map_tensor, map_meta)
        loss += feas_w * feas_loss
    else:
        feas_loss = torch.tensor(0.0, device=DEVICE)

    
    # scores_fake = discriminator(pred_traj_fake_rel)
    # discriminator_loss = loss_fns['GAN_generator_loss'](scores_fake)
    # feas_loss = loss_fns['GAN_generator_feasibility_loss'](pred_traj_fake_rel, map_tensor, map_meta)
    

    # loss = feas_w * feas_loss + traj_w * g_l2_loss + discriminator_loss
    losses['G_feas_loss'] = (feas_w * feas_loss).item()
    losses['G_l2_loss_rel'] = (traj_w*g_l2_loss).item()
    losses['G_discriminator_loss'] = discriminator_loss.item()

    losses['G_total_loss'] = loss.item()

    loss.backward()
    # nn.utils.clip_grad_norm_(generator.parameters(), 5.0)

    # if t % 256 == 0:            # every 200 steps
    #     with torch.no_grad():
    #         # pick first weight tensor that requires_grad
    #         for name, p in generator.named_parameters():
    #             if p.grad is not None:
    #                 print(f"[{t:05d}] G grad ‖{name}‖ = {p.grad.abs().mean():.4e}")
    #                 break
                    
    optimizer_g.step()
    # ema.update(generator)
    return losses

def check_accuracy(loader, generator, discriminator, metric_fn, Q_model, dts, target_type, t):
    metrics = {}
    d_losses = []
    g_traj_loss = []
    disp_error = []
    f_disp_error = []
    ds_score = []
    cr_score = []
    ate_score = []
    total_samples = 0
    generator.eval()
    with torch.no_grad():
        for batch in loader:
            feat, targ, initial_pos, traj_real, map_tensor, map_meta, ts = batch
            # dts = (ts[:,1:] - ts[:,:-1]).to(DEVICE)
            feat, targ, initial_pos, traj_real, map_tensor = feat.to(DEVICE), targ.to(DEVICE), initial_pos.to(DEVICE), traj_real.to(DEVICE), map_tensor.to(DEVICE)
            pred_traj_fake_rel, _, _, quantiles, imu_feat = generator(map_tensor, initial_pos, Q_model, feat, map_meta, t)
            pred_traj_fake = relative_to_abs(pred_traj_fake_rel, initial_pos, dts, target_type)

            # g_l2_loss_rel = metric_fn['ade'](pred_traj_fake_rel, targ)

            # ade = metric_fn['ade'](pred_traj_fake, traj_real)
            traj_loss = metric_fn['traj_loss'](pred_traj_fake_rel, targ)
            ate = metric_fn['ate'](pred_traj_fake, traj_real)
            fde = metric_fn['fde'](pred_traj_fake, traj_real)
            ds = metric_fn['ds'](pred_traj_fake, map_tensor, map_meta)
            cr = metric_fn['cr'](pred_traj_fake, map_tensor, map_meta)

            scores_fake = discriminator(pred_traj_fake_rel, quantiles, map_tensor)#, imu_feat)
            scores_real = discriminator(targ, quantiles, map_tensor)#, imu_feat)
            d_loss = metric_fn['bce'](scores_real, scores_fake)

            total_samples += pred_traj_fake.shape[0]
            d_losses.append(d_loss.item())
            g_traj_loss.append(traj_loss.item())
            disp_error.append(ate.sum().item())
            f_disp_error.append(fde.sum().item())
            ds_score.append(ds.sum().item())
            cr_score.append(cr.sum().item())

    metrics['d_loss'] = sum(d_losses) / len(d_losses)
    metrics['traj_loss'] = sum(g_traj_loss) / len(loader)
    # metrics['g_l2_loss_rel'] = sum(g_l2_losses_rel) / total_samples
    # metrics['ade'] = sum(disp_error) / total_samples
    metrics['fde'] = sum(f_disp_error) / total_samples
    metrics['ds'] = sum(ds_score) / total_samples
    metrics['cr'] = sum(cr_score) / total_samples
    metrics['ate'] = sum(disp_error) / total_samples

    generator.train()
    return metrics
    
def trainGAN(args, train_loader, val_loader, Q_model, GANtraining_epochs, loss_fns, dts):
    print(PHASES)
    d_steps = 3
    g_steps = 1
    g_model = Generator(args.input_channel, args)
    g_model = g_model.to(DEVICE)
    d_model = TrajectoryDiscriminator(args)
    d_model = d_model.to(DEVICE)
    g_model.train()
    d_model.train()
    # Define loss function and optimizer
    # gan_g_loss, gan_d_loss  = loss_fns['GAN_generator_loss'], loss_fns['GAN_discrimentator_loss']
    # gan_g_feas_loss = loss_fns['GAN_generator_feasibility_loss']
    # gan_g_aux_loss = loss_fns['GAN_generator_aux_attn_loss']
    g_optimizer = torch.optim.Adam(g_model.parameters(), lr=3e-4, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(d_model.parameters(), lr=1e-4, betas=(0.5, 0.999))
    # ema = EMA(g_model, decay=0.999)
    # def lr_curve(step,
    #          warmup=5_000,
    #          start=3e-4,            # LR_G initial (your current value)
    #          end=3e-5,
    #          total=PHASES["cum_boundaries"][-1]):   # last iter
    #     if step < warmup:
    #         return start * step / warmup
    #     progress = (step - warmup) / max(1, total - warmup)
    #     return start - progress * (start - end)

    # sched_G = torch.optim.lr_scheduler.LambdaLR(
    # g_optimizer,
    # lr_lambda=lambda s: lr_curve(s, start=3e-4, end=3e-5))

    # sched_D = torch.optim.lr_scheduler.LambdaLR(
    # d_optimizer,
    # lr_lambda=lambda s: lr_curve(s, start=1e-4, end=1e-5))

    # sched_G = torch.optim.lr_scheduler.LambdaLR(
    #     g_optimizer, lr_lambda=lambda s: lr_curve(s))

    # sched_D = torch.optim.lr_scheduler.LambdaLR(
    #     d_optimizer, lr_lambda=lambda s:       # keep 4× ratio
    #     (lr_curve(s) * 4) / 3e-4)


    # lr = 1e-4             # usually same LR for G & D
    
    # g_optimizer = torch.optim.Adam(g_model.parameters(), lr=lr, betas=(0.0, 0.9))
    # d_optimizer = torch.optim.Adam(d_model.parameters(), lr=2.5e-5, betas=(0.0, 0.9))
    # Training loop
    val_loader.dataset.transform = None
    # train_loader.dataset.transform = None
    def phase_transition_actions(t, prev=[-1]):
        phase = current_phase(t)
        if phase == prev[0]:
            return                      # already in this phase
        prev[0] = phase

        if phase == 0:                  # freeze D, keep Q_model frozen
            for p in d_model.parameters():  p.requires_grad = False
            # train_loader.dataset.enable_rotation = True
            
        elif phase == 1:                # unfreeze D
            train_loader.dataset.transform = None
            for p in d_model.parameters():  p.requires_grad = True
            # train_loader.dataset.enable_rotation = True
        elif phase == 2:                # rotation off, real map on
            train_loader.dataset.transform = None
        elif phase == 3:                # freeze D, unfreeze LSTM with 0.1× LR
            for p in d_model.parameters():  p.requires_grad = False
            # for p in Q_model.parameters():  p.requires_grad = True
            # base_lr = g_optimizer.param_groups[0]['lr']
            # g_optimizer.add_param_group(dict(params=Q_model.parameters(),
            #                                  lr=base_lr * 0.1))
            # for n, p in Q_model.named_parameters():
            #     if n.startswith(("linear1", "linear2")):  # only heads
            #         p.requires_grad = True
            # g_optimizer.add_param_group(
            # dict(params=(p for p in Q_model.parameters() if p.requires_grad),
            #  lr=3e-4 * 0.1))
            train_loader.dataset.transform = None

    iterations_per_epoch = len(train_loader)
    GANtraining_epochs = np.ceil(TOTAL_ITERATIONS / iterations_per_epoch)    # I_ep from Step-2

    # iterations_per_epoch = len(train_loader) / d_steps
    # num_iterations = int(iterations_per_epoch * GANtraining_epochs)
    print('There are {} iterations per epoch'.format(iterations_per_epoch))
    t, epoch = 0, 0
    print_every = int(iterations_per_epoch)
    # traj_w = 100.0
    # obstacle_dynamic_w = 10.0
    best_min_ate = np.inf
    best_min_ate_1 = np.inf
    reset_ade = True
    metric_fn = {'fde': final_displacement_error,
                 'bce': gan_d_loss,
                 'ds': get_distance_score,
                 'cr': get_collision_rate,
                'ate': absolute_trajectory_error,
                'traj_loss': TrajLoss(args)}
    history = {
    'iter':        [],
    'd_loss':      [],
    'g_loss':      [],
    'g_traj_loss': [],
    'g_aux_loss': [],
    'g_feas_loss': [],
    'ate_val': [],
    'ade_val':     [],
    'fde_val':     [],
    'ds_val':      [],
    'cr_val':      [],
    'ade_train':     [],
    'fde_train':     [],
    'ds_train':      [],
    'cr_train':      [],
    'traj_loss_eval': [],
    }
    while t < TOTAL_ITERATIONS:
        d_steps_left = d_steps
        g_steps_left = g_steps

        epoch += 1
        for batch in train_loader:
            phase_transition_actions(t)
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
                                          g_optimizer, Q_model, dts, args.target_type, t, use_map=args.use_map)
                history['g_loss'].append(losses_g['G_discriminator_loss'])
                history['g_traj_loss'].append(losses_g['G_l2_loss_rel'])
                history['g_feas_loss'].append(losses_g['G_feas_loss'])

                # history['g_feas_loss'].append(losses_g['G_feasibility_loss'])
                # history['g_aux_loss'].append(losses_g['G_aux_loss'])
                g_steps_left -= 1

            if d_steps_left > 0 or g_steps_left > 0:
                continue

            if t > 0 and t % print_every == 0:
                g_traj_scaled = losses_g['G_l2_loss_rel']          # already traj_w * ...
                g_adv = losses_g['G_discriminator_loss']
                ratio = g_traj_scaled / (g_adv + 1e-8)
                print(f"[{t:05d}] G_traj={g_traj_scaled:.3f}  "
                      f"G_adv={g_adv:.3f}  ratio={ratio:.2f}")
            
                np.savez_compressed(os.path.join(args.output_directory, 'train_history.npz'), **history)

                print('t = {} / {}'.format(t + 1, TOTAL_ITERATIONS))
                for k, v in sorted(losses_d.items()):
                    print('  [D] {}: {:.3f}'.format(k, v))
                for k, v in sorted(losses_g.items()):
                    print('  [G] {}: {:.3f}'.format(k, v))

                print('Checking stats on validation ...')
                # g_eval = deepcopy(g_model).to(DEVICE)
                # ema.copy_to(g_eval)
                metrics_val = check_accuracy(val_loader, g_model, d_model, metric_fn, Q_model, dts, args.target_type, t)
                # metrics_train = check_accuracy(train_loader, g_model, d_model, metric_fn, Q_model, dts, args.target_type, t)
                history['fde_val'].append(metrics_val['fde'])
                history['ds_val'] .append(metrics_val['ds'])
                history['cr_val'] .append(metrics_val['cr'])
                history['ate_val'].append(metrics_val['ate'])
                history['traj_loss_eval'].append(metrics_val['traj_loss'])

                # history['ade_train'].append(metrics_train['ade'])
                # history['fde_train'].append(metrics_train['fde'])
                # history['ds_train'].append(metrics_train['ds'])
                # history['cr_train'].append(metrics_train['cr'])
                for k, v in sorted(metrics_val.items()):
                    print('  [val] {}: {:.3f}'.format(k, v))

                # for k, v in sorted(metrics_train.items()):
                #     print('  [train] {}: {:.3f}'.format(k, v))
                phase = current_phase(t)
                if phase == 2 and metrics_val['traj_loss'] < best_min_ate:
                    best_min_ate = metrics_val['traj_loss']
                    # g_eval = deepcopy(g_model).to(DEVICE)
                    # ema.copy_to(g_eval)
                    if args.output_directory:
                        GAN_model_path = osp.join(args.output_directory, 'checkpoints', 'gan_checkpoint_ph1_%d.pt' % epoch)
                        torch.save({'g_best_state': g_model.state_dict(),
                                    'd_best_state': d_model.state_dict(),
                                    'best_t': t,
                                    'g_optim_state': g_optimizer.state_dict(),
                                    'd_optim_state': d_optimizer.state_dict()}, GAN_model_path)
                        print('Best Validation Model saved to ', GAN_model_path)

                if phase == 3 and metrics_val['traj_loss'] < best_min_ate_1:
                    best_min_ate_1 = metrics_val['traj_loss']
                    # g_eval = deepcopy(g_model).to(DEVICE)
                    # ema.copy_to(g_eval)
                    if args.output_directory:
                        GAN_model_path = osp.join(args.output_directory, 'checkpoints', 'gan_checkpoint_ph2_%d.pt' % epoch)
                        torch.save({'g_best_state': g_model.state_dict(),
                                    'd_best_state': d_model.state_dict(),
                                    'best_t': t,
                                    'g_optim_state': g_optimizer.state_dict(),
                                    'd_optim_state': d_optimizer.state_dict()}, GAN_model_path)
                        print('Best Validation Model saved to ', GAN_model_path)

            # sched_G.step()           # one call per iteration
            # sched_D.step()
            t += 1
            d_steps_left = d_steps
            g_steps_left = g_steps
            if t >= TOTAL_ITERATIONS:
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
    train_dataset = get_data_list(args.dataset_directory, args.train_list, args, mode=args.mode)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4, shuffle=True, drop_last=True, pin_memory=True)
    # if args.dataset == 'kaust':
    #     dts = (train_dataset.ts[0][1:] - train_dataset.ts[0][:-1])[:, None].mean()/1000
    # else:
    #     dts = (train_dataset.ts[0][1:] - train_dataset.ts[0][:-1])[:, None].mean()
    dts = (train_dataset.ts[0][1:] - train_dataset.ts[0][:-1])[:, None].mean()
    #if args.use_map:
     #   train_dist_map, boundries = get_map_data(train_dataset.gt_pos, args.sigma, args.map_size)

    val_dataset, val_loader = None, None
    if args.val_list is not None:
        val_dataset =get_data_list(args.dataset_directory, args.val_list, args, mode='val')
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=4, shuffle=True, drop_last=True, pin_memory=True)
        #if args.use_map:
         #   val_dist_map, boundries = get_map_data(val_dataset.gt_pos, args.sigma, args.map_size)
        print('Validation set loaded')

    print('\nNumber of train samples: {}'.format(len(train_dataset)))
    print('Number of val samples: {}'.format(len(val_dataset)))
    loss_fns = get_losses(args, dts)

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

def get_eval_model_type(args):
    if args.perturb and args.use_map:
        model_type = 'umgloc_perturb_' + str(args.pi) + '_' + str(args.noise_level)
    elif args.perturb and not args.use_map:
        model_type = 'umgloc_nomap_perturb_' + str(args.pi) + '_' + str(args.noise_level)
    elif args.use_map:
        model_type = 'umgloc'
    else:
        model_type = 'umgloc_nomap'
    return model_type
    
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
    seq_dataset = get_data_list(data_dir, args.test_list, args, mode=args.mode)



    gan_checkpoint = torch.load(args.gan_path)
    g_model = Generator(args.input_channel, args)

    g_model.load_state_dict(gan_checkpoint.get('g_best_state'))
    g_model.eval().to(args.device)
    print('Model {} loaded to device {}.'.format(args.gan_path, args.device))
    log_file = None
    if args.test_list and args.output_directory:
        log_file = osp.join(args.output_directory, osp.split(args.test_list)[-1].split('.')[0] + '_log.txt')
        with open(log_file, 'w') as f:
            f.write(args.lstm_path + '\n')
            f.write('Sequence Seq_length traj_error ATE RTE FDE DS CR\n')
    if args.dataset == 'kaust':
        fs = 60

    elif args.dataset == 'ronin':
        fs = 200
    else:
        fs = 100

    pred_per_min = fs * 60

    metric_fn = {'fde': final_displacement_error,
                'cr': get_collision_rate,
                'ate': absolute_trajectory_error,
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
            # if args.dataset == 'kaust':
            #     dts = (seq_dataset.ts[idx][1:] - seq_dataset.ts[idx][:-1])[:, None].mean()/1000
            # else:
            #     dts = (seq_dataset.ts[idx][1:] - seq_dataset.ts[idx][:-1])[:, None].mean()

            dts = (seq_dataset.ts[idx][1:] - seq_dataset.ts[idx][:-1])[:, None].mean()
            gt = seq_dataset.gt_pos[idx]
            feat, targ, map_tensor, map_meta = seq_dataset.get_test_seq(idx, gt)
            # dts = (seq_dataset.ts[idx][1:] - seq_dataset.ts[idx][:-1])[None, :]
            feat = torch.Tensor(feat).to(args.device)
            map_tensor = torch.Tensor(map_tensor).to(args.device)
            if args.perturb:
                feat = perturb_imu(feat, args.noise_level)
            initial_pos = gt[[0], :]

            initial_pos = torch.Tensor(initial_pos).to(args.device)
            gt_traj_real = torch.Tensor(gt).unsqueeze(0).to(args.device)
            targ = torch.Tensor(targ).unsqueeze(0).to(args.device)
            # dts = torch.Tensor(dts).to(args.device)

            best_fde = torch.inf
            all_traj = []
            model_name = get_eval_model_type(args)
            # breakpoint()
            for k in range(20):
                #breakpoint()
                pred_traj_fake_rel, _, _, quantiles, imu_feat = g_model(map_tensor, initial_pos, lstm_model, feat, map_meta, 1)
                pred_traj_fake = relative_to_abs(pred_traj_fake_rel, initial_pos, dts, args.target_type)
                all_traj.append(pred_traj_fake.squeeze(0).cpu().detach().numpy())
                # np.save(osp.join(args.output_directory, data + '_' + model_name + str(k) + '.npy'), traj_data)
                # pred_traj_real = relative_to_abs(targ, initial_pos, dts, args.target_type)
                
                delta_position = gt_traj_real[:, 1:, :] - gt_traj_real[:, :-1, :]
                delta_length = torch.norm(delta_position, dim=-1)
                moving_len = torch.sum(delta_length, dim=1)
    

                # breakpoint()
                fde = metric_fn['fde'](pred_traj_fake, gt_traj_real)
                if fde < best_fde:
                    best_trajetory = pred_traj_fake
                    best_fde = fde

            ds = metric_fn['ds'](pred_traj_fake, map_tensor, map_meta)
            cr = metric_fn['cr'](pred_traj_fake, map_tensor, map_meta)
            X_quantiles, Y_quantiles, _ = lstm_model(feat)

            X_quantiles = relative_to_abs(X_quantiles, initial_pos[:, [0]], dts, args.target_type)
            Y_quantiles = relative_to_abs(Y_quantiles, initial_pos[:, [1]], dts, args.target_type)
            
            best_fde = best_fde.item()
            ds = ds.item()
            cr = cr.item()

            gt_traj_real = gt_traj_real.squeeze(0).cpu().detach().numpy()
            # pred_traj_real = pred_traj_real.squeeze(0).cpu().detach().numpy()
            pred_traj = best_trajetory.squeeze(0).cpu().detach().numpy()
            X_quantiles = X_quantiles.squeeze(0).cpu().detach().numpy()
            Y_quantiles = Y_quantiles.squeeze(0).cpu().detach().numpy()

                

            fde_outer.append(best_fde)
            ds_all.append(ds)
            cr_all.append(cr)

            ate, rte = compute_ate_rte(pred_traj, gt_traj_real, pred_per_min)
            pos_losses = np.mean((gt_traj_real - pred_traj) ** 2, axis=0)
            pos_cum_error = np.linalg.norm(pred_traj - gt_traj_real, axis=1)
            if args.output_directory is not None and osp.isdir(args.output_directory):
                model_name = get_eval_model_type(args)
                np.save(osp.join(args.output_directory, data + '_' + model_name + '.npy'), pred_traj)
                np.save(osp.join(args.output_directory, data + '_gt.npy'), gt_traj_real)
                np.save(osp.join(args.output_directory, data + '_quantiles_' + model_name + '.npy'),
                        np.stack([X_quantiles, Y_quantiles], axis=-1))
                np.save(osp.join(args.output_directory, data + '_' + model_name + 'all.npy'), all_traj)
                # np.save(osp.join(args.output_directory, data + '_umgloc_error.npy'), pos_cum_error)
                            
            error = np.sort(pos_cum_error)
            cumulative_prob = np.arange(1, len(error) + 1) / len(error)
            log_line = format_string(data, moving_len, np.mean(pos_losses), ate, rte, best_fde, ds, cr)
            if log_file is not None:
                with open(log_file, 'a') as f:
                    log_line += '\n'
                    f.write(log_line)

            ate_all.append(ate)
            rte_all.append(rte)
            print('Sequence {}, Seq_length: {}, ATE: {}, RTE: {}, FDE: {}, DS: {}, CR: {}'.format(data, moving_len, ate, rte, best_fde, ds, cr))

            # Assuming the first column is 'x' and the second column is 'y' for both preds and gt
            time = np.linspace(0, len(pred_traj)//fs, len(pred_traj))
            plot_result(pred_traj, gt_traj_real, X_quantiles[:, 0], Y_quantiles[:, 0], X_quantiles[:, -1], Y_quantiles[:, -1], error, cumulative_prob, pos_cum_error, data, args)
    ate_all = np.array(ate_all)
    rte_all = np.array(rte_all)
    fde_outer = np.array(fde_outer)
    ds_all = np.array(ds_all)
    cr_all = np.array(cr_all)

    measure = format_string('ATE', 'RTE', 'FDE', 'DS', 'CR', sep='\t')
    values = format_string(np.mean(ate_all), np.mean(rte_all), np.mean(fde_outer), np.mean(ds_all), np.mean(cr_all), sep='\t')
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
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate for LSTM')
    parser.add_argument('--latent_dim', type=int, default=32, help='Latent dimension for the decoder in generator')
    parser.add_argument('--kernel_size', type=int, default=3, help='Kernel size for ConvLSTM in encoder')
    parser.add_argument('--input_channel', type=int, default=1, help='Kernel size for ConvLSTM in encoder')
    parser.add_argument('--noise_level', type=float, default=0.5, help='Kernel size for ConvLSTM in encoder')
    parser.add_argument('--pi', type=float, default=95, help='Kernel size for ConvLSTM in encoder')
    parser.add_argument('--drop_level', type=float, default=0.1, help='Kernel size for ConvLSTM in encoder')

    # Data argument
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--step_size', default=60, type=int) #
    parser.add_argument('--window_size', default=10*60, type=int) #
    parser.add_argument('--seq_len', default=10*60, type=int) #
    parser.add_argument('--map_size', type=int, default=256, help='Hidden dimension for LSTM/ConvLSTM')
    parser.add_argument('--sigma', default=2, type=int)
    parser.add_argument('--feat_sigma', default=0.001, type=float)
    parser.add_argument('--targ_sigma', default=0, type=float)

    # Design argumnet
    parser.add_argument('--target_type', default='global_vel', type=str, choices=['disp', 'vel', 'global_vel'])
    parser.add_argument('--dataset', type=str, default='kaust', choices=['rnin', 'ronin', 'kaust', 'idol'])
    parser.add_argument('--use_map', action='store_true')
    parser.add_argument('--perturb', action='store_true')
    parser.add_argument('--model_type', type=str, default='Q_LSTM_GAN', choices=['Q_LSTM', 'GAN', 'Q_LSTM_GAN'])


    mode = parser.add_subparsers(title='mode', dest='mode', help='Operation: [train] train model, [test] evaluate model')
    mode.required = True

    # Train argument
    train_cmd = mode.add_parser('train')
    train_cmd.add_argument('--train_list', type=str, default='lists/kaust/list_train.txt') #
    train_cmd.add_argument('--val_list', type=str, default='lists/kaust/list_val.txt') #
    train_cmd.add_argument('--continue_from', type=str, default=None)
    train_cmd.add_argument('--epochs', type=int, default=400)
    train_cmd.add_argument('--save_interval', type=int, default=20)
    train_cmd.add_argument('--lr', '--learning_rate', type=float, default=1e-3)
    train_cmd.add_argument('--lstm_path', type=str, default=None)
    parser.add_argument('--use_scheduler', action='store_true')

    # Test argument
    test_cmd = mode.add_parser('test')
    test_cmd.add_argument('--test_list', type=str, default='lists/kaust/test_list.txt') #
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
