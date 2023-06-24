import glob
import os
import subprocess
import sys
import argparse
import logging
import math

from copy import deepcopy
from threading import Thread

import numpy as np
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import yaml

from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import test  # import test.py to get mAP after each epoch
from models.experimental import attempt_load
from models.yolo import Model
from utils.autoanchor import check_anchors
from utils.datasets import create_dataloader
from utils.general import labels_to_class_weights, increment_path, labels_to_image_weights, init_seeds, \
    fitness, strip_optimizer, get_latest_run, check_dataset, check_file, check_img_size, \
    print_mutation, set_logging, one_cycle, colorstr
from utils.google_utils import attempt_download
from utils.loss import ComputeLoss, ComputeLossOTA
from utils.plots import plot_images, plot_results, plot_evolution
from utils.torch_utils import ModelEMA, select_device, intersect_dicts, torch_distributed_zero_first, is_parallel
from utils.wandb_logging.wandb_utils import WandbLogger, check_wandb_resume
# --------------------------------------------------------------------------


import time

import matplotlib.pyplot as plt
from PyQt5 import QtWidgets

from PyQt5.QtCore import pyqtSignal, QThread
from PyQt5.QtWidgets import QMainWindow, QFileDialog
from matplotlib.offsetbox import AnchoredText
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import main_ui
import random
import shutil
import globals as gl
from pathlib import Path
import FrontEnd
import thread_dialog
from Predict_view import PredictView


logger = logging.getLogger(__name__)


def train(hyp, opt, device, tb_writer=None):

    logger.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))
    save_dir, epochs, batch_size, total_batch_size, weights, rank, freeze = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.total_batch_size, opt.weights, opt.global_rank, opt.freeze

    # Directories 訓練後生成的檔案
    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)  # make dir
    last = wdir / 'last.pt'
    best = wdir / 'best.pt'
    results_file = save_dir / 'results.txt'

    # Save run settings 儲存運行設定
    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.dump(hyp, f, sort_keys=False)
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.dump(vars(opt), f, sort_keys=False)

    # Configure
    plots = not opt.evolve  # create plots
    cuda = device.type != 'cpu'
    init_seeds(2 + rank)
    with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)  # data dict
    is_coco = opt.data.endswith('coco.yaml')

    # Logging- Doing this before checking the dataset. Might update data_dict 在檢查數據集之前執行此操作。可能會更新 data_dict
    loggers = {'wandb': None}  # loggers dict
    if rank in [-1, 0]:
        opt.hyp = hyp  # add hyperparameters
        run_id = torch.load(weights, map_location=device).get('wandb_id') if weights.endswith('.pt') and os.path.isfile(weights) else None
        wandb_logger = WandbLogger(opt, Path(opt.save_dir).stem, run_id, data_dict)
        loggers['wandb'] = wandb_logger.wandb
        data_dict = wandb_logger.data_dict
        if wandb_logger.wandb:
            weights, epochs, hyp = opt.weights, opt.epochs, opt.hyp  # WandbLogger might update weights, epochs if resuming

    nc = 1 if opt.single_cls else int(data_dict['nc'])  # number of classes
    names = ['item'] if opt.single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names
    assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, opt.data)  # check

    # Model
    pretrained = weights.endswith('.pt')
    if pretrained:
        with torch_distributed_zero_first(rank):
            attempt_download(weights)  # download if not found locally
        ckpt = torch.load(weights, map_location=device)  # load checkpoint
        model = Model(opt.cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
        exclude = ['anchor'] if (opt.cfg or hyp.get('anchors')) and not opt.resume else []  # exclude keys
        state_dict = ckpt['model'].float().state_dict()  # to FP32
        state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)  # intersect
        model.load_state_dict(state_dict, strict=False)  # load

        logger.info('Transferred %g/%g items from %s' % (len(state_dict), len(model.state_dict()), weights))  # report
    else:
        model = Model(opt.cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
    with torch_distributed_zero_first(rank):
        check_dataset(data_dict)  # check
    train_path = data_dict['train']
    test_path = data_dict['val']

    # Freeze
    freeze = [f'model.{x}.' for x in (freeze if len(freeze) > 1 else range(freeze[0]))]  # parameter names to freeze (full or partial)
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        if any(x in k for x in freeze):
            print('freezing %s' % k)
            v.requires_grad = False

    # Optimizer 優化器
    nbs = 64  # nominal batch size 虛設
    accumulate = max(round(nbs / total_batch_size), 1)  # accumulate loss before optimizing 優化前累計損失
    hyp['weight_decay'] *= total_batch_size * accumulate / nbs  # scale weight_decay 權重衰變

    logger.info(f"Scaled weight_decay = {hyp['weight_decay']}")

    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in model.named_modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)  # biases
        if isinstance(v, nn.BatchNorm2d):
            pg0.append(v.weight)  # no decay
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)  # apply decay
        if hasattr(v, 'im'):
            if hasattr(v.im, 'implicit'):           
                pg0.append(v.im.implicit)
            else:
                for iv in v.im:
                    pg0.append(iv.implicit)
        if hasattr(v, 'imc'):
            if hasattr(v.imc, 'implicit'):           
                pg0.append(v.imc.implicit)
            else:
                for iv in v.imc:
                    pg0.append(iv.implicit)
        if hasattr(v, 'imb'):
            if hasattr(v.imb, 'implicit'):           
                pg0.append(v.imb.implicit)
            else:
                for iv in v.imb:
                    pg0.append(iv.implicit)
        if hasattr(v, 'imo'):
            if hasattr(v.imo, 'implicit'):           
                pg0.append(v.imo.implicit)
            else:
                for iv in v.imo:
                    pg0.append(iv.implicit)
        if hasattr(v, 'ia'):
            if hasattr(v.ia, 'implicit'):           
                pg0.append(v.ia.implicit)
            else:
                for iv in v.ia:
                    pg0.append(iv.implicit)
        if hasattr(v, 'attn'):
            if hasattr(v.attn, 'logit_scale'):   
                pg0.append(v.attn.logit_scale)
            if hasattr(v.attn, 'q_bias'):   
                pg0.append(v.attn.q_bias)
            if hasattr(v.attn, 'v_bias'):  
                pg0.append(v.attn.v_bias)
            if hasattr(v.attn, 'relative_position_bias_table'):  
                pg0.append(v.attn.relative_position_bias_table)
        if hasattr(v, 'rbr_dense'):
            if hasattr(v.rbr_dense, 'weight_rbr_origin'):  
                pg0.append(v.rbr_dense.weight_rbr_origin)
            if hasattr(v.rbr_dense, 'weight_rbr_avg_conv'): 
                pg0.append(v.rbr_dense.weight_rbr_avg_conv)
            if hasattr(v.rbr_dense, 'weight_rbr_pfir_conv'):  
                pg0.append(v.rbr_dense.weight_rbr_pfir_conv)
            if hasattr(v.rbr_dense, 'weight_rbr_1x1_kxk_idconv1'): 
                pg0.append(v.rbr_dense.weight_rbr_1x1_kxk_idconv1)
            if hasattr(v.rbr_dense, 'weight_rbr_1x1_kxk_conv2'):   
                pg0.append(v.rbr_dense.weight_rbr_1x1_kxk_conv2)
            if hasattr(v.rbr_dense, 'weight_rbr_gconv_dw'):   
                pg0.append(v.rbr_dense.weight_rbr_gconv_dw)
            if hasattr(v.rbr_dense, 'weight_rbr_gconv_pw'):   
                pg0.append(v.rbr_dense.weight_rbr_gconv_pw)
            if hasattr(v.rbr_dense, 'vector'):   
                pg0.append(v.rbr_dense.vector)

    if opt.adam:
        optimizer = optim.Adam(pg0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
    else:
        optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)

    logger.info('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
    del pg0, pg1, pg2

    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#OneCycleLR
    if opt.linear_lr:
        lf = lambda x: (1 - x / (epochs - 1)) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
    else:
        lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    # plot_lr_scheduler(optimizer, scheduler, epochs)

    # EMA
    ema = ModelEMA(model) if rank in [-1, 0] else None

    # Resume
    start_epoch, best_fitness = 0, 0.0
    if pretrained:
        # Optimizer
        if ckpt['optimizer'] is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
            best_fitness = ckpt['best_fitness']

        # EMA
        if ema and ckpt.get('ema'):
            ema.ema.load_state_dict(ckpt['ema'].float().state_dict())
            ema.updates = ckpt['updates']

        # Results
        if ckpt.get('training_results') is not None:
            results_file.write_text(ckpt['training_results'])  # write results.txt

        # Epochs
        start_epoch = ckpt['epoch'] + 1
        if opt.resume:
            assert start_epoch > 0, '%s training to %g epochs is finished, nothing to resume.' % (weights, epochs)
        if epochs < start_epoch:
            logger.info('%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' %
                        (weights, ckpt['epoch'], epochs))
            epochs += ckpt['epoch']  # finetune additional epochs 微調額外的次數

        del ckpt, state_dict

    # Image sizes
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    nl = model.model[-1].nl  # number of detection layers (used for scaling hyp['obj'])
    imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.img_size]  # verify imgsz are gs-multiples

    # DP mode
    if cuda and rank == -1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # SyncBatchNorm
    if opt.sync_bn and cuda and rank != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        logger.info('Using SyncBatchNorm()')

    # Trainloader
    dataloader, dataset = create_dataloader(train_path, imgsz, batch_size, gs, opt,
                                            hyp=hyp, augment=True, cache=opt.cache_images, rect=opt.rect, rank=rank,
                                            world_size=opt.world_size, workers=opt.workers,
                                            image_weights=opt.image_weights, quad=opt.quad, prefix=colorstr('train: '))
    mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  # max label class
    nb = len(dataloader)  # number of batches
    assert mlc < nc, 'Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g' % (mlc, nc, opt.data, nc - 1)

    # Process 0
    if rank in [-1, 0]:
        testloader = create_dataloader(test_path, imgsz_test, batch_size * 2, gs, opt,  # testloader
                                       hyp=hyp, cache=opt.cache_images and not opt.notest, rect=True, rank=-1,
                                       world_size=opt.world_size, workers=opt.workers,
                                       pad=0.5, prefix=colorstr('val: '))[0]

        if not opt.resume:
            labels = np.concatenate(dataset.labels, 0)
            c = torch.tensor(labels[:, 0])  # classes
            # cf = torch.bincount(c.long(), minlength=nc) + 1.  # frequency
            # model._initialize_biases(cf.to(device))
            if plots:
                #plot_labels(labels, names, save_dir, loggers)
                if tb_writer:
                    tb_writer.add_histogram('classes', c, 0)

            # Anchors
            if not opt.noautoanchor:
                check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)
            model.half().float()  # pre-reduce anchor precision

    # DDP mode
    if cuda and rank != -1:
        model = DDP(model, device_ids=[opt.local_rank], output_device=opt.local_rank,
                    # nn.MultiheadAttention incompatibility with DDP https://github.com/pytorch/pytorch/issues/26698
                    find_unused_parameters=any(isinstance(layer, nn.MultiheadAttention) for layer in model.modules()))

    # Model parameters
    hyp['box'] *= 3. / nl  # scale to layers
    hyp['cls'] *= nc / 80. * 3. / nl  # scale to classes and layers
    hyp['obj'] *= (imgsz / 640) ** 2 * 3. / nl  # scale to image size and layers
    hyp['label_smoothing'] = opt.label_smoothing
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
    model.names = names

    # Start training
    t0 = time.time()
    nw = max(round(hyp['warmup_epochs'] * nb), 1000)  # number of warmup iterations, max(3 epochs, 1k iterations)
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler = amp.GradScaler(enabled=cuda)
    compute_loss_ota = ComputeLossOTA(model)  # init loss class
    compute_loss = ComputeLoss(model)  # init loss class

    logger.info(f'Image sizes {imgsz} train, {imgsz_test} test\n'
                f'Using {dataloader.num_workers} dataloader workers\n'
                f'Logging results to {save_dir}\n'
                f'Starting training for {epochs} epochs...')
    torch.save(model, wdir / 'init.pt')
    # 調整記憶體使用量比例
    # torch.cuda.set_per_process_memory_fraction(0.9, device=None)

    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------

        model.train()
        # Update image weights (optional)
        if opt.image_weights:
            # Generate indices
            if rank in [-1, 0]:
                cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # class weights
                iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # image weights
                dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # rand weighted idx
            # Broadcast if DDP
            if rank != -1:
                indices = (torch.tensor(dataset.indices) if rank == 0 else torch.zeros(dataset.n)).int()
                dist.broadcast(indices, 0)
                if rank != 0:
                    dataset.indices = indices.cpu().numpy()

        # Update mosaic border
        # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
        # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders
        # 創建 mloss
        mloss = torch.zeros(4, device=device)  # mean losses

        if rank != -1:
            dataloader.sampler.set_epoch(epoch)
        pbar = enumerate(dataloader)

        logger.info(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'total', 'labels', 'img_size'))
        if rank in [-1, 0]:
            pbar = tqdm(pbar, total=nb)  # progress bar
        optimizer.zero_grad()

        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------

            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0

            # Warmup 暖身
            if ni <= nw:
                xi = [0, nw]  # x interp
                # model.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / total_batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            # Multi-scale
            if opt.multi_scale:
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                    imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            # Forward
            with amp.autocast(enabled=cuda):
                pred = model(imgs)  # forward
                if 'loss_ota' not in hyp or hyp['loss_ota'] == 1:
                    loss, loss_items = compute_loss_ota(pred, targets.to(device), imgs)  # loss scaled by batch_size
                else:
                    loss, loss_items = compute_loss(pred, targets.to(device))  # loss scaled by batch_size
                                                                                # 損失按照 batch_size 縮放
                if rank != -1:
                    loss *= opt.world_size  # gradient averaged between devices in DDP mode
                                            # DDP 模式下設備之間的梯度平均
                if opt.quad:
                    loss *= 4.

            # Backward
            scaler.scale(loss).backward()

            # Optimize
            if ni % accumulate == 0:
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)

            # Print
            if rank in [-1, 0]:

                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses  [tensor_array]
                # print(f'更新損失總值{loss_value}')

                global win
                loss_value = float((list(mloss.cpu().numpy())[3]))
                time.sleep(0.1)
                win.SIGNAL_LOSS.emit(loss_value)
                # print(f'損失函數{loss_value}')
                # 記憶體內存保留量
                mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)

                s = ('%10s' * 2 + '%10.4g' * 6) % (
                    '%g/%g' % (epoch, epochs - 1), mem, *mloss, targets.shape[0], imgs.shape[-1])

                pbar.set_description(s)

                # Plot
                if plots and ni < 10:
                    f = save_dir / f'train_batch{ni}.jpg'  # filename
                    Thread(target=plot_images, args=(imgs, targets, paths, f), daemon=True).start()
                    # if tb_writer:
                    #     tb_writer.add_image(f, result, dataformats='HWC', global_step=epoch)
                    #     tb_writer.add_graph(torch.jit.trace(model, imgs, strict=False), [])  # add model graph
                elif plots and ni == 10 and wandb_logger.wandb:
                    wandb_logger.log({"Mosaics": [wandb_logger.wandb.Image(str(x), caption=x.name) for x in
                                                  save_dir.glob('train*.jpg') if x.exists()]})

            # end batch ------------------------------------------------------------------------------------------------
        # end epoch ----------------------------------------------------------------------------------------------------

        # Scheduler 調度程序
        lr = [x['lr'] for x in optimizer.param_groups]  # for tensorboard
        scheduler.step()

        # DDP process 0 or single-GPU
        if rank in [-1, 0]:
            # mAP
            ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride', 'class_weights'])
            final_epoch = epoch + 1 == epochs
            if not opt.notest or final_epoch:  # Calculate mAP
                wandb_logger.current_epoch = epoch + 1
                results, maps, times = test.test(data_dict,
                                                 batch_size=batch_size * 2,
                                                 imgsz=imgsz_test,
                                                 model=ema.ema,
                                                 single_cls=opt.single_cls,
                                                 dataloader=testloader,
                                                 save_dir=save_dir,
                                                 verbose=nc < 50 and final_epoch,
                                                 plots=plots and final_epoch,
                                                 wandb_logger=wandb_logger,
                                                 compute_loss=compute_loss,
                                                 is_coco=is_coco,
                                                 v5_metric=opt.v5_metric)

            # Write
            with open(results_file, 'a') as f:
                f.write(s + '%10.4g' * 7 % results + '\n')  # append metrics, val_loss
            if len(opt.name) and opt.bucket:
                os.system('gsutil cp %s gs://%s/results/results%s.txt' % (results_file, opt.bucket, opt.name))

            # Log
            tags = ['train/box_loss', 'train/obj_loss', 'train/cls_loss',  # train loss
                    'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95',
                    'val/box_loss', 'val/obj_loss', 'val/cls_loss',  # val loss
                    'x/lr0', 'x/lr1', 'x/lr2']  # params

            for x, tag in zip(list(mloss[:-1]) + list(results) + lr, tags):

                if tb_writer:
                    tb_writer.add_scalar(tag, x, epoch)  # tensorboard
                if wandb_logger.wandb:
                    wandb_logger.log({tag: x})  # W&B

            # Update best mAP
            fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            if fi > best_fitness:
                best_fitness = fi
            wandb_logger.end_epoch(best_result=best_fitness == fi)

            # Save model
            if (not opt.nosave) or (final_epoch and not opt.evolve):  # if save
                ckpt = {'epoch': epoch,
                        'best_fitness': best_fitness,
                        'training_results': results_file.read_text(),
                        'model': deepcopy(model.module if is_parallel(model) else model).half(),
                        'ema': deepcopy(ema.ema).half(),
                        'updates': ema.updates,
                        'optimizer': optimizer.state_dict(),
                        'wandb_id': wandb_logger.wandb_run.id if wandb_logger.wandb else None}

                # Save last, best and delete
                torch.save(ckpt, last)
                if best_fitness == fi:
                    torch.save(ckpt, best)
                if (best_fitness == fi) and (epoch >= 200):
                    torch.save(ckpt, wdir / 'best_{:03d}.pt'.format(epoch))
                if epoch == 0:
                    torch.save(ckpt, wdir / 'epoch_{:03d}.pt'.format(epoch))
                elif ((epoch+1) % 25) == 0:
                    torch.save(ckpt, wdir / 'epoch_{:03d}.pt'.format(epoch))
                elif epoch >= (epochs-5):
                    torch.save(ckpt, wdir / 'epoch_{:03d}.pt'.format(epoch))
                if wandb_logger.wandb:
                    if ((epoch + 1) % opt.save_period == 0 and not final_epoch) and opt.save_period != -1:
                        wandb_logger.log_model(
                            last.parent, opt, epoch, fi, best_model=best_fitness == fi)
                del ckpt

        # end epoch ----------------------------------------------------------------------------------------------------
    # end training
    if rank in [-1, 0]:
        # Plots
        if plots:
            plot_results(save_dir=save_dir)  # save as results.png
            if wandb_logger.wandb:
                files = ['results.png', 'confusion_matrix.png', *[f'{x}_curve.png' for x in ('F1', 'PR', 'P', 'R')]]
                wandb_logger.log({"Results": [wandb_logger.wandb.Image(str(save_dir / f), caption=f) for f in files
                                              if (save_dir / f).exists()]})
        # Test best.pt

        logger.info('%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, (time.time() - t0) / 3600))
        if opt.data.endswith('coco.yaml') and nc == 80:  # if COCO
            for m in (last, best) if best.exists() else (last):  # speed, mAP tests
                results, _, _ = test.test(opt.data,
                                          batch_size=batch_size * 2,
                                          imgsz=imgsz_test,
                                          conf_thres=0.001,
                                          iou_thres=0.7,
                                          model=attempt_load(m, device).half(),
                                          single_cls=opt.single_cls,
                                          dataloader=testloader,
                                          save_dir=save_dir,
                                          save_json=True,
                                          plots=False,
                                          is_coco=is_coco,
                                          v5_metric=opt.v5_metric)

        # Strip optimizers
        final = best if best.exists() else last  # final model
        for f in last, best:
            if f.exists():
                strip_optimizer(f)  # strip optimizers
        if opt.bucket:
            os.system(f'gsutil cp {final} gs://{opt.bucket}/weights')  # upload
        if wandb_logger.wandb and not opt.evolve:  # Log the stripped model
            wandb_logger.wandb.log_artifact(str(final), type='model',
                                            name='run_' + wandb_logger.wandb_run.id + '_model',
                                            aliases=['last', 'best', 'stripped'])
        wandb_logger.finish_run()
    else:
        dist.destroy_process_group()
    torch.cuda.empty_cache()  # ---------------------
    return results


def yolo_task():

    parser = argparse.ArgumentParser()
    # parser.add_argument('--weights', type=str, default='yolov7.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    # parser.add_argument('--data', type=str, default='data/coco.yaml', help='data.yaml path')
    parser.add_argument('--hyp', type=str, default='data/hyp.scratch.p5.yaml', help='hyperparameters path')
    # parser.add_argument('--epochs', type=int, default=300)
    # parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='[train, test] image sizes')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    # parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--workers', type=int, default=0, help='maximum number of dataloader workers')
    # parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--entity', default=None, help='W&B entity')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--linear-lr', action='store_true', help='linear LR')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--upload_dataset', action='store_true', help='Upload dataset as W&B artifact table')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='Set bounding-box image logging interval for W&B')
    parser.add_argument('--save_period', type=int, default=-1, help='Log model after every "save_period" epoch')
    parser.add_argument('--artifact_alias', type=str, default="latest", help='version of dataset artifact to be used')
    parser.add_argument('--freeze', nargs='+', type=int, default=[0], help='Freeze layers: backbone of yolov7=50, first3=0 1 2')
    parser.add_argument('--v5-metric', action='store_true', help='assume maximum recall as 1.0 in AP calculation')

    opt = parser.parse_args()

    opt.project = gl.PROJECT_FOLDER
    save_run_dir = os.path.join(gl.PROJECT_FOLDER, 'runs/train')
    opt.weights = gl.TRAIN_WEIGHTS
    opt.data = gl.DATA
    opt.img_size = gl.TRAIN_IMG_SIZE
    opt.batch_size = gl.TRAIN_BATCH_SIZE
    opt.epochs = gl.TRAIN_EPOCHS
    opt.device = gl.TRAIN_DEVICE

    # Set DDP variables
    opt.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1

    opt.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1
    set_logging(opt.global_rank)
    #if opt.global_rank in [-1, 0]:
    #    check_git_status()
    #    check_requirements()

    # Resume 恢復
    wandb_run = check_wandb_resume(opt)
    if opt.resume and not wandb_run:  # resume an interrupted run
        ckpt = opt.resume if isinstance(opt.resume, str) else get_latest_run()  # specified or most recent path 指定或最近的路徑
        assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'
        apriori = opt.global_rank, opt.local_rank
        with open(Path(ckpt).parent.parent / 'opt.yaml') as f:
            opt = argparse.Namespace(**yaml.load(f, Loader=yaml.SafeLoader))  # replace
        opt.cfg, opt.weights, opt.resume, opt.batch_size, opt.global_rank, opt.local_rank = '', ckpt, True, opt.total_batch_size, *apriori  # reinstate 恢復
        logger.info('Resuming training from %s' % ckpt)
    else:
        # opt.hyp = opt.hyp or ('hyp.finetune.yaml' if opt.weights else 'hyp.scratch.yaml')
        opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_file(opt.cfg), check_file(opt.hyp)  # check files
        assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
        opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))  # extend to 2 sizes (train, test)
        opt.name = 'evolve' if opt.evolve else opt.name
        opt.save_dir = increment_path(Path(save_run_dir) / opt.name, exist_ok=opt.exist_ok | opt.evolve) # increment run

    # DDP mode
    opt.total_batch_size = opt.batch_size

    device = select_device(opt.device, batch_size=opt.batch_size)
    if opt.local_rank != -1:
        assert torch.cuda.device_count() > opt.local_rank
        torch.cuda.set_device(opt.local_rank)
        device = torch.device('cuda', opt.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')  # distributed backend 分布式後端
        # 批量大小必須是 CUDA 設備數的倍數
        assert opt.batch_size % opt.world_size == 0, '--batch-size must be multiple of CUDA device count'
        opt.batch_size = opt.total_batch_size // opt.world_size

    # Hyperparameters
    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps

    # Train
    logger.info(opt)
    if not opt.evolve:
        tb_writer = None  # init loggers
        if opt.global_rank in [-1, 0]:
            prefix = colorstr('tensorboard: ')
            
            logger.info(f"{prefix}Start with 'tensorboard --logdir {opt.project}', view at http://localhost:6006/")
            tb_writer = SummaryWriter(opt.save_dir)  # Tensorboard
        train(hyp, opt, device, tb_writer)

    # Evolve hyperparameters (optional)
    else:
        # Hyperparameter evolution metadata (mutation scale 0-1, lower_limit, upper_limit)
        meta = {'lr0': (1, 1e-5, 1e-1),  # initial learning rate (SGD=1E-2, Adam=1E-3)
                'lrf': (1, 0.01, 1.0),  # final OneCycleLR learning rate (lr0 * lrf)
                'momentum': (0.3, 0.6, 0.98),  # SGD momentum/Adam beta1
                'weight_decay': (1, 0.0, 0.001),  # optimizer weight decay
                'warmup_epochs': (1, 0.0, 5.0),  # warmup epochs (fractions ok)
                'warmup_momentum': (1, 0.0, 0.95),  # warmup initial momentum
                'warmup_bias_lr': (1, 0.0, 0.2),  # warmup initial bias lr
                'box': (1, 0.02, 0.2),  # box loss gain
                'cls': (1, 0.2, 4.0),  # cls loss gain
                'cls_pw': (1, 0.5, 2.0),  # cls BCELoss positive_weight
                'obj': (1, 0.2, 4.0),  # obj loss gain (scale with pixels)
                'obj_pw': (1, 0.5, 2.0),  # obj BCELoss positive_weight
                'iou_t': (0, 0.1, 0.7),  # IoU training threshold
                'anchor_t': (1, 2.0, 8.0),  # anchor-multiple threshold
                'anchors': (2, 2.0, 10.0),  # anchors per output grid (0 to ignore)
                'fl_gamma': (0, 0.0, 2.0),  # focal loss gamma (efficientDet default gamma=1.5)
                'hsv_h': (1, 0.0, 0.1),  # image HSV-Hue augmentation (fraction)
                'hsv_s': (1, 0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
                'hsv_v': (1, 0.0, 0.9),  # image HSV-Value augmentation (fraction)
                'degrees': (1, 0.0, 45.0),  # image rotation (+/- deg)
                'translate': (1, 0.0, 0.9),  # image translation (+/- fraction)
                'scale': (1, 0.0, 0.9),  # image scale (+/- gain)
                'shear': (1, 0.0, 10.0),  # image shear (+/- deg)
                'perspective': (0, 0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
                'flipud': (1, 0.0, 1.0),  # image flip up-down (probability)
                'fliplr': (0, 0.0, 1.0),  # image flip left-right (probability)
                'mosaic': (1, 0.0, 1.0),  # image mixup (probability)
                'mixup': (1, 0.0, 1.0),   # image mixup (probability)
                'copy_paste': (1, 0.0, 1.0),  # segment copy-paste (probability)
                'paste_in': (1, 0.0, 1.0)}    # segment copy-paste (probability)
        
        with open(opt.hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
            if 'anchors' not in hyp:  # anchors commented in hyp.yaml
                hyp['anchors'] = 3
                
        assert opt.local_rank == -1, 'DDP mode not implemented for --evolve'
        opt.notest, opt.nosave = True, True  # only test/save final epoch
        # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # evolvable indices
        yaml_file = Path(opt.save_dir) / 'hyp_evolved.yaml'  # save best result here
        if opt.bucket:
            os.system('gsutil cp gs://%s/evolve.txt .' % opt.bucket)  # download evolve.txt if exists

        for _ in range(300):  # generations to evolve
            if Path('evolve.txt').exists():  # if evolve.txt exists: select best hyps and mutate
                # Select parent(s)
                parent = 'single'  # parent selection method: 'single' or 'weighted'
                x = np.loadtxt('evolve.txt', ndmin=2)
                n = min(5, len(x))  # number of previous results to consider
                x = x[np.argsort(-fitness(x))][:n]  # top n mutations
                w = fitness(x) - fitness(x).min()  # weights
                if parent == 'single' or len(x) == 1:
                    # x = x[random.randint(0, n - 1)]  # random selection
                    x = x[random.choices(range(n), weights=w)[0]]  # weighted selection
                elif parent == 'weighted':
                    x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination

                # Mutate
                mp, s = 0.8, 0.2  # mutation probability, sigma
                npr = np.random
                npr.seed(int(time.time()))
                g = np.array([x[0] for x in meta.values()])  # gains 0-1
                ng = len(meta)
                v = np.ones(ng)
                while all(v == 1):  # mutate until a change occurs (prevent duplicates)
                    v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)

                for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
                    hyp[k] = float(x[i + 7] * v[i])  # mutate

            # Constrain to limits 約束極限
            for k, v in meta.items():
                hyp[k] = max(hyp[k], v[1])  # lower limit
                hyp[k] = min(hyp[k], v[2])  # upper limit
                hyp[k] = round(hyp[k], 5)  # significant digits  有效數字

            # Train mutation 突變
            results = train(hyp.copy(), opt, device)

            # Write mutation results
            print_mutation(hyp.copy(), results, yaml_file, opt.bucket)

        # Plot results
        plot_evolution(yaml_file)
        print(f'Hyperparameter evolution complete. Best results saved as: {yaml_file}\n'
              f'Command to train a new model with these hyperparameters: $ python train.py --hyp {yaml_file}')


'''檔案前處理'''
def io_convert():
    source_img = []
    source_file = []
    ext_count = {}
    ext_filenames = {}
    print('創建訓練集與驗證集資料夾')
    gl.TRAIN_FOLDER = os.path.join(gl.PROJECT_FOLDER, 'train')
    gl.VAL_FOLDER = os.path.join(gl.PROJECT_FOLDER, 'val')
    Path(gl.TRAIN_FOLDER).mkdir(parents=True, exist_ok=True)
    Path(gl.VAL_FOLDER).mkdir(parents=True, exist_ok=True)
    # 創建 train,val 的 images 資料夾
    train_imgsf = os.path.join(gl.TRAIN_FOLDER, 'images')
    val_imgsf = os.path.join(gl.VAL_FOLDER, 'images')
    Path(train_imgsf).mkdir(parents=True, exist_ok=True)
    Path(val_imgsf).mkdir(parents=True, exist_ok=True)
    # 創建 train,val 的 labels 資料夾
    train_lbsf = os.path.join(gl.TRAIN_FOLDER, 'labels')
    val_lbsf = os.path.join(gl.VAL_FOLDER, 'labels')
    Path(train_lbsf).mkdir(parents=True, exist_ok=True)
    Path(val_lbsf).mkdir(parents=True, exist_ok=True)
    # --------------------------------------------------------------------------------------
    cls_path = os.path.join(gl.PROJECT_FOLDER, 'classes.txt')

    source_img = [os.path.splitext(file)[0] for file in os.listdir(gl.PROJECT_FOLDER) if
                  os.path.splitext(file)[1] in ['.jpg', '.png', '.jpeg']]
    source_file = [os.path.splitext(file)[0] for file in os.listdir(gl.PROJECT_FOLDER) if
                   os.path.splitext(file)[1] in ['.txt']]
    files = set(source_img) & set(source_file)

    cls_dict = {}
    for file in files:
        # 將 classes.txt 忽略
        if file != 'classes':
            # 開啟將符合訓練的群體，開啟它們的txt
            with open(os.path.join(gl.PROJECT_FOLDER, file + '.txt'), 'r', encoding='utf-8') as f:

                cls_num = set()
                for line in f.readlines():
                    line = line.strip('')
                    cls_num.add(line[0])  # 將每行字首放進集合中
                # 排序 set 讓字首的類別代碼有順序性
                num_list = sorted(cls_num)

                cls = ''  # 宣告空的字串來連接類別代號
                # 拼接字首
                for word in num_list:
                    cls += word
                # 判斷類別名稱是否存在字典的 key 中
                if cls in cls_dict.keys():
                    cls_dict[cls]['num'] += 1  # 類計class數量
                    cls_dict[cls]['img_list'].append(file)  # 將檔案名稱放進陣列中
                # 否則創建字典
                else:
                    cls_dict[cls] = {'num': 1, 'img_list': [file]}

    # content = ''
    '''取得每個類別的影像數量與影像陣列內容'''
    for key, value in cls_dict.items():

        # 影像的數量
        cls_amount = cls_dict[key]['num']
        # 影像的檔名
        imgnames = cls_dict[key]['img_list']

        img_list = range(cls_amount)
        # 比重   =  數量 * 比例(假設 90 / 100)
        train_percent = int(cls_amount * gl.PERCENT / 100)

        # 訓練數組 = 亂數選取多個檔案(數量範圍內 , 取得多少張)
        train_num = random.sample(img_list, train_percent)

        # content += key + '  :  ' + str(train_percent) + '\n'

        for idx in img_list:

            name = imgnames[idx]
            # 判斷陣列的數組是否有在訓練數組中
            if idx in train_num:
                # 將索引的數組名稱加上副檔名
                shutil.copyfile(os.path.join(gl.PROJECT_FOLDER, name + '.png'),
                                os.path.join(train_imgsf, name + '.png'))
                shutil.copyfile(os.path.join(gl.PROJECT_FOLDER, name + '.txt'),
                                os.path.join(train_lbsf, name + '.txt'))
            else:
                # 剩下的給 val
                shutil.copyfile(os.path.join(gl.PROJECT_FOLDER, name + '.png'),
                                os.path.join(val_imgsf, name + '.png'))
                shutil.copyfile(os.path.join(gl.PROJECT_FOLDER, name + '.txt'),
                                os.path.join(val_lbsf, name + '.txt'))

    # 打開 labelimg 生成的 classes.txt
    cls_list = []
    with open(cls_path, 'r') as file:
        for line in file.readlines():
            # 暫存類別名稱
            cls_list.append(line.strip('\n'))
    # 將路徑反斜線 取代成 單斜線
    train_img_dir = str(train_imgsf).replace('\\', '/')
    val_img_dir = str(val_imgsf).replace('\\', '/')

    # yaml 的內容
    data = 'train' + ': ' + train_img_dir + '\n' \
           + 'val' + ': ' + val_img_dir + '\n' \
           + 'nc' + ': ' + str(len(cls_list)) + '\n' \
           + 'names' + ': ' + str(cls_list)

    # 在專案資料夾內創建 data 資料夾
    train_data = os.path.join(gl.PROJECT_FOLDER, 'data')
    Path(train_data).mkdir(parents=True, exist_ok=True)
    # 創建 yaml 檔案
    gl.DATA = os.path.join(train_data, 'data.yaml')

    with open(gl.DATA, 'w') as file:
        file.write(data)

    print('將數據寫入yaml')
    del cls_list, cls_path, train_img_dir, val_img_dir, train_data


'''訓練的執行緒'''
class ThreadTrain(QThread):
    SIGNAL_FINISHED = pyqtSignal()
    SIGNAL_OPEN_DIALOG = pyqtSignal(str)

    def __init__(self, main):
        super(ThreadTrain, self).__init__()
        self.main = main
        self.running = True

        self.thread_dialog = None
        self.SIGNAL_FINISHED.connect(self.main.finish)
        self.SIGNAL_OPEN_DIALOG.connect(self.open_err)
        self.main.SIGNAL_LOSS.connect(self.main.display_chart)

    def run(self):
        self.running = True  # 重置 running 狀態
        if self.running is True:
            print('訓練start')
            try:
                io_convert()
                yolo_task()
                self.SIGNAL_FINISHED.emit()
                print('信號發送')

            except Exception as e:
                self.SIGNAL_OPEN_DIALOG.emit(str(e))  # 傳遞信號給dialog顯示

    # 訓練被中斷，開啟Dialog執行緒
    def open_err(self, msg):
        self.running = False
        self.thread_dialog = thread_dialog.ThreadDialog(msg)
        self.thread_dialog.start()


class TabWidget(QMainWindow, main_ui.Ui_MainWindow):
    SIGNAL_LOSS = pyqtSignal(float)
    SIGNAL_PROCESS_BAR = pyqtSignal(int)

    def __init__(self):
        super(TabWidget, self).__init__()
        self.setupUi(self)
        self.btn_project.clicked.connect(self.open_project)
        # 訓練傳出的 LOSS 信號
        self.SIGNAL_LOSS.connect(self.display_chart)
        self.SIGNAL_PROCESS_BAR.connect(self.update_process)
        # 實力化 python

        self.thread_dialog = None
        self.setClose = False
        self.predict_view = PredictView(self)
        self.thread_train = ThreadTrain(self)
        # 將 predict_view 加入 main layout中
        self.predictLayout.addWidget(self.predict_view)

        # 設置初始頁面
        self.tab_widget.setCurrentIndex(0)
        self.tab_widget.currentChanged.connect(self.tab_changed)
        self.cbb_select()

        # train -------------------------------------------------------------------------------------------------
        '''條件未成立前禁用[model, train, predict] button '''
        FrontEnd.set_button_enable([self.btn_train_run, self.predict_view.btn_model, self.predict_view.btn_predict], 0, False)

        self.train_time = 0  # 圖表初始時間
        self.btn_train_run.clicked.connect(self.run_train)
        self.btn_model.clicked.connect(self.select_weights)
        # 初始化數據
        self.xdata = []
        self.ydata = []

        '''創建一個Matplotlib圖形容器和軸'''
        self.figure = plt.Figure()
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)

        '''將圖表放進 layout 中'''
        self.chartLayout.addWidget(self.canvas)
        self.setLayout(self.chartLayout)
        self.set_plot()

    '''設定圖表'''
    def set_plot(self):

        # 設定圖表外圍背景顏色
        self.figure.patch.set_facecolor('gray')
        self.ax.set_xlabel("Time(ms)", color="white", fontsize=14)
        self.ax.set_ylabel("loss", color="white", fontsize=14)
        # 設定曲線範圍內的顏色
        self.ax.set_facecolor('gray')
        # 設定網格
        self.ax.grid(color='white')

        # 在圖表右上方顯示x y軸項目
        self.ax.add_artist(AnchoredText("x axis time\ny axis loss", loc='upper right'))

        # 自動設定 x, y 軸的範圍
        self.ax.set_xlim(auto=True)
        self.ax.set_ylim(auto=True)

    def update_process(self, process):
        self.progressBar.setValue(process)


    def open_project(self):

        gl.PROJECT_FOLDER = QFileDialog.getExistingDirectory(self, '選擇專案路徑', 'D:/',
                                                             options=QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks)

        if gl.PROJECT_FOLDER:
            self.lb_project_path.setText(f'專案路徑: {gl.PROJECT_FOLDER}')

        if gl.PROJECT_FOLDER == '':

            print("\n取消選擇")
            return

    '''訓練參數選擇'''
    def cbb_select(self):
        # 選擇類別比重
        self.cbb_classScale_select.addItems([str(i) + '%' for i in range(50, 101, 10)])
        self.cbb_classScale_select.activated[str].connect(self.select_cls_percent)
        self.cbb_classScale_select.setCurrentIndex(4)
        # 選擇影像解析度
        self.cbb_imgSize_select.addItems(gl.IMG_SIZE)
        self.cbb_imgSize_select.activated[str].connect(self.select_img_size)
        self.cbb_imgSize_select.setCurrentIndex(1)
        # 選擇批次數量
        self.cbb_batchSize_select.addItems([str(i) for i in range(2, 17)])
        self.cbb_batchSize_select.activated[str].connect(self.select_batch_size)
        # 選擇訓練次數
        self.cbb_epoch_select.addItems([str(i) for i in range(1, 101)])
        self.cbb_epoch_select.activated[str].connect(self.select_epoch)
        self.cbb_epoch_select.setCurrentIndex(0)
        # 選擇硬體設備
        self.cbb_device_select.addItems(gl.DEVICE)
        self.cbb_device_select.activated[str].connect(self.select_device)

    # 獲得訓練比重 int
    def select_cls_percent(self, text):
        percent = text
        gl.PERCENT = int(percent[:2])
        print(f'獲得訓練比重{int(gl.PERCENT)}')

    # 解析度 [int ,int]        text => # <class 'str'>
    def select_img_size(self, text):
        for img_size in gl.IMG_SIZE:
            if text == img_size:
                resolution = img_size.split('x')  # <class 'list'>
                # resolution[0] # <class 'str'>
                gl.TRAIN_IMG_SIZE = [int(resolution[0]), int(resolution[1])]
                print(f'實際獲得解析度{gl.TRAIN_IMG_SIZE}')

    # 批次數量 int
    def select_batch_size(self, text):

        gl.TRAIN_BATCH_SIZE = int(text)
        print(f'批次數量{gl.TRAIN_BATCH_SIZE}')

    # 訓練次數 int
    def select_epoch(self, text):
        gl.TRAIN_EPOCHS = int(text)
        print(f'訓練次數{gl.TRAIN_EPOCHS}')

    # 選擇硬體 str
    def select_device(self, text):
        if text == 'CPU':
            gl.TRAIN_DEVICE = 'cpu'
        elif text == 'GPU 1顆':
            gl.TRAIN_DEVICE = '0'
        elif text == 'GPU 2顆':
            gl.TRAIN_DEVICE = '0,1'
        elif text == 'GPU 3顆':
            gl.TRAIN_DEVICE = '0,1,2'

        print(f'選擇硬體{gl.TRAIN_DEVICE}')

    # 選擇權重
    def select_weights(self):
        
        if gl.PROJECT_FOLDER != '':
            # print('專案成立', gl.PROJECT_FOLDER)
            # 專案路徑成立，打開選擇模型按鈕
            gl.TRAIN_WEIGHTS, _ = QFileDialog.getOpenFileName(self, '選擇訓練模型', os.path.realpath('yolov7.pt'), 'weights file(*.pt)')

            # 有選擇權重檔，打開訓練按鈕
            if gl.TRAIN_WEIGHTS:
                FrontEnd.set_button_enable([self.btn_train_run], 1, True)

            if gl.TRAIN_WEIGHTS == '':
                FrontEnd.set_button_enable([self.btn_train_run], 0, False)
                print("\n取消選擇")
                return
        else:

            self.thread_dialog = thread_dialog.ThreadDialog('尚未選擇專案路徑')
            self.thread_dialog.start()

    # train ----------------------------------------------------------------------
    '''每次訓練，清空當前圖表內容，重新設置圖表'''
    def run_train(self):
        # 清空圖表
        self.figure.clf()
        self.xdata = []
        self.ydata = []
        self.ax = self.figure.add_subplot(111)
        self.set_plot()
        self.train_time = 0

        if gl.PROJECT_FOLDER != '':
            if 'classes.txt' not in os.listdir(gl.PROJECT_FOLDER):
                print('not have classes.txt')
                self.thread_dialog = thread_dialog.ThreadDialog('classes.txt not exists')
                self.thread_dialog.start()

            if 'classes.txt' in os.listdir(gl.PROJECT_FOLDER):
                print('classes exists')
                '''這部分是為了重複訓練導致疊加資料'''
                if os.path.exists(gl.TRAIN_FOLDER) or os.path.exists(gl.VAL_FOLDER):
                    if os.listdir(gl.TRAIN_FOLDER) != [] or os.listdir(gl.VAL_FOLDER) != []:
                        print(f'資料夾存在時，刪除資料夾')
                        shutil.rmtree(gl.TRAIN_FOLDER)
                        shutil.rmtree(gl.VAL_FOLDER)

                self.thread_train.start()
                self.lb_train_param.setText(f'訓練模型: {gl.TRAIN_WEIGHTS}\n'
                                            f'影像尺寸: {gl.TRAIN_IMG_SIZE} \n'
                                            f'批次數量: {gl.TRAIN_BATCH_SIZE} \n'
                                            f'訓練次數: {gl.TRAIN_EPOCHS} \n'
                                            f'硬體設備: {gl.TRAIN_DEVICE}\n')
                # 訓練過程按鈕停用
                FrontEnd.set_button_enable([self.btn_train_run], 0, False)

    '''顯示圖表 傳入 loss'''
    def display_chart(self, loss):
        self.ax.cla()
        # 平化數據更新圖表
        self.ax.relim()
        self.ax.autoscale_view(True, True, True)
        self.train_time += 1
        self.xdata.append(self.train_time)
        self.ydata.append(loss)
        # 更新圖形
        self.ax.plot(self.xdata, self.ydata, color='yellow', linewidth=2)
        self.canvas.draw()

    # train -----------------------------------------------------------------------------------------------

    # tab page
    def tab_changed(self, index):
        # 當選擇的標籤頁更改時，顯示相應的內容
        if index == 0:
            print('Train view')
        elif index == 1:
            print('Predict view')

    def finish(self):
        self.thread_train.running = False
        self.figure.savefig(os.path.join(gl.PROJECT_FOLDER, 'figure.png'))
        print('Release Thread')
        self.close()

        python_executable = sys.executable
        # python_directory = os.path.dirname(python_executable)

        current_directory = os.path.dirname(os.path.abspath(__file__))
        dialog_path = os.path.join(current_directory, "dialog.py")

        subprocess.Popen(['python', os.path.abspath('dialog.py')], executable=python_executable).wait()



if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    win = TabWidget()
    # win.showMaximized()
    win.show()
    sys.exit(app.exec_())
