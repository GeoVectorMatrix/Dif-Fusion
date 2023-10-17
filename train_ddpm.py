import torch
import models as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import os
import numpy as np
from data.VIDataset import FusionDataset as FD


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/ddpm_train.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                        help='Run either train(training) or val(generation)', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_eval', action='store_true')

    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    # dataset
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train' and args.phase != 'val':
            print("Creating train dataloader.")
            train_dataset = FD(split='train',
                               crop_size=dataset_opt['resolution'],
                               ir_path='Path_IR',
                               vi_path='Path_VIS',
                               is_crop=True)
            print("the training dataset is length:{}".format(train_dataset.length))
            train_loader = DataLoader(
                dataset=train_dataset,
                batch_size=dataset_opt['batch_size'],
                shuffle=True,
                num_workers=4,
                pin_memory=True,
                drop_last=True,
            )
            train_loader.n_iter = len(train_loader)

    logger.info('Initial Dataset Finished')

    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    # Train
    current_step = diffusion.begin_step
    current_epoch = diffusion.begin_epoch
    n_iter = opt['train']['n_iter']

    if opt['path']['resume_state']:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            current_epoch, current_step))

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])

    if opt['phase'] == 'train':
        while current_step < n_iter:
            current_epoch += 1
            for _, (train_data, _) in enumerate(train_loader):
                current_step += 1
                if current_step > n_iter:
                    break
                diffusion.feed_data(train_data)
                diffusion.optimize_parameters()
                # log
                if current_step % opt['train']['print_freq'] == 0:
                    logs = diffusion.get_current_log()
                    message = '<epoch:{:3d}, iter:{:8,d}> '.format(
                        current_epoch, current_step)
                    for k, v in logs.items():
                        message += '{:s}: {:.4e} '.format(k, v)
                        tb_logger.add_scalar(k, v, current_step)
                    logger.info(message)

                # validation
                if current_step % opt['train']['val_freq'] == 0:
                    result_path = '{}/{}'.format(opt['path']
                                                 ['results'], current_epoch)
                    os.makedirs(result_path, exist_ok=True)

                    diffusion.set_new_noise_schedule(
                        opt['model']['beta_schedule']['val'], schedule_phase='val')
                    for idx in range(0, opt['datasets']['val']['data_len'], 1):
                        diffusion.test_concat(in_channels=4,
                                              img_size_w=opt['datasets']['val']['image_size_w'],
                                              img_size_h=opt['datasets']['val']['image_size_h'],
                                              x_vis=train_data['vis'][idx:idx+1, :, :, :],
                                              x_ir=train_data['ir'][idx:idx+1, :, :, :],
                                              continous=False)

                        visuals = diffusion.get_current_visuals()
                        sam_img = Metrics.tensor2img(visuals['SAM'][:, 0:3, :, :])

                        # generation
                        Metrics.save_img(
                            sam_img, '{}/sample_{}_{}.png'.format(result_path, current_step, idx))

                        tb_logger.add_image(
                            'Iter_{}'.format(current_step),
                            np.transpose(sam_img, [2, 0, 1]),
                            idx)

                    diffusion.set_new_noise_schedule(
                        opt['model']['beta_schedule']['train'], schedule_phase='train')
                    # log
                    logger_val = logging.getLogger('val')  # validation logger
                    logger_val.info('<epoch:{:3d}, iter:{:8,d}> Sample generation completed.'.format(
                        current_epoch, current_step))

                if current_step % opt['train']['save_checkpoint_freq'] == 0:
                    logger.info('Saving models and training states.')
                    diffusion.save_network(current_epoch, current_step)

        # save model
        logger.info('End of training.')
