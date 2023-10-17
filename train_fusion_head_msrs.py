import torch
import models as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import os
from data.VIDataset import FusionDataset as FD

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/fusion_train.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'test'],
                        help='Run either train(training + validation) or testing', default='train')
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
    Logger.setup_logger('test', opt['path']['log'], 'test', level=logging.INFO)
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
                num_workers=1,
                pin_memory=True,
                drop_last=True,
            )
            train_loader.n_iter = len(train_loader)

    logger.info('Initial Dataset Finished')

    # Loading diffusion model
    diffusion = Model.create_model(opt)
    logger.info('Initial Diffusion Model Finished')

    # Set noise schedule for the diffusion model
    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])

    # Creating Fusion model
    fussion_net = Model.create_fusion_model(opt)

    #################
    # Training loop #
    #################
    n_epoch = opt['train']['n_epoch']
    best_mF1 = 0.0
    start_epoch = 0
    if opt['phase'] == 'train':
        for current_epoch in range(start_epoch, n_epoch):
            train_result_path = '{}/train/{}'.format(opt['path']
                                                     ['results'], current_epoch)
            os.makedirs(train_result_path, exist_ok=True)
            ################
            ### training ###
            ################
            message = 'lr: %0.7f\n \n' % fussion_net.optDF.param_groups[0]['lr']
            logger.info(message)
            for current_step, (train_data, _) in enumerate(train_loader):
                diffusion.feed_data(train_data)
                fes = []
                fds = []
                for t in opt['model_df']['t']:
                    fe_t, fd_t = diffusion.get_feats(t=t)
                    if opt['model_df']['feat_type'] == "dec":
                        fds.append(fd_t)
                        del fd_t
                    else:
                        fes.append(fe_t)
                        del fe_t

                # Feeding features
                fussion_net.feed_data(fds, train_data)
                fussion_net.optimize_parameters()

                # log running batch status
                if current_step % opt['train']['train_print_freq'] == 0:
                    # message
                    fussion_net.update_loss()
                    logs = fussion_net.get_current_log()
                    message = '[Training FS]. epoch: [%d/%d]. Itter: [%d/%d], ' \
                              'All_loss: %.5f,Intensity_loss: %.5f, Grad_loss: %.5f' % \
                              (current_epoch, n_epoch - 1, current_step, len(train_loader), logs['l_all'],
                               logs['l_in'], logs['l_grad'])
                    logger.info(message)

            visuals = fussion_net.get_current_visuals()
            grid_img = torch.cat((visuals['pred_rgb'].detach(),
                                  visuals['gt_vis'],
                                  visuals['gt_ir'].repeat(1, 3, 1, 1)), dim=0)
            grid_img = Metrics.tensor2img(grid_img)
            Metrics.save_img(grid_img, '{}/img_fused_e{}_b{}.png'.format(train_result_path,
                                                                         current_epoch,
                                                                         current_step))
            if (current_epoch > 1) & ((current_epoch+1) % 10 == 0):
                fussion_net.save_network(current_epoch)
        fussion_net._update_lr_schedulers()
        logger.info('End of fusion training.')
        fussion_net.save_network(current_epoch)