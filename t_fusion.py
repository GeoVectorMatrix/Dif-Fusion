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
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/fusion_test.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'test'],
                        help='Run either train(training + validation) or testing', default='test')
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
            test_dataset = FD(split='val',
                              crop_size=dataset_opt['resolution'],
                              ir_path='Path_IR',
                              vi_path='Path_VIS',
                              is_crop=False)
            print("the training dataset is length:{}".format(test_dataset.length))
            test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=dataset_opt['batch_size'],
                shuffle=False,
                num_workers=1,
                pin_memory=True,
                drop_last=False,
            )
            test_loader.n_iter = len(test_loader)

    logger.info('Initial Dataset Finished')

    # Loading diffusion model
    diffusion = Model.create_model(opt)
    logger.info('Initial Diffusion Model Finished')

    # Creating Fusion model
    fussion_net = Model.create_fusion_model(opt)

    logger.info('Begin Model Evaluation (testing).')
    test_result_path = '{}/test/'.format(opt['path']['results'])
    os.makedirs(test_result_path, exist_ok=True)
    logger_test = logging.getLogger('test')  # test logger

    time_list = []
    for current_step, (test_data, file_names) in enumerate(test_loader):
        start = time.time()
        diffusion.feed_data(test_data)
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
        fussion_net.feed_data(fds, test_data)
        fussion_net.test()
        visuals = fussion_net.get_current_visuals()
        grid_img = visuals['pred_rgb'].detach()
        grid_img = Metrics.tensor2img(grid_img)
        Metrics.save_img(grid_img, '{}/{}'.format(test_result_path, file_names[0]))
    logger.info('End of Testing.')
