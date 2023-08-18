import logging
from collections import OrderedDict
import torch
import torch.nn as nn
import os
import numpy as np
import models.networks as networks
from models.base_model import BaseModel
from data.util import get_scheduler
from models.fs_loss import Fusionloss
logger = logging.getLogger('base')


class DFFM(BaseModel):
    def __init__(self, opt):
        super(DFFM, self).__init__(opt)
        # define network and load pretrained models
        self.netDF = self.set_device(networks.define_DFFM(opt))

        # set loss and load resume state
        self.loss_func = Fusionloss().to(self.device)

        if self.opt['phase'] == 'train':
            self.netDF.train()
            # find the parameters to optimize
            optim_df_params = list(self.netDF.parameters())

            if opt['train']["optimizer"]["type"] == "adam":
                self.optDF = torch.optim.Adam(
                    optim_df_params, lr=opt['train']["optimizer"]["lr"])
            elif opt['train']["optimizer"]["type"] == "adamw":
                self.optDF = torch.optim.AdamW(
                    optim_df_params, lr=opt['train']["optimizer"]["lr"])
            else:
                raise NotImplementedError(
                    'Optimizer [{:s}] not implemented'.format(opt['train']["optimizer"]["type"]))
            self.log_dict = OrderedDict()

            # Define learning rate sheduler
            self.exp_lr_scheduler_netDF = get_scheduler(optimizer=self.optDF, args=opt['train'])
        else:
            self.netDF.eval()
            self.log_dict = OrderedDict()

        self.load_network()
        self.print_network()
        self.loss_all = []
        self.loss_in = []
        self.loss_grad = []
        self.len_train_dataloader = opt["len_train_dataloader"]
        self.len_val_dataloader = opt["len_val_dataloader"]
        self.alpha = 1.0

    # Feeding all data to the DF model
    def feed_data(self, feats, data):
        self.feats = feats
        self.data = self.set_device(data)

    # Optimize the parameters of the DF model
    def optimize_parameters(self):
        self.optDF.zero_grad()
        self.pred_rgb = self.netDF(self.feats)
        loss_in, loss_grad = self.loss_func(image_vis=self.data["vis"], image_ir=self.data["ir"],  generate_img=self.pred_rgb)
        loss_fs = loss_in + self.alpha * loss_grad
        loss_fs.backward()
        self.optDF.step()
        self.loss_all.append(loss_fs.item())
        self.loss_in.append(loss_in.item())
        self.loss_grad.append(loss_grad.item())

    def update_loss(self):
        self.log_dict['l_all'] = np.average(self.loss_all)
        self.log_dict['l_in'] = np.average(self.loss_in)
        self.log_dict['l_grad'] = np.average(self.loss_grad)
        ###
        self.loss_all = []
        self.loss_in = []
        self.loss_grad = []

    # Testing on given data
    def test(self):
        self.netDF.eval()
        with torch.no_grad():
            self.pred_rgb = self.netDF(self.feats)
            loss_in, loss_grad = self.loss_func(image_vis=self.data["vis"],
                                                image_ir=self.data["ir"],
                                                generate_img=self.pred_rgb)
            loss_fs = loss_in + loss_grad
            self.loss_all.append(loss_fs.item())
            self.loss_in.append(loss_in.item())
            self.loss_grad.append(loss_grad.item())
        self.netDF.train()

    # Get current log
    def get_current_log(self):
        return self.log_dict

    # Get current visuals
    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['pred_rgb'] = self.pred_rgb
        out_dict['gt_vis'] = self.data["vis"]
        out_dict['gt_ir'] = self.data["ir"]
        return out_dict

    # Printing the DF network
    def print_network(self):
        s, n = self.get_network_description(self.netDF)
        if isinstance(self.netDF, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netDF.__class__.__name__,
                                             self.netDF.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netDF.__class__.__name__)

        logger.info(
            'Change Detection Network structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)

    # Saving the network parameters
    def save_network(self, epoch, is_best_model=False):
        df_gen_path = os.path.join(
            self.opt['path']['checkpoint'], 'df_model_E{}_gen.pth'.format(epoch))
        df_opt_path = os.path.join(
            self.opt['path']['checkpoint'], 'df_model_E{}_opt.pth'.format(epoch))

        if is_best_model:
            best_df_gen_path = os.path.join(
                self.opt['path']['checkpoint'], 'best_df_model_gen.pth'.format(epoch))
            best_df_opt_path = os.path.join(
                self.opt['path']['checkpoint'], 'best_df_model_opt.pth'.format(epoch))

        # Save DF model parameters
        network = self.netDF
        if isinstance(self.netDF, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, df_gen_path)
        if is_best_model:
            torch.save(state_dict, best_df_gen_path)

        # Save DF optimizer parameters
        opt_state = {'epoch': epoch,
                     'scheduler': None,
                     'optimizer': None}
        opt_state['optimizer'] = self.optDF.state_dict()
        torch.save(opt_state, df_opt_path)
        if is_best_model:
            torch.save(opt_state, best_df_opt_path)

        # Print info
        logger.info(
            'Saved current DF model in [{:s}] ...'.format(df_gen_path))
        if is_best_model:
            logger.info(
                'Saved best DF model in [{:s}] ...'.format(best_df_gen_path))

    # Loading pre-trained DF network
    def load_network(self):
        load_path = self.opt['path_df']['resume_state']
        if load_path is not None:
            logger.info(
                'Loading pretrained model for Fusion head model [{:s}] ...'.format(load_path))
            gen_path = '{}_gen.pth'.format(load_path)
            opt_path = '{}_opt.pth'.format(load_path)

            # change detection model
            network = self.netDF
            if isinstance(self.netDF, nn.DataParallel):
                network = network.module
            network.load_state_dict(torch.load(
                gen_path), strict=True)

            if self.opt['phase'] == 'train':
                opt = torch.load(opt_path)
                self.optDF.load_state_dict(opt['optimizer'])
                self.begin_step = opt['iter']
                self.begin_epoch = opt['epoch']

    # Functions related to computing performance metrics for DF
    def _update_metric(self):
        """
        update metric
        """
        G_pred = self.pred_cm.detach()
        G_pred = torch.argmax(G_pred, dim=1)

        current_score = self.running_metric.update_cm(pr=G_pred.cpu().numpy(), gt=self.data['L'].detach().cpu().numpy())
        return current_score

    # Collecting status of the current running batch
    def _collect_running_batch_states(self):
        self.running_acc = self._update_metric()
        self.log_dict['running_acc'] = self.running_acc.item()

    # Collect the status of the epoch
    def _collect_epoch_states(self):
        scores = self.running_metric.get_scores()
        self.epoch_acc = scores['mf1']
        self.log_dict['epoch_acc'] = self.epoch_acc.item()

        for k, v in scores.items():
            self.log_dict[k] = v
            # message += '%s: %.5f ' % (k, v)

    # Rest all the performance metrics
    # def _clear_cache(self):
    #     self.running_metric.clear()

    # Finctions related to learning rate sheduler
    def _update_lr_schedulers(self):
        self.exp_lr_scheduler_netDF.step()