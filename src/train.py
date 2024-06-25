# ReginalWFN

import os
import sys
import re
import time
import socket
import logging
import random
import numpy as np
import multiprocessing as mp
import torch
import torch.distributed as dist
from torchvision.utils import save_image
import torch.cuda.amp as amp
from torch.nn.parallel import DistributedDataParallel

import logging_utils
logging_utils.config_logger()
from YParams import YParams
from data_loader_multifiles import get_data_loader
from darcy_loss import LpLoss
from afnonet import AFNONet, PrecipNet

HERE = os.path.dirname(__file__)
HOME = os.path.abspath(os.path.join(HERE, ".."))

pat_envvar = re.compile(r'^([_\d\w]+)=(.*)$')
pat_param  = re.compile(r'^--([_\d\w]+)=(.*)$')

def usage():
    print("usgae: python3 train.py [env_name=env_value ...] [--param_name=param_value ...]")

def parse_args(vargs):

    envs = {}
    params = {}

    for arg in vargs:

        E = pat_envvar.match(arg)
        P = pat_param.match(arg)

        if E:
            matched = E.groups(1)
            envs[matched[0]] = matched[1]

        elif P:
            matched = P.groups(1)
            params[matched[0]] = matched[1]

        else:
            print(f"ERROR: command-line syntax error near {arg}\n")
            print(usage())
            exit(-1)
 

    return envs, params


def read_params(path, user_params):

    params = YParams()

    misc_path  = os.environ.get("RWFN_MISC_YAML", os.path.join(path, "misc.yaml"))
    model_path = os.environ.get("RWFN_MODEL_YAML", os.path.join(path, "model.yaml"))
    train_path = os.environ.get("RWFN_TRAIN_YAML", os.path.join(path, "train.yaml"))
    data_path  = os.environ.get("RWFN_DATA_YAML", os.path.join(path, "data.yaml"))

    misc_cfg  = os.environ.get("RWFN_MISC_CONFIG", "default")
    model_cfg = os.environ.get("RWFN_MODEL_CONFIG", "default")
    train_cfg = os.environ.get("RWFN_TRAIN_CONFIG", "default")
    data_cfg  = os.environ.get("RWFN_DATA_CONFIG", "default")

    # read misc params
    if os.path.isfile(misc_path):
        misc_params = YParams(misc_path, misc_cfg)

        if user_params["world_rank"] == 0:
            misc_params.log()

        params.update_params(misc_params)

    # read model params
    if os.path.isfile(model_path):
        model_params = YParams(model_path, model_cfg)

        if user_params["world_rank"] == 0:
            model_params.log()

        params.update_params(model_params)

    # read train params
    if os.path.isfile(train_path):
        train_params = YParams(train_path, train_cfg)

        if user_params["world_rank"] == 0:
            train_params.log()

        params.update_params(train_params)

    # read data params
    if os.path.isfile(data_path):
        data_params = YParams(data_path, data_cfg)
        
        if user_params["world_rank"] == 0:
            data_params.log()
 
        params.update_params(data_params)

    # update params with user-specified items
    for key, val in user_params.items():
        if key in params:
            val = type(params[key])(val)
        params[key] = val
        params.__setattr__(key, val)

    return params


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


class Trainer():

  def count_parameters(self):
    return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

  def __init__(self, params):
    
    self.params = params
    self.device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
    self.tbdir = f"{params.experiment_dir}/profile/tblogs_{params.world_rank}"

    logging.info('rank %d, begin data loader init'%params.world_rank)

    self.train_data_loader, self.train_dataset, self.train_sampler = \
        get_data_loader(params, params.train_data_path, dist.is_initialized(), train=True)

    self.valid_data_loader, self.valid_dataset = \
        get_data_loader(params, params.valid_data_path, dist.is_initialized(), train=False)

    self.loss_obj = LpLoss()

    logging.info('rank %d, data loader initialized'%params.world_rank)

    params.crop_size_x = self.valid_dataset.crop_size_x
    params.crop_size_y = self.valid_dataset.crop_size_y
    params.img_shape_x = self.valid_dataset.img_shape_x
    params.img_shape_y = self.valid_dataset.img_shape_y

    # precip models
    self.precip = True if "precip" in params else False
    
    if self.precip:
      if 'model_wind_path' not in params:
        raise Exception("no backbone model weights specified")
      # load a wind model 
      # the wind model has out channels = in channels
      out_channels = np.array(params['in_channels'])
      params['N_out_channels'] = len(out_channels)

      if params.nettype_wind == 'afno':
        self.model_wind = AFNONet(params).to(self.device)

      else:
        raise Exception("not implemented")

      if dist.is_initialized():
        self.model_wind = DistributedDataParallel(self.model_wind,
                                            device_ids=[params.local_rank],
                                            output_device=[params.local_rank],
                                            find_unused_parameters=True)
      self.load_model_wind(params.model_wind_path)
      self.switch_off_grad(self.model_wind) # no backprop through the wind model

    # reset out_channels for precip models
    if self.precip:
      params['N_out_channels'] = len(params['out_channels'])

    if params.nettype == 'afno':
      self.model = AFNONet(params).to(self.device) 

    else:
      raise Exception("not implemented")
     
    # precip model
    if self.precip:
      self.model = PrecipNet(params, backbone=self.model).to(self.device)

    if self.params.enable_nhwc:
      # NHWC: Convert model to channels_last memory format
      self.model = self.model.to(memory_format=torch.channels_last)

    if params.optimizer_type == 'FusedAdam':
      self.optimizer = optimizers.FusedAdam(self.model.parameters(), lr = params.lr)

    else:
      self.optimizer = torch.optim.Adam(self.model.parameters(), lr = params.lr)

    if params.enable_amp == True:
      self.gscaler = amp.GradScaler()

    if dist.is_initialized():
      self.model = DistributedDataParallel(self.model,
                                           device_ids=[params.local_rank],
                                           output_device=[params.local_rank],
                                           find_unused_parameters=True)

    self.iters = 0
    self.startEpoch = 0

    if params.resuming:
      logging.info("Loading checkpoint %s"%params.checkpoint_path)
      self.restore_checkpoint(params.checkpoint_path)

    if params.two_step_training:
      if params.resuming == False and params.pretrained == True:
        logging.info("Starting from pretrained one-step afno model at %s"%params.pretrained_ckpt_path)
        self.restore_checkpoint(params.pretrained_ckpt_path)
        self.iters = 0
        self.startEpoch = 0
        #logging.info("Pretrained checkpoint was trained for %d epochs"%self.startEpoch)
        #logging.info("Adding %d epochs specified in config file for refining pretrained model"%self.params.max_epochs)
        #self.params.max_epochs += self.startEpoch

    self.epoch = self.startEpoch

    if params.scheduler == 'ReduceLROnPlateau':
      self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                            self.optimizer, factor=0.2, patience=5, mode='min')

    elif params.scheduler == 'CosineAnnealingLR':
      self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                            self.optimizer, T_max=params.max_epochs,
                            last_epoch=self.startEpoch-1)
    else:
      self.scheduler = None

    '''if params.log_to_screen:
      logging.info(self.model)'''
    if params.log_to_screen:
      logging.info("Number of trainable model parameters: {}".format(self.count_parameters()))

  def switch_off_grad(self, model):
    for param in model.parameters():
      param.requires_grad = False

  def train(self):
    if self.params.log_to_screen:
      logging.info("Starting Training Loop...")

    best_valid_loss = 1.e6
    for epoch in range(self.startEpoch, self.params.max_epochs):
      if dist.is_initialized():
        self.train_sampler.set_epoch(epoch)
#        self.valid_sampler.set_epoch(epoch)

      start = time.time()

      tr_time, data_time, train_logs = self.train_one_epoch()

      valid_time, valid_logs = self.validate_one_epoch()

      if epoch==self.params.max_epochs-1 and self.params.prediction_type == 'direct':
        valid_weighted_rmse = self.validate_final()

      if self.params.scheduler == 'ReduceLROnPlateau':
        self.scheduler.step(valid_logs['valid_loss'])

      elif self.params.scheduler == 'CosineAnnealingLR':
        self.scheduler.step()

        if self.epoch >= self.params.max_epochs:
          logging.info("Terminating training after reaching params.max_epochs "
                        "while LR scheduler is set to CosineAnnealingLR")
          exit()

      if self.world_rank == 0:
        if self.params.save_checkpoint:
          #checkpoint at the end of every epoch
          self.save_checkpoint(self.params.checkpoint_path)

          if valid_logs['valid_loss'] <= best_valid_loss:
            #logging.info('Val loss improved from {} to {}'.format(best_valid_loss, valid_logs['valid_loss']))
            self.save_checkpoint(self.params.best_checkpoint_path)
            best_valid_loss = valid_logs['valid_loss']

      if self.params.log_to_screen:
        logging.info('Time taken for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
        #logging.info('train data time={}, train step time={}, valid step time={}'.format(data_time, tr_time, valid_time))
        logging.info('Train loss: {}. Valid loss: {}'.format(train_logs['loss'], valid_logs['valid_loss']))
#        if epoch==self.params.max_epochs-1 and self.params.prediction_type == 'direct':
#          logging.info('Final Valid RMSE: Z500- {}. T850- {}, 2m_T- {}'.format(valid_weighted_rmse[0], valid_weighted_rmse[1], valid_weighted_rmse[2]))

  def train_one_epoch(self):
    self.epoch += 1
    tr_time = 0
    data_time = 0
    self.model.train()
    
    for i, data in enumerate(self.train_data_loader, 0):
      self.iters += 1
      # adjust_LR(optimizer, params, iters)
      data_start = time.time()
      inp, tar = map(lambda x: x.to(self.device, dtype = torch.float), data)      
      if self.params.orography and self.params.two_step_training:
          orog = inp[:,-2:-1] 

      if self.params.enable_nhwc:
        inp = inp.to(memory_format=torch.channels_last)
        tar = tar.to(memory_format=torch.channels_last)

      if 'residual_field' in self.params.target:
        tar -= inp[:, 0:tar.size()[1]]
      data_time += time.time() - data_start

      tr_start = time.time()

      self.model.zero_grad()

      if self.params.two_step_training:
          with amp.autocast(self.params.enable_amp):
            gen_step_one = self.model(inp).to(self.device, dtype = torch.float)
            loss_step_one = self.loss_obj(gen_step_one, tar[:,0:self.params.N_out_channels])

            if self.params.orography:
                gen_step_two = self.model(torch.cat( (gen_step_one, orog), axis = 1)
                                    ).to(self.device, dtype = torch.float)
            else:
                gen_step_two = self.model(gen_step_one
                                    ).to(self.device, dtype = torch.float)

            loss_step_two = self.loss_obj(gen_step_two,
                                tar[:,self.params.N_out_channels:2*self.params.N_out_channels])
            loss = loss_step_one + loss_step_two
      else:
          with amp.autocast(self.params.enable_amp):
            if self.precip: # use a wind model to predict 17(+n) channels at t+dt
              with torch.no_grad():
                inp = self.model_wind(inp).to(self.device, dtype = torch.float)

              gen = self.model(inp.detach()).to(self.device, dtype = torch.float)

            else:
              gen = self.model(inp).to(self.device, dtype = torch.float)

            loss = self.loss_obj(gen, tar)

      if self.params.enable_amp:
        self.gscaler.scale(loss).backward()
        self.gscaler.step(self.optimizer)

      else:
        loss.backward()
        self.optimizer.step()

      if self.params.enable_amp:
        self.gscaler.update()

      tr_time += time.time() - tr_start
    
    try:
        logs = {'loss': loss, 'loss_step_one': loss_step_one,
                'loss_step_two': loss_step_two}
    except:
        logs = {'loss': loss}

    if dist.is_initialized():
      for key in sorted(logs.keys()):
        dist.all_reduce(logs[key].detach())
        logs[key] = float(logs[key]/dist.get_world_size())

    return tr_time, data_time, logs

  def validate_one_epoch(self):
    self.model.eval()

    n_valid_batches = 20 #do validation on first 20 images, just for LR scheduler

    if self.params.normalization == 'minmax':
        raise Exception("minmax normalization not supported")

    elif self.params.normalization == 'zscore':
        mult = torch.as_tensor(np.load(self.params.global_stds_path
                                      )[0, self.params.out_channels, 0, 0]
                              ).to(self.device)

    valid_buff = torch.zeros((3), dtype=torch.float32, device=self.device)
    valid_loss = valid_buff[0].view(-1)
    valid_l1 = valid_buff[1].view(-1)
    valid_steps = valid_buff[2].view(-1)
    valid_weighted_rmse = torch.zeros((self.params.N_out_channels),
                            dtype=torch.float32, device=self.device)
    valid_weighted_acc = torch.zeros((self.params.N_out_channels),
                            dtype=torch.float32, device=self.device)

    valid_start = time.time()

    sample_idx = np.random.randint(len(self.valid_data_loader))

    with torch.no_grad():
      for i, data in enumerate(self.valid_data_loader, 0):
        if (not self.precip) and i>=n_valid_batches:
          break    

        inp, tar  = map(lambda x: x.to(self.device, dtype = torch.float), data)

        if self.params.orography and self.params.two_step_training:
            orog = inp[:,-2:-1]

        if self.params.two_step_training:
            gen_step_one = self.model(inp).to(self.device, dtype = torch.float)
            loss_step_one = self.loss_obj(gen_step_one, tar[:,0:self.params.N_out_channels])

            if self.params.orography:
                gen_step_two = self.model(torch.cat( (gen_step_one, orog), axis = 1)
                                         ).to(self.device, dtype = torch.float)
            else:
                gen_step_two = self.model(gen_step_one
                                         ).to(self.device, dtype = torch.float)

            loss_step_two = self.loss_obj(gen_step_two,
                                tar[:,self.params.N_out_channels:2*self.params.N_out_channels])
            valid_loss += loss_step_one + loss_step_two
            valid_l1 += nn.functional.l1_loss(gen_step_one,
                                tar[:,0:self.params.N_out_channels])
        else:
            if self.precip:
                with torch.no_grad():
                    inp = self.model_wind(inp).to(self.device, dtype = torch.float)

                gen = self.model(inp.detach())

            else:
                gen = self.model(inp).to(self.device, dtype = torch.float)

            valid_loss += self.loss_obj(gen, tar) 
            valid_l1 += nn.functional.l1_loss(gen, tar)

        valid_steps += 1.

        # save fields for vis before log norm 
        if (i == sample_idx) and (self.precip and self.params.log_to_wandb):
          fields = [gen[0,0].detach().cpu().numpy(), tar[0,0].detach().cpu().numpy()]

        if self.precip:
          gen = unlog_tp_torch(gen, self.params.precip_eps)
          tar = unlog_tp_torch(tar, self.params.precip_eps)

        #direct prediction weighted rmse
        if self.params.two_step_training:
            if 'residual_field' in self.params.target:
                valid_weighted_rmse += weighted_rmse_torch((gen_step_one + inp),
                                        (tar[:,0:self.params.N_out_channels] + inp))
            else:
                valid_weighted_rmse += weighted_rmse_torch(gen_step_one,
                                        tar[:,0:self.params.N_out_channels])
        else:
            if 'residual_field' in self.params.target:
                valid_weighted_rmse += weighted_rmse_torch((gen + inp), (tar + inp))
            else:
                valid_weighted_rmse += weighted_rmse_torch(gen, tar)

        if not self.precip:
            try:
                os.mkdir(params['experiment_dir'] + "/" + str(i))
            except:
                pass

            #save first channel of image
            image_path = params['experiment_dir'] + "/" + str(i) + "/" + str(self.epoch) + ".png"
            image_temp = torch.zeros((self.valid_dataset.img_shape_x, 4)
                                    ).to(self.device, dtype = torch.float)

            if self.params.two_step_training:
                image_data = torch.cat((gen_step_one[0,0], image_temp, tar[0,0]), axis = 1)
            else:
                image_data = torch.cat((gen[0,0], image_temp, tar[0,0]), axis = 1)

            save_image(image_data, image_path)
           
    if dist.is_initialized():
      dist.all_reduce(valid_buff)
      dist.all_reduce(valid_weighted_rmse)

    # divide by number of steps
    valid_buff[0:2] = valid_buff[0:2] / valid_buff[2]
    valid_weighted_rmse = valid_weighted_rmse / valid_buff[2]
    if not self.precip:
      valid_weighted_rmse *= mult

    # download buffers
    valid_buff_cpu = valid_buff.detach().cpu().numpy()
    valid_weighted_rmse_cpu = valid_weighted_rmse.detach().cpu().numpy()

    valid_time = time.time() - valid_start
    valid_weighted_rmse = mult*torch.mean(valid_weighted_rmse, axis = 0)

    if self.precip:
      logs = {'valid_l1': valid_buff_cpu[1], 'valid_loss': valid_buff_cpu[0],
                'valid_rmse_tp': valid_weighted_rmse_cpu[0]}
    else:
      try:
        logs = {'valid_l1': valid_buff_cpu[1], 'valid_loss': valid_buff_cpu[0],
                'valid_rmse_u10': valid_weighted_rmse_cpu[0],
                'valid_rmse_v10': valid_weighted_rmse_cpu[1]}
      except:
        logs = {'valid_l1': valid_buff_cpu[1], 'valid_loss': valid_buff_cpu[0],
                'valid_rmse_u10': valid_weighted_rmse_cpu[0]}#, 'valid_rmse_v10': valid_weighted_rmse[1]}
    
    return valid_time, logs

  def validate_final(self):
    self.model.eval()

    n_valid_batches = int(self.valid_dataset.n_patches_total /
                            self.valid_dataset.n_patches) #validate on whole dataset
    valid_weighted_rmse = torch.zeros(n_valid_batches,
                                    self.params.N_out_channels)

    if self.params.normalization == 'minmax':
        raise Exception("minmax normalization not supported")

    elif self.params.normalization == 'zscore':
        mult = torch.as_tensor(np.load(self.params.global_stds_path
                            )[0, self.params.out_channels, 0, 0]).to(self.device)

    with torch.no_grad():
        for i, data in enumerate(self.valid_data_loader):
          if i>100:
            break

          inp, tar = map(lambda x: x.to(self.device, dtype = torch.float), data)

          if self.params.orography and self.params.two_step_training:
              orog = inp[:,-2:-1]

          if 'residual_field' in self.params.target:
            tar -= inp[:, 0:tar.size()[1]]

        if self.params.two_step_training:
            gen_step_one = self.model(inp).to(self.device, dtype = torch.float)
            loss_step_one = self.loss_obj(gen_step_one,
                                    tar[:,0:self.params.N_out_channels])

            if self.params.orography:
                gen_step_two = self.model(torch.cat((gen_step_one, orog),
                                axis=1)).to(self.device, dtype=torch.float)
            else:
                gen_step_two = self.model(gen_step_one).to(self.device,
                                dtype=torch.float)

            loss_step_two = self.loss_obj(gen_step_two,
                                tar[:,self.params.N_out_channels:2 *
                                    self.params.N_out_channels])
            valid_loss[i] = loss_step_one + loss_step_two
            valid_l1[i] = nn.functional.l1_loss(gen_step_one,
                            tar[:,0:self.params.N_out_channels])
        else:
            gen = self.model(inp)
            valid_loss[i] += self.loss_obj(gen, tar) 
            valid_l1[i] += nn.functional.l1_loss(gen, tar)

        if self.params.two_step_training:
            for c in range(self.params.N_out_channels):
              if 'residual_field' in self.params.target:
                valid_weighted_rmse[i, c] = weighted_rmse_torch(
                                            (gen_step_one[0,c] + inp[0,c]),
                                            (tar[0,c]+inp[0,c]),
                                            self.device)
              else:
                valid_weighted_rmse[i, c] = weighted_rmse_torch(
                                            gen_step_one[0,c],
                                            tar[0,c],
                                            self.device)
        else:
            for c in range(self.params.N_out_channels):
              if 'residual_field' in self.params.target:
                valid_weighted_rmse[i, c] = weighted_rmse_torch(
                                            (gen[0,c] + inp[0,c]),
                                            (tar[0,c]+inp[0,c]),
                                            self.device)
              else:
                valid_weighted_rmse[i, c] = weighted_rmse_torch(
                                            gen[0,c],
                                            tar[0,c],
                                            self.device)
            
        #un-normalize
        valid_weighted_rmse = mult*torch.mean(
                                valid_weighted_rmse[0:100],
                                axis = 0).to(self.device)

    return valid_weighted_rmse


  def load_model_wind(self, model_path):

    if self.params.log_to_screen:
      logging.info('Loading the wind model weights from {}'.format(model_path))

    checkpoint = torch.load(model_path, map_location='cuda:{}'.format(self.params.local_rank))

    if dist.is_initialized():
      self.model_wind.load_state_dict(checkpoint['model_state'])

    else:
      new_model_state = OrderedDict()
      model_key = 'model_state' if 'model_state' in checkpoint else 'state_dict'

      for key in checkpoint[model_key].keys():
          if 'module.' in key: # model was stored using ddp which prepends module
              name = str(key[7:])
              new_model_state[name] = checkpoint[model_key][key]

          else:
              new_model_state[key] = checkpoint[model_key][key]

      self.model_wind.load_state_dict(new_model_state)
      self.model_wind.eval()

  def save_checkpoint(self, checkpoint_path, model=None):
    """ We intentionally require a checkpoint_dir to be passed
        in order to allow Ray Tune to use this function """

    if not model:
      model = self.model

    torch.save({'iters': self.iters, 'epoch': self.epoch,
                'model_state': model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict()},
               checkpoint_path)

  def restore_checkpoint(self, checkpoint_path):
    """ We intentionally require a checkpoint_dir to be passed
        in order to allow Ray Tune to use this function """

    checkpoint = torch.load(checkpoint_path,
                    map_location='cuda:{}'.format(self.params.local_rank))

    try:
        self.model.load_state_dict(checkpoint['model_state'])

    except:
        new_state_dict = OrderedDict()

        for key, val in checkpoint['model_state'].items():
            name = key[7:]
            new_state_dict[name] = val 

        self.model.load_state_dict(new_state_dict)

    self.iters = checkpoint['iters']
    self.startEpoch = checkpoint['epoch']

    if self.params.resuming:  #restore checkpoint is used for finetuning as well as resuming. If finetuning (i.e., not resuming), restore checkpoint does not load optimizer state, instead uses config specified lr.
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

def main(myrank, user_params):

    # set rank info
    if user_params["job_launcher"] == "multiprocessing":
        user_params["world_rank"] = myrank
        user_params["local_rank"] = myrank

    # read parameter files
    params = read_params(
                os.environ.get("RWFN_CONFIG_PATH", os.path.join(HOME, "cfg")),
                user_params
            )

    seed_everything()

    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = socket.gethostname()

    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "29500"

    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        torch.cuda.set_device(params.local_rank % torch.cuda.device_count())
        dist.init_process_group('nccl', rank=params.world_rank, world_size=params.world_size)

    else:
        device = "cpu"
        dist.init_process_group('gloo', rank=params.world_rank, world_size=params.world_size)

    expDir = os.path.abspath(params.exp_dir)

    if  params.world_rank==0:
        if not os.path.isdir(expDir):
            os.makedirs(expDir)
            os.makedirs(os.path.join(expDir, 'training_checkpoints/'))

    params['experiment_dir'] = expDir
    params['checkpoint_path'] = os.path.join(expDir, 'training_checkpoints/ckpt.tar')
    params['best_checkpoint_path'] = os.path.join(expDir, 'training_checkpoints/best_ckpt.tar')

    # Do not comment this line out please:
    #resuming = True if os.path.isfile(params.checkpoint_path) else False
    params.resuming = False
    params['log_to_screen'] = (params['world_rank']==0) and params['log_to_screen']

    params['in_channels'] = np.array(params['in_channels'])
    params['out_channels'] = np.array(params['out_channels'])
    if params.orography:
        params['N_in_channels'] = len(params['in_channels']) +1
    else:
        params['N_in_channels'] = len(params['in_channels'])

    params['N_out_channels'] = len(params['out_channels'])

    params['embed_dim'] = 712
    params['depth'] = 12
    params['prediction_length'] = 49
    params['dt'] = 1
    params['epsilon_factor'] = 0.0

    trainer = Trainer(params)

    trainer.train()

    logging.info('DONE ---- rank %d'%params.world_rank)

    return 0

if __name__ == "__main__":

    # parse command-line options
    user_envs, user_params = parse_args(sys.argv[1:])

    # set environment variables
    for env_name, env_value in user_envs.items():
        os.environ[env_name] = env_value

    # launch main
    if "SLURM_NTASKS" in os.environ:
        user_params["job_launcher"] = "slurm"
        user_params["world_size"] = int(os.environ['SLURM_NTASKS'])
        user_params["world_rank"] = int(os.environ['SLURM_PROCID'])
        user_params["local_rank"] = int(os.environ['SLURM_LOCALID'])

        main(os.environ["SLURM_PROCID"], user_params)

    else:
        ntasks = int(os.environ.get("RWFN_MP_NTASKS", mp.cpu_count()))

        user_params["job_launcher"] ="multiprocessing"
        user_params["world_size"] = ntasks
        user_params["world_rank"] = None
        user_params["local_rank"] = None

        with mp.Pool(ntasks) as p:
            args = [(rank, user_params) for rank in range(ntasks)]
            p.starmap(main, args)
