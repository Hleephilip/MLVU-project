import torch
import torch.nn as nn
import copy
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import *
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.loops import EvaluationLoop
from torch.cuda import amp
from torch.utils.data.dataset import TensorDataset
from torch.optim.optimizer import Optimizer
from torch.nn import CosineSimilarity
from renderer import *
from config import *


class CustomValidationLoop(EvaluationLoop):
    pass



class LatentModel(pl.LightningModule):
    def __init__(self, conf: TrainConfig, train_data_path: str, val_data_path: str, cfg_prob: float, cfg_guidance: float):
        super().__init__()

        if conf.seed is not None:
            pl.seed_everything(conf.seed)

        self.save_hyperparameters(conf.as_dict_jsonable())

        self.conf = conf
        self.train_data_path = train_data_path
        self.val_data_path = val_data_path
        self.cfg_prob = cfg_prob
        self.cfg_guidance = cfg_guidance
        self.do_cfg = self.cfg_guidance > 1.0

        self.model = conf.make_model_conf().make_model()
        self.ema_model = copy.deepcopy(self.model)
        self.ema_model.requires_grad_(False)
        self.ema_model.eval()

        self.cos_sim = CosineSimilarity(dim=1)
        
        self.T_sampler = conf.make_T_sampler()
        # if conf.train_mode.use_latent_net():
        assert conf.train_mode.use_latent_net()
        self.latent_sampler = conf.make_latent_diffusion_conf().make_sampler() # DDPM
        self.eval_latent_sampler = conf.make_latent_eval_diffusion_conf().make_sampler() # DDIM

        # initial variables for consistent sampling
        self.register_buffer(
            'x_T',
            torch.randn(conf.sample_size, 3, conf.img_size, conf.img_size))

        if conf.pretrain is not None:
            print(f'loading pretrain ... {conf.pretrain.name}')
            state = torch.load(conf.pretrain.path, map_location='cpu')
            print('step:', state['global_step'])
            self.load_state_dict(state['state_dict'], strict=False)

        if conf.latent_infer_path is not None:
            print('loading latent stats ...')
            state = torch.load(conf.latent_infer_path)
            self.conds = state['conds']
            self.register_buffer('conds_mean', state['conds_mean'][None, :])
            self.register_buffer('conds_std', state['conds_std'][None, :])
        else:
            self.x_mean = None
            self.x_std = None
            self.conds_mean = None
            self.conds_std = None

            self.val_x_mean = None
            self.val_x_std = None
            self.val_conds_mean = None
            self.val_conds_std = None
        
        self.lmb1 = 1.
        self.lmb2 = 1.
        
    def normalize_cond(self, cond):
        cond2 = (cond - self.conds_mean.to(self.device)) / self.conds_std.to(self.device)
        return cond2

    def denormalize_cond(self, cond):
        cond2 = (cond * self.conds_std.to(self.device)) + self.conds_mean.to(self.device)
        return cond2
    
    def normalize_x(self, x):
        x2 = (x - self.x_mean.to(self.device)) / self.x_std.to(self.device)
        return x2

    def denormalize_x(self, x):
        x2 = (x * self.x_std.to(self.device)) + self.x_mean.to(self.device)
        return x2
    
    def forward(self, noise=None, x_start=None, ema_model: bool = False):
        with amp.autocast(False):
            if ema_model:
                model = self.ema_model
            else:
                model = self.model
            gen = self.eval_sampler.sample(model=model, noise=noise, x_start=x_start)
            return gen

    def setup(self, stage=None) -> None:
        """
        make datasets & seeding each worker separately
        """
        ##############################################
        # NEED TO SET THE SEED SEPARATELY HERE
        if self.conf.seed is not None:
            seed = self.conf.seed * get_world_size() + self.global_rank
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            print('local seed:', seed)
        ##############################################
        
        print('Preparing train data')
        self.train_data = ZipDataset(self.train_data_path) # self.conf.make_dataset()
        print('train data:', len(self.train_data))

        print('Preparing val data')
        self.val_data = ZipDataset(self.val_data_path)
        print('val data:', len(self.val_data))

    def _train_dataloader(self, drop_last=False):
        """
        really make the dataloader
        """
        # make sure to use the fraction of batch size
        # the batch size is global!
        conf = self.conf.clone()
        conf.batch_size = self.batch_size

        # print(conf.batch_size)
        # print(len(self.train_data))
        dataloader = conf.make_loader(self.train_data, shuffle=True, drop_last=drop_last)
        return dataloader
    
    def train_dataloader(self):
        """
        return the dataloader, if diffusion mode => return image dataset
        if latent mode => return the inferred latent dataset
        ** In this implementation, we use only latent mode.
        """
        print('on train dataloader start ...')
        if self.conf.train_mode.require_dataset_infer():
            raise NotImplementedError()
        
            if self.conds is None:
                # usually we load self.conds from a file
                # so we do not need to do this again!
                # self.conds = self.infer_whole_dataset()
                # print(self.conds)

                # need to use float32! unless the mean & std will be off!
                # (1, c)
                self.conds_mean.data = self.conds.float().mean(dim=0, keepdim=True)
                self.conds_std.data = self.conds.float().std(dim=0, keepdim=True)
            print('mean:', self.conds_mean.mean(), 'std:', self.conds_std.mean())

            # return the dataset with pre-calculated conds
            conf = self.conf.clone()
            conf.batch_size = self.batch_size
            data = TensorDataset(self.conds)
            return conf.make_loader(data, shuffle=True)
        else:
            self.conds_mean = self.train_data.clip_img_mean
            self.conds_std = self.train_data.clip_img_std
            
            self.x_mean = self.train_data.clip_txt_mean
            self.x_std = self.train_data.clip_txt_std
            # print('x_mean', self.x_mean.shape)
            # print('x_std', self.x_std.shape)

            return self._train_dataloader()
        

    def _val_dataloader(self, drop_last=False):
            """
            really make the dataloader
            """
            # make sure to use the fraction of batch size
            # the batch size is global!
            conf = self.conf.clone()
            conf.batch_size = self.batch_size

            dataloader = conf.make_loader(self.val_data, shuffle=False, drop_last=drop_last)
            return dataloader

    def val_dataloader(self):
        """
        return the dataloader, if diffusion mode => return image dataset
        if latent mode => return the inferred latent dataset
        ** In this implementation, we use only latent mode.
        """
        # print('on validation dataloader start ...')
        if self.conf.train_mode.require_dataset_infer():
            raise NotImplementedError()
        
            if self.conds is None:
                # usually we load self.conds from a file
                # so we do not need to do this again!
                # self.conds = self.infer_whole_dataset()
                # print(self.conds)

                # need to use float32! unless the mean & std will be off!
                # (1, c)
                self.conds_mean.data = self.conds.float().mean(dim=0, keepdim=True)
                self.conds_std.data = self.conds.float().std(dim=0, keepdim=True)
            print('mean:', self.conds_mean.mean(), 'std:', self.conds_std.mean())

            # return the dataset with pre-calculated conds
            conf = self.conf.clone()
            conf.batch_size = self.batch_size
            data = TensorDataset(self.conds)
            return conf.make_loader(data, shuffle=True)
        else:
            
            self.val_conds_mean = self.val_data.clip_img_mean
            self.val_conds_std = self.val_data.clip_img_std
            
            self.val_x_mean = self.val_data.clip_txt_mean
            self.val_x_std = self.val_data.clip_txt_std
            
            return self._val_dataloader()
        
    @property
    def batch_size(self):
        """
        local batch size for each worker
        """
        ws = get_world_size()
        assert self.conf.batch_size % ws == 0
        return self.conf.batch_size // ws
    
    @property
    def num_samples(self):
        """
        (global) batch size * iterations
        """
        # batch size here is global!
        # global_step already takes into account the accum batches
        return self.global_step * self.conf.batch_size_effective
    
    def is_last_accum(self, batch_idx):
        """
        is it the last gradient accumulation loop? 
        used with gradient_accum > 1 and to see if the optimizer will perform "step" in this iteration or not
        """
        return (batch_idx + 1) % self.conf.accum_batches == 0

    def infer_whole_dataset(self, with_render=False, T_render=None, render_save_path=None):
        raise NotImplementedError()
    
    def training_step(self, batch, batch_idx):
        """
        given an input, calculate the loss function
        no optimization at this stage.
        """
        with amp.autocast(False):
            # batch size here is local!
            # forward
            if self.conf.train_mode.require_dataset_infer():
                # this mode as pre-calculated cond
                raise NotImplementedError()
                cond = batch[0]
                if self.conf.latent_znormalize:
                    cond = (cond - self.conds_mean.to(self.device)) / self.conds_std.to(self.device)
            else:
                # imgs, idxs = batch['img'], batch['index']
                z_img, z_txt = batch
                # print(f'(rank {self.global_rank}) batch size:', len(imgs))
                
                if self.conf.latent_znormalize:
                    z_img = (z_img - self.conds_mean.to(self.device)) / self.conds_std.to(self.device)
                    z_txt = (z_txt - self.x_mean.to(self.device)) / self.x_std.to(self.device)

            if self.conf.train_mode.is_latent_diffusion():
                """
                training the latent DDIM model!
                """
                # diffusion on the latent
                t, weight = self.T_sampler.sample(len(z_txt), self.device)
                cfg_prob = np.random.uniform(0, 1, 1)[0] < self.cfg_prob # self.cfg_prob = p_uncond = 0.1
                if cfg_prob: 
                    # drop condition (use None)
                    latent_losses = self.latent_sampler.training_losses(
                        model=self.model, x_start=z_txt, t=t, c=None)
                    
                    recon_loss = 0.0

                else:
                    # use condition
                    latent_losses = self.latent_sampler.training_losses(
                        model=self.model, x_start=z_txt, t=t, c=z_img)
                    
                    z_txt_hat = render_latent(
                            conf=self.conf,
                            model=self.model,
                            x_T=z_txt,
                            cond=z_img,
                            latent_sampler=self.eval_latent_sampler, # use DDIM : 50 step
                            x_mean=self.x_mean,
                            x_std=self.x_std)

                    # print('sampled :', z_txt_hat.shape) # [B, 768]
                    # print('original :', z_txt.shape) # [B, 768]
                    # print(torch.min(self.denormalize_x(z_txt)), torch.max(self.denormalize_x(z_txt)), torch.min(z_txt_hat), torch.max(z_txt_hat))
                    # reconstruction loss
                    if self.conf.latent_znormalize:
                        z_txt_unnorm = (z_txt * self.x_std.to(self.device)) + self.x_mean.to(self.device)
                        recon_loss = nn.L1Loss()(z_txt_unnorm, z_txt_hat)
                    else:
                        recon_loss = nn.L1Loss()(z_txt, z_txt_hat)
                    # train only do the latent diffusion
                
                      
                # print(latent_losses['loss'], recon_loss)
                losses = {
                    'latent': latent_losses['loss'] * self.lmb1,
                    'recon': recon_loss * self.lmb2,
                    'loss': latent_losses['loss'] * self.lmb1 + recon_loss * self.lmb2
                }

            else:
                raise NotImplementedError()

            loss = losses['loss'].mean()
            # divide by accum batches to make the accumulated gradient exact!
            for key in ['loss', 'latent', 'recon'] : # in ['loss', 'vae', 'latent', 'mmd', 'chamfer', 'arg_cnt']:
                if key in losses:
                    losses[key] = self.all_gather(losses[key]).mean()

            if self.global_rank == 0:
                self.logger.experiment.add_scalar('loss', losses['loss'])
                for key in ['latent', 'recon'] : # in ['vae', 'latent', 'mmd', 'chamfer', 'arg_cnt']:
                    if key in losses:
                        self.logger.experiment.add_scalar(
                            f'loss/{key}', losses[key], self.num_samples)
            
            # self.log("latent_loss", losses['latent'], on_step=False, on_epoch=True, prog_bar=True, logger=False)

        return {'loss': loss, 'latent_loss': losses['latent'], 'recon_loss': losses['recon']}


    def on_train_batch_end(self, outputs, batch, batch_idx: int, dataloader_idx: int) -> None:
        """
        after each training step ...
        """

        if self.is_last_accum(batch_idx):
            # only apply ema on the last gradient accumulation step,
            # if it is the iteration that has optimizer.step()
            if self.conf.train_mode == TrainMode.latent_diffusion:
                # it trains only the latent hence change only the latent
                ema(self.model, self.ema_model, self.conf.ema_decay)
            else:
                raise NotImplementedError()

            # Do not use below two lines in this implementation
            '''
            # logging
            if self.conf.train_mode.require_dataset_infer():
                imgs = None
            else:
                imgs = batch['img']

            self.log_sample(x_start=imgs)
            self.evaluate_scores()
            '''

    def on_train_epoch_end(self, outputs):
        """
        after each training epoch ...
        """
        print(f"       Epoch {self.current_epoch}  loss : {outputs[0]['loss']:.4f}  latent_loss : {outputs[0]['latent_loss']:.4f}  recon_loss : {outputs[0]['recon_loss']:.4f}")
        epoch_sim = self.calc_cos_sim().item()
        print(f"       Similarity : {epoch_sim:.4f}")


    def on_before_optimizer_step(self, optimizer: Optimizer, optimizer_idx: int) -> None:
        # fix the fp16 + clip grad norm problem with pytorch lightinng
        # this is the currently correct way to do it
        if self.conf.grad_clip > 0:
            # from trainer.params_grads import grads_norm, iter_opt_params
            params = [
                p for group in optimizer.param_groups for p in group['params']
            ]
            # print('before:', grads_norm(iter_opt_params(optimizer)))
            torch.nn.utils.clip_grad_norm_(params, max_norm=self.conf.grad_clip)
            # print('after:', grads_norm(iter_opt_params(optimizer)))

    def calc_cos_sim(self):
        # set up for validation
        self.model.eval()
        torch.set_grad_enabled(False)

        out_lst = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_dataloader()):
                out = self.cos_sim_step(batch, batch_idx)
                out_lst.append(out)

        out_lst = torch.cat(out_lst)

        # set up for train
        self.model.train()
        torch.set_grad_enabled(True)
        return out_lst.mean()

    def cos_sim_step(self, batch, batch_idx):
        with amp.autocast(False):
            z_img, z_txt = batch
            z_img, z_txt = z_img.to(self.device), z_txt.to(self.device)

            if self.conf.latent_znormalize:
                z_img = (z_img - self.val_conds_mean.to(self.device)) / self.val_conds_std.to(self.device)
                z_txt = (z_txt - self.val_x_mean.to(self.device)) / self.val_x_std.to(self.device)
            
            if self.conf.train_mode.is_latent_diffusion():
                """
                calculating cosine sim. between z_txt and \hat{z}_txt
                """
                z_txt_hat_cond = render_latent(
                            conf=self.conf,
                            model=self.model,
                            x_T=z_txt,
                            cond=z_img,
                            latent_sampler=self.eval_latent_sampler, # use DDIM : 50 step
                            x_mean=self.val_x_mean,
                            x_std=self.val_x_std)

                z_txt_hat_uncond = render_latent(
                            conf=self.conf,
                            model=self.model,
                            x_T=z_txt,
                            cond=None,
                            latent_sampler=self.eval_latent_sampler, # use DDIM : 50 step
                            x_mean=self.val_x_mean,
                            x_std=self.val_x_std)
                
                if not self.do_cfg: 
                    print('Not using classifier-free guidance')
                    z_txt_hat = z_txt_hat_cond
                else:
                    # apply CFG
                    z_txt_hat = (1. + self.cfg_guidance) * z_txt_hat_cond - self.cfg_guidance * z_txt_hat_uncond
                
                if self.conf.latent_znormalize:
                    z_txt_unnorm = (z_txt * self.val_x_std.to(self.device)) + self.val_x_mean.to(self.device)
                    output = self.cos_sim(z_txt_hat, z_txt_unnorm)
                else:
                    output = self.cos_sim(z_txt_hat, z_txt)
            else:
                raise NotImplementedError()

        return output
    
    def log_sample(self, x_start):
        raise NotImplementedError()
    
    def evaluate_scores(self):
        raise NotImplementedError()
    
    def configure_optimizers(self):
        out = {}
        if self.conf.optimizer == OptimizerType.adam:
            optim = torch.optim.Adam(self.model.parameters(),
                                     lr=self.conf.lr,
                                     weight_decay=self.conf.weight_decay)
        elif self.conf.optimizer == OptimizerType.adamw:
            optim = torch.optim.AdamW(self.model.parameters(),
                                      lr=self.conf.lr,
                                      weight_decay=self.conf.weight_decay)
        else:
            raise NotImplementedError()
        
        out['optimizer'] = optim

        if self.conf.warmup > 0:
            sched = torch.optim.lr_scheduler.LambdaLR(optim,
                                                      lr_lambda=WarmupLR(
                                                          self.conf.warmup))
            out['lr_scheduler'] = {
                'scheduler': sched,
                'interval': 'step',
            }
        return out

    def split_tensor(self, x):
        # only used in log_sample()
        raise NotImplementedError()
    
    def test_step(self, batch, *args, **kwargs):
        # TODO : implement here!
        raise NotImplementedError()
    

class WarmupLR:
    def __init__(self, warmup) -> None:
        self.warmup = warmup

    def __call__(self, step):
        return min(step, self.warmup) / self.warmup


def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(target_dict[key].data * decay + source_dict[key].data * (1 - decay))


def train(conf: TrainConfig, nodes=1, mode: str = 'train', device = 'cuda', args: argparse = None):
    print('conf:', conf.name)
    gpus = args.gpus
    # assert not (conf.fp16 and conf.grad_clip > 0
    #             ), 'pytorch lightning has bug with amp + gradient clipping'
    # model = LatentModel(conf, '../DATA/sample_train.zip')
    model = LatentModel(conf, args.train_data_path, args.val_data_path, args.cfg_prob, args.cfg_guidance)

    if not os.path.exists(conf.logdir):
        os.makedirs(conf.logdir)

    checkpoint = ModelCheckpoint(dirpath=f'{conf.logdir}',
                                 save_last=True,
                                 save_top_k=args.checkpoint_top_k,
                                 every_n_epochs=args.checkpoint_interval
                                 )
    checkpoint_path = f'{conf.logdir}/last.ckpt'

    print('ckpt path:', checkpoint_path)
    if not args.use_pretrained :
        resume = None
    elif os.path.exists(checkpoint_path):
        resume = checkpoint_path
        print('resume!')
    else:
        if conf.continue_from is not None:
            # continue from a checkpoint
            resume = conf.continue_from.path
        else:
            resume = None

    # tb_logger = pl_loggers.TensorBoardLogger(save_dir=conf.logdir, name=args.log_name, version=args.log_version)


    plugins = []
    if len(gpus) == 1 and nodes == 1:
        accelerator = None
    else:
        accelerator = 'ddp'
        from pytorch_lightning.plugins import DDPPlugin

        # important for working with gradient checkpoint
        plugins.append(DDPPlugin(find_unused_parameters=False))

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        resume_from_checkpoint=resume,
        gpus=gpus,
        num_nodes=nodes,
        accelerator=accelerator,
        precision=16 if conf.fp16 else 32,
        callbacks=[
            checkpoint,
            LearningRateMonitor(),
        ],
        replace_sampler_ddp=True,
        # logger=tb_logger,
        accumulate_grad_batches=conf.accum_batches,
        plugins=plugins,
        # progress_bar_refresh_rate=0
    )

    if mode == 'train':
        trainer.fit(model)
    elif mode == 'eval':
        # TODO : implement here!
        raise NotImplementedError()
    else:
        raise NotImplementedError()

    # print(model.device)
    # print(model.conf.train_mode.require_dataset_infer())
    # print(model.conf.optimizer)
    # print(model.conf.logdir)
    print('finish')
    return 
