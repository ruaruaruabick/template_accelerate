import os,sys
import logging
import argparse
from tqdm.auto import tqdm
import time,datetime
import yaml
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed, gather_object
from accelerate.data_loader import prepare_data_loader
import itertools

from math import ceil

import torch
from deepspeed.ops.adam import FusedAdam
from utils.ema import EMA as ExponentialMovingAverage
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

from utils.util import load_yaml,load_json,compute_trainable_parameters,init_weights
from utils.scheduler import CosineLRScheduler

from datasets_.dataset import MyDataset, CollateFn
import torch.distributed as dist
from torch.utils.data import DataLoader,RandomSampler

torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = False
dist.init_process_group(backend='nccl', init_method='env://', timeout=datetime.timedelta(hours=3.))
logger = logging.getLogger(__name__)

def main(args):
    #load config
    config = load_yaml(args.config)
    
    #set seed
    set_seed(config['seed'])
    
    #define accelerator
    ckptdir = f"logs/ckpt/{args.expname}"
    logdir = f"logs/tb_log/{args.expname}"
    if args.finetune:
        newckptdir = ckptdir + "_ft"
        logdir += "_ft"
    else:
        newckptdir = ckptdir
    os.makedirs(logdir, exist_ok=True)
    prjconfig = ProjectConfiguration(project_dir="logs", logging_dir=logdir)
    accelerator = Accelerator(project_config=prjconfig,log_with="tensorboard",gradient_accumulation_steps=config['training']['gradient_accumulation_steps'])#,kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
    device = accelerator.device
    num_gpus = accelerator.num_processes
    cur_index = accelerator.local_process_index
    
    #define dataloader
    accelerator.print(f"###################Loading dataset###################")
    dataconfig = config['dataset']
    train_list, valid_list, test_list = load_json(dataconfig['train_file'],use_split=args.split,index=cur_index), load_json(dataconfig['valid_file']), load_json(dataconfig['test_file'])
    
    training_data = MyDataset(train_list,dataconfig)
    
    sampler = None
    if args.split:
        sampler = RandomSampler(data_source=training_data)
        config['dataloader']['shuffle'] = False
        
    train_loader = DataLoader(training_data,  sampler=sampler, collate_fn=CollateFn(), **config['dataloader'])
    valid_data = MyDataset(valid_list,config['audio'],dataconfig)
    config['dataloader']['drop_last'] = False
    valid_loader = DataLoader(valid_data, collate_fn=CollateFn(), **config['dataloader'])

    del train_list, valid_list, test_list, training_data, valid_data
    
    accelerator.wait_for_everyone()
    steps_per_epoch = len(train_loader) if args.split else len(train_loader)// num_gpus
    
    #define model
    accelerator.print(f"###################Loading model####################")
    generator = 
    discriminator = 
    
    accelerator.init_trackers('',{
        "num of params_g": f"{compute_trainable_parameters(generator)/1e6} M ",
        "num of params_d": f"{compute_trainable_parameters(discriminator)/1e6} M",
    })
    with open(f"{logdir}/config.yaml", "w") as f :
        yaml.dump(config, f)
        
    #define ema
    ema = ExponentialMovingAverage(generator, decay=config["ema"],)
    ema.to(device)
        
    if args.finetune:
        generator.load_state_dict(torch.load(ckptdir+f"/{args.restore_step}/pytorch_model.bin",map_location=device),strict=False)
        
        ema = ExponentialMovingAverage(generator, decay=config["ema"],)
        ema.to(device)

        torch.cuda.empty_cache()

    #define optimizer and scheduler
    config['scheduler']['warmup_steps'] = config['scheduler'].get('warmup_steps', steps_per_epoch)
    if args.finetune:
        ftparams = itertools.chain(generator.encoder.parameters(),generator.decoder.quantizer.parameters(),generator.decoder.timbre_encoder.parameters(),generator.decoder.melspec_encoder.parameters())
        retraineparams = generator.decoder.model.parameters()
        ftcfg = config['optimizer']['g'].copy()
        ftcfg['lr'] /= 10
        optimizer_g = FusedAdam([
            {'params': ftparams,**ftcfg},
            {'params': retraineparams,**config['optimizer']['g']}
        ])
        
    else:
        optimizer_g = FusedAdam(filter(lambda p: p.requires_grad,generator.parameters()), **config['optimizer']['g'])
    
    fun_sch_g = CosineLRScheduler(init_lr=config['optimizer']['g']['lr'],max_steps=config['training']['steps'],num_gpus = num_gpus,  **config['scheduler']['g'])
    scheduler_g = LambdaLR(optimizer_g, fun_sch_g)
    
    optimizer_d = FusedAdam(discriminator.parameters(), **config['optimizer']['d'])
    fun_sch_d = CosineLRScheduler(init_lr=config['optimizer']['d']['lr'],max_steps=config['training']['steps'],num_gpus = num_gpus,  **config['scheduler']['d'])
    scheduler_d = LambdaLR(optimizer_d, fun_sch_d)
    
    #prepare
    generator, discriminator, valid_loader, optimizer_g, optimizer_d, scheduler_g, scheduler_d, ema = accelerator.prepare(
        generator, discriminator, valid_loader, optimizer_g, optimizer_d, scheduler_g, scheduler_d, ema
    )
    if not args.split:
        train_loader = accelerator.prepare(train_loader)
    else:
        train_loader = prepare_data_loader(train_loader,device,num_processes=1,put_on_device=True,non_blocking=True)
    accelerator.register_for_checkpointing(ema)
       
    #load ckpt
    cursteps = 0
    if not args.finetune and args.restore_step >=0 and os.path.exists(ckptdir+f"/{args.restore_step}"):
        accelerator.load_state(ckptdir+f"/{args.restore_step}",strict=False)
        cursteps = args.restore_step
        ema.to(device)
    
    #training
    maxsteps = config['training']['steps']
    total_epochs = ceil(maxsteps / steps_per_epoch)
    cur_epoch = cursteps // steps_per_epoch
    progress_bar = tqdm(range(maxsteps), disable=not accelerator.is_main_process, initial=cursteps)
    logger = accelerator.get_tracker("tensorboard")
    
    for epoch in range(cur_epoch, total_epochs + 1):
        st = time.perf_counter()
        with accelerator.accumulate(generator,discriminator):
            for batch in train_loader:
                wav_real = batch['wav'].unsqueeze(1)
                
                #train generator
                
                
                #advloss
                
                
                #optimize
                optimizer_g.zero_grad()
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    norm_g = accelerator.clip_grad_norm_(generator.parameters(), config['training']['max_grad_norm_g']).item()
                optimizer_g.step()
                scheduler_g.step()
                ema.update()
                
                #logging
                if accelerator.is_main_process:
                    cur_lr = scheduler_g.get_lr()[0]
                    lossdict.update({"lr/g":cur_lr,"gradient/g":norm_g})
                    accelerator.log(lossdict,step=cursteps)
                
                #train discriminator
                train_d = (cursteps % config['training']['d_interval'] == 0 and cursteps > config['training']['d_start_steps'])
                if train_d:
                    loss
                    #optimize
                    optimizer_d.zero_grad()
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        norm_d = accelerator.clip_grad_norm_(discriminator.parameters(), config['training']['max_grad_norm_d']).item()
                    optimizer_d.step()
                    scheduler_d.step()
                    #logging
                    if accelerator.is_main_process:
                        cur_lr = scheduler_d.get_lr()[0]
                        lossdict.update({"lr/d":cur_lr,"gradient/d":norm_d})
                        accelerator.log(lossdict,step=cursteps)
                
                cursteps += 1
                progress_bar.update(1)
                progress_bar.set_description(f"Epoch {epoch}/{total_epochs}, Step {cursteps%steps_per_epoch}/{steps_per_epoch}")
                
                #validation
                if cursteps % config['training']['valid_interval'] == 1:
                    with torch.no_grad(), ema.average_parameters():
                        generator.eval()
                        total_loss = []
                        for idx, batch in enumerate(valid_loader):
                            if batch == None:
                                continue
                            
                            
                            progress_bar.set_description(f"Validating... {idx}/{len(valid_loader)}")
                        generator.train()
                        #log
                        total_loss = gather_object(total_loss)
                    accelerator.wait_for_everyone()
                
                #save ckpt
                if cursteps % config['training']['save_interval'] == 0:
                    accelerator.save_state(output_dir=newckptdir+f'/{cursteps}', safe_serialization=False)
                accelerator.wait_for_everyone()
                if cursteps >= maxsteps:
                    accelerator.print("Training Finished!")
                    accelerator.wait_for_everyone()
                    accelerator.end_training()
                    sys.exit()                
        et = time.perf_counter()
        accelerator.log({"epoch_time":et-st,},step=cursteps)
        accelerator.print(f"epoch time:{et-st:.2f}s")
    accelerator.end_training()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",
        type=str,
        default="configs/default.yaml",
        help="config file path",
    )
    parser.add_argument("--expname",
        type=str,
        default="test",
        help="config file path",
    )
    parser.add_argument("--restore_step",
        type=int,
        default=-1,
        help="restore_step of ckpt",
    )
    parser.add_argument('--finetune', '-ft',
                        action='store_true',
                        default=False,
                        help=''
    )
    parser.add_argument('--split', 
                        action='store_true',
                        default=False,
                        help='split training data or not'
    )
    args = parser.parse_args()
    main(args)