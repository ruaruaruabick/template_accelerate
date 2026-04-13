import os,sys
import logging
logging.basicConfig(level=f"{os.environ['LOG_LEVEL']}")
from accelerate.logging import get_logger
import argparse
from tqdm.auto import tqdm
import time,datetime
import yaml
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed, gather_object
from accelerate.data_loader import prepare_data_loader
import itertools
from omegaconf import OmegaConf
from math import ceil
import swanlab

import torch
from torch.optim import AdamW
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
logger = get_logger(__name__)

def main(args):
    #load config
    config = load_yaml(args.config)
    
    #set seed
    set_seed(config['seed'])
    
    #define accelerator
    ckptdir = f"logs/ckpt/{args.expname}"
    expname = args.expname
    if args.finetune:
        newckptdir = ckptdir + "_ft"
        expname += "_ft"
        if args.lora:
            newckptdir += "_lora"+f'_{args.suffix}'
            expname += "_lora"+f'_{args.suffix}'
        else:
            newckptdir += f'_{args.suffix}'
            expname += f'_{args.suffix}'
    else:
        newckptdir = ckptdir
    
    prjconfig = ProjectConfiguration(project_dir="logs")
    accelerator = Accelerator(project_config=prjconfig,log_with="swanlab",gradient_accumulation_steps=config['training']['gradient_accumulation_steps'])#,kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
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
    valid_data = MyDataset(valid_list,dataconfig) #MyIterableDataset
    config['dataloader']['drop_last'] = True
    valid_loader = DataLoader(valid_data, collate_fn=CollateFn(), **config['dataloader'])

    del train_list, valid_list, test_list, training_data, valid_data
    
    accelerator.wait_for_everyone()
    # steps_per_epoch = len(train_loader) if args.split else len(train_loader)// num_gpus
    
    #define model
    accelerator.print(f"###################Loading model####################")
    generator = None
    discriminator = None
    if args.ckpt_path != '':
        generator.load_state_dict(torch.load(args.ckpt_path,map_location=device))
        
    #define ema
    ema = ExponentialMovingAverage(generator, decay=config["ema"],)
    ema.to(device)
        
    if args.finetune:       
        ema.load_state_dict(torch.load(ckptdir+f"/{args.restore_step}/custom_checkpoint_0.pkl",map_location=device))
        ema.apply_shadow()
        
        if args.lora:
            from peft import get_peft_model,LoraConfig, TaskType
            generator.requires_grad_(False)
            peft_config = LoraConfig(inference_mode=False,**config['lam']['lora'])
            generator.llm.model = get_peft_model(generator.llm.model, peft_config)
            generator.llm.model.print_trainable_parameters()
            ema = ExponentialMovingAverage(generator, decay=config["ema"],)
            ema.to(device)
    
    paramsstr = [
        swanlab.Text(f"num of params_g: {compute_trainable_parameters(model.generator)} M "),
        swanlab.Text(f"num of params_d: {compute_trainable_parameters(model.discriminator)} M"),
    ]
    if args.resumeid != '':
        resume = True
        accelerator.init_trackers(args.project,OmegaConf.to_container(config),init_kwargs={"swanlab": {
            "experiment_name": expname,
            "resume" : 'must',
            "logdir": "logs/swanlab_log",
            'id':args.resumeid,
        }})
    else:
        resume = False
        accelerator.init_trackers(args.project,OmegaConf.to_container(config),init_kwargs={"swanlab": {
            "experiment_name": expname,
            "resume" : 'never',
            "logdir": "logs/swanlab_log",
        }})
    logger = accelerator.get_tracker("swanlab",unwrap=True)
    if accelerator.is_main_process:
        swanlab.log({"model size":paramsstr},step=None,print_to_console=True)
    
    #define optimizer and scheduler   
    optimizer_g = AdamW(filter(lambda p: p.requires_grad,generator.parameters()), **config['optimizer']['g'])
    fun_sch_g = CosineLRScheduler(init_lr=config['optimizer']['g']['lr'],num_gpus = num_gpus,  **config['scheduler']['g'])
    scheduler_g = LambdaLR(optimizer_g, fun_sch_g)
    
    optimizer_d = AdamW(discriminator.parameters(), **config['optimizer']['d'])
    fun_sch_d = CosineLRScheduler(init_lr=config['optimizer']['d']['lr'],num_gpus = num_gpus,  **config['scheduler']['d'])
    scheduler_d = LambdaLR(optimizer_d, fun_sch_d)
    
    #prepare
    generator, discriminator, valid_loader, optimizer_g, optimizer_d, scheduler_g, scheduler_d, ema = accelerator.prepare(
        generator, discriminator, valid_loader, optimizer_g, optimizer_d, scheduler_g, scheduler_d, ema
    )
    if accelerator.distributed_type == "DEEPSPEED" and not args.split: #DEEPSPEED finetune
        train_loader = accelerator.prepare(train_loader)
    elif not args.split: #torch ddp finetune
        train_loader = prepare_data_loader(train_loader,device,put_on_device=True,non_blocking=True,dispatch_batches=False)
    else: #iterable dataset且每个GPU加载的数据不同
        train_loader = prepare_data_loader(train_loader,device,num_processes=1,put_on_device=True,non_blocking=True,dispatch_batches=False)
    accelerator.register_for_checkpointing(ema)
       
    #load ckpt
    cursteps = 0
    if not args.finetune and args.restore_step >=0 and os.path.exists(ckptdir+f"/{args.restore_step}"):
        accelerator.load_state(ckptdir+f"/{args.restore_step}",{'weights_only':False},strict=False, )
        cursteps = args.restore_step
        ema.to(device)
    
    #training
    maxsteps = config['training']['steps']
    cur_epoch = 0
    progress_bar = tqdm(range(maxsteps), disable=not accelerator.is_main_process, initial=cursteps)
    
    
    for _ in range(cursteps, maxsteps + 1):
        st = time.perf_counter()
        with accelerator.accumulate(generator,discriminator):
            for batch in train_loader:
                wav_real = batch['wav'].unsqueeze(1)
                
                #train generator
                generator.module.train_step(batch)
                
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
                if accelerator.is_main_process and cursteps % config['training']['log_interval'] == 0:
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
                    if accelerator.is_main_process and cursteps % config['training']['log_interval'] == 0:
                        cur_lr = scheduler_d.get_lr()[0]
                        lossdict.update({"lr/d":cur_lr,"gradient/d":norm_d})
                        accelerator.log(lossdict,step=cursteps)
                
                cursteps += 1
                progress_bar.update(1)
                progress_bar.set_description(f"Epoch {cur_epoch}:")
                
                #validation
                if cursteps % config['training']['valid_interval'] == 1:
                    generator.eval()
                    with torch.no_grad(), ema.average_parameters():
                        total_loss = 0
                        for idx, batch in enumerate(valid_loader):
                            if batch == None:
                                continue
                            loss = accelerator.gather(loss).mean() # tensor要有shape
                            total_loss += accelerator.gather(loss).mean()
                            progress_bar.set_description(f"Validating... {idx}/{len(valid_loader)}")
                        
                        #log
                        if accelerator.is_main_process:
                            total_loss = total_loss / (idx+1)
                            accelerator.log()
                    accelerator.wait_for_everyone()
                    generator.train()
                
                #save ckpt
                if cursteps % config['training']['save_interval'] == 0:
                    accelerator.save_state(output_dir=newckptdir+f'/{cursteps}', safe_serialization=False)
                accelerator.wait_for_everyone()
                
                #end training
                if cursteps >= maxsteps:
                    accelerator.print("Training Finished!")
                    accelerator.wait_for_everyone()
                    accelerator.end_training()
                    sys.exit()                
            et = time.perf_counter()
            accelerator.log({"epoch_time":et-st,},step=cursteps)
            accelerator.print(f"epoch time:{et-st:.2f}s")
            cur_epoch += 1
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",
        type=str,
        default="configs/default.yaml",
        help="config file path",
    )
    parser.add_argument("--project",
        type=str,
        default="test",
        help="project name",
    )
    parser.add_argument("--expname",
        type=str,
        default="test",
        help="exp name",
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
    parser.add_argument('--lora', 
                        action='store_true',
                        default=False,
                        help='use lora or not'
    )
    parser.add_argument('--suffix', 
                        type=str,
                        default='',
                        help='expname suffix'
    )
    parser.add_argument('--ckpt_path',
                        type=str,
                        default='',
                        help='Specify the ckpt path.'
    )
    parser.add_argument('--ema_path',
                        type=str,
                        default='',
                        help='Specify the ema path.'
    )
    parser.add_argument('--resumeid',
                        type=str,
                        default='',
                        help='Specify the resumeid for swanlab'
    )
    args = parser.parse_args()
    main(args)