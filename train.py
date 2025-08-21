# %%

import torch.utils
import os
import torch
import numpy as np
import wandb
import torch.nn as nn
from tqdm import tqdm
import random
import argparse
import shutil
from src.datasets.MidiDataset import MidiDateset
from src.datasets.MidiRandomDataset import MidiRandomDateset
from src.datasets.MapsMidiRandomDataset import MapsMidiRandomDataset
from src.datasets.MapsMidiDataset import MapsMidiDataset
from src.models.PianoSSM import *
from src.loss import *
from src.scheduler.CosineAnnealing import CosineAnnealingWarmupRestarts
import src.s_edge as s_edge

def main():

    parser = argparse.ArgumentParser(description='Script description')
    parser.add_argument('--wandb', type=bool, default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--sample_rate', type=int, default=16000)
    parser.add_argument('--midi_rate', type=int, default=1000, help="Sample rate has to be a multiple of midi rate")
    parser.add_argument('--sample_length', type=float, default=4)
    parser.add_argument('--dilation', type=float, default=4)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--epoch_size', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--step_size', type=int, default=0) 
    parser.add_argument('--criterion', nargs='+', default=["CombinedSpectralLoss"], help="List of criteria: CombinedSpectralLoss, SpectralLoss")
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--model', type=str, default="PianoSSM_S", help="see PianoSSM.py")
    parser.add_argument('--train_year', type=str, default='single', help="2004, 2006, 2008, 2009, 2011, 2013, 2014, 2015, 2017, 2018, all, single")
    parser.add_argument('--test_year', type=str, default='single', help="2004, 2006, 2008, 2009, 2011, 2013, 2014, 2015, 2017, 2018, all, single")
    parser.add_argument('--scheduler', type=bool, default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--test', type=bool, default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--valid', type=bool, default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--normalize', type=bool, default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--weights', type=str, default=None)
    parser.add_argument('--activation', type=str, default="Tanh", help="ReLU, LeakyReLU, Identity, Tanh")
    parser.add_argument('--dataset', type=str, default="maestro", help="maestro, maps")
    parser.add_argument('--dataset_path', type=str, default="data/")
    parser.add_argument('--finetune', type=bool, default=True, action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    print(args)

    assert args.sample_rate % args.midi_rate == 0, "Midi rate has to be a multiple of the sample rate"
    
    # seed = 42
    # random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True
    
    if 'LOCAL_RANK' in os.environ:
        local_rank = int(os.environ['LOCAL_RANK'])
        print("local_rank = " + str(local_rank))
    else:
        local_rank = args.gpu
        
    if 'ROLE_RANK' in os.environ:
        role_rank = int(os.environ['ROLE_RANK'])
        print("role_rank = " + str(role_rank))
    else:
        role_rank = 0
    
    path = args.dataset_path
    
    print(f"local_rank={local_rank} role_rank={role_rank} Start Training")
    
    if args.dataset == "maestro":
        dataset_class_train = MidiRandomDateset
        dataset_class_test = MidiDateset
    elif args.dataset == "maps":
        dataset_class_train = MapsMidiRandomDataset
        dataset_class_test = MapsMidiDataset
        args.valid = False
    else:
        assert False, "Dataset not recognised"
    dataset_train = dataset_class_train(data_dir=path, 
                                    sample_rate=args.sample_rate, 
                                    midi_rate=args.midi_rate,
                                    sample_length=args.sample_length, 
                                    epoch_size=args.epoch_size,      
                                    normalize=args.normalize, 
                                    mode='train', 
                                    year=args.train_year)
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)

    if args.test:
        dataset_test =  dataset_class_test(data_dir=path, 
                                    sample_rate=args.sample_rate, 
                                    midi_rate=args.midi_rate,
                                    sample_length=args.sample_length, 
                                    dilation=args.sample_length, 
                                    normalize=args.normalize, 
                                    mode='test', 
                                    year=args.test_year)
        dataloader_test =  torch.utils.data.DataLoader(dataset_test,  batch_size=1, shuffle=False)

    if args.valid:
        dataset_valid = dataset_class_test(data_dir=path, 
                                    sample_rate=args.sample_rate, 
                                    midi_rate=args.midi_rate,
                                    sample_length=args.sample_length, 
                                    dilation=args.sample_length, 
                                    normalize=args.normalize, 
                                    mode='valid', 
                                    year=args.test_year)
        dataloader_valid = torch.utils.data.DataLoader(dataset_valid, batch_size=args.batch_size, shuffle=False)

    if not os.path.exists("output"):
        os.makedirs("output")

    if ((local_rank == 0 and role_rank == 0) or 'LOCAL_RANK' not in os.environ) and args.wandb:
        wandb.require("core")

        wandb.login(key = "12345") 

        run = wandb.init(project="piano-ssm",
                    config = vars(args),
                    )
        
        output_file = f"output/{os.path.basename(__file__)}_{run.name}.py"
        shutil.copy(__file__, output_file)
        wandb.save(output_file)

        for root, _, files in os.walk('src'):
            for file in files:
                if file.endswith(".py"):
                    output_file = f"output/{run.name}_{os.path.basename(file)}"
                    shutil.copy(f"{root}/{file}", output_file)
                    wandb.save(output_file)

    if args.activation == 'ReLU':
        activation = torch.nn.ReLU()
    elif args.activation == 'LeakyReLU':
        activation = torch.nn.LeakyReLU()
    elif args.activation == 'Identity':
        activation = torch.nn.Identity()
    elif args.activation == 'Tanh':
        activation = torch.nn.Tanh()
    else:
        print(f'act: {args.activation}')
        raise NotImplementedError('Activation function not implemented')
    
    model_base = s_edge
    
    if args.dataset == "maestro":
        if args.train_year == "all":
            n_instruments=10
        else:
            n_instruments=1
    elif args.dataset == "maps":
        if args.train_year == "all":
            n_instruments=2
        else:
            n_instruments=1
    try:
        model_class = globals()[args.model]
        model = model_class(sample_rate=args.sample_rate, midi_rate=args.midi_rate, activation=activation, model_base=model_base, n_instruments=n_instruments).to(local_rank)
        if args.weights is not None:
            model.load_state_dict(torch.load(args.weights, weights_only=True))
    except KeyError:
        print(f"No class named {args.model} found in the global scope.")            
    
    
    if torch.cuda.device_count() > 1 and 'LOCAL_RANK' in os.environ:
        print("Multiple GPUs detected")
        torch.distributed.init_process_group(backend="nccl")
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])#, find_unused_parameters=True)

        sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
        dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=False, num_workers=0, sampler=sampler)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0001)

    loss_fn_train_list = []
    loss_fn_test_list = []

    for c in args.criterion:
        if c == "CombinedSpectralLoss":
            loss_fn = CombinedSpectralLoss(frame_length=args.sample_rate, 
                                           stride=args.sample_rate//10, 
                                           sample_rate=args.sample_rate)
        elif c == "SpectralLoss":
            fft_sizes = [int(args.sample_rate/48000*i) for i in [192,384,768,1526,3072,6144,12288]]
            loss_fn = SpectralLoss(loss_type='L1', mag_weight=1.0, logmag_weight=1.0,fft_sizes=fft_sizes)
        else:
            assert False, f"No criterion named {c} found in the global scope."
        loss_fn_train_list.append(loss_fn)

        loss_fn_test_list.append(CombinedSpectralLoss(frame_length=args.sample_rate, 
                                           stride=args.sample_rate//4, 
                                           sample_rate=args.sample_rate))
        fft_sizes = [int(args.sample_rate/48000*i) for i in [192,384,768,1526,3072,6144,12288]]
        loss_fn_test_list.append(SpectralLoss(loss_type='L1', mag_weight=1.0, logmag_weight=1.0,fft_sizes=fft_sizes))

    if args.scheduler:
        max_lr = args.lr*5
        min_lr = args.lr/10
        first_cycle_steps = int(args.epochs*0.4)
        cycle_mult  = 1.0
        warmup_steps = int(args.epochs*0.25)
        gamma = (min_lr/max_lr)**(1/(args.epochs/first_cycle_steps))
        scheduler = CosineAnnealingWarmupRestarts(optimizer, 
                                                first_cycle_steps=first_cycle_steps, 
                                                cycle_mult=cycle_mult, 
                                                max_lr=max_lr, 
                                                min_lr=min_lr, 
                                                warmup_steps=warmup_steps, 
                                                gamma=gamma)
    else:
        scheduler = None

    train_log_path = None
    test_log_path = None
    valid_log_path = None

    len_epochs = args.epochs if not args.finetune else args.epochs+20
    for epoch in range(args.epochs+20):

        if args.finetune and epoch == args.epochs:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr/5)
            scheduler = None
            fft_sizes = [int(args.sample_rate/48000*i) for i in [192,384,768,1526,3072,6144,12288]]
            loss_fn_train_list = [SpectralLoss(loss_type='L1', mag_weight=1.0, logmag_weight=1.0,fft_sizes=fft_sizes)]
            loss_fn_test_list = [SpectralLoss(loss_type='L1', mag_weight=1.0, logmag_weight=1.0,fft_sizes=fft_sizes)]

        train_dict =     train(model=model, dataloader=dataloader_train, dataset=dataset_train, optimizer=optimizer, loss_fn_train_list=loss_fn_train_list, loss_fn_test_list=loss_fn_test_list, args=args, local_rank=local_rank, role_rank=role_rank, epoch=epoch, scheduler=scheduler, log_path=train_log_path, mode='train')
        if args.valid:
            valid_dict = train(model=model, dataloader=dataloader_valid, dataset=dataset_valid, optimizer=optimizer, loss_fn_train_list=loss_fn_train_list, loss_fn_test_list=loss_fn_test_list, args=args, local_rank=local_rank, role_rank=role_rank, epoch=epoch, scheduler=scheduler, log_path=valid_log_path, mode='valid')
        if args.test:
            test_dict =  train(model=model, dataloader=dataloader_test,  dataset=dataset_test,  optimizer=optimizer, loss_fn_train_list=loss_fn_train_list, loss_fn_test_list=loss_fn_test_list, args=args, local_rank=local_rank, role_rank=role_rank, epoch=epoch, scheduler=scheduler, log_path=test_log_path,  mode='test ')

        train_running_loss = train_dict["running_loss"]
        train_running_loss_dict = train_dict["running_loss_dict"]
        train_log_path = train_dict["log_path"]
        train_counter = train_dict["counter"]

        if args.valid:
            valid_running_loss = valid_dict["running_loss"]
            valid_running_loss_dict = valid_dict["running_loss_dict"]
            valid_log_path = valid_dict["log_path"]
            valid_counter = valid_dict["counter"]

        if args.test:
            test_running_loss = test_dict["running_loss"]
            test_running_loss_dict = test_dict["running_loss_dict"]
            test_log_path = test_dict["log_path"]
            test_counter = test_dict["counter"]
        
        if args.wandb and ((local_rank == 0 and role_rank == 0) or 'LOCAL_RANK' not in os.environ):
            test_epoch_loss = 0
            valid_epoch_loss = 0
            train_epoch_loss = train_running_loss.item()  / train_counter
            if args.valid:
                valid_epoch_loss = valid_running_loss.item() / valid_counter
            if args.test:
                test_epoch_loss = test_running_loss.item() / test_counter
            lr = scheduler.get_lr()[0] if scheduler is not None else optimizer.param_groups[0]['lr']
            
            wandb.log({ "epoch": epoch, 
                        "train_loss": train_epoch_loss,
                        "valid_loss": valid_epoch_loss,
                        "test_loss": test_epoch_loss,
                        "lr": lr,
                        
                    }, step=epoch)
            
            for key in train_running_loss_dict.keys():
                wandb.log({f"train_{key}": train_running_loss_dict[key].item() / train_counter}, step=epoch)
            if args.valid:
                for key in valid_running_loss_dict.keys():
                    wandb.log({f"valid_{key}": valid_running_loss_dict[key].item() / valid_counter}, step=epoch)
            if args.test:
                for key in test_running_loss_dict.keys():
                    wandb.log({f"test_{key}": test_running_loss_dict[key].item() / test_counter}, step=epoch)
            
        if epoch % 1 == 0:
            model_name = os.path.join("output", args.model + "_" + str(epoch) + "_model.pth")
            torch.save(model.state_dict(), model_name)
            if args.wandb and ((local_rank == 0 and role_rank == 0) or 'LOCAL_RANK' not in os.environ):
                model_name = os.path.join("output", str(run.name) + "_" + str(epoch) + "_model.pth")
                wandb.save(model_name)
        
        if scheduler is not None:
            scheduler.step()

def train(model, dataloader, dataset, optimizer, loss_fn_train_list, loss_fn_test_list, args, local_rank, role_rank, epoch, scheduler, log_path, mode: str = "train"):
    if mode == "train":
        model.train()
    else:
        model.eval()

    descriptor = mode
    running_loss = 0.0
    running_loss_dict = {}
    counter = 0


    step_size = args.step_size
    step_counter = 1

    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    pbar.set_description(f"Epoch: 0, {descriptor} loss: {0:.4f}, LR: {0:.6f}")

    for batch_idx, d in pbar:
        audio = d["audio"]
        midi = d["midi"]
        path = d["path"]
        year = d["year"]
        loss = torch.tensor(0.0, device=local_rank)
        loss_dict = {}

        counter += 1

        if log_path is None and args.wandb and ((local_rank == 0 and role_rank == 0) or 'LOCAL_RANK' not in os.environ):
            log_path = path[0]
            sample = audio[0]
            sample = sample * dataset.audio_std + dataset.audio_mean
            waveform_label_train = wandb.Audio(sample.detach().cpu().flatten().numpy(), sample_rate=args.sample_rate)
            wandb.log({f"label_{descriptor}" : waveform_label_train}, step=epoch)
            
            sample = midi[0]
            image = wandb.Image(sample.unsqueeze(2).detach().cpu().numpy().transpose())

            wandb.log({f"label_midi_{descriptor}" : image}, step=epoch)

        midi = midi.to(local_rank)
        audio = audio.to(local_rank)
        year = year.to(local_rank)
        
        if mode == "train":
            pred = model({"midi": midi, "year": year})
            for loss_fn in loss_fn_train_list:
                loss_, loss_dict_ = loss_fn(pred, {"audio" : audio}, epoch)
                loss += loss_
                loss_dict.update(loss_dict_)

            loss.backward() 
            if step_counter > step_size:
                optimizer.step()
                optimizer.zero_grad(True)
                step_counter = 0
            step_counter += 1
        
        else:
            with torch.no_grad():
                pred = model({"midi": midi, "year": year})
                audio = audio * dataset.audio_std + dataset.audio_mean
                pred["audio"] = pred["audio"] * dataset.audio_std + dataset.audio_mean
                for loss_fn in loss_fn_test_list:
                    loss_, loss_dict_ = loss_fn(pred, {"audio" : audio}, epoch)
                    loss_dict.update({f"{loss_fn.__class__.__name__}": loss_})
                    loss += loss_
                    loss_dict.update(loss_dict_)

                audio = audio / dataset.audio_std - dataset.audio_mean
                pred["audio"] = pred["audio"] / dataset.audio_std - dataset.audio_mean
                
        with torch.no_grad():
            running_loss += loss.detach()
            for key in loss_dict.keys():
                if key in running_loss_dict:
                    running_loss_dict[key] += loss_dict[key]
                else:
                    running_loss_dict[key] = loss_dict[key]
        
        # Accumulate total loss
        pbar.set_description(f"Epoch: {epoch}, {descriptor} loss: {running_loss.item() / (batch_idx + 1):.4f}, LR: {scheduler.get_lr()[0] if scheduler is not None else optimizer.param_groups[0]['lr']:.6f}")
        if log_path in path and args.wandb and ((local_rank == 0 and role_rank == 0)or 'LOCAL_RANK' not in os.environ):
            sample = pred["audio"][path.index(log_path)]
            sample = sample * dataset.audio_std + dataset.audio_mean
            waveform_pred_train = wandb.Audio(sample.detach().cpu().flatten().float().numpy(), sample_rate=args.sample_rate)
            wandb.log( {f"pred_{descriptor}" : waveform_pred_train}, step=epoch)

            if mode == 'train':
                sample = audio[path.index(log_path)]
                sample = sample * dataset.audio_std + dataset.audio_mean
                waveform_label_train = wandb.Audio(sample.detach().cpu().flatten().float().numpy(), sample_rate=args.sample_rate)
                wandb.log({f"label_{descriptor}" : waveform_label_train}, step=epoch)

    dict = {"running_loss": running_loss,
            "running_loss_dict": running_loss_dict,
            "log_path": log_path,
            "counter": counter
            }
    
    del audio
    del midi
    del pred
    del loss

    return dict
# %%
if __name__ == "__main__":
    main()
    
