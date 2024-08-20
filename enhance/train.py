#!/usr/bin/env python
# coding=utf-8
'''
FilePath     : /enh_yov5/enhance/train.py
Description  :  
Author       : Zhang Xiuyu
LastEditors  : Zhang Xiuyu
LastEditTime : 2024-08-20 09:18:23
'''
import argparse
import os
import torch
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import wandb
from enhance.val import evaluate
from models.unet import UNet
from enhance_utils.dataloader import EnhanceDataset
from enhance_utils.loss import HybridLoss

dir_checkpoint = Path('./checkpoints/')


def train_model(
        model,
        device,
        epochs: int = 5,
        batch_size: int = 1,
        data_dir:str = 'data/images',
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        mix_flag: bool = True,
        alpha: float = 0.5,
        channels: int = 3,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
):
    # 1. Create dataset
    dataset = EnhanceDataset(data_dir, 3)

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging)
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
             val_percent=val_percent, save_checkpoint=save_checkpoint, alpha=alpha)
    )

    print(f'''Starting training:
        Epochs:            {epochs}
        Batch size:        {batch_size}
        Learning rate:     {learning_rate}
        Training size:     {n_train}
        Validation size:   {n_val}
        Checkpoints:       {save_checkpoint}
        Device:            {device.type}
        mix precision:     {mix_flag}
        Hybrid loss alpha: {alpha}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=mix_flag)
    loss_executer = HybridLoss(alpha=alpha, channel=channels)
    global_step = 0

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, gts = batch['image'], batch['gt']

                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                gts = gts.to(device=device, dtype=torch.float32)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=mix_flag):
                    preds = model(images)
                    loss = loss_executer(preds, gts)

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                division_step = n_train // batch_size
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in model.named_parameters():
                            tag = tag.replace('/', '.')
                            if not (torch.isinf(value) | torch.isnan(value)).any():
                                histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            if value.grad is not None:  # 先检查梯度是否为 None
                                if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                                    histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())
                            else:
                                print(f"Parameter {tag} has no gradient.")

                        val_score = evaluate(model, val_loader, device, mix_flag)
                        scheduler.step(val_score)

                        print('Validation PSNR score: {}'.format(val_score))
                        try:
                            experiment.log({
                                'learning rate': optimizer.param_groups[0]['lr'],
                                'validation PSNR': val_score,
                                'images': wandb.Image(images[0].cpu()),
                                'masks': {
                                    'true': wandb.Image(gts[0].float().cpu()),
                                    'pred': wandb.Image(preds.argmax(dim=1)[0].float().cpu()),
                                },
                                'step': global_step,
                                'epoch': epoch,
                                **histograms
                            })
                        except:
                            pass

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
            print(f'Checkpoint {epoch} saved!')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--dataset', '-d', type=str, default='data/images', help='Load model from a .pth file')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--mix', '-m', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--alpha', '-a', type=float, default=0.5, help='Hybrid L1 loss ratio')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--channels', '-c', type=int, default=3, help='Number of channels')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}')
    
    model = UNet(n_channels=args.channels, bilinear=args.bilinear)
    model = model.to(memory_format=torch.channels_last)

    print(f'Network:\n'
          f'\t{model.n_channels} images channels\n'
          f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        del state_dict['mask_values']
        model.load_state_dict(state_dict)
        print(f'Model loaded from {args.load}')

    model.to(device=device)
    try:
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            data_dir=args.dataset,
            device=device,
            val_percent=args.val / 100,
            mix_flag=args.mix,
            alpha=args.alpha,
            channels=args.channels
        )
    except torch.cuda.OutOfMemoryError:
        print('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            data_dir=args.dataset,
            device=device,
            val_percent=args.val / 100,
            mix_flag=args.mix,
            alpha=args.alpha,
            channels=args.channels

        )
