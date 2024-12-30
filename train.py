import os
import pickle
import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
import torch.cuda.amp as amp
import pandas as pd
import numpy as np
from tqdm import tqdm

from config import Config
from dataset import LungDataset
from models.efficient_net import ImprovedEfficientNet
from utils.metrics import IoULoss, AverageMeter, BBoxLoss  # Added BBoxLoss
from utils.bbox_utils import adjust_predictions, validate_with_tta  # Added new utilities


def print_gpu_memory_status():
    """Print GPU memory usage"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i} Memory: {torch.cuda.memory_allocated(i)/1e9:.2f}GB / "
                  f"{torch.cuda.max_memory_allocated(i)/1e9:.2f}GB")

def setup():
    """Initialize distributed training"""
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = '0'
    
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        torch.cuda.empty_cache()
        
    if world_size > 1:
        dist.init_process_group(backend="nccl", init_method='env://')
    
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    return device, rank, world_size, local_rank

def load_data():
    """Load all necessary data"""
    # Load bounding box data
    df = pd.read_csv(Config.bbox_csv_path)
    bbox_dict = {row['Image']: [
        max(0.0, row['Xmin']), max(0.0, row['Ymin']),
        min(1.0, row['Xmax']), min(1.0, row['Ymax'])
    ] for _, row in df.iterrows()}

    # Load pickle files
    with open(Config.series_list_path, 'rb') as f:
        series_list_train = pickle.load(f)
    with open(Config.series_dict_path, 'rb') as f:
        series_dict = pickle.load(f)
    with open(Config.image_dict_path, 'rb') as f:
        image_dict = pickle.load(f)

    return bbox_dict, series_list_train, series_dict, image_dict

def create_data_loader(image_dict, bbox_dict, image_list, world_size, rank):
    dataset = LungDataset(
        image_dict=image_dict,
        bbox_dict=bbox_dict,
        image_list=image_list,
        target_size=Config.image_size,
        transform=Config.train_transforms
    )

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank
    ) if world_size > 1 else None

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=Config.batch_size,
        sampler=sampler,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    return loader, sampler

def main():
    # Setup distributed training
    device, rank, world_size, local_rank = setup()
    
    if rank == 0:
        print(f"CUDA Available: {torch.cuda.is_available()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print_gpu_memory_status()

    # Set random seed
    torch.manual_seed(Config.seed)
    np.random.seed(Config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(Config.seed)

    try:
        # Load data
        bbox_dict, series_list_train, series_dict, image_dict = load_data()

        # Create image list
        image_list_train = []
        for series_id in series_list_train:
            sorted_image_list = series_dict[series_id]['sorted_image_list']
            num_image = len(sorted_image_list)
            selected_idx = [
                int(0.2*num_image), int(0.3*num_image),
                int(0.4*num_image), int(0.5*num_image)
            ]
            image_list_train.extend([sorted_image_list[i] for i in selected_idx])

        if rank == 0:
            print(f'Number of training images: {len(image_list_train)}')

        # Create data loader
        train_loader, train_sampler = create_data_loader(
            image_dict, bbox_dict, image_list_train, world_size, rank
        )

        # Create model
        model = ImprovedEfficientNet(Config.model_name).to(device)
        if world_size > 1:
            model = DistributedDataParallel(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=True,
                broadcast_buffers=False
            )

        # Create optimizer and scheduler
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=Config.base_lr,
            weight_decay=Config.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=Config.epochs,
            eta_min=Config.min_lr
        )

        # Create loss functions and scaler
        criterion = BBoxLoss(size_penalty_weight=Config.BBOX_SIZE_PENALTY_WEIGHT).to(device)
        iou_criterion = IoULoss().to(device)
        scaler = torch.amp.GradScaler()

         # Training loop
        for epoch in range(Config.epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            
            model.train()
            losses = AverageMeter()
            bbox_losses = AverageMeter()
            iou_losses = AverageMeter()
    
            # Warmup learning rate
            if epoch < Config.warmup_epochs:
                lr = Config.base_lr * (epoch + 1) / Config.warmup_epochs
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
    
            iterator = tqdm(train_loader) if rank == 0 else train_loader
            # In the training loop
            for step, (images, targets) in enumerate(iterator):
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
            
                # Clear gradients
                optimizer.zero_grad(set_to_none=True)
            
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(images)
                    
                    # Calculate losses
                    bbox_loss = criterion(outputs, targets)
                    iou_loss = iou_criterion(outputs, targets)
                    
                    # Combined loss with adjusted weights
                    loss = Config.BBOX_LOSS_WEIGHT * bbox_loss + Config.IOU_LOSS_WEIGHT * iou_loss
            
                # Update loss meters
                losses.update(loss.item(), images.size(0))
                bbox_losses.update(bbox_loss.item(), images.size(0))
                iou_losses.update(iou_loss.item(), images.size(0))
            
                # Gradient scaling and backward pass
                scaler.scale(loss).backward()
                
                # Unscale before optimizer step
                scaler.unscale_(optimizer)
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Optimizer step
                scaler.step(optimizer)
                scaler.update()
    
                if rank == 0 and step % 50 == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    print(f'Epoch [{epoch+1}/{Config.epochs}] '
                          f'Step [{step+1}/{len(train_loader)}] '
                          f'Total Loss: {losses.avg:.4f} '
                          f'BBox Loss: {bbox_losses.avg:.4f} '
                          f'IoU Loss: {iou_losses.avg:.4f} '
                          f'LR: {current_lr:.6f}')
    
            if epoch >= Config.warmup_epochs:
                scheduler.step()
    
            # Save model with additional metrics
            if rank == 0 and (epoch + 1) % 5 == 0:
                save_path = os.path.join('weights', f'model_epoch_{epoch+1}.pth')
                os.makedirs('weights', exist_ok=True)
                save_dict = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.module.state_dict() if hasattr(model, 'module') 
                                      else model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': losses.avg,
                    'bbox_loss': bbox_losses.avg,
                    'iou_loss': iou_losses.avg,
                }
                torch.save(save_dict, save_path)
                print(f'Saved checkpoint: {save_path}')
    
                # Validate with TTA if enabled
                if Config.USE_TTA:
                    model.eval()
                    val_predictions = validate_with_tta(model, images)
                    # You might want to add validation metrics here
    
            if rank == 0 and epoch % 5 == 0:
                print("\nGPU Memory Status:")
                print_gpu_memory_status()

    except Exception as e:
        print(f"Error on rank {rank}: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise e

    finally:
        if world_size > 1:
            dist.destroy_process_group()

if __name__ == '__main__':
    main()