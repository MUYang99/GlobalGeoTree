import torch
import pandas as pd
import sys
from torchvision import transforms
import numpy as np
import os
from tqdm import tqdm
from torch.optim import AdamW
from torch.utils.data import DataLoader
import webdataset as wds
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# Set environment variables before importing other modules
os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.path.append('..')
from dataloader import GGTDataset
from models import GeoTreeClip, CLIPContrastiveLoss  # Import the custom model and loss


def get_text_queries(df):
    """Get unique text queries from DataFrame."""
    # Get unique combinations of the text levels
    unique_texts = df[['level0', 'level1_family', 'level2_genus', 'level3_species']].drop_duplicates()

    text_queries = []
    sep_token = " / "
    for _, row in unique_texts.iterrows():
        # Format text query more like natural language
        text = sep_token.join([
            row['level0'],
            row['level1_family'],
            row['level2_genus'],
            row['level3_species']
        ])
        text_queries.append(text)

    return text_queries, unique_texts


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    valid_batches = 0
    progress_bar = tqdm(train_loader, desc='Training')

    # 梯度累积步数
    accumulation_steps = 1
    optimizer.zero_grad()

    for i, batch in enumerate(progress_bar):
        try:
            # 获取数据
            images, text_data, auxiliary_data, image_mask, aux_mask = batch

            # 转换数据类型并移动到设备
            images = images.to(device).float()
            image_mask = image_mask.to(device)
            aux_mask = aux_mask.to(device)

            # 前向传播
            image_features, text_features = model(images, text_data, auxiliary_data, image_mask, aux_mask)

            # 计算损失
            loss = criterion(image_features, text_features)
            loss = loss / accumulation_steps

            # 反向传播
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # 梯度累积
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * accumulation_steps
            valid_batches += 1
            progress_bar.set_postfix({'loss': loss.item() * accumulation_steps})

        except Exception as e:
            print(f"Error in batch {i}: {str(e)}")
            continue

    return total_loss / valid_batches if valid_batches > 0 else float('inf')


def save_checkpoint(model, optimizer, scheduler, epoch, train_loss, checkpoint_dir):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss': train_loss,
    }
    torch.save(checkpoint, f"{checkpoint_dir}/latest_model.pth")


def main():
    # Set device
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Set environment variables for memory optimization
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    # Enable CUDA memory caching
    torch.backends.cudnn.benchmark = True

    # Create checkpoint directory
    checkpoint_dir = './checkpoints/3_shot'
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Load and prepare the text queries from CSV
    df = pd.read_csv('../GlobalGeoTree-10kEval-90.csv')
    text_queries, unique_texts = get_text_queries(df)

    print(f"Number of unique text queries: {len(text_queries)}")

    # Load the GeoTreeCLIP model
    print("Loading GeoTreeCLIP model...")
    model = GeoTreeClip().to(device)
    checkpoint = torch.load('../results/20250429_161750/checkpoints/best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model loaded successfully")

    # Initialize dataloader
    support_shards = "./3_shot_splits/support_3shot.tar"
    support_dataset = GGTDataset(support_shards, batch_size=128, num_workers=1, resampled=False)
    train_loader = support_dataset.get_dataloader()

    # 设置超参数
    learning_rate = 1e-5
    weight_decay = 1e-4
    warmup_epochs = 5
    num_epochs = 10

    # 创建损失函数
    criterion = CLIPContrastiveLoss().to(device)

    # 创建优化器
    optimizer = AdamW([
        {'params': model.temporal_extractor.parameters(), 'lr': learning_rate},
        {'params': model.auxiliary_encoder.parameters(), 'lr': learning_rate},
        {'params': model.text_encoder.parameters(), 'lr': learning_rate * 0.1}
    ], weight_decay=weight_decay)

    # 创建学习率调度器
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2,
        eta_min=learning_rate * 0.01
    )

    # Training loop
    for epoch in range(num_epochs):
        # 学习率预热
        if epoch < warmup_epochs:
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate * (epoch + 1) / warmup_epochs

        # Train for one epoch
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f'Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}')

        # 更新学习率
        scheduler.step()

        # Save checkpoint
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            save_checkpoint(
                model, optimizer, scheduler,
                epoch, train_loss, checkpoint_dir
            )
            print(f'Checkpoint saved at epoch {epoch + 1}')


if __name__ == '__main__':
    main()