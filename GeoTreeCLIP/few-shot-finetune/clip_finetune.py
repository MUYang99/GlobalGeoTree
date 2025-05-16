import torch
from transformers import CLIPProcessor, CLIPModel
import pandas as pd
import sys
from torchvision import transforms
import numpy as np
import os
from tqdm import tqdm
from torch.optim import AdamW
from torch.utils.data import DataLoader
import webdataset as wds

# Set environment variables before importing other modules
os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.path.append('..')
from dataloader import GGTDataset


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


def train_epoch(model, processor, train_loader, optimizer, device, text_queries, unique_texts):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0  # Counter for number of batches processed

    # Process text inputs once
    text_inputs = processor(
        text=text_queries,
        padding=True,
        return_tensors="pt",
        truncation=True,
        max_length=77
    )
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

    # Compute text features
    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs)  # [num_texts, embed_dim]
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    for batch_idx, (images, text_data, auxiliary_data, image_mask, aux_mask) in enumerate(tqdm(train_loader)):
        batch_size = images.shape[0]  # batch_size = 8

        # Get ground truth text for each sample
        gt_texts = []
        for i in range(batch_size):
            text = " / ".join([
                text_data[i]['level0'],
                text_data[i]['level1_family'],
                text_data[i]['level2_genus'],
                text_data[i]['level3_species']
            ])
            gt_texts.append(text)

        # Find indices of ground truth texts in text_queries
        gt_indices = torch.tensor([text_queries.index(text) for text in gt_texts], device=device)  # [batch_size]

        # Process all valid timesteps at once
        batch_loss = 0
        valid_timesteps = 0

        # Count valid timesteps per sample
        valid_timesteps_per_sample = image_mask.sum(dim=1)  # [batch_size]

        # Reshape images to process all timesteps at once
        all_images = images[:, :, :3].reshape(-1, 3, images.shape[3], images.shape[4])  # [batch_size*12, 3, H, W]
        all_masks = image_mask.reshape(-1)  # [batch_size*12]

        # Only process valid images
        valid_images = all_images[all_masks]  # [num_valid, 3, H, W]
        if len(valid_images) == 0:
            continue

        # Convert BGR to RGB for all valid images at once
        valid_images = valid_images.flip(1)  # [num_valid, 3, H, W]

        # Convert to PIL Images and process
        pil_images = [transforms.ToPILImage()(img) for img in valid_images]
        image_inputs = processor(
            images=pil_images,
            return_tensors="pt",
            padding=True
        )
        image_inputs = {k: v.to(device) for k, v in image_inputs.items()}

        # Forward pass for all valid images at once
        image_features = model.get_image_features(**image_inputs)  # [num_valid, embed_dim]
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Calculate logits
        logits_per_image = model.logit_scale.exp() * image_features @ text_features.t()  # [num_valid, num_texts]

        # Process each sample separately
        start_idx = 0
        for i in range(batch_size):
            num_valid = int(valid_timesteps_per_sample[i].item())
            if num_valid == 0:
                continue

            # Get logits for valid timesteps of this sample
            sample_logits = logits_per_image[start_idx:start_idx + num_valid]  # [num_valid, num_texts]

            # Calculate loss
            loss = torch.nn.functional.cross_entropy(sample_logits, gt_indices[i].expand(num_valid))
            batch_loss += loss
            valid_timesteps += num_valid

            start_idx += num_valid

        if valid_timesteps > 0:
            # Average loss over valid samples
            batch_loss = batch_loss / valid_timesteps

            # Backward pass
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            total_loss += batch_loss.item()
            num_batches += 1  # Increment batch counter

            # Clear cache after each batch
            torch.cuda.empty_cache()

    # Return average loss over all batches
    return total_loss / num_batches if num_batches > 0 else float('inf')


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

    # Load the model and processor
    print("Loading CLIP model...")
    model_name = "openai/clip-vit-base-patch16"
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)

    # Initialize dataloader for support set with smaller batch size
    support_shards = "./3_shot_splits/support_3shot.tar"

    # Initialize dataset with resampled=False
    support_dataset = GGTDataset(
        support_shards,
        batch_size=8,
        num_workers=1,
        resampled=False  # Set to False to avoid infinite sampling
    )
    train_loader = support_dataset.get_dataloader()

    # Print dataset size
    print("Starting training...")

    # Freeze text encoder parameters
    for param in model.text_model.parameters():
        param.requires_grad = False

    # Freeze most of the vision encoder parameters
    # Only keep the last 4 transformer blocks trainable
    for name, param in model.vision_model.named_parameters():
        if 'encoder.layers' in name:
            layer_num = int(name.split('.')[2])
            if layer_num < 8:  # Freeze first 8 layers (keep last 4 trainable)
                param.requires_grad = False
        elif any(layer in name for layer in ['embeddings', 'pre_layrnorm']):
            param.requires_grad = False

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params / total_params * 100:.2f}% of total)")

    # Create parameter groups for different learning rates
    param_groups = [
        {'params': [p for n, p in model.named_parameters() if p.requires_grad and 'vision_model' in n], 'lr': 5e-5},
        {'params': [model.logit_scale], 'lr': 5e-5}
    ]

    optimizer = AdamW(param_groups)

    # Training loop
    num_epochs = 10
    best_loss = float('inf')

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # Train for one epoch
        train_loss = train_epoch(model, processor, train_loader, optimizer, device,
                                 text_queries, unique_texts)

        print(f"Training Loss: {train_loss:.4f}")

        # Save checkpoint if loss improved
        if train_loss < best_loss:
            best_loss = train_loss
            checkpoint_path = os.path.join(checkpoint_dir, f'clip_3shot_epoch_{epoch + 1}.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")


if __name__ == '__main__':
    main()