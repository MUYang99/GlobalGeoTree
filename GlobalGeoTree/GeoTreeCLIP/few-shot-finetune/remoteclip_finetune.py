import torch
import open_clip
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


def train_epoch(model, tokenizer, train_loader, optimizer, device, text_queries, unique_texts, preprocess):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0  # Counter for number of batches processed

    # Process text inputs once
    text_tokens = tokenizer(text_queries).to(device)  # [num_texts, max_len]

    # Compute text features
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)  # [num_texts, embed_dim]
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # Create transform pipeline
    to_pil = transforms.ToPILImage()

    for batch_idx, (images, text_data, auxiliary_data, image_mask, aux_mask) in enumerate(tqdm(train_loader)):
        batch_size = images.shape[0]

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

        # Convert tensors to PIL images and process
        processed_images = []
        for img in valid_images:
            pil_img = transforms.ToPILImage()(img)
            processed_img = preprocess(pil_img)
            processed_images.append(processed_img)

        # Stack processed images and move to device
        processed_images = torch.stack(processed_images).to(device)  # [num_valid, 3, 224, 224]

        # Forward pass
        image_features = model.encode_image(processed_images)  # [num_valid, embed_dim]
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
            num_batches += 1

            # Clear cache after each batch
            torch.cuda.empty_cache()

    # Return average loss over all batches
    return total_loss / num_batches if num_batches > 0 else float('inf')


def main():
    # Set device

    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
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

    # Load the model and tokenizer
    print("Loading RemoteCLIP model...")
    model_name = 'ViT-B-32'
    model, preprocess, _ = open_clip.create_model_and_transforms(model_name)
    tokenizer = open_clip.get_tokenizer(model_name)

    # Load RemoteCLIP checkpoint from local path
    path_to_your_checkpoints = 'checkpoints/models--chendelong--RemoteCLIP/snapshots/bf1d8a3ccf2ddbf7c875705e46373bfe542bce38'
    checkpoint_path = f"{path_to_your_checkpoints}/RemoteCLIP-{model_name}.pt"
    print(f"Loading checkpoint from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    message = model.load_state_dict(ckpt)
    print(message)
    model = model.to(device)

    # Initialize dataloader for support set with smaller batch size
    support_shards = "./3_shot_splits/support_3shot.tar"
    support_dataset = GGTDataset(support_shards, batch_size=8, num_workers=1,
                                 resampled=False)  # Match CLIP's configuration
    train_loader = support_dataset.get_dataloader()

    # Freeze most of the model parameters
    # Only keep the last 4 transformer blocks trainable
    for name, param in model.named_parameters():
        if 'visual' in name:
            # Keep only the last 4 transformer blocks trainable
            if 'transformer.resblocks' in name:
                block_num = int(name.split('.')[3])  # Get block number
                if block_num < len(model.visual.transformer.resblocks) - 4:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            # Freeze early layers
            elif any(layer in name for layer in ['conv1', 'class_embedding', 'positional_embedding', 'ln_pre']):
                param.requires_grad = False
            # Keep ln_post and proj trainable
            else:
                param.requires_grad = True
        else:
            # Freeze all non-visual parameters (text encoder, etc.)
            param.requires_grad = False

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params / total_params * 100:.2f}% of total)")

    # Create parameter groups for different learning rates
    param_groups = [
        {'params': [p for n, p in model.named_parameters() if p.requires_grad and 'visual' in n], 'lr': 5e-5},
        # Vision encoder
        {'params': [model.logit_scale], 'lr': 5e-5}  # Logit scale parameter
    ]

    optimizer = AdamW(param_groups)

    # Training loop
    num_epochs = 10
    best_loss = float('inf')

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # Train for one epoch
        train_loss = train_epoch(model, tokenizer, train_loader, optimizer, device,
                                 text_queries, unique_texts, preprocess)

        print(f"Training Loss: {train_loss:.4f}")

        # Save checkpoint if loss improved
        if train_loss < best_loss:
            best_loss = train_loss
            checkpoint_path = os.path.join(checkpoint_dir, f'remoteclip_3shot_epoch_{epoch + 1}.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")


if __name__ == '__main__':
    main()