import torch
from transformers import CLIPProcessor, CLIPModel
import pandas as pd
import sys
from torchvision import transforms
import numpy as np
import argparse
from typing import Dict

sys.path.append('..')
from dataloader import GGTDataset


def load_trained_checkpoint(checkpoint_path, device):
    """Load a trained CLIP model from checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load the model to

    Returns:
        model: Loaded CLIP model
        processor: CLIP processor
    """
    print(f"Loading checkpoint from {checkpoint_path}")

    # Load the base model and processor
    model_name = "openai/clip-vit-base-patch16"
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} with loss {checkpoint['loss']:.4f}")

    return model, processor


def run_experiment(seed: int = None) -> Dict[str, Dict[str, float]]:
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Set device
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load and prepare the text queries from CSV
    df = pd.read_csv('../GlobalGeoTree-10kEval-300.csv')
    # df = pd.read_csv('../GlobalGeoTree-10kEval-90.csv')
    # Get unique combinations of the text levels
    unique_texts = df[['level0', 'level1_family', 'level2_genus', 'level3_species']].drop_duplicates()
    print(unique_texts)

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
    print(text_queries)

    level_to_idx = {
        'level3_species': {row['level3_species']: i for i, row in unique_texts.iterrows()},
        'level2_genus': {row['level2_genus']: i for i, row in unique_texts.iterrows()},
        'level1_family': {row['level1_family']: i for i, row in unique_texts.iterrows()}
    }

    print(f"Number of unique taxa:")
    for level, mapping in level_to_idx.items():
        print(f"{level}: {len(mapping)}")

    # Load the model and processor
    print("Loading CLIP model...")

    # Choose whether to use base model or trained checkpoint
    use_trained_checkpoint = True  # Set to True to use trained checkpoint
    if use_trained_checkpoint:
        # checkpoint_path = './checkpoints/3_shot/clip_3shot_epoch_10.pt'  # Change to your checkpoint path
        # checkpoint_path = './checkpoints/1_shot-90/clip_1shot_epoch_10.pt'

        # checkpoint_path = './checkpoints/1_shot-300/clip_1shot_epoch_10.pt'
        checkpoint_path = './checkpoints/3_shot-300/clip_3shot_epoch_10.pt'

        model, processor = load_trained_checkpoint(checkpoint_path, device)
    else:
        model_name = "openai/clip-vit-base-patch16"
        model = CLIPModel.from_pretrained(model_name).to(device)
        processor = CLIPProcessor.from_pretrained(model_name)
        model.eval()

    # Initialize dataloader
    # test_shards = "../GGT-Eval/webdataset/test/test-000000.tar"
    test_shards = "../GGT-Eval/GGT-10kEval-300.tar"
    # test_shards = "./3_shot_splits/query_3shot.tar"

    # test_shards = "./1_shot_splits/query_1shot-300.tar"
    # test_shards = "./3_shot_splits/query_3shot-300.tar"

    test_dataset = GGTDataset(test_shards, batch_size=256, num_workers=1, resampled=False)
    test_loader = test_dataset.get_dataloader()

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
        text_features = model.get_text_features(**text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # Initialize accuracy metrics for each level
    metrics = {
        level: {'correct_top1': 0, 'correct_top5': 0, 'total': 0}
        for level in ['level3_species', 'level2_genus', 'level1_family']
    }

    # Process all samples in the batch
    for batch_idx, (images, text_data, auxiliary_data, image_mask, aux_mask) in enumerate(test_loader):
        batch_size = images.shape[0]
        print(f"\nProcessing Batch {batch_idx + 1} with {batch_size} samples")

        # Get ground truth for all levels
        gt_indices = {}
        for level in ['level3_species', 'level2_genus', 'level1_family']:
            gt_names = [text_data[i][level] for i in range(batch_size)]
            gt_indices[level] = torch.tensor([level_to_idx[level][name] for name in gt_names], device=device)

        # Process all valid timesteps for the entire batch
        all_probs = []  # Will store probabilities for all samples and timesteps
        valid_timesteps = np.zeros(batch_size)

        # Process each timestep
        for t in range(12):
            try:
                # Use image_mask to check if this timestep is valid
                valid_samples = image_mask[:, t]  # Shape: [batch_size]
                if not valid_samples.any():
                    print(f"Skipping timestep {t} - no valid samples")
                    continue

                # Get images at timestep t for valid samples and select RGB channels
                valid_batch_images = images[valid_samples, t, :3]  # Shape: [n_valid, 3, 5, 5]

                # Convert BGR to RGB for all valid images
                valid_batch_images = valid_batch_images.flip(1)  # Flip the channel dimension

                # Convert to PIL Images and process using CLIP processor
                pil_images = [transforms.ToPILImage()(img) for img in valid_batch_images]
                image_inputs = processor(
                    images=pil_images,
                    return_tensors="pt",
                    padding=True
                )
                image_inputs = {k: v.to(device) for k, v in image_inputs.items()}

                with torch.no_grad():
                    # Encode all valid images in the batch at once
                    image_features = model.get_image_features(**image_inputs)
                    # Normalize image features
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                    logits_per_image = model.logit_scale.exp() * image_features @ text_features.t()
                    text_probs = logits_per_image.softmax(dim=-1).cpu().numpy()

                    # Create a batch_size array of probabilities, filling in zeros for invalid samples
                    batch_probs = np.zeros((batch_size, len(text_queries)))
                    batch_probs[valid_samples.cpu().numpy()] = text_probs
                    all_probs.append(batch_probs)
                    valid_timesteps[valid_samples.cpu().numpy()] += 1

            except Exception as e:
                print(f"Error processing timestep {t}: {str(e)}")
                continue

        # Process results for each sample
        for sample_idx in range(batch_size):
            if valid_timesteps[sample_idx] == 0:
                print(f"Warning: No valid predictions for sample {sample_idx}")
                continue

            # Calculate mean probabilities across valid timesteps for this sample
            sample_probs = np.array([timestep_probs[sample_idx] for timestep_probs in all_probs])
            valid_mask = sample_probs.sum(axis=1) != 0  # shape: [num_timesteps]
            mean_probs = sample_probs[valid_mask].mean(axis=0)
            # mean_probs = np.mean(sample_probs, axis=0)
            # print("sum(mean_probs):", f"{mean_probs.sum() * 100: 5.1f}%")

            top_5_indices = mean_probs.argsort()[-5:][::-1]
            # For each taxonomic level
            for level in ['level3_species', 'level2_genus', 'level1_family']:
                pred_taxa = [unique_texts.iloc[idx][level] for idx in top_5_indices]
                pred_level_indices = [level_to_idx[level][taxon] for taxon in pred_taxa]

                # Update accuracy metrics
                if gt_indices[level][sample_idx] == pred_level_indices[0]:  # Top-1 accuracy
                    metrics[level]['correct_top1'] += 1
                if gt_indices[level][sample_idx] in pred_level_indices:  # Top-5 accuracy
                    metrics[level]['correct_top5'] += 1
                metrics[level]['total'] += 1

            # print(f"\nResults for Sample {sample_idx + 1}:")
            # print(
            #     f"Ground truth: {text_data[sample_idx]['level1_family']} / {text_data[sample_idx]['level2_genus']} / {text_data[sample_idx]['level3_species']}")
            # print(f"Top 5 predictions:")
            # for idx in top_5_indices:
            #     print(f"{text_queries[idx]:<100} {mean_probs[idx] * 100:5.1f}%")

    # Calculate accuracies and return results
    results = {}
    print(f"\nResults for seed: {seed if seed is not None else 'None'}")
    for level in ['level3_species', 'level2_genus', 'level1_family']:
        total = metrics[level]['total']
        if total > 0:
            top1_accuracy = (metrics[level]['correct_top1'] / total) * 100
            top5_accuracy = (metrics[level]['correct_top5'] / total) * 100
            print(f"\n{level} Results on {total} samples:")
            print(f"Top-1 Accuracy: {top1_accuracy:.2f}%")
            print(f"Top-5 Accuracy: {top5_accuracy:.2f}%")
            results[level] = {'top1': top1_accuracy, 'top5': top5_accuracy}
        else:
            print(f"\nNo valid samples processed for {level}")
            results[level] = {'top1': 0.0, 'top5': 0.0}

    return results


def run_multiple_times(num_runs: int = 5, start_seed: int = 42) -> None:
    all_results = []
    for i in range(num_runs):
        seed = start_seed + i
        print(f"\nRunning experiment {i + 1}/{num_runs} with seed {seed}")
        results = run_experiment(seed)
        all_results.append(results)

    # Calculate mean and std for each metric
    print("\n=== Final Statistics ===")
    for level in ['level3_species', 'level2_genus', 'level1_family']:
        top1_scores = [r[level]['top1'] for r in all_results]
        top5_scores = [r[level]['top5'] for r in all_results]

        print(f"\n{level} Statistics:")
        print(f"Top-1 Accuracy: {np.mean(top1_scores):.2f}% ± {np.std(top1_scores):.2f}%")
        print(f"Top-5 Accuracy: {np.mean(top5_scores):.2f}% ± {np.std(top5_scores):.2f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--multi_run', default=True, help='Run multiple times with different seeds')
    parser.add_argument('--num_runs', type=int, default=3, help='Number of runs when using multi_run')
    parser.add_argument('--start_seed', type=int, default=42, help='Starting seed for multiple runs')
    args = parser.parse_args()

    if args.multi_run:
        run_multiple_times(args.num_runs, args.start_seed)
    else:
        run_experiment()


if __name__ == '__main__':
    main()
