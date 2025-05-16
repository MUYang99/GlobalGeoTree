from huggingface_hub import hf_hub_download
import torch
import open_clip
import pandas as pd
import sys
from torchvision import transforms
import argparse
from typing import Dict

sys.path.append('..')
from dataloader import GGTDataset
import multiprocessing
import numpy as np


# for model_name in ['RN50'] #, 'ViT-B-32', 'ViT-L-14']: #faster loading
# for model_name in ['ViT-B-32']:
#     checkpoint_path = hf_hub_download("chendelong/RemoteCLIP", f"RemoteCLIP-{model_name}.pt", cache_dir='checkpoints')
#     print(f'{model_name} is downloaded to {checkpoint_path}.')

def run_experiment(seed: int = None) -> Dict[str, Dict[str, float]]:
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Set device
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load and prepare the text queries from CSV
    df = pd.read_csv('../GlobalGeoTree-10kEval-90.csv')
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

    # Create mappings from taxon name to index for each level
    level_to_idx = {
        'level3_species': {row['level3_species']: i for i, row in unique_texts.iterrows()},
        'level2_genus': {row['level2_genus']: i for i, row in unique_texts.iterrows()},
        'level1_family': {row['level1_family']: i for i, row in unique_texts.iterrows()}
    }

    print(f"Number of unique taxa:")
    for level, mapping in level_to_idx.items():
        print(f"{level}: {len(mapping)}")

    model_name = 'ViT-B-32'  # @param ['RN50', 'ViT-B-32', 'ViT-L-14']
    model, preprocess, _ = open_clip.create_model_and_transforms(model_name)
    tokenizer = open_clip.get_tokenizer(model_name)

    use_trained_checkpoint = False  # Set to True to use trained checkpoint
    if use_trained_checkpoint:
        path_to_your_checkpoints = './checkpoints/3_shot/remoteclip_3shot_epoch_10.pt'  # Change to your checkpoint path
        checkpoint = torch.load(path_to_your_checkpoints, map_location=device)
        ckpt = checkpoint['model_state_dict']  # Extract only the model weights
    else:
        path_to_your_checkpoints = 'checkpoints/models--chendelong--RemoteCLIP/snapshots/bf1d8a3ccf2ddbf7c875705e46373bfe542bce38'
        ckpt = torch.load(f"{path_to_your_checkpoints}/RemoteCLIP-{model_name}.pt", map_location=device)

    message = model.load_state_dict(ckpt)
    print(message)

    # Convert model to float32 and move to device
    model = model.to(device).eval()

    # Initialize dataloader with num_workers=0 to avoid multiprocessing issues
    # test_shards = "../GGT-Eval/webdataset/test/test-000000.tar"
    # test_shards = "../GGT-Eval/webdataset/test/test-000000.tar"
    test_shards = "../GGT-Eval/GGT-10kEval-90.tar"
    # test_shards = "./3_shot_splits/query_3shot.tar"
    test_dataset = GGTDataset(test_shards, batch_size=32, num_workers=1, resampled=False)
    test_loader = test_dataset.get_dataloader()

    text = tokenizer(text_queries)
    # Move text tokens to device
    text = text.to(device)

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
        text_features = model.encode_text(text)
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

                # Convert to PIL Images and preprocess in a list comprehension
                processed_images = torch.stack([
                    preprocess(transforms.ToPILImage()(img))
                    for img in valid_batch_images
                ]).to(device)

                with torch.no_grad(), torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    # Encode all valid images in the batch at once
                    image_features = model.encode_image(processed_images)

                    # Normalize image features
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                    # Calculate probabilities for all valid samples at once
                    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1).cpu().numpy()

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
            # print("sum(mean_probs):", f"{mean_probs.sum() * 100: 5.1f}%")

            # Get top predictions
            top_5_indices = mean_probs.argsort()[-5:][::-1]

            # For each taxonomic level
            for level in ['level3_species', 'level2_genus', 'level1_family']:
                pred_taxa = [unique_texts.iloc[idx][level] for idx in top_5_indices]
                pred_level_indices = [level_to_idx[level][taxon] for taxon in pred_taxa]

                # Update accuracy metrics
                if gt_indices[level][sample_idx].cpu().item() == pred_level_indices[0]:  # Top-1 accuracy
                    metrics[level]['correct_top1'] += 1
                if gt_indices[level][sample_idx].cpu().item() in pred_level_indices:  # Top-5 accuracy
                    metrics[level]['correct_top5'] += 1
                metrics[level]['total'] += 1

            # Print results for this sample
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
    multiprocessing.freeze_support()
    main()
