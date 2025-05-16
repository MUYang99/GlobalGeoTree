import torch
import pandas as pd
import sys
import numpy as np
import multiprocessing
from typing import Dict, List
import argparse

sys.path.append('..')
from dataloader import GGTDataset
from models import GeoTreeClip


def create_text_data(row):
    """Create text data dictionary in the expected format"""
    return {
        'level0': row['level0'],
        'level1_family': row['level1_family'],
        'level2_genus': row['level2_genus'],
        'level3_species': row['level3_species']
    }


def run_experiment(seed: int = None) -> Dict[str, Dict[str, float]]:
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load and prepare the text queries from CSV
    df = pd.read_csv('../GlobalGeoTree-10kEval-90.csv')

    # Get unique combinations of the text levels
    unique_texts = df[['level0', 'level1_family', 'level2_genus', 'level3_species']].drop_duplicates()

    # Create text data in the expected format
    text_queries = [create_text_data(row) for _, row in unique_texts.iterrows()]

    # Create mappings from taxon name to index for each level
    level_to_idx = {
        'level3_species': {text['level3_species']: i for i, text in enumerate(text_queries)},
        'level2_genus': {text['level2_genus']: i for i, text in enumerate(text_queries)},
        'level1_family': {text['level1_family']: i for i, text in enumerate(text_queries)}
    }

    print(f"Number of unique taxa:")
    for level, mapping in level_to_idx.items():
        print(f"{level}: {len(mapping)}")

    # Load the trained GeoTreeClip model
    model = GeoTreeClip().to(device)
    # checkpoint = torch.load('../results/20250513_014804/checkpoints/latest_model_13.pth', map_location=device)
    checkpoint = torch.load('../results/20250512_103419/checkpoints/best_model.pth', map_location=device)
    # checkpoint = torch.load('../results/20250429_161750/checkpoints/best_model.pth', map_location=device)
    # checkpoint = torch.load('./checkpoints/1_shot/latest_model.pth', map_location=device)
    # checkpoint = torch.load('./checkpoints/1_shot-90/latest_model.pth', map_location=device)
    # checkpoint = torch.load('./checkpoints/1_shot-300/latest_model.pth', map_location=device)
    # checkpoint = torch.load('./checkpoints/3_shot/latest_model.pth', map_location=device)
    # checkpoint = torch.load('./checkpoints/3_shot-90/latest_model.pth', map_location=device)
    # checkpoint = torch.load('./checkpoints/3_shot-300/latest_model.pth', map_location=device)
    # checkpoint = torch.load('../results/20250508_095629/checkpoints/best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Initialize dataloader
    test_shards = "../GGT-Eval/GGT-10kEval-90.tar"
    test_dataset = GGTDataset(test_shards, batch_size=512, num_workers=1, resampled=False)
    test_loader = test_dataset.get_dataloader()

    # Pre-compute text features for all queries
    print("Computing text features for all queries...")
    with torch.no_grad():
        text_features = model.text_encoder(text_queries)
        print(f"Text features shape: {text_features.shape}")
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # Initialize accuracy metrics for each level
    metrics = {
        level: {'correct_top1': 0, 'correct_top5': 0, 'total': 0}
        for level in ['level3_species', 'level2_genus', 'level1_family']
    }

    # Process all samples in the batch
    print("\nStarting evaluation...")
    for batch_idx, (images, text_data, auxiliary_data, image_mask, aux_mask) in enumerate(test_loader):
        batch_size = images.shape[0]
        print(f"\nProcessing Batch {batch_idx + 1} with {batch_size} samples")

        # Move data to device
        images = images.to(device).float()
        image_mask = image_mask.to(device)
        aux_mask = aux_mask.to(device)

        # Get ground truth for all levels
        gt_indices = {}
        for level in ['level3_species', 'level2_genus', 'level1_family']:
            gt_names = [text_data[i][level] for i in range(batch_size)]
            gt_indices[level] = torch.tensor([level_to_idx[level][name] for name in gt_names], device=device)

        # Process batch with model
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            combined_features, _ = model(images, text_data, auxiliary_data, image_mask, aux_mask)

            # Normalize combined features
            combined_features = combined_features / combined_features.norm(dim=-1, keepdim=True)

            # Calculate similarity scores
            logits = 100.0 * combined_features @ text_features.T
            text_probs = logits.softmax(dim=-1).cpu().numpy()

            # For each taxonomic level
            for level in ['level3_species', 'level2_genus', 'level1_family']:
                # Get predictions
                top_5_indices = text_probs.argsort(axis=1)[:, -5:][:, ::-1]  # [batch_size, 5]
                pred_taxa = [text_queries[idx][level] for idx in top_5_indices.flatten()]
                top_5_indices = np.array([level_to_idx[level][taxon] for taxon in pred_taxa]).reshape(
                    batch_size, 5)

                # Update metrics
                metrics[level]['correct_top1'] += (top_5_indices[:, 0] == gt_indices[level].cpu().numpy()).sum()
                metrics[level]['correct_top5'] += np.any(top_5_indices == gt_indices[level].cpu().numpy()[:, None],
                                                         axis=1).sum()
                metrics[level]['total'] += batch_size

    # Calculate and print final accuracies for each level
    results = {}
    print("\nResults for seed:", seed if seed is not None else "None")
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
    parser.add_argument('--num_runs', type=int, default=5, help='Number of runs when using multi_run')
    parser.add_argument('--start_seed', type=int, default=42, help='Starting seed for multiple runs')
    args = parser.parse_args()

    if args.multi_run:
        run_multiple_times(args.num_runs, args.start_seed)
    else:
        run_experiment()


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
