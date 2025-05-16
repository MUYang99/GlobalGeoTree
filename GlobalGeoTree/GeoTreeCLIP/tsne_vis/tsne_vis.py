import torch
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import argparse
import sys
import os
import random
import json
import warnings

# Add parent directory to path to import custom modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataloader import GGTDataset  # Assuming GGTDataset is in dataloader.py
from models import GeoTreeClip  # Assuming GeoTreeClip is in models.py

# 忽略来自 threadpoolctl 的特定 AttributeError
warnings.filterwarnings("ignore", category=UserWarning, module="threadpoolctl", message=".*AttributeError.*")


# 或者更精确地匹配，但这可能需要知道具体的警告信息格式
# warnings.filterwarnings("ignore", message=".*'NoneType' object has no attribute 'split'.*", category=UserWarning)

def parse_args():
    parser = argparse.ArgumentParser(
        description="t-SNE visualization of GeoTreeClip combined features.")
    parser.add_argument('--csv_path', type=str, default="../GlobalGeoTree-10kEval-300.csv",
                        help="Path to the metadata CSV file (e.g., GlobalGeoTree-10kEval-90.csv). Used for context if needed, primary selections from JSON.")
    parser.add_argument('--shards_path', type=str, default="../GGT-Eval/GGT-10kEval-300.tar",
                        help="Path to the dataset shards (e.g., GGT-10kEval-90.tar)")
    parser.add_argument('--checkpoint_path', type=str,
                        default="../results/20250512_103419/checkpoints/latest_model_43.pth",
                        help="Path to the model checkpoint (.pth file)")
    parser.add_argument('--selections_json_path', type=str, default='./tsne_selections.json',
                        help="Path to the JSON file with pre-selected taxa.")
    parser.add_argument('--level', type=str, default='family', choices=['family', 'genus', 'species'],
                        help="Taxonomic level for t-SNE visualization.")
    parser.add_argument('--output_dir', type=str, default='./tsne_results', help="Directory to save the t-SNE plot")
    # parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument('--batch_size', type=int, default=256, help="Batch size for feature extraction")
    parser.add_argument('--num_workers', type=int, default=1, help="Number of workers for dataloader")
    parser.add_argument('--tsne_perplexity', type=float, default=30.0, help="Perplexity for t-SNE")
    parser.add_argument('--tsne_n_iter', type=int, default=1000, help="Number of iterations for t-SNE optimization")
    return parser.parse_args()


def main(seed):
    args = parse_args()

    # Setup seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Set device
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load pre-selected taxa from JSON
    try:
        with open(args.selections_json_path, 'r') as f:
            selections_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Selections JSON file not found at {args.selections_json_path}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {args.selections_json_path}")
        sys.exit(1)

    level = args.level
    target_names_for_plot = []
    category_map_for_plot = {}
    plot_title_suffix = ""
    legend_title = ""
    filter_source_family = None
    filter_source_genus = None

    try:
        if level == 'family':
            selected_taxa_details = selections_data['family_level']['selected_families']
            target_names_for_plot = [t['name'] for t in selected_taxa_details]
            category_map_for_plot = {t['name']: t['category'] for t in selected_taxa_details}
            plot_title_suffix = "Family-Level"
            legend_title = "Family (Rarity Category)"
            if not target_names_for_plot:
                raise ValueError("No families selected for family level.")
        elif level == 'genus':
            filter_source_family = selections_data['genus_level']['source_family_from_initial_selection']
            target_names_for_plot = selections_data['genus_level']['selected_genera']
            category_map_for_plot = {name: name for name in target_names_for_plot}  # Genus names themselves
            plot_title_suffix = f"Genus-Level within Family '{filter_source_family}'"
            legend_title = "Genus"
            if not filter_source_family or not target_names_for_plot:
                raise ValueError("Missing source_family or selected_genera for genus level.")
        elif level == 'species':
            filter_source_family = selections_data['species_level']['source_family']
            filter_source_genus = selections_data['species_level']['source_genus_from_selected_genera']
            target_names_for_plot = selections_data['species_level']['selected_species']
            category_map_for_plot = {name: name for name in target_names_for_plot}  # Species names themselves
            plot_title_suffix = f"Species-Level within Genus '{filter_source_genus}' (Family '{filter_source_family}')"
            legend_title = "Species"
            if not filter_source_family or not filter_source_genus or not target_names_for_plot:
                raise ValueError("Missing source_family, source_genus, or selected_species for species level.")
    except (KeyError, TypeError) as e:
        print(
            f"Error: JSON file {args.selections_json_path} is not in the expected format for level '{level}'. Error: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"Error in selection data for level '{level}': {e}")
        sys.exit(1)

    print(f"Loaded {len(target_names_for_plot)} pre-selected taxa for level '{level}' from {args.selections_json_path}")
    if filter_source_family:
        print(f"  Filtering for Family: {filter_source_family}")
    if filter_source_genus:
        print(f"  Filtering for Genus: {filter_source_genus}")

    # Load model
    model = GeoTreeClip().to(device)
    try:
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
    except FileNotFoundError:
        print(f"Error: Model checkpoint not found at {args.checkpoint_path}")
        sys.exit(1)
    except KeyError:
        print(f"Error: 'model_state_dict' not found in checkpoint. Please ensure it's a valid model checkpoint.")
        sys.exit(1)
    print(f"Model loaded from {args.checkpoint_path}")

    # Initialize Dataloader
    print("Initializing Dataloader...")
    try:
        actual_dataset = GGTDataset(args.shards_path, batch_size=args.batch_size, num_workers=args.num_workers,
                                    resampled=False)
        dataloader = actual_dataset.get_dataloader()
    except Exception as e:
        print(f"Error initializing dataset/dataloader: {e}")
        sys.exit(1)

    # Extract features
    print(f"Extracting features for selected taxa at {level} level...")
    all_features = []
    all_labels_for_plot = []  # Renamed from all_family_labels_for_plot

    with torch.no_grad():
        for batch_idx, (images, text_data_batch, aux_data, img_mask, aux_mask) in enumerate(dataloader):
            images = images.to(device).float()
            img_mask = img_mask.to(device)
            aux_mask = aux_mask.to(device)

            batch_features, _ = model(images, text_data_batch, aux_data, img_mask, aux_mask)

            norm = batch_features.norm(dim=-1, keepdim=True)
            batch_features = batch_features / (norm + 1e-8)
            batch_features = torch.nan_to_num(batch_features, nan=0.0, posinf=1.0, neginf=-1.0)
            batch_features = torch.clamp(batch_features, min=-1e4, max=1e4)

            for i in range(len(text_data_batch)):
                sample_text_data = text_data_batch[i]
                if not isinstance(sample_text_data, dict):
                    print(f"Warning: sample_text_data is not a dict in batch {batch_idx}, sample {i}.")
                    continue

                current_taxon_name_from_sample = None
                include_sample = False

                if level == 'family':
                    if 'level1_family' in sample_text_data:
                        current_taxon_name_from_sample = sample_text_data['level1_family']
                        if current_taxon_name_from_sample in target_names_for_plot:
                            include_sample = True
                elif level == 'genus':
                    if 'level1_family' in sample_text_data and 'level2_genus' in sample_text_data:
                        if sample_text_data['level1_family'] == filter_source_family:
                            current_taxon_name_from_sample = sample_text_data['level2_genus']
                            if current_taxon_name_from_sample in target_names_for_plot:
                                include_sample = True
                elif level == 'species':
                    if 'level1_family' in sample_text_data and \
                            'level2_genus' in sample_text_data and \
                            'level3_species' in sample_text_data:
                        if sample_text_data['level1_family'] == filter_source_family and \
                                sample_text_data['level2_genus'] == filter_source_genus:
                            current_taxon_name_from_sample = sample_text_data['level3_species']
                            if current_taxon_name_from_sample in target_names_for_plot:
                                include_sample = True

                if include_sample:
                    all_features.append(batch_features[i].cpu().numpy())
                    all_labels_for_plot.append(current_taxon_name_from_sample)

            if (batch_idx + 1) % 10 == 0:
                print(
                    f"  Processed batch {batch_idx + 1}, collected {len(all_features)} features so far.")

    if not all_features:
        print("No features extracted for the selected taxa. Check selection criteria, JSON file, and dataset content.")
        sys.exit(0)

    features_np = np.array(all_features)
    labels_np = np.array(all_labels_for_plot)
    print(f"Total features extracted: {features_np.shape[0]}")

    if not np.all(np.isfinite(features_np)):
        print("Warning: Features contain NaN or Inf values even after cleaning.")
        nan_inf_rows = ~np.all(np.isfinite(features_np), axis=1)
        print(f"Number of rows with NaN/Inf values: {np.sum(nan_inf_rows)}")
        features_np = features_np[~nan_inf_rows]
        labels_np = labels_np[~nan_inf_rows]
        print(f"Removed problematic rows. New feature shape: {features_np.shape}")
        if features_np.shape[0] == 0:
            print("Error: All features were NaN/Inf. Cannot proceed with t-SNE.")
            sys.exit(1)
        if features_np.shape[0] < args.tsne_perplexity:
            print(
                f"Warning: Number of samples ({features_np.shape[0]}) is less than perplexity ({args.tsne_perplexity}). Adjusting perplexity.")
            args.tsne_perplexity = max(1.0, float(features_np.shape[0] - 1))

    print(f"Total features for t-SNE after cleaning: {features_np.shape[0]}")

    # Perform t-SNE
    print(f"Performing t-SNE (perplexity={args.tsne_perplexity}, n_iter={args.tsne_n_iter})...")
    tsne = TSNE(n_components=2, random_state=seed, perplexity=min(args.tsne_perplexity, features_np.shape[0] - 1),
                n_iter=args.tsne_n_iter, init='random', learning_rate='auto')
    tsne_results = tsne.fit_transform(features_np)
    print("t-SNE completed.")

    # Plotting
    os.makedirs(args.output_dir, exist_ok=True)
    unique_labels_in_plot = sorted(list(set(all_labels_for_plot)))

    if len(unique_labels_in_plot) <= 10:
        colors = plt.cm.get_cmap('tab10', len(unique_labels_in_plot))
    else:
        colors = plt.cm.get_cmap('viridis', len(unique_labels_in_plot))
    color_map = {label: colors(i) for i, label in enumerate(unique_labels_in_plot)}

    # Apply demo plot style
    plt.figure(figsize=(15, 8))

    for i, label_name in enumerate(unique_labels_in_plot):
        indices = np.where(labels_np == label_name)
        plot_label_text = label_name
        plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1],
                    label=plot_label_text,
                    color=color_map[label_name],
                    alpha=0.8,  # from demo (was 0.7)
                    s=50)  # from demo

    # Title from demo style
    title_fontsize = 16  # from demo
    # plot_title_suffix is already defined in the script (e.g., "Family-Level")
    # For this script, the model is GeoTreeClip
    model_specific_title_suffix = "(GeoTreeClip)"
    plt.title(
        f't-SNE Visualization ({plot_title_suffix} {model_specific_title_suffix})\n'
        f'Seed: {seed}, Perplexity: {args.tsne_perplexity}, Iterations: {args.tsne_n_iter}',
        fontsize=title_fontsize,
        pad=20  # from demo
    )

    # Remove axis labels, ticks, and grid as per demo
    # plt.xlabel('t-SNE Dimension 1') # Removed
    # plt.ylabel('t-SNE Dimension 2') # Removed
    plt.xticks([])
    plt.yticks([])
    plt.tick_params(axis='both', which='both', length=0)
    plt.grid(False)

    # Legend from demo style (keeping dynamic legend_title from script)
    plt.legend(title=legend_title,  # legend_title is already defined
               bbox_to_anchor=(1.03, 1),
               loc='upper left',
               fontsize=15,  # from demo (was 10)
               # title_fontsize=12, # Not used in demo's direct plt.legend call, can be added if needed
               frameon=True,
               # framealpha=0.5, # Not in demo, can add if desired
               # shadow=True, # Not in demo, can add if desired
               borderpad=1,
               labelspacing=0.8
               )

    # Layout from demo
    plt.tight_layout(rect=[0, 0, 0.82, 1])

    plot_filename = os.path.join(args.output_dir,
                                 f'{level}_tsne_seed{seed}_perp{int(args.tsne_perplexity)}.png')
    plt.savefig(plot_filename, bbox_inches='tight', dpi=300)  # Added dpi=300 from demo
    print(f"t-SNE plot saved to {plot_filename}")


if __name__ == '__main__':
    seeds = [21, 42, 62]
    for seed in seeds:
        main(seed)