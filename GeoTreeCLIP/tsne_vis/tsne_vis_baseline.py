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
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel  # For HuggingFace CLIP
import open_clip  # For RemoteCLIP

# Add parent directory to path to import custom modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataloader import GGTDataset  # Assuming GGTDataset is in dataloader.py

# Note: GeoTreeClip model is not used here

# Ignore specific AttributeError from threadpoolctl if it occurs
warnings.filterwarnings("ignore", category=UserWarning, module="threadpoolctl", message=".*AttributeError.*")


def parse_args():
    parser = argparse.ArgumentParser(
        description="t-SNE visualization of CLIP-extracted mean features over timesteps.")
    parser.add_argument('--shards_path', type=str, default="../GGT-Eval/GGT-10kEval-300.tar",
                        help="Path to the dataset shards.")
    parser.add_argument('--selections_json_path', type=str, default="./tsne_selections.json",
                        help="Path to the JSON file with pre-selected taxa.")
    parser.add_argument('--level', type=str, default='species', choices=['family', 'genus', 'species'],
                        help="Taxonomic level for t-SNE visualization.")

    # Model choice arguments
    parser.add_argument('--model_type', type=str, default='huggingface_clip',
                        choices=['huggingface_clip', 'remote_clip'],
                        help="Type of CLIP model to use.")
    parser.add_argument('--hf_clip_model_name', type=str, default='openai/clip-vit-base-patch16',
                        help="Name of the Hugging Face CLIP model (if model_type is huggingface_clip).")
    parser.add_argument('--remote_clip_model_name', type=str, default='ViT-B-32',
                        help="Name of the RemoteCLIP model from open_clip (e.g., 'ViT-B-32', 'RN50').")
    parser.add_argument('--remote_clip_checkpoint_path', type=str,
                        default='checkpoints/models--chendelong--RemoteCLIP/snapshots/bf1d8a3ccf2ddbf7c875705e46373bfe542bce38',
                        help="Path to the .pt checkpoint for RemoteCLIP (if model_type is remote_clip).")

    parser.add_argument('--output_dir', type=str, default='./tsne_clip_model_results',  # Generalizing output dir name
                        help="Directory to save the t-SNE plot")
    # parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument('--batch_size', type=int, default=256,  # Smaller default batch for timestep processing
                        help="Batch size for feature extraction")
    parser.add_argument('--num_workers', type=int, default=1, help="Number of workers for dataloader")
    parser.add_argument('--tsne_perplexity', type=float, default=30.0, help="Perplexity for t-SNE")
    parser.add_argument('--tsne_n_iter', type=int, default=250, help="Number of iterations for t-SNE optimization")
    return parser.parse_args()


def main(seed):
    args = parse_args()

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if args.model_type == 'remote_clip' and not args.remote_clip_checkpoint_path:
        print("Error: --remote_clip_checkpoint_path is required when --model_type is 'remote_clip'.")
        sys.exit(1)

    # Load selections from JSON
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
    plot_title_suffix_model = "(HuggingFace CLIP Baseline)" if args.model_type == 'huggingface_clip' else "(RemoteCLIP Baseline)"
    plot_title_level_detail = ""
    legend_title = ""
    filter_source_family = None
    filter_source_genus = None

    try:
        if level == 'family':
            selected_taxa_details = selections_data['family_level']['selected_families']
            target_names_for_plot = [t['name'] for t in selected_taxa_details]
            category_map_for_plot = {t['name']: t['category'] for t in selected_taxa_details}
            plot_title_level_detail = "Family-Level"
            legend_title = "Family (Rarity Category)"
            if not target_names_for_plot: raise ValueError("No families selected.")
        elif level == 'genus':
            filter_source_family = selections_data['genus_level']['source_family_from_initial_selection']
            target_names_for_plot = selections_data['genus_level']['selected_genera']
            category_map_for_plot = {name: name for name in target_names_for_plot}
            plot_title_level_detail = f"Genus-Level within '{filter_source_family}'"
            legend_title = "Genus"
            if not filter_source_family or not target_names_for_plot: raise ValueError("Missing data for genus level.")
        elif level == 'species':
            filter_source_family = selections_data['species_level']['source_family']
            filter_source_genus = selections_data['species_level']['source_genus_from_selected_genera']
            target_names_for_plot = selections_data['species_level']['selected_species']
            category_map_for_plot = {name: name for name in target_names_for_plot}
            plot_title_level_detail = f"Species-Level in '{filter_source_genus}' (Fam: '{filter_source_family}')"
            legend_title = "Species"
            if not filter_source_family or not filter_source_genus or not target_names_for_plot: raise ValueError(
                "Missing data.")
    except (KeyError, TypeError, ValueError) as e:
        print(f"Error parsing JSON for level '{level}' from {args.selections_json_path}: {e}")
        sys.exit(1)

    print(f"Loaded {len(target_names_for_plot)} taxa for level '{level}'. Model: {args.model_type}")
    if filter_source_family: print(f"  Filtering for Family: {filter_source_family}")
    if filter_source_genus: print(f"  Filtering for Genus: {filter_source_genus}")

    # Conditional Model Loading
    model = None
    processor = None
    feature_dim = 512  # Default, will try to update

    if args.model_type == 'huggingface_clip':
        print(f"Loading HuggingFace CLIP model: {args.hf_clip_model_name}")
        try:
            model = CLIPModel.from_pretrained(args.hf_clip_model_name).to(device)
            processor = CLIPProcessor.from_pretrained(args.hf_clip_model_name)
            model.eval()
            feature_dim = model.config.projection_dim
        except Exception as e:
            print(f"Error loading HuggingFace CLIP model or processor: {e}")
            sys.exit(1)
    elif args.model_type == 'remote_clip':
        print(
            f"Loading RemoteCLIP model: {args.remote_clip_model_name} from checkpoint: {args.remote_clip_checkpoint_path}")
        try:
            model, processor, _ = open_clip.create_model_and_transforms(
                args.remote_clip_model_name,
                pretrained=None  # We are loading a specific checkpoint
            )
            model = model.to(device)
            # Load the checkpoint
            checkpoint = torch.load(f"{args.remote_clip_checkpoint_path}/RemoteCLIP-{args.remote_clip_model_name}.pt",
                                    map_location=device)

            model.load_state_dict(checkpoint)
            model.eval()
            # Try to get feature dimension for RemoteCLIP (often 512 for ViT-B based models)
            if hasattr(model, 'visual') and hasattr(model.visual, 'output_dim'):
                feature_dim = model.visual.output_dim
            elif hasattr(model, 'text_projection') and model.text_projection is not None:
                feature_dim = model.text_projection.shape[0]  # common for open_clip models
            else:  # Fallback if not easily found
                print(f"Warning: Could not dynamically determine feature_dim for RemoteCLIP. Assuming {feature_dim}.")

        except Exception as e:
            print(f"Error loading RemoteCLIP model or processor: {e}")
            sys.exit(1)

    # Initialize Dataloader
    print("Initializing Dataloader...")
    try:
        actual_dataset = GGTDataset(args.shards_path, batch_size=args.batch_size, num_workers=args.num_workers,
                                    resampled=False)
        dataloader = actual_dataset.get_dataloader()
    except Exception as e:
        print(f"Error initializing dataset/dataloader: {e}")
        sys.exit(1)

    print(f"Extracting mean features using {args.model_type} for selected taxa at {level} level...")
    all_features_for_tsne = []
    all_labels_for_tsne = []

    for batch_idx, (images, text_data_batch, _, image_mask, _) in enumerate(dataloader):
        current_batch_size = images.shape[0]
        batch_features_sum = torch.zeros(current_batch_size, feature_dim, device=device, dtype=torch.float32)
        batch_valid_timestep_counts = torch.zeros(current_batch_size, device=device, dtype=torch.int32)

        images_gpu = images.to(device)
        image_mask_gpu = image_mask.to(device)

        for t in range(images_gpu.shape[1]):  # Iterate through timesteps
            valid_sample_indices_at_t = image_mask_gpu[:, t].nonzero(as_tuple=True)[0]
            if len(valid_sample_indices_at_t) == 0:
                continue

            timestep_imgs_for_valid_samples = images_gpu[valid_sample_indices_at_t, t, :3, :, :]
            timestep_imgs_for_valid_samples_rgb = timestep_imgs_for_valid_samples.flip(dims=[1])
            pil_images = [transforms.ToPILImage()(img.cpu()) for img in timestep_imgs_for_valid_samples_rgb]

            timestep_features = None
            try:
                with torch.no_grad():
                    if args.model_type == 'huggingface_clip':
                        image_inputs = processor(images=pil_images, return_tensors="pt", padding=True)
                        image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
                        timestep_features = model.get_image_features(**image_inputs)
                    elif args.model_type == 'remote_clip':
                        # RemoteCLIP's preprocess typically handles normalization and to_tensor
                        processed_images = torch.stack([processor(pil_img) for pil_img in pil_images]).to(device)
                        timestep_features = model.encode_image(processed_images)

                    if timestep_features is not None:
                        timestep_features = timestep_features / (timestep_features.norm(dim=-1, keepdim=True) + 1e-8)
                        timestep_features = torch.nan_to_num(timestep_features, nan=0.0, posinf=1.0, neginf=-1.0)
                        timestep_features = torch.clamp(timestep_features, min=-1e4, max=1e4)
                        batch_features_sum[valid_sample_indices_at_t] += timestep_features
                        batch_valid_timestep_counts[valid_sample_indices_at_t] += 1

            except Exception as e:
                print(
                    f"Warning: Feature extraction error ({args.model_type}) at batch {batch_idx}, timestep {t}: {e}. Skipping timestep.")
                continue

        for s_idx in range(current_batch_size):
            sample_text_data = text_data_batch[s_idx]
            if not isinstance(sample_text_data, dict): continue
            current_taxon_name_from_sample = None
            include_sample = False
            if level == 'family':
                if 'level1_family' in sample_text_data:
                    current_taxon_name_from_sample = sample_text_data['level1_family']
                    if current_taxon_name_from_sample in target_names_for_plot: include_sample = True
            elif level == 'genus':
                if 'level1_family' in sample_text_data and sample_text_data['level1_family'] == filter_source_family and \
                        'level2_genus' in sample_text_data:
                    current_taxon_name_from_sample = sample_text_data['level2_genus']
                    if current_taxon_name_from_sample in target_names_for_plot: include_sample = True
            elif level == 'species':
                if 'level1_family' in sample_text_data and sample_text_data['level1_family'] == filter_source_family and \
                        'level2_genus' in sample_text_data and sample_text_data[
                    'level2_genus'] == filter_source_genus and \
                        'level3_species' in sample_text_data:
                    current_taxon_name_from_sample = sample_text_data['level3_species']
                    if current_taxon_name_from_sample in target_names_for_plot: include_sample = True

            if include_sample and batch_valid_timestep_counts[s_idx] > 0:
                mean_feature = batch_features_sum[s_idx] / batch_valid_timestep_counts[s_idx].float()
                all_features_for_tsne.append(mean_feature.cpu().numpy())
                all_labels_for_tsne.append(current_taxon_name_from_sample)
            elif include_sample:  # No valid timesteps but was selected
                # print(f"Warning: Sample for {current_taxon_name_from_sample} had no valid timesteps. Skipping.")
                pass

        if (batch_idx + 1) % 1 == 0:
            print(
                f"  Processed batch {batch_idx + 1}, collected {len(all_features_for_tsne)} features.")

    if not all_features_for_tsne:
        print("No features extracted. Check criteria, JSON, dataset, and model processing.")
        sys.exit(0)

    features_np = np.array(all_features_for_tsne)
    labels_np = np.array(all_labels_for_tsne)
    print(f"Total features extracted for t-SNE: {features_np.shape[0]}")

    if not np.all(np.isfinite(features_np)):
        print("Warning: Mean features contain NaN/Inf values.")
        nan_inf_rows = ~np.all(np.isfinite(features_np), axis=1)
        features_np = features_np[~nan_inf_rows]
        labels_np = labels_np[~nan_inf_rows]
        print(f"Removed {np.sum(nan_inf_rows)} problematic rows. New shape: {features_np.shape}")
        if features_np.shape[0] == 0: sys.exit("Error: All features were NaN/Inf.")

    current_perplexity = args.tsne_perplexity
    if features_np.shape[0] <= current_perplexity:
        print(f"Warning: n_samples ({features_np.shape[0]}) <= perplexity ({current_perplexity}). Adjusting.")
        current_perplexity = max(1.0, float(features_np.shape[0] - 1))
        if features_np.shape[0] <= 1:
            print("Error: Not enough samples for t-SNE after filtering. Exiting.")
            sys.exit(1)

    print(f"Performing t-SNE (perplexity={current_perplexity}, n_iter={args.tsne_n_iter})...")
    tsne = TSNE(n_components=2, random_state=seed, perplexity=current_perplexity,
                n_iter=args.tsne_n_iter, init='random', learning_rate='auto')
    tsne_results = tsne.fit_transform(features_np)
    print("t-SNE completed.")

    os.makedirs(args.output_dir, exist_ok=True)
    unique_labels_in_plot = sorted(list(set(all_labels_for_tsne)))

    cmap_name = 'tab10' if len(unique_labels_in_plot) <= 10 else 'viridis'
    colors = plt.cm.get_cmap(cmap_name, len(unique_labels_in_plot) if len(unique_labels_in_plot) > 0 else 1)
    color_map = {label: colors(i) for i, label in enumerate(unique_labels_in_plot)}

    # Apply demo plot style
    plt.figure(figsize=(15, 8))

    for i, label_name in enumerate(unique_labels_in_plot):
        indices = np.where(labels_np == label_name)
        plot_label_text = label_name

        plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1],
                    label=plot_label_text,
                    color=color_map[label_name],
                    alpha=0.8, # from demo (was 0.7)
                    s=50)      # from demo

    # Title from demo style
    title_fontsize = 16 # from demo
    # plot_title_level_detail and plot_title_suffix_model are already defined in this script
    plt.title(
        f't-SNE Visualization ({plot_title_level_detail} {plot_title_suffix_model})\n' 
        f'Seed: {seed}, Perplexity: {current_perplexity}, Iterations: {args.tsne_n_iter}',
        fontsize=title_fontsize,
        pad=20 # from demo
    )

    # Remove axis labels, ticks, and grid as per demo
    # plt.xlabel('t-SNE Dimension 1') # Removed
    # plt.ylabel('t-SNE Dimension 2') # Removed
    plt.xticks([])
    plt.yticks([])
    plt.tick_params(axis='both', which='both', length=0)
    plt.grid(False)

    # Legend from demo style (keeping dynamic legend_title from script)
    plt.legend(title=legend_title, # legend_title is already defined
               bbox_to_anchor=(1.03, 1),
               loc='upper left',
               fontsize=15, # from demo (was 10)
               # title_fontsize=12, # Not used in demo's direct plt.legend call, can be added if needed
               frameon=True,
               # framealpha=0.5, # Not in demo
               # shadow=True, # Not in demo
               borderpad=1,
               labelspacing=0.8
               )

    # Layout from demo
    plt.tight_layout(rect=[0, 0, 0.82, 1])

    model_type_tag = args.model_type.replace("_", "")
    plot_filename = os.path.join(args.output_dir,
                                 f'{level}_{model_type_tag}_tsne_seed{seed}_perp{int(current_perplexity)}.png')
    # Save with DPI from demo
    plt.savefig(plot_filename, bbox_inches='tight', dpi=300)
    print(f"t-SNE plot saved to {plot_filename}")


if __name__ == '__main__':
    seeds = [22, 32, 42, 52, 62]
    for seed in seeds:
        main(seed)
