import pandas as pd
import numpy as np
import os
import tarfile
import json
import torch
import io
from tqdm import tqdm


def get_3shot_split_ids(csv_path: str, random_seed: int = 42) -> tuple[set, set, dict]:
    """Get support and query sample IDs for 3-shot split.

    Args:
        csv_path: Path to the CSV file containing the dataset
        random_seed: Random seed for reproducibility

    Returns:
        support_ids: Set of sample IDs for support set
        query_ids: Set of sample IDs for query set
        class_mapping: Dictionary mapping class names to their support sample IDs
    """
    print("Loading dataset...")
    df = pd.read_csv(csv_path)

    support_ids = set()
    query_ids = set()
    class_mapping = {
        'species': {},
        'genus': {},
        'family': {}
    }

    # Process each taxonomic level
    for level, col_name in [
        ('species', 'level3_species'),
        ('genus', 'level2_genus'),
        ('family', 'level1_family')
    ]:
        print(f"\nProcessing {level} level...")

        # Group by the taxonomic level
        groups = df.groupby(col_name)
        n_classes = len(groups)
        print(f"Number of {level} classes: {n_classes}")

        # For each class, randomly select three samples for support set
        for class_name, group in groups:
            # Shuffle the group
            group = group.sample(frac=1, random_state=random_seed)

            # Take first three samples for support set (or all if less than 3)
            shots = min(3, len(group))
            for i in range(shots):
                support_sample_id = group.iloc[i]['sample_id']
                support_ids.add(support_sample_id)

                # Store the first sample ID in class_mapping for reference
                if i == 0:
                    class_mapping[level][class_name] = support_sample_id

            # Take remaining samples for query set
            for _, row in group.iloc[shots:].iterrows():
                query_ids.add(row['sample_id'])

    print(f"\nTotal support samples: {len(support_ids)}")
    print(f"Total query samples: {len(query_ids)}")

    return support_ids, query_ids, class_mapping


def split_tar_by_ids(
        tar_path: str,
        support_ids: set,
        query_ids: set,
        class_mapping: dict,
        support_tar_path: str,
        query_tar_path: str
) -> None:
    """Split tar file into support and query sets based on sample IDs.

    Args:
        tar_path: Path to the input tar file
        support_ids: Set of sample IDs for support set
        query_ids: Set of sample IDs for query set
        class_mapping: Dictionary mapping class names to their support sample IDs
        support_tar_path: Path to save support set tar file
        query_tar_path: Path to save query set tar file
    """
    print("\nSplitting tar file...")

    # Create output directories if they don't exist
    os.makedirs(os.path.dirname(support_tar_path), exist_ok=True)
    os.makedirs(os.path.dirname(query_tar_path), exist_ok=True)

    # Statistics for validation
    stats = {
        'support': {'samples': 0, 'files': 0},
        'query': {'samples': 0, 'files': 0},
        'skipped': 0,
        'errors': 0
    }

    # Open tar files
    with tarfile.open(tar_path, 'r') as src_tar, \
            tarfile.open(support_tar_path, 'w') as sup_tar, \
            tarfile.open(query_tar_path, 'w') as qry_tar:

        # Get all members
        members = src_tar.getmembers()
        print(f"Total files in tar: {len(members)}")

        # Group files by sample identifier (filename prefix)
        sample_files = {}
        for member in members:
            if member.isdir():  # Skip directory entries explicitly
                continue

            # Extract sample identifier from filename (part before the first '.')
            filename = os.path.basename(member.name)
            parts = filename.split('.', 1)
            if len(parts) != 2: # Skip files that don't match the expected 'ID.type' format
                print(f"Skipping unexpected file format: {member.name}")
                stats['skipped'] += 1
                continue

            sample_identifier = parts[0]
            if sample_identifier not in sample_files:
                sample_files[sample_identifier] = []
            sample_files[sample_identifier].append(member)

        print(f"\nFound {len(sample_files)} unique sample identifiers")

        # Process each sample identifier
        for sample_identifier, files in tqdm(sample_files.items()):
            try:
                # Find auxiliary.json file for this sample identifier
                aux_file_name = f"{sample_identifier}.auxiliary.json"
                aux_member = next((m for m in files if os.path.basename(m.name) == aux_file_name), None)

                if aux_member is None:
                    print(f"Error: auxiliary.json not found for sample {sample_identifier}")
                    stats['errors'] += 1
                    continue

                content = src_tar.extractfile(aux_member).read()
                aux_data = json.loads(content)
                sample_id = aux_data['sample_id']

                # Determine which tar to write to
                if sample_id in support_ids:
                    target_tar = sup_tar
                    # Avoid double counting samples if multiple files exist per sample
                    # We count samples based on finding the aux file
                    stats['support']['samples'] += 1 
                else:
                    target_tar = qry_tar
                    stats['query']['samples'] += 1

                # Write all files for this sample
                files_written_count = 0
                for member in files:
                    content = src_tar.extractfile(member).read()
                    # Use basename to ensure flat structure in output tars
                    info = tarfile.TarInfo(name=os.path.basename(member.name))
                    info.size = len(content)
                    target_tar.addfile(info, io.BytesIO(content))
                    files_written_count += 1
                
                # Update file counts after writing
                if target_tar == sup_tar:
                    stats['support']['files'] += files_written_count
                else:
                    stats['query']['files'] += files_written_count

            except Exception as e:
                print(f"Error processing sample {sample_identifier}: {str(e)}")
                stats['errors'] += 1
                continue

    # Print detailed statistics
    print("\nSplitting Statistics:")
    print(f"Support set: {stats['support']['samples']} samples, {stats['support']['files']} files")
    print(f"Query set: {stats['query']['samples']} samples, {stats['query']['files']} files")
    print(f"Skipped files: {stats['skipped']}")
    print(f"Errors: {stats['errors']}")

    # Validate class distribution
    print("\nValidating class distribution...")
    for level in ['species', 'genus', 'family']:
        n_classes = len(class_mapping[level])
        print(f"{level.capitalize()} level: {n_classes} classes")

    print(f"\nSupport set saved to: {support_tar_path}")
    print(f"Query set saved to: {query_tar_path}")


def main():
    # Set paths
    csv_path = '../GlobalGeoTree-10kEval-300.csv'
    tar_path = '../GGT-Eval/GGT-10kEval-300.tar'
    support_tar_path = './3_shot_splits/support_3shot-300.tar'
    query_tar_path = './3_shot_splits/query_3shot-300.tar'

    # Set random seed
    random_seed = 42

    # Get support and query IDs
    support_ids, query_ids, class_mapping = get_3shot_split_ids(csv_path, random_seed)

    # Split tar file
    split_tar_by_ids(tar_path, support_ids, query_ids, class_mapping, support_tar_path, query_tar_path)


if __name__ == '__main__':
    main()