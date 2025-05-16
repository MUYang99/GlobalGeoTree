import os
import torch
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from dataloader import GGTDataset
from models import GeoTreeClip, CLIPContrastiveLoss
from tqdm import tqdm
import logging
from datetime import datetime, timedelta
import threading
from queue import Queue
import glob
from pathlib import Path
import time
from huggingface_hub import list_repo_files, hf_hub_url
import requests
import io
import argparse


# 创建异步日志处理器
class AsyncLogHandler(logging.Handler):
    def __init__(self, filename):
        super().__init__()
        self.queue = Queue()
        self.filename = filename
        self.thread = threading.Thread(target=self._process_queue)
        self.thread.daemon = True
        self.thread.start()

    def emit(self, record):
        self.queue.put(record)

    def _process_queue(self):
        with open(self.filename, 'a') as f:
            while True:
                record = self.queue.get()
                if record is None:
                    break
                f.write(self.format(record) + '\n')
                f.flush()


# 创建输出目录
def setup_output_dirs():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = f"results/{timestamp}"
    dirs = {
        'base': base_dir,
        'logs': f"{base_dir}/logs",
        'checkpoints': f"{base_dir}/checkpoints"
    }

    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    return dirs


# 设置日志
def setup_logging(log_dir):
    # 创建异步日志处理器
    async_handler = AsyncLogHandler(f'{log_dir}/training.log')
    async_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    # 设置日志记录器
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(async_handler)
    logger.addHandler(logging.StreamHandler())


def train_epoch(model, dataloader, criterion, optimizer, device, accumulation_steps=4, target_iterations=0,
                is_resampled=False):
    model.train()
    total_loss = 0
    valid_batches = 0
    local_batch_count = 0

    optimizer.zero_grad()

    if target_iterations == 0:
        if dist.get_rank() == 0:
            logging.info("train_epoch received target_iterations=0, loop will not run.")
        return 0.0

    dataloader_iter = iter(dataloader)
    progress_bar_desc = f'Training (Resampled {target_iterations} iter)' if is_resampled else f'Training (Finite {target_iterations} iter)'
    progress_bar = tqdm(range(target_iterations), desc=progress_bar_desc, disable=not (dist.get_rank() == 0))

    all_ranks_should_continue = torch.ones(1, device=device, dtype=torch.int)

    for i in progress_bar:
        if not is_resampled:
            dist.all_reduce(all_ranks_should_continue, op=dist.ReduceOp.MIN)
            if all_ranks_should_continue.item() == 0:
                if dist.get_rank() == 0:
                    logging.info(
                        f"A rank signalled to stop processing (finite dataset). Epoch will end after iteration {i - 1}.")
                break

        try:
            try:
                batch = next(dataloader_iter)
                local_batch_count += 1
            except StopIteration:
                if not is_resampled:
                    logging.warning(
                        f"Rank {dist.get_rank()}: Dataloader exhausted unexpectedly (finite dataset) at iter {i}/{target_iterations}. Signalling stop.")
                    all_ranks_should_continue[0] = 0
                    dist.all_reduce(all_ranks_should_continue, op=dist.ReduceOp.MIN)
                    break
                else:
                    logging.error(
                        f"Rank {dist.get_rank()}: Dataloader (resampled) raised StopIteration at iter {i}/{target_iterations}. This is unexpected. Check dataset/shard integrity.")
                    all_ranks_should_continue[0] = 0
                    dist.all_reduce(all_ranks_should_continue, op=dist.ReduceOp.MIN)
                    break

            images, text_data, auxiliary_data, image_mask, aux_mask = batch

            if images is None:
                if dist.get_rank() == 0:
                    logging.warning(f"Batch {i} contains a corrupted sample and will be skipped.")
                if dist.get_world_size() > 1:
                    try:
                        dist.barrier()
                    except Exception as e:
                        logging.error(
                            f"Rank {dist.get_rank()}: Barrier before skipping batch failed: {str(e)}, attempting to continue epoch.")
                continue

            data_start = time.time()
            images = images.to(device, dtype=torch.float32, non_blocking=True)
            image_mask = image_mask.to(device, non_blocking=True)
            aux_mask = aux_mask.to(device, non_blocking=True)
            if dist.get_rank() == 0 and i % 100 == 0:
                logging.info(f"Data transfer took {time.time() - data_start:.2f}s")

            forward_start = time.time()
            with torch.amp.autocast('cuda'):
                image_features, text_features = model(images, text_data, auxiliary_data, image_mask, aux_mask)
                loss = criterion(image_features, text_features)
                loss = loss / accumulation_steps

            if i % 100 == 0:
                if dist.get_rank() == 0:
                    logging.info(f"Forward pass took {time.time() - forward_start:.2f}s")

            backward_start = time.time()
            loss.backward()

            if i % 100 == 0:
                if dist.get_rank() == 0:
                    logging.info(f"Backward pass took {time.time() - backward_start:.2f}s")

            if (i + 1) % accumulation_steps == 0:
                sync_start = time.time()

                try:
                    torch.cuda.synchronize()

                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                    if dist.get_rank() == 0 and i % 100 == 0:
                        logging.info(f"Gradient sync and update took {time.time() - sync_start:.2f}s")
                except Exception as e:
                    logging.error(f"Rank {dist.get_rank()}: Error during gradient sync: {str(e)}")
                    raise

            total_loss += loss.item() * accumulation_steps
            valid_batches += 1

            if i % 100 == 0:
                if dist.get_rank() == 0:
                    progress_bar.set_postfix({
                        'loss': f"{loss.item() * accumulation_steps:.4f}",
                        'step_time': f"{time.time() - forward_start:.2f}s",
                        'lr': f"{optimizer.param_groups[0]['lr']:.6f}"
                    })

            if i % 50 == 0:
                torch.cuda.empty_cache()

            if i % 500 == 0:
                try:
                    local_count = torch.tensor([local_batch_count], dtype=torch.float32, device=device)
                    gathered_counts = [torch.zeros(1, dtype=torch.float32, device=device) for _ in
                                       range(dist.get_world_size())]
                    dist.all_gather(gathered_counts, local_count)

                    # Initialize batches_per_rank for this specific logging scope
                    batches_per_rank = [0] * dist.get_world_size()
                    for r in range(dist.get_world_size()):
                        batches_per_rank[r] = int(gathered_counts[r].item())

                    torch.cuda.synchronize()
                    dist.barrier()

                    if dist.get_rank() == 0:
                        logging.info(f"Batch progress - ranks processed: {batches_per_rank}")

                except Exception as e:
                    logging.error(f"Process {dist.get_rank()} failed at sync: {str(e)}")
                    raise

        except Exception as e:
            logging.error(f"Error in batch {i}: {str(e)}")
            continue

    local_count = torch.tensor([local_batch_count], dtype=torch.float32, device=device)
    gathered_counts = [torch.zeros(1, dtype=torch.float32, device=device) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered_counts, local_count)

    if dist.get_rank() == 0:
        for r in range(dist.get_world_size()):
            batches_per_rank[r] = int(gathered_counts[r].item())
        logging.info(f"Epoch completed - batches processed per rank: {batches_per_rank}")

        min_batches = min(batches_per_rank)
        max_batches = max(batches_per_rank)
        if max_batches > min_batches * 1.1:
            logging.warning(f"Significant imbalance detected: min={min_batches}, max={max_batches}")

    return total_loss / valid_batches if valid_batches > 0 else float('inf')


def save_checkpoint(model, optimizer, scheduler, epoch, train_loss, checkpoint_dir):
    try:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
        }
        save_path = f"{checkpoint_dir}/latest_model_{epoch}.pth"
        torch.save(checkpoint, save_path, _use_new_zipfile_serialization=True)
        logging.info(f"Successfully saved checkpoint at epoch {epoch + 1} to {save_path}")
        logging.info(f"Checkpoint size: {os.path.getsize(save_path) / 1024 / 1024:.2f} MB")
    except Exception as e:
        logging.error(f"Failed to save checkpoint at epoch {epoch + 1}: {str(e)}")
        raise


def setup(rank, world_size):
    os.environ['NCCL_DEBUG'] = 'INFO'

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'

    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=world_size,
    )


def cleanup():
    try:
        torch.cuda.synchronize()

        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
    except Exception as e:
        logging.error(f"Error during cleanup: {str(e)}")
    finally:
        torch.cuda.empty_cache()


def get_tar_files(train_path_config) -> list[str]:
    collected_files = set()

    if isinstance(train_path_config, dict):
        repo_id = train_path_config.get('repo_id')
        folder_path = train_path_config.get('folder_path', '')
        repo_type = train_path_config.get('repo_type', 'dataset')

        if not repo_id:
            logging.error("Hugging Face config missing 'repo_id'.")
            return []

        logging.info(
            f"Fetching file list from Hugging Face Hub: repo_id='{repo_id}', folder_path='{folder_path}', repo_type='{repo_type}'")
        try:
            all_repo_files_flat = list_repo_files(repo_id=repo_id, repo_type=repo_type)

            target_prefix = folder_path.strip("/") + "/" if folder_path else ""

            for f_path_in_repo in all_repo_files_flat:
                normalized_f_path = f_path_in_repo.replace("\\", "/")
                if normalized_f_path.startswith(target_prefix) and normalized_f_path.endswith(".tar"):
                    file_url = hf_hub_url(repo_id=repo_id, filename=normalized_f_path, repo_type=repo_type)
                    collected_files.add(file_url)

            if not collected_files:
                logging.warning(
                    f"No .tar files found in Hugging Face Hub: '{repo_id}' under folder '{folder_path}'. Checked {len(all_repo_files_flat)} files.")

        except Exception as e:
            logging.error(f"Error listing or constructing URLs from Hugging Face Hub: {e}")
            return []

    elif isinstance(train_path_config, list):
        for path_item in train_path_config:
            if not isinstance(path_item, str):
                logging.warning(f"路径列表中的项目不是字符串，已跳过: {path_item}")
                continue

            if not os.path.exists(path_item):
                logging.warning(f"提供的路径不存在，已跳过: {path_item}")
                continue

            if os.path.isfile(path_item) and path_item.endswith('.tar'):
                collected_files.add(os.path.abspath(path_item))
            elif os.path.isdir(path_item):
                tar_pattern = os.path.join(path_item, '*.tar')
                found_in_dir = glob.glob(tar_pattern)
                if not found_in_dir:
                    logging.warning(f"目录中未找到 .tar 文件: {path_item}")
                for tar_file in found_in_dir:
                    collected_files.add(os.path.abspath(tar_file))
            else:
                logging.warning(f"路径既不是 .tar 文件也不是目录，已跳过: {path_item}")
    else:
        logging.error(f"Invalid train_path_config type: {type(train_path_config)}. Expected dict or list.")
        return []

    if not collected_files:
        logging.error("在所有提供的路径中均未找到 .tar 文件。")
        return []

    return sorted(list(collected_files))


def train(rank, world_size, dirs, train_path_config, use_resampled_arg, iterations_per_epoch_arg,
          shard_counts_json_path_arg):
    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'

    setup(rank, world_size)

    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)

    if rank == 0:
        setup_logging(dirs['logs'])
        logging.info(f'Visible CUDA devices: {os.environ["CUDA_VISIBLE_DEVICES"]}')
        logging.info(f'Using device: cuda:{rank} (Actual GPU: {os.environ["CUDA_VISIBLE_DEVICES"].split(",")[rank]})')

    batch_size = 192
    num_epochs = 100
    learning_rate = 1e-5
    weight_decay = 1e-4
    warmup_epochs = 5
    accumulation_steps = 2

    bucket_cap_mb = 25

    tar_files = get_tar_files(train_path_config)
    if not tar_files:
        if rank == 0:
            logging.error("No training files found. Exiting.")
        cleanup()
        return

    if rank == 0:
        logging.info(f"Found {len(tar_files)} tar files/URLs for training")
        for tar_file in tar_files:
            logging.info(f"Training file: {tar_file}")

    train_dataset = GGTDataset(
        tar_files,
        num_workers=4,
        batch_size=batch_size,
        resampled=use_resampled_arg,
        shardshuffle=True,
        world_size=world_size,
        rank=rank,
        shard_counts_json_path=shard_counts_json_path_arg
    )

    train_loader = train_dataset.get_dataloader()

    # Initialize effective_iterations_per_epoch for all ranks
    # For resampled data, this is the user-specified value.
    # For finite data, this will be overridden by the minimum across ranks.
    effective_iterations_per_epoch = iterations_per_epoch_arg if use_resampled_arg else 0

    if not use_resampled_arg:
        # Finite dataset: all ranks determine the number of iterations
        # based on the rank with the fewest batches.
        local_loader_length = torch.tensor([train_loader.length], dtype=torch.long, device=device)
        all_loader_lengths_tensors = [torch.zeros_like(local_loader_length) for _ in range(world_size)]
        dist.all_gather(all_loader_lengths_tensors, local_loader_length)
        
        actual_batches_per_rank = [tensor.item() for tensor in all_loader_lengths_tensors]
        if actual_batches_per_rank: # Ensure the list is not empty
            effective_iterations_per_epoch = min(actual_batches_per_rank)
            if effective_iterations_per_epoch < 0: # Safety check
                effective_iterations_per_epoch = 0
        else: # Should not happen if world_size >= 1
            effective_iterations_per_epoch = 0
            
        if rank == 0:
            logging.info(
                f"Finite dataset: Min batches per rank = {effective_iterations_per_epoch}. All ranks will run for this many iterations.")
            logging.info(f"Dataset size (finite, rank 0 perspective): {train_dataset.num_samples}, Batches for rank 0: {train_loader.length}")
            logging.info(f"Actual batches per rank: {actual_batches_per_rank}")
    else:
        # Resampled dataset: iterations are fixed by user argument
        if rank == 0:
            logging.info(
                f"Resampled dataset: Each epoch will run for a fixed {effective_iterations_per_epoch} iterations.")
            logging.info(f"(Resampled) Estimated total samples (one pass, rank 0 perspective): {train_dataset.num_samples}")
            # For resampled, train_loader.length might reflect one pass for its shards, not the target iterations.
            # logging.info(f"Dataloader length for rank 0 (resampled, one pass): {train_loader.length}")


    try:
        has_data = torch.tensor([train_loader.length > 0 or use_resampled_arg], dtype=torch.float32, device=device) # Resampled always "has data" for target iterations
        all_has_data = [torch.zeros(1, dtype=torch.float32, device=device) for _ in range(world_size)]
        dist.all_gather(all_has_data, has_data)

        if not all(t.item() > 0.5 for t in all_has_data):
            missing_ranks = [i for i, t in enumerate(all_has_data) if t.item() <= 0.5]
            if rank == 0:
                logging.error(f"Ranks without data: {missing_ranks}")

            if has_data.item() <= 0.5:
                logging.warning(f"Rank {rank} does not have any data! Using data from rank 0")

        dist.barrier()
    except Exception as e:
        logging.error(f"Error during dataloader verification: {str(e)}")

    model = GeoTreeClip().to(device)

    if rank == 0:
        for idx, (name, param) in enumerate(model.named_parameters()):
            logging.info(f"Parameter {idx}: {name}, Shape: {param.shape}")

        params = list(model.named_parameters())
        if len(params) > 198:
            logging.info(f"Problem parameter 197: {params[197][0]}")
            logging.info(f"Problem parameter 198: {params[198][0]}")

    def register_param_hook(model):
        unused_params = []
        for name, param in model.named_parameters():
            if "post_layernorm" in name:
                unused_params.append(param)

        def hook_fn(grad):
            return grad

        hooks = []
        for param in unused_params:
            hooks.append(param.register_hook(hook_fn))

        return hooks

    param_hooks = register_param_hook(model)

    model = DDP(
        model,
        device_ids=[rank],
        find_unused_parameters=True,
        broadcast_buffers=False,
        bucket_cap_mb=bucket_cap_mb
    )

    criterion = CLIPContrastiveLoss().to(device)

    optimizer = optim.AdamW([
        {'params': model.module.temporal_extractor.parameters(), 'lr': learning_rate},
        {'params': model.module.auxiliary_encoder.parameters(), 'lr': learning_rate},
        {'params': model.module.text_encoder.parameters(), 'lr': learning_rate * 0.1}
    ], weight_decay=weight_decay)

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2,
        eta_min=learning_rate * 0.01
    )

    for epoch in range(num_epochs):
        try:
            if rank == 0:
                logging.info(f"\nStarting epoch {epoch + 1}/{num_epochs}")

            torch.cuda.synchronize()
            try:
                dist.barrier()
            except Exception as e:
                logging.error(f"Rank {rank}: Barrier at epoch start failed: {str(e)}")

            if epoch > 0 and hasattr(train_dataset, 'use_sample_filtering') and train_dataset.use_sample_filtering:
                if rank == 0:
                    logging.info(f"Epoch {epoch + 1}: Recreating DataLoader to reset sample filter state.")
                train_loader = train_dataset.get_dataloader()
            elif hasattr(train_loader, 'sampler') and \
                    hasattr(train_loader.sampler, 'set_epoch') and \
                    train_loader.sampler is not None:
                train_loader.sampler.set_epoch(epoch)

            if epoch < warmup_epochs:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = learning_rate * (epoch + 1) / warmup_epochs
                if rank == 0:
                    logging.info(f"Warmup learning rate: {learning_rate * (epoch + 1) / warmup_epochs:.6f}")

            current_iterations_for_epoch = effective_iterations_per_epoch
            if current_iterations_for_epoch == 0 and rank == 0:
                logging.warning("Effective iterations per epoch is 0. Skipping training for this epoch.")

            train_loss = train_epoch(model, train_loader, criterion, optimizer, device, accumulation_steps,
                                     current_iterations_for_epoch, use_resampled_arg)

            loss_tensor = torch.tensor([train_loss], device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            avg_loss = loss_tensor.item() / world_size

            if rank == 0:
                logging.info(f"Epoch {epoch + 1}/{num_epochs} - Average Train Loss: {avg_loss:.4f}")

                if (epoch + 1) % 1 == 0 or epoch == num_epochs - 1:
                    save_checkpoint(
                        model.module, optimizer, scheduler,
                        epoch, avg_loss, dirs['checkpoints']
                    )
                    logging.info(f"Checkpoint saved at epoch {epoch + 1}")

            scheduler.step()

            torch.cuda.synchronize()
            try:
                dist.barrier()
                if rank == 0:
                    logging.info(f"All processes completed epoch {epoch + 1}")
            except Exception as e:
                logging.error(f"Rank {rank}: Error at epoch barrier: {str(e)}")
                raise

            torch.cuda.empty_cache()

        except Exception as e:
            logging.error(f"Rank {rank}: Error in epoch {epoch + 1}: {str(e)}")
            try:
                cleanup()
            except:
                pass
            raise

    for hook in param_hooks:
        hook.remove()

    cleanup()


def main():
    os.environ['NCCL_DEBUG'] = 'INFO'
    os.environ['NCCL_IB_DISABLE'] = '1'
    os.environ['NCCL_P2P_DISABLE'] = '0'
    os.environ['NCCL_SHM_DISABLE'] = '0'
    os.environ['NCCL_SOCKET_NTHREADS'] = '4'
    os.environ['NCCL_NSOCKS_PERTHREAD'] = '4'
    os.environ['NCCL_BUFFSIZE'] = '2097152'
    os.environ['NCCL_MAX_NCHANNELS'] = '4'
    os.environ['NCCL_MIN_NCHANNELS'] = '1'
    os.environ['TORCH_NCCL_BLOCKING_WAIT'] = '1'
    os.environ['TORCH_NCCL_ASYNC_ERROR_HANDLING'] = '1'

    os.environ['CUDA_VISIBLE_DEVICES'] = '2,3,6'
    os.environ['NCCL_IGNORE_DISABLED_P2P'] = '1'

    dirs = setup_output_dirs()

    setup_logging(dirs['logs'])
    logging.info("Starting training process...")


    parser = argparse.ArgumentParser(description="Train GeoTreeClip model with DDP, supporting local and HF streaming.")
    parser.add_argument('--data_source_type', type=str, default='huggingface', choices=['local', 'huggingface'],
                        help='Source of the training data: "local" or "huggingface".')
    parser.add_argument('--local_paths', nargs='+', default=[],
                        help='List of local paths (files or directories) for tar files. Used if data_source_type is "local".')
    parser.add_argument('--hf_repo_id', type=str, default='yann111/GlobalGeoTree',
                        help='Hugging Face repository ID. Used if data_source_type is "huggingface".')
    parser.add_argument('--hf_folder_path', type=str, default='GlobalGeoTree-6M',
                        help='Folder path within the Hugging Face repository. Used if data_source_type is "huggingface".')
    parser.add_argument('--hf_repo_type', type=str, default='dataset',
                        help='Type of the Hugging Face repository. Used if data_source_type is "huggingface".')

    parser.add_argument('--use_resampled_dataset', default=True,
                        help='Enable resampling for the WebDataset (infinite stream).')
    parser.add_argument('--iterations_per_epoch', type=int, default=4500,
                        help='Number of training iterations (steps) per epoch, used if --use_resampled_dataset is set.')
    parser.add_argument('--shard_counts_json_path', type=str, default='precomputed_shard_counts.json',
                        help='Optional path to a JSON file with precomputed shard sample counts.')

    main_args = parser.parse_args()

    if main_args.data_source_type == 'local':
        if not main_args.local_paths:
            logging.error("Error: --local_paths must be provided when --data_source_type is 'local'.")
            return
        train_path_config = main_args.local_paths
    elif main_args.data_source_type == 'huggingface':
        train_path_config = {
            "repo_id": main_args.hf_repo_id,
            "folder_path": main_args.hf_folder_path,
            "repo_type": main_args.hf_repo_type
        }
    else:
        logging.error(f"Invalid --data_source_type: {main_args.data_source_type}")
        return

    if main_args.use_resampled_dataset and main_args.iterations_per_epoch <= 0:
        logging.error(
            "Error: --iterations_per_epoch must be a positive integer when --use_resampled_dataset is enabled.")
        return

    world_size = 3
    try:
        mp.spawn(
            train,
            args=(world_size, dirs, train_path_config, main_args.use_resampled_dataset, main_args.iterations_per_epoch,
                  main_args.shard_counts_json_path),
            nprocs=world_size,
            join=True
        )
    except KeyboardInterrupt:
        logging.info("Training interrupted by user")
    except Exception as e:
        logging.error(f"Training failed with error: {str(e)}")
    finally:
        cleanup()


if __name__ == '__main__':
    main()