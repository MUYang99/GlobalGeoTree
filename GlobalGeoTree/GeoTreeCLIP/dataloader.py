import torch
import webdataset as wds
from torch.utils.data.distributed import DistributedSampler
from typing import Dict, Tuple, List, Optional
import math
import os
import tarfile
import json
import requests
import io


# 可序列化的全局索引范围过滤器类
class GlobalIndexRangeFilter:
    def __init__(self, rank_start_global_idx: int, rank_end_global_idx: int):
        self.rank_start_global_idx = rank_start_global_idx
        self.rank_end_global_idx = rank_end_global_idx
        # 每个worker将有自己的计数器，从头开始处理全局一致的样本流
        self.current_sample_global_idx = -1

    def __call__(self, sample: Dict) -> bool:
        self.current_sample_global_idx += 1
        if self.rank_start_global_idx <= self.current_sample_global_idx < self.rank_end_global_idx:
            return True  # 此样本属于当前worker对应的rank的处理范围
        return False


class WebDatasetDistributedSampler:
    """为WebDataset实现的分布式采样器"""

    def __init__(self, num_samples, num_replicas, rank, shuffle=True, seed=0):
        self.num_samples = num_samples
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.shuffle = shuffle
        self.seed = seed

        # 确保每个进程的样本数量相同
        self.num_samples_per_replica = math.ceil(num_samples / num_replicas)
        self.total_size = self.num_samples_per_replica * num_replicas

    def set_epoch(self, epoch):
        """设置当前epoch，用于shuffle"""
        self.epoch = epoch


class GGTDataset:
    """
    WebDataset-based dataloader for GGT dataset.
    Each sample contains:
    - Sentinel-2 images (12 months, 10 bands, 5x5 pixels)
    - Text labels for classification
    - Auxiliary environmental data
    """

    def __init__(
            self,
            shards_path: str,
            batch_size: int = 32,
            num_workers: int = 4,
            shuffle: int = 1000,
            shardshuffle: bool = True,
            resampled: bool = True,
            world_size: Optional[int] = None,
            rank: Optional[int] = None,
            shard_counts_json_path: Optional[str] = None
    ):
        """
        Args:
            shards_path: Path to the WebDataset shards
            batch_size: Batch size for the dataloader
            num_workers: Number of worker processes
            shuffle: Shuffle buffer size
            shardshuffle: Whether to shuffle shards
            resampled: Whether to resample the dataset
            world_size: Total number of processes for distributed training
            rank: Process rank for distributed training
            shard_counts_json_path: Optional path to a JSON file with precomputed shard sample counts.
        """
        self.shards_path = shards_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.shardshuffle = shardshuffle
        self.resampled = resampled
        self.world_size = world_size
        self.rank = rank
        self.shard_counts_json_path = shard_counts_json_path

        # Define aux_fields for consistent use and testing
        self.aux_fields_for_test = [
                                       'longitude', 'latitude', 'elevation',
                                       'slope', 'aspect',
                                       'soil_moisture_0_5cm', 'soil_moisture_5_15cm', 'soil_moisture_15_30cm',
                                   ] + [f'bio{i:02d}' for i in range(1, 20)]

        # 准确计算数据集大小
        self.shards_with_samples = self._count_samples_per_shard()
        self.num_samples = sum(count for _, count in self.shards_with_samples)

        if self.num_samples == 0 and (rank == 0 or world_size is None):
            print("警告: 数据集总样本数为 0。请检查shards路径和内容。")
            # return # Early exit if no samples? Or let it proceed and WebDataset will yield nothing.

        if rank == 0 or (world_size is None and rank is None):
            print(f"数据集总样本数: {self.num_samples}")
            if self.num_samples > 0:
                # 打印每个shard的样本数
                for shard_idx, (shard, count) in enumerate(self.shards_with_samples):
                    print(f"  Shard {shard_idx}: {os.path.basename(shard)} - {count} 个样本")

        self.use_sample_filtering = False  # Initialize this flag
        self.distributed_rank_shards_lists = self._distribute_shards_and_set_filter_flag()

    def _count_samples_per_shard(self) -> List[Tuple[str, int]]:
        """准确计算每个tar文件中的样本数量，并返回(shard_path_or_url, sample_count)列表"""
        shards_input = self.shards_path
        if isinstance(self.shards_path, str):
            shards_input = [self.shards_path]

        shards_with_samples = []
        precomputed_counts_by_basename = {}

        if self.shard_counts_json_path and os.path.exists(self.shard_counts_json_path):
            try:
                with open(self.shard_counts_json_path, 'r') as f:
                    loaded_counts = json.load(f)
                # Convert keys (full paths) to basenames for matching with URLs
                for path_key, count_val in loaded_counts.items():
                    precomputed_counts_by_basename[os.path.basename(path_key)] = count_val
                if self.rank == 0 or self.world_size is None:
                    print(
                        f"信息: 成功从 {self.shard_counts_json_path} 加载了 {len(precomputed_counts_by_basename)} 条预计算样本数。")
            except Exception as e:
                if self.rank == 0 or self.world_size is None:
                    print(f"警告: 加载预计算样本数文件 {self.shard_counts_json_path} 失败: {e}。将回退到在线计数。")

        for shard_ref in shards_input:  # shard_ref can be a local path or a URL
            sample_count = 0
            tar_basename = os.path.basename(shard_ref)  # Get basename for matching

            if tar_basename in precomputed_counts_by_basename:
                sample_count = precomputed_counts_by_basename[tar_basename]
                if self.rank == 0 or self.world_size is None:  # Log only from rank 0 or non-DDP
                    print(f"信息: 使用预计算的样本数 {sample_count} for {tar_basename}")
                shards_with_samples.append((shard_ref, sample_count))
                continue  # Move to next shard_ref

            # Fallback to online counting if not found in precomputed_counts
            if self.rank == 0 or self.world_size is None:
                print(f"信息: {tar_basename} 未在预计算JSON中找到，将执行在线样本计数。")

            try:
                if shard_ref.startswith("http://") or shard_ref.startswith("https://"):
                    # Handle URL
                    if self.rank == 0 or self.world_size is None:
                        print(f"Counting samples from URL: {shard_ref}")
                    response = requests.get(shard_ref, stream=False)  # stream=False to get content at once
                    response.raise_for_status()  # Ensure the request was successful

                    # Use io.BytesIO to treat the byte content as a file-like object
                    tar_content_stream = io.BytesIO(response.content)
                    with tarfile.open(fileobj=tar_content_stream, mode='r') as tar:
                        sample_count = sum(1 for member in tar.getmembers()
                                           if member.name.endswith('text.json'))
                    if self.rank == 0 or self.world_size is None:
                        print(f"Counted {sample_count} samples from URL: {os.path.basename(shard_ref)}")
                else:
                    # Handle local file path
                    if not os.path.exists(shard_ref):
                        print(f"Warning: Local shard path does not exist: {shard_ref}. Assigning 0 samples.")
                        shards_with_samples.append((shard_ref, 0))
                        continue  # Skip to the next shard_ref

                    with tarfile.open(shard_ref, 'r') as tar:
                        sample_count = sum(1 for member in tar.getmembers()
                                           if member.name.endswith('text.json'))

                if sample_count == 0 and (self.rank == 0 or self.world_size is None):
                    print(
                        f"Info: No 'text.json' files found in {os.path.basename(shard_ref)}. Assuming 0 samples for this shard.")

                shards_with_samples.append((shard_ref, sample_count))

            except requests.exceptions.RequestException as e:
                print(f"HTTP Error for URL {shard_ref}: {str(e)}. Assigning 0 samples to this shard.")
                shards_with_samples.append((shard_ref, 0))
            except tarfile.ReadError as e:
                print(
                    f"Error reading tar content from {os.path.basename(shard_ref)} (local or URL): {str(e)}. Assigning 0 samples.")
                shards_with_samples.append((shard_ref, 0))
            except Exception as e:
                print(
                    f"An unexpected error occurred while processing {os.path.basename(shard_ref)}: {str(e)}. Assigning 0 samples.")
                shards_with_samples.append((shard_ref, 0))

        return shards_with_samples

    def __len__(self):
        """返回数据集的大小"""
        return self.num_samples

    def _distribute_shards_and_set_filter_flag(self) -> List[List[str]]:
        """
        决定shard分配策略 (贪心 vs 样本过滤) 并设置 self.use_sample_filtering 标志。
        返回一个列表的列表，其中每个内部列表是分配给对应rank的shards。
        """
        all_shards_paths_list = [s_path for s_path, _ in self.shards_with_samples]

        # 处理非分布式或单卡情况
        if self.world_size is None or self.rank is None or self.world_size == 1:
            self.use_sample_filtering = False
            if self.rank == 0 or (self.world_size is None and self.rank is None):
                if self.num_samples > 0:
                    print("信息: 运行于单进程或非分布式模式。所有shards分配给当前进程。")
                    print(f"       共 {len(all_shards_paths_list)} 个shards, {self.num_samples} 个样本。")
            return [all_shards_paths_list]

        # --- DDP 模式下的分配逻辑 ---
        if not all_shards_paths_list:  # 没有shards (可能是因为num_samples为0)
            if self.rank == 0:
                print("信息: 无可用shards进行分配。")
            self.use_sample_filtering = False
            return [[] for _ in range(self.world_size)]

        # 初始贪心分配尝试
        rank_shards_greedy = [[] for _ in range(self.world_size)]
        rank_samples_greedy = [0] * self.world_size
        # self.shards_with_samples 已经是 (path, count) 的列表
        sorted_shards_tuples = sorted(self.shards_with_samples, key=lambda x: x[1], reverse=True)

        for i, (shard_path, count) in enumerate(sorted_shards_tuples):
            if i < self.world_size:
                target_rank = i % self.world_size  # 更公平的初始分配
                rank_shards_greedy[target_rank].append(shard_path)
                rank_samples_greedy[target_rank] += count
            else:
                min_rank_idx = rank_samples_greedy.index(min(rank_samples_greedy))
                rank_shards_greedy[min_rank_idx].append(shard_path)
                rank_samples_greedy[min_rank_idx] += count

        # 计算贪心分配后的不平衡度
        min_assigned_samples = min(rank_samples_greedy) if rank_samples_greedy else 0
        max_assigned_samples = max(rank_samples_greedy) if rank_samples_greedy else 0
        imbalance_threshold = 20.0  # 不平衡阈值 e.g., 20%

        current_imbalance_percent = 0.0
        if max_assigned_samples > 0:  # 避免除以零
            current_imbalance_percent = (max_assigned_samples - min_assigned_samples) / max_assigned_samples * 100

        if self.rank == 0:
            print(f"信息: 初步贪心Shard分配导致的不平衡度: {current_imbalance_percent:.2f}% "
                  f"(最少样本: {min_assigned_samples}, 最多样本: {max_assigned_samples})")

        # 决定是否使用样本级过滤
        self.use_sample_filtering = False
        if len(all_shards_paths_list) == 1 and self.world_size > 1:
            if self.rank == 0:
                print(f"信息: 数据集中只有一个shard。将为 {self.world_size} 个ranks启用样本级过滤以平衡负载。")
            self.use_sample_filtering = True
        elif current_imbalance_percent > imbalance_threshold and self.world_size > 1:
            if self.rank == 0:
                print(f"信息: 检测到高负载不平衡 ({current_imbalance_percent:.2f}% > {imbalance_threshold}%)."
                      " 将切换到样本级过滤策略。")
            self.use_sample_filtering = True
        elif len(all_shards_paths_list) < self.world_size and len(all_shards_paths_list) > 0 and self.world_size > 1:
            if self.rank == 0:
                print(f"信息: Shard数量 ({len(all_shards_paths_list)}) 少于Rank数量 ({self.world_size})。"
                      " 将切换到样本级过滤策略。")
            self.use_sample_filtering = True

        final_distributed_shards_list: List[List[str]]
        if self.use_sample_filtering:
            final_distributed_shards_list = [all_shards_paths_list for _ in range(self.world_size)]
        else:
            final_distributed_shards_list = rank_shards_greedy

        # 日志记录最终分配策略
        if self.rank == 0 and self.num_samples > 0:  # 只在主进程且有数据时打印详细分配
            print("\n最终Shard/样本分配策略:")
            if self.use_sample_filtering:
                print(f"  策略: 所有Ranks ({self.world_size}个) 将读取所有 {len(all_shards_paths_list)} 个Shards。")
                print(f"        样本将通过内部样本级过滤机制在Ranks间分配。")
                print(f"        每个Rank理论上将处理约 {math.ceil(self.num_samples / self.world_size)} 个样本。")
            else:  # 贪心分配策略被采用
                print(f"  策略: 基于贪心算法的Shard分配 (计算不平衡度: {current_imbalance_percent:.2f}%).")
                for r_idx, r_s_list_paths in enumerate(final_distributed_shards_list):
                    r_s_total_samples = sum(c for s, c in self.shards_with_samples if s in r_s_list_paths)
                    print(f"    Rank {r_idx}: 分配到 {len(r_s_list_paths)} 个shards, 共 {r_s_total_samples} 个样本。")
                    # for shard_p_log in r_s_list_paths:
                    #     _c_log = next(c_val for s_val,c_val in self.shards_with_samples if s_val == shard_p_log)
                    #     print(f"      - {os.path.basename(shard_p_log)}: {_c_log} samples")

            # 验证Shard是否都分配了 (仅在非样本过滤时有意义)
            if not self.use_sample_filtering:
                assigned_check_paths = []
                for s_list_check_paths in final_distributed_shards_list:
                    assigned_check_paths.extend(s_list_check_paths)
                if sorted(assigned_check_paths) != sorted(all_shards_paths_list):
                    print("警告: (贪心分配后) 部分shard似乎未被分配! 请检查逻辑。")  # 应不太可能发生
                else:
                    print("  验证: 所有shard都已通过贪心策略在各Rank间完成分配。")
            else:
                print("  验证: 所有shard将由所有rank共享处理 (采用样本过滤策略)。")
            print("-" * 30)

        return final_distributed_shards_list

    def process_sample(self, sample: Dict) -> Dict:
        """简化的样本处理，添加辅助数据掩码，并处理潜在错误"""
        sample_key_for_error = sample.get('__key__', 'UNKNOWN_KEY')
        sample_url_for_error = sample.get('__url__', 'UNKNOWN_URL')
        try:
            images = sample["images.pth"]
            raw_text_data = sample["text.json"]
            raw_auxiliary_data = sample["auxiliary.json"]

            # --- 修改：处理 text_data 的两种可能格式 ---
            if isinstance(raw_text_data, dict):
                text_data = raw_text_data
            elif isinstance(raw_text_data, list) and raw_text_data and isinstance(raw_text_data[0], dict):
                text_data = raw_text_data[0]
            else:
                raise ValueError(
                    f"Unexpected format for text_data in sample {sample_key_for_error} from {sample_url_for_error}. Expected dict or list with one dict.")
            # --- 结束修改 ---

            # --- 修改：处理 auxiliary_data 的两种可能格式 ---
            if isinstance(raw_auxiliary_data, dict):
                auxiliary_data = raw_auxiliary_data
            elif isinstance(raw_auxiliary_data, list) and raw_auxiliary_data and isinstance(raw_auxiliary_data[0],
                                                                                            dict):
                auxiliary_data = raw_auxiliary_data[0]
            else:
                raise ValueError(
                    f"Unexpected format for auxiliary_data in sample {sample_key_for_error} from {sample_url_for_error}. Expected dict or list with one dict.")
            # --- 结束修改 ---

            image_mask = ~torch.isnan(images).view(images.shape[0], -1).all(dim=1)

            # Use the class attribute for aux_fields
            aux_fields = self.aux_fields_for_test

            aux_mask_list = []
            # Make a copy if we are going to modify it, to avoid changing the cached dict in WebDataset
            auxiliary_data_processed = auxiliary_data.copy()

            for field in aux_fields:
                value = auxiliary_data_processed.get(field)
                if value is None:
                    if field in ['longitude', 'latitude']:
                        raise KeyError(
                            f"Critical field '{field}' missing in auxiliary_data for sample {sample_key_for_error}")
                    else:
                        # 对于其他缺失的辅助变量，设为0，并在mask中标记为False (表示原始缺失)
                        auxiliary_data_processed[field] = 0
                        aux_mask_list.append(False)
                else:
                    aux_mask_list.append(True)

            aux_mask = torch.tensor(aux_mask_list, dtype=torch.bool)

            # ---- DEBUG PRINT ----
            # print(f"DEBUG process_sample: type(text_data)={type(text_data)}, type(auxiliary_data_processed)={type(auxiliary_data_processed)}") # Commented out
            # ---- END DEBUG ----

            # Return a dictionary instead of a tuple
            return {
                "images_out": images,
                "text_data_out": text_data,  # text_data is already the extracted dict
                "auxiliary_data_out": auxiliary_data_processed,  # This is the processed dict
                "image_mask_out": image_mask,
                "aux_mask_out": aux_mask
            }

        except KeyError as e:
            print(
                f"KeyError processing sample {sample_key_for_error} from {sample_url_for_error}: {str(e)}. Skipping sample.")
            return None
        except ValueError as e:  # Catch the new ValueError for format issues
            print(
                f"ValueError processing sample {sample_key_for_error} from {sample_url_for_error}: {str(e)}. Skipping sample.")
            return None
        except Exception as e:
            # sample_key and sample_url are already defined at the start of the try block
            print(
                f"Error processing sample {sample_key_for_error} from {sample_url_for_error}: {str(e)}. Skipping sample.")
            return None

    def distribute_shards_by_samples(self) -> List[List[str]]:
        """基于样本数量均衡分配shard到每个进程，而不是简单地按照文件数量分配"""
        # This method is now replaced by _distribute_shards_and_set_filter_flag
        # and its results are stored in self.distributed_rank_shards_lists
        # Kept for compatibility if other parts of the code were calling it,
        # but ideally, it should be removed or refactored.
        # For now, it can return the pre-calculated distribution.
        if hasattr(self, 'distributed_rank_shards_lists'):
            return self.distributed_rank_shards_lists
        else:
            # Fallback or error, should have been initialized
            print("错误: distributed_rank_shards_lists 未在 GGTDataset 初始化时设置。")
            all_shards = [shard for shard, _ in self.shards_with_samples]
            if self.world_size and self.world_size > 1:
                return [all_shards for _ in range(self.world_size)]  # give all shards to everyone as a guess
            return [all_shards]

    def get_dataloader(self) -> wds.WebLoader:
        """创建优化的数据加载器，支持分布式训练"""

        current_rank_shards_to_process: List[str]
        samples_for_this_rank: int = 0

        if self.world_size is not None and self.rank is not None:  # DDP mode
            current_rank_shards_to_process = self.distributed_rank_shards_lists[self.rank]
            if not self.use_sample_filtering:  # Greedy shard assignment
                samples_for_this_rank = sum(
                    c for s, c in self.shards_with_samples if s in current_rank_shards_to_process)
                # Log for greedy assignment will be done by _distribute_shards_and_set_filter_flag
            # else: for sample filtering, samples_for_this_rank will be calculated later
        else:  # Non-DDP / single rank
            current_rank_shards_to_process = self.distributed_rank_shards_lists[0]
            samples_for_this_rank = self.num_samples
            # Log for non-DDP will be done by _distribute_shards_and_set_filter_flag

        if not current_rank_shards_to_process and self.num_samples > 0:
            # This might happen if a rank gets no shards in greedy and we didn't switch to sample filtering
            print(f"警告 (Rank {self.rank}): 未分配到任何shards，但数据集非空。将尝试处理空列表。")
        elif not current_rank_shards_to_process and self.num_samples == 0:
            # Expected if dataset is empty
            pass

        # 动态调整 shardshuffle: 如果启用样本级过滤，则禁用 shardshuffle 以确保各rank数据处理顺序一致
        _effective_shardshuffle = self.shardshuffle
        if self.use_sample_filtering and self.world_size is not None and self.world_size > 1:
            _effective_shardshuffle = False  # Crucial for consistent sample ordering across ranks
            if self.shardshuffle and (self.rank == 0 or self.world_size is None):  # Log once
                print(f"信息 (Rank {self.rank}): 由于启用样本级过滤, shardshuffle 被临时设为 False 以保证数据一致性。")

        # 调整worker数量
        num_shards_for_loader = len(current_rank_shards_to_process)

        if self.use_sample_filtering and self.world_size is not None and self.world_size > 1:
            effective_workers = 1  # 强制为1个worker以确保样本过滤的正确性
            if self.num_workers > 1 and (self.rank == 0 or self.world_size is None):  # Log only if changed
                print(f"信息 (Rank {self.rank if self.rank is not None else 0}): 由于启用样本级过滤, "
                      f"num_workers 从 {self.num_workers} 被强制调整为 {effective_workers} 以保证样本过滤的正确性。")
        else:
            # 在非样本过滤模式下，或者单GPU模式下，可以有多个workers
            effective_workers = min(self.num_workers, num_shards_for_loader) if num_shards_for_loader > 0 else 0

        # if self.rank == 0 or self.world_size is None : # Log worker adjustment once
        #     if self.num_workers != effective_workers:
        #          print(f"信息 (Rank {self.rank if self.rank is not None else 0}): 根据可用shards数量 ({num_shards_for_loader}), "
        #                f"dataloader workers 从 {self.num_workers} 调整为 {effective_workers}。")

        dataset = wds.WebDataset(
            current_rank_shards_to_process,  # List of shard paths for this rank
            resampled=self.resampled,  # Should be False for DDP determinism per epoch
            handler=wds.warn_and_continue,
            shardshuffle=_effective_shardshuffle,
            empty_check=False
        )

        # 应用样本内洗牌 (如果设置了 self.shuffle > 0)
        if self.shuffle > 0:  # self.shuffle is a buffer size
            dataset = dataset.shuffle(self.shuffle)

        dataset = dataset.decode()  # .decode("pil") or other specific decoders as needed

        # 应用样本级过滤 (如果 self.use_sample_filtering is True)
        if self.use_sample_filtering and self.world_size is not None and self.rank is not None and self.num_samples > 0:
            total_dataset_samples = self.num_samples

            # 计算当前rank应该处理的全局样本索引范围
            # 使用与 DistributedSampler 类似的方法，确保覆盖所有样本且尽可能均匀
            num_samples_per_replica_ceil = math.ceil(total_dataset_samples / self.world_size)
            rank_start_global_idx = self.rank * num_samples_per_replica_ceil
            rank_end_global_idx = min(rank_start_global_idx + num_samples_per_replica_ceil, total_dataset_samples)

            samples_for_this_rank = max(0, rank_end_global_idx - rank_start_global_idx)

            if self.rank == 0:  # Log the filtering strategy once
                print(f"信息: Rank 0 (及其他Ranks) 将应用样本级过滤。")
                # print(f"       总样本数: {total_dataset_samples}, Ranks: {self.world_size}")
                # print(f"       每个Rank将从全局样本流中按计算范围选取样本。")

            # 使用可序列化的过滤器类实例
            range_filter = GlobalIndexRangeFilter(rank_start_global_idx, rank_end_global_idx)
            dataset = dataset.select(range_filter)

            # print(f"Rank {self.rank}: 应用样本过滤器，处理全局索引范围 [{rank_start_global_idx}, {rank_end_global_idx-1}) 的样本。预计样本数: {samples_for_this_rank}")
        elif not self.use_sample_filtering and (
                self.world_size is None or self.rank is None):  # single rank no filtering
            samples_for_this_rank = self.num_samples
        # else: samples_for_this_rank already calculated for greedy DDP

        dataset = dataset.map(self.process_sample, handler=wds.warn_and_continue)
        # Convert the dictionary output of process_sample to a tuple
        dataset = dataset.to_tuple(
            "images_out",
            "text_data_out",
            "auxiliary_data_out",
            "image_mask_out",
            "aux_mask_out",
            handler=wds.warn_and_continue
        )

        # partial=False is important if the loss reduction averages over batch size.
        # If any rank has a smaller final batch, it could cause issues with DDP grad sync if not handled.
        dataset = dataset.batched(self.batch_size, partial=False)

        dataloader = wds.WebLoader(
            dataset,
            batch_size=None,  # WebDataset流已批处理
            num_workers=effective_workers,
            pin_memory=True,
            prefetch_factor=2 if effective_workers > 0 else None,  # None if 0 workers
            persistent_workers=True if effective_workers > 0 else False  # good for multiple epochs
        )

        # 计算当前rank的批次数
        if samples_for_this_rank > 0 and self.batch_size > 0:
            batches_per_rank = math.ceil(samples_for_this_rank / self.batch_size)
        else:
            batches_per_rank = 0

        dataloader.length = batches_per_rank  # Inform the training loop

        log_rank_id = self.rank if self.rank is not None else 0
        print(
            f"Rank {log_rank_id}: Dataloader准备就绪。有效Workers: {effective_workers}。预计样本数: {samples_for_this_rank}, "
            f"批次数: {batches_per_rank} (batch_size={self.batch_size})。")

        return dataloader


if __name__ == "__main__":
    # Example usage for testing GGTDataset
    # 这个测试用例已经使用了您指定的文件路径
    test_shard_path = "GGT-Eval/webdataset/train/train-000000.tar"
    # 您也可以在这里替换为包含多个tar文件的列表，或者一个包含tar文件的目录路径进行测试
    # test_shard_path = ["./path/to/shard1.tar", "./path/to/shard2.tar"]
    # test_shard_path = "./path/to/shards_directory/"

    print(f"--- Testing GGTDataset with: {test_shard_path} ---")
    # 为了测试，我们通常在非分布式模式下运行 (world_size=None, rank=None)
    # batch_size=1 方便逐条查看样本
    try:
        ggt_dataset = GGTDataset(
            shards_path=test_shard_path,
            batch_size=1,
            num_workers=0,  # 使用0个worker方便调试，避免多进程问题
            shuffle=0,  # 关闭样本内shuffle
            shardshuffle=False,  # 关闭shard shuffle
            resampled=False,
            world_size=None,  # 非DDP模式
            rank=None,  # 非DDP模式
            shard_counts_json_path=None  # Added
        )

        if ggt_dataset.num_samples == 0:
            print("No samples found in the dataset. Please check the shard path and content.")
        else:
            print(f"Total samples found by GGTDataset: {ggt_dataset.num_samples}")
            train_loader = ggt_dataset.get_dataloader()
            print(f"Dataloader length (batches): {train_loader.length}")

            print("\n--- Iterating through the first 3 samples (if available) ---")
            for i, batch_data in enumerate(train_loader):
                if i >= 3:  # 只打印前3个样本
                    break

                print(f"\n--- Sample {i} (from Batch {i}) ---")
                if batch_data is None or batch_data[0] is None:  # 检查样本是否被process_sample跳过
                    print("Sample was skipped by process_sample (likely due to error during processing).")
                    continue

                images, text_data, auxiliary_data, image_mask, aux_mask = batch_data

                print(f"images.shape: {images.shape}, images.dtype: {images.dtype}")
                print(f"text_data (type: {type(text_data)}): {text_data}")
                print(f"auxiliary_data (type: {type(auxiliary_data)}): {auxiliary_data}")
                print(f"image_mask.shape: {image_mask.shape}, image_mask.dtype: {image_mask.dtype}")
                # print(f"image_mask values:\n{image_mask}")
                print(f"aux_mask.shape: {aux_mask.shape}, aux_mask.dtype: {aux_mask.dtype}")
                # print(f"aux_mask values:\n{aux_mask}")

                # 验证 auxiliary_data 中的 None 值是否已按要求处理
                if isinstance(auxiliary_data, dict):
                    missing_non_critical_fields = []
                    for key, val in auxiliary_data.items():
                        if key not in ['longitude', 'latitude', 'sample_id'] and val == 0:
                            # 检查aux_mask对应位置是否为False (表示原始为None，后被设为0)
                            try:
                                field_idx = ggt_dataset.aux_fields_for_test.index(key)  # 需要aux_fields在ggt_dataset中可访问
                                if not aux_mask[0, field_idx].item():  # batch_size=1,所以取[0,idx]
                                    missing_non_critical_fields.append(key)
                            except (AttributeError, ValueError, IndexError):
                                # aux_fields_for_test 可能未定义或key不在里面,暂时忽略此检查细节
                                pass
                    if missing_non_critical_fields:
                        print(
                            f"INFO: Non-critical auxiliary fields originally missing and set to 0: {missing_non_critical_fields}")
                else:
                    print("WARNING: auxiliary_data is not a dictionary as expected after processing.")

            if i < 2 and i < ggt_dataset.num_samples - 1:
                print(
                    "\nINFO: Less than 3 samples were printed. The dataset might have fewer than 3 valid samples or iteration stopped early.")
            print("\n--- Test iteration finished ---")

    except Exception as e:
        print(f"ERROR during GGTDataset test: {str(e)}")
        import traceback

        traceback.print_exc()

    print("\nReminder: Ensure you are running dataloader_parallel.py for this test output.")
    print("The GGTDataset class within this file contains the latest parsing and error handling logic.")
