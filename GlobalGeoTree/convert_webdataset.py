import os
import json
import pandas as pd
import torch
import webdataset as wds
import numpy as np
from glob import glob
from tqdm import tqdm
import logging
import argparse
import rasterio
from skimage.transform import resize
# import cv2
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle
import time

logging.getLogger('rasterio._env').setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def read_csv(csv_path):
    """读取CSV文件并返回字典 {sample_id: row_dict}"""
    df = pd.read_csv(csv_path)
    return df.set_index('sample_id').T.to_dict()

def get_csv_data(csv_index, sample_id):
    """按需从CSV文件中读取特定样本ID的数据
    
    Args:
        csv_index: 由read_csv创建的索引
        sample_id: 要读取的样本ID
        
    Returns:
        样本的CSV数据，如果不存在则返回空字典
    """
    csv_path = csv_index.get('csv_path')
    if csv_path and os.path.exists(csv_path):
        try:
            # 使用pd.read_csv与SQL查询方式高效读取特定行
            query = f'sample_id == {sample_id}'
            df = pd.read_csv(csv_path, query=query, engine='python')
            if not df.empty:
                return df.iloc[0].to_dict()
        except Exception as e:
            logging.warning(f"无法加载样本 {sample_id} 的CSV数据: {e}")
            # 备用方法：如果上面的方法失败，尝试读取全部然后过滤
            try:
                df = pd.read_csv(csv_path)
                filtered = df[df['sample_id'] == int(sample_id)]
                if not filtered.empty:
                    return filtered.iloc[0].to_dict()
            except Exception as e2:
                logging.warning(f"备用方法也无法加载样本 {sample_id} 的CSV数据: {e2}")
    return {}

def read_auxiliary_data(auxiliary_dir, load_all=False):
    """读取辅助数据，可选择只读取特定样本ID的数据
    文件名格式为: sample_{sample_id}_{lon}_{lat}.json，其中lon和lat是具体的经纬度数值
    
    Args:
        auxiliary_dir: 辅助数据目录路径
        load_all: 是否加载所有数据到内存，默认为False只建立索引
        
    Returns:
        如果load_all=True，返回完整的辅助数据字典{sample_id: data}
        如果load_all=False，返回文件索引字典{sample_id: file_path}
    """
    auxiliary_files = glob(os.path.join(auxiliary_dir, '*.json'))
    
    if load_all:
        # 原来的方法：加载所有数据到内存
        auxiliary_data = {}
        for file in auxiliary_files:
            with open(file, 'r') as f:
                data = json.load(f)
                sample_id = str(data['sample_id'])
                auxiliary_data[sample_id] = data
        return auxiliary_data
    else:
        # 新方法：只建立索引，按需加载
        auxiliary_index = {}
        for file in auxiliary_files:
            # 从文件名解析sample_id
            basename = os.path.basename(file)
            if basename.startswith('sample_'):
                parts = basename.split('_')
                if len(parts) >= 2:
                    sample_id = parts[1]
                    auxiliary_index[str(sample_id)] = file
        return auxiliary_index

def get_auxiliary_data(auxiliary_index, sample_id):
    """按需从索引中加载特定样本ID的辅助数据
    
    Args:
        auxiliary_index: 由read_auxiliary_data创建的索引
        sample_id: 要加载的样本ID
        
    Returns:
        样本的辅助数据，如果不存在则返回空字典
    """
    file_path = auxiliary_index.get(str(sample_id))
    if file_path and os.path.exists(file_path):
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logging.warning(f"无法加载样本 {sample_id} 的辅助数据: {e}")
    return {}

def resize_image_efficient(img, target_size=(5, 5)):
    h, w = img.shape[1:]
    if (h, w) == target_size:
        return img
    if h >= target_size[0] and w >= target_size[1]:
        h_start = (h - target_size[0]) // 2
        w_start = (w - target_size[1]) // 2
        return img[:, h_start:h_start+target_size[0], w_start:w_start+target_size[1]]
    resized = resize(
        img,
        (img.shape[0], target_size[0], target_size[1]),
        order=0,
        mode='edge',
        preserve_range=True,
        anti_aliasing=False
    ).astype(img.dtype)
    return resized

# def resize_image_efficient(img, target_size=(5, 5)):
#     h, w = img.shape[1:]
#     if (h, w) == target_size:
#         return img
#     if h >= target_size[0] and w >= target_size[1]:
#         h_start = (h - target_size[0]) // 2
#         w_start = (w - target_size[1]) // 2
#         return img[:, h_start:h_start + target_size[0], w_start:w_start + target_size[1]]
#
#     # 使用OpenCV的resize替代skimage.transform.resize
#     resized = np.zeros((img.shape[0], target_size[0], target_size[1]), dtype=img.dtype)
#     for i in range(img.shape[0]):
#         resized[i] = cv2.resize(img[i], target_size, interpolation=cv2.INTER_NEAREST)
#     return resized

def process_s2_images(image_dir, sample_id):
    monthly_data = np.full((12, 10, 5, 5), np.nan)
    for month in range(1, 13):
        tif_path = os.path.join(image_dir, f's2_{month}.tif')
        if os.path.exists(tif_path):
            try:
                with rasterio.open(tif_path) as src:
                    img = src.read()
                    img = resize_image_efficient(img, (5, 5))
                    monthly_data[month-1] = img
            except Exception as e:
                pass
    return torch.from_numpy(monthly_data)

def process_one_sample(args):
    sample_id, csv_info, auxiliary_index, sample_dir = args
    try:
        # 处理图像数据
        images = process_s2_images(sample_dir, sample_id)
        
        text_data = {
            "level0": csv_info.get('level0', ''),
            "level1_family": csv_info.get('level1_family', ''),
            "level2_genus": csv_info.get('level2_genus', ''),
            "level3_species": csv_info.get('level3_species', '')
        }
        
        # 按需加载辅助数据
        auxiliary_info = get_auxiliary_data(auxiliary_index, sample_id)
        
        sample = {
            "__key__": str(sample_id),
            "images.pth": images,
            "text.json": json.dumps(text_data),
            "auxiliary.json": json.dumps(auxiliary_info)
        }
        return sample, None
    except Exception as e:
        return None, str(e)

def find_sample_dirs(data_dir):
    """返回 {sample_id: sample_dir} 的字典"""
    sample_dirs = glob(os.path.join(data_dir, "sample_*"))
    mapping = {}
    for d in sample_dirs:
        # 假设目录名格式为 sample_12345_xxx
        basename = os.path.basename(d)
        parts = basename.split('_')
        if len(parts) >= 2:
            sample_id = parts[1]
            mapping[sample_id] = d
    return mapping

def get_checkpoint_path(ggt_folder_name, output_dir):
    """获取检查点文件路径"""
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    return os.path.join(checkpoint_dir, f"{ggt_folder_name}_checkpoint.pkl")

def save_checkpoint(checkpoint_path, processed_samples, stats):
    """保存处理进度检查点"""
    with open(checkpoint_path, 'wb') as f:
        pickle.dump({
            'processed_samples': processed_samples,
            'stats': stats,
            'timestamp': time.time()
        }, f)
    logging.info(f"进度保存至检查点文件: {checkpoint_path}")

def load_checkpoint(checkpoint_path):
    """加载处理进度检查点"""
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, 'rb') as f:
                checkpoint = pickle.load(f)
            logging.info(f"已加载检查点文件 ({len(checkpoint['processed_samples'])} 已处理样本)")
            return checkpoint['processed_samples'], checkpoint['stats']
        except Exception as e:
            logging.warning(f"无法加载检查点文件: {e}")
    return set(), {'total_samples': 0, 'processed': 0, 'failed': 0}

def create_webdataset(csv_data, auxiliary_index, data_dir, output_dir, output_prefix, num_workers=8, resume=True, batch_size=5000):
    """创建WebDataset，支持断点续传"""
    os.makedirs(output_dir, exist_ok=True)
    # 创建文件模式
    pattern = os.path.join(output_dir, f"{output_prefix}-%06d.tar")
    
    # 获取检查点路径
    checkpoint_path = get_checkpoint_path(output_prefix, output_dir)
    
    # 加载检查点，如果存在且启用了断点续传
    processed_samples = set()
    stats = {'total_samples': 0, 'processed': 0, 'failed': 0}
    if resume:
        processed_samples, stats = load_checkpoint(checkpoint_path)
    
    # 建立图像数据索引
    sample_dir_map = find_sample_dirs(data_dir)
    args_list = []
    
    # 准备要处理的样本列表
    for sample_id, csv_info in csv_data.items():
        # 如果已经处理过，跳过
        if str(sample_id) in processed_samples:
            continue
            
        sample_dir = sample_dir_map.get(str(sample_id))
        if sample_dir:
            args_list.append((sample_id, csv_info, auxiliary_index, sample_dir))
    
    if not args_list:
        logging.info(f"没有新样本需要处理，已完成 {len(processed_samples)} 个样本。")
        return stats
    
    # 更新总样本数
    stats['total_samples'] = len(processed_samples) + len(args_list)
    logging.info(f"总共 {stats['total_samples']} 个样本, 已处理 {len(processed_samples)} 个, 剩余 {len(args_list)} 个待处理")
    
    # 处理样本并定期保存检查点
    checkpoint_interval = max(1, min(5000, len(args_list) // 10))  # 至少每1000个样本保存一次，或处理10%的样本
    last_checkpoint_time = time.time()
    processed_count = 0
    
    # 确定起始分片索引，以避免覆盖现有文件
    start_shard_idx = 0
    existing_shards = glob(os.path.join(output_dir, f"{output_prefix}-*.tar"))
    if existing_shards and resume:
        # 从现有分片名称中提取最大索引
        shard_indices = []
        for shard in existing_shards:
            try:
                idx = int(os.path.basename(shard).split('-')[-1].split('.')[0])
                shard_indices.append(idx)
            except (ValueError, IndexError):
                pass
        
        if shard_indices:
            # 使用最大索引 + 1作为起始索引
            start_shard_idx = max(shard_indices) + 1
            logging.info(f"检测到现有分片，将从索引 {start_shard_idx} 开始写入新分片")
    
    # 创建ShardWriter，直接使用原始模式和起始索引参数
    logging.info(f"将新数据写入到: {pattern} 分片，起始索引: {start_shard_idx}")
    
    with wds.ShardWriter(pattern, maxsize=2e9, start_shard=start_shard_idx) as sink:
        # 将大量样本分批处理
        total_args = len(args_list)
        for batch_start in range(0, total_args, batch_size):
            batch_end = min(batch_start + batch_size, total_args)
            current_batch = args_list[batch_start:batch_end]
            
            logging.info(f"处理批次 {batch_start//batch_size + 1}/{(total_args+batch_size-1)//batch_size}: 样本 {batch_start}-{batch_end-1}")
            
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(process_one_sample, args) for args in current_batch]
                
                # 使用tqdm显示进度条
                pbar = tqdm(as_completed(futures), total=len(futures), desc=f"处理样本 ({output_prefix})")
                
                for f in pbar:
                    sample, err = f.result()
                    processed_count += 1
                    
                    if sample:
                        sink.write(sample)
                        processed_samples.add(sample["__key__"])
                        stats['processed'] += 1
                        pbar.set_postfix(processed=stats['processed'], failed=stats['failed'])
                    else:
                        stats['failed'] += 1
                        pbar.set_postfix(processed=stats['processed'], failed=stats['failed'])
                    
                    # 定期保存检查点
                    current_time = time.time()
                    if processed_count % checkpoint_interval == 0 or current_time - last_checkpoint_time > 6000:  # 每5分钟或处理一定数量后保存
                        save_checkpoint(checkpoint_path, processed_samples, stats)
                        last_checkpoint_time = current_time
    
    # 处理完成后保存最终检查点
    save_checkpoint(checkpoint_path, processed_samples, stats)
    logging.info(f"处理完成: {stats}")
    return stats

def find_directories_in_ggt_folder(ggt_folder):
    """在GGT文件夹中查找所需的目录和CSV文件"""
    # 查找Sentinel-2目录
    sentinel_dirs = glob(os.path.join(ggt_folder, "Sentinel-2")) + glob(os.path.join(ggt_folder, "*Sentinel-2*"))
    if not sentinel_dirs:
        raise ValueError(f"找不到Sentinel-2目录在 {ggt_folder}")
    
    # 查找Auxiliary data目录
    auxiliary_dirs = glob(os.path.join(ggt_folder, "Auxiliary data")) + glob(os.path.join(ggt_folder, "*Auxiliary*data*"))
    if not auxiliary_dirs:
        raise ValueError(f"找不到Auxiliary data目录在 {ggt_folder}")
    
    # 查找CSV文件 - 首先查找与文件夹同名的CSV
    ggt_folder_name = os.path.basename(ggt_folder)
    csv_path = os.path.join(ggt_folder, f"{ggt_folder_name}.csv")
    
    # 如果找不到同名CSV，尝试查找任何CSV文件
    if not os.path.exists(csv_path):
        csv_files = glob(os.path.join(ggt_folder, "*.csv"))
        if not csv_files:
            raise ValueError(f"找不到CSV文件在 {ggt_folder}")
        csv_path = csv_files[0]  # 使用第一个找到的CSV文件
    
    # 将webdataset保存在当前文件夹下的webdataset子目录
    output_dir = "./GGT-webdataset"
    
    return {
        "data_dir": sentinel_dirs[0],
        "auxiliary_dir": auxiliary_dirs[0],
        "csv_path": csv_path,
        "output_dir": output_dir
    }

def main(args):
    # 处理多个GGT文件夹
    if args.ggt_folders:
        ggt_folder_list = args.ggt_folders.split(',')
        logging.info(f"将处理 {len(ggt_folder_list)} 个GGT文件夹")
        
        for ggt_folder in ggt_folder_list:
            ggt_folder = ggt_folder.strip()
            logging.info(f"开始处理GGT文件夹: {ggt_folder}")
            try:
                process_single_ggt_folder(ggt_folder, args)
                logging.info(f"完成处理GGT文件夹: {ggt_folder}")
            except Exception as e:
                logging.error(f"处理文件夹 {ggt_folder} 时出错: {str(e)}")
    # 处理单个GGT文件夹（向后兼容）
    elif args.ggt_folder:
        process_single_ggt_folder(args.ggt_folder, args)
    else:
        # 如果没有指定GGT文件夹，使用命令行参数
        output_prefix = "dataset"
        logging.info(f"使用以下参数:")
        logging.info(f"  CSV文件: {args.csv_path}")
        logging.info(f"  数据目录: {args.data_dir}")
        logging.info(f"  辅助数据目录: {args.auxiliary_dir}")
        logging.info(f"  输出目录: {args.output_dir}")
        logging.info(f"  输出前缀: {output_prefix}")
        logging.info(f"  断点续传: {'启用' if args.resume else '禁用'}")
        
        csv_data = read_csv(args.csv_path)
        print("csv read successfully")
        auxiliary_index = read_auxiliary_data(args.auxiliary_dir)
        print("auxiliary data read successfully")
        os.makedirs(args.output_dir, exist_ok=True)
        print("output dir created")
        logging.info("创建WebDataset...")
        create_webdataset(
            csv_data,
            auxiliary_index,
            args.data_dir,
            args.output_dir,
            output_prefix,
            num_workers=args.num_workers,
            resume=args.resume
        )
        logging.info("WebDataset创建完成.")

def process_single_ggt_folder(ggt_folder, args):
    """处理单个GGT文件夹"""
    try:
        logging.info(f"分析GGT文件夹: {ggt_folder}")
        dirs = find_directories_in_ggt_folder(ggt_folder)
        
        # 使用找到的目录
        csv_path = dirs["csv_path"]
        data_dir = dirs["data_dir"]
        auxiliary_dir = dirs["auxiliary_dir"]
        output_dir = dirs["output_dir"]
        
        # 使用GGT文件夹名称作为输出前缀
        output_prefix = os.path.basename(ggt_folder)
        
        logging.info(f"使用以下参数:")
        logging.info(f"  CSV文件: {csv_path}")
        logging.info(f"  数据目录: {data_dir}")
        logging.info(f"  辅助数据目录: {auxiliary_dir}")
        logging.info(f"  输出目录: {output_dir}")
        logging.info(f"  输出前缀: {output_prefix}")
        logging.info(f"  断点续传: {'启用' if args.resume else '禁用'}")
        
        csv_data = read_csv(csv_path)
        print("csv read successfully")
        auxiliary_index = read_auxiliary_data(auxiliary_dir)
        print("auxiliary data read successfully")
        os.makedirs(output_dir, exist_ok=True)
        print("output dir created")
        
        logging.info("创建WebDataset...")
        create_webdataset(
            csv_data,
            auxiliary_index,
            data_dir,
            output_dir,
            output_prefix,
            num_workers=args.num_workers,
            resume=args.resume
        )
        logging.info(f"文件夹 {ggt_folder} 的WebDataset创建完成.")
    except ValueError as e:
        logging.error(str(e))
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从GGT文件夹创建WebDataset。")
    
    # 添加多个GGT文件夹参数
    parser.add_argument('--ggt_folders', type=str, default=None,
                        help='多个GGT文件夹的路径，用逗号分隔，如"GGT-5.5M_5.75M,GGT-5.75M_6M"')
    
    # 保留单个文件夹参数以便向后兼容
    parser.add_argument('--ggt_folder', type=str, default=None,
                        help='单个GGT文件夹的路径，包含所需的所有数据')
    
    # 添加断点续传参数
    parser.add_argument('--resume', action='store_true', default=True,
                        help='是否从上次中断的地方继续处理')
    
    # 保留原有参数以便向后兼容
    parser.add_argument('--csv_path', type=str, default='GlobalGeoTree-10kEval-300.csv',
                        help='CSV文件的路径.')
    parser.add_argument('--data_dir', type=str, default='GGT-Eval-300/Sentinel-2',
                        help='包含图像数据的目录.')
    parser.add_argument('--auxiliary_dir', type=str, default='GGT-Eval-300/Auxiliary data',
                        help='包含辅助数据的目录.')
    parser.add_argument('--output_dir', type=str, default='GGT-Eval-300/webdataset',
                        help='保存WebDataset输出的目录.')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='并行工作进程数.')
    
    args = parser.parse_args()
    main(args)
