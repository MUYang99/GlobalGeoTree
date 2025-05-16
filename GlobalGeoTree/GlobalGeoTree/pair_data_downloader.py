import ee
from multiprocessing.dummy import Lock, Pool
import os
import time
import requests
import csv
import json
import argparse
import pandas as pd


class Counter:
    def __init__(self, start=0):
        self.value = start
        self.lock = Lock()

    def update(self, delta=1):
        with self.lock:
            self.value += delta
            return self.value


def get_monthly_dates(year, month):
    """生成月度日期范围"""
    start_date = f"{year}-{month:02d}-01"
    if month == 12:
        end_date = f"{year + 1}-01-01"
    else:
        end_date = f"{year}-{month + 1:02d}-01"
    return start_date, end_date


# def create_square_region(lon, lat, size_m=50):
#     """以(lon, lat)为中心，创建边长为size_m的正方形Polygon"""
#     import math
#     import ee
#
#     # 经纬度每度对应的米数
#     lat_per_meter = 1 / 110574  # 纬度方向
#     lon_per_meter = 1 / (111319 * math.cos(math.radians(lat)))  # 经度方向
#
#     half_side_lat = (size_m / 2) * lat_per_meter
#     half_side_lon = (size_m / 2) * lon_per_meter
#
#     coords = [
#         [lon - half_side_lon, lat - half_side_lat],
#         [lon - half_side_lon, lat + half_side_lat],
#         [lon + half_side_lon, lat + half_side_lat],
#         [lon + half_side_lon, lat - half_side_lat],
#         [lon - half_side_lon, lat - half_side_lat]
#     ]
#     return ee.Geometry.Polygon([coords])


def download_sentinel2(sample_id, lon, lat, year, month, save_dir, radius=20):
    """下载Sentinel-2月度中值合成影像"""
    # 构建存储路径
    str_id = str(sample_id)
    str_lon = str(f"{lon:.5f}")
    str_lat = str(f"{lat:.5f}")

    sample_dir = os.path.join(save_dir, f"sample_{str_id}_{str_lon}_{str_lat}")
    os.makedirs(sample_dir, exist_ok=True)

    save_path = os.path.join(sample_dir, f"s2_{month}.tif")

    # 如果文件已存在且大小正常，则跳过
    if os.path.exists(save_path) and os.path.getsize(save_path) > 1000:
        return {"status": 2, "message": "文件已存在"}

    # 设置日期范围
    start_date, end_date = get_monthly_dates(year, month)

    # 创建GEE点和区域(5x5像素补丁，约25米半径)
    point = ee.Geometry.Point([lon, lat])
    region = point.buffer(radius).bounds()
    # region = create_square_region(lon, lat, size_m=50)

    # 获取Sentinel-2数据
    try:
        collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        collection = collection.filterDate(start_date, end_date)
        collection = collection.filterBounds(region)
        collection = collection.filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))

        # 检查是否有可用图像
        count = collection.size().getInfo()
        if count == 0:
            return {"status": 0, "message": f"无可用影像: {start_date} 至 {end_date}"}

        # 计算中值合成
        image = collection.median()

        # 选择需要的波段
        bands = ['B2', 'B3', 'B4', 'B8', 'B5', 'B6', 'B7', 'B8A', 'B11', 'B12']
        image = image.select(bands)

        # 获取下载URL
        url = image.getDownloadUrl({
            'bands': bands,
            'region': region,
            # 'dimensions': [5, 5],
            'scale': 10,  # 10米分辨率
            'format': 'GEO_TIFF'
        })

        # 下载图像
        response = requests.get(url)
        if response.status_code != 200:
            return {"status": 0, "message": f"下载失败: HTTP {response.status_code}"}

        with open(save_path, 'wb') as fd:
            fd.write(response.content)

        return {"status": 1, "message": "下载成功", "path": save_path}

    except Exception as e:
        return {"status": 0, "message": f"处理失败: {str(e)}"}


def collect_auxiliary_data(sample_id, lon, lat, save_dir):
    """收集辅助数据: 地形、土壤和生物气候变量"""
    # 构建存储路径
    str_id = str(sample_id)
    str_lon = str(f"{lon:.5f}")
    str_lat = str(f"{lat:.5f}")

    save_path = os.path.join(save_dir, f"sample_{str_id}_{str_lon}_{str_lat}.json")

    # 如果文件已存在，则跳过
    if os.path.exists(save_path):
        return {"status": 2, "message": "辅助数据已存在"}

    try:
        # 创建点
        point = ee.Geometry.Point([lon, lat])

        # 初始化数据字典
        aux_data = {
            'sample_id': sample_id,
            'longitude': lon,
            'latitude': lat
        }

        # 1. 获取地形数据
        try:
            srtm = ee.Image('USGS/SRTMGL1_003')
            terrain = ee.Terrain.products(srtm)
            terrain_sample = terrain.select(['elevation', 'slope', 'aspect']).sample(point, 30).first()
            terrain_data = terrain_sample.getInfo()['properties']

            aux_data.update({
                'elevation': terrain_data.get('elevation', None),
                'slope': terrain_data.get('slope', None),
                'aspect': terrain_data.get('aspect', None)
            })
        except Exception as e:
            print(f"地形数据获取失败: {e}")
            aux_data.update({
                'elevation': None,
                'slope': None,
                'aspect': None
            })

        # 2. 获取土壤湿度数据
        try:
            # 加载正确的数据集
            soilgrids = ee.ImageCollection("ISRIC/SoilGrids250m/v20").first()

            # 选择33kPa下的体积含水量波段（田间持水量）
            soil_moisture_bands = [
                'wv0033_0-5cm',  # 0-5cm深度
                'wv0033_5-15cm',  # 5-15cm深度
                'wv0033_15-30cm'  # 15-30cm深度
            ]

            # 选择需要的波段
            soil_moisture = soilgrids.select(soil_moisture_bands)

            # 在指定点位采样
            soil_sample = soil_moisture.sample(point, 250).first()
            soil_data = soil_sample.getInfo()['properties']

            # 更新辅助数据字典
            aux_data.update({
                'soil_moisture_0_5cm': soil_data.get('wv0033_0-5cm', None),
                'soil_moisture_5_15cm': soil_data.get('wv0033_5-15cm', None),
                'soil_moisture_15_30cm': soil_data.get('wv0033_15-30cm', None)
            })

        except Exception as e:
            print(f"土壤数据获取失败: {e}")
            aux_data.update({
                'soil_moisture_0_5cm': None,
                'soil_moisture_5_15cm': None,
                'soil_moisture_15_30cm': None
            })

        # 3. 获取生物气候变量
        try:
            worldclim = ee.Image('WORLDCLIM/V1/BIO')
            bioclim_sample = worldclim.sample(point, 927.67).first()
            bioclim_data = bioclim_sample.getInfo()['properties']

            for i in range(1, 20):
                band_name = f'bio{i:02d}'
                aux_data[band_name] = bioclim_data.get(band_name, None)
        except Exception as e:
            print(f"生物气候数据获取失败: {e}")
            for i in range(1, 20):
                band_name = f'bio{i:02d}'
                aux_data[band_name] = None

        # 写入文件
        with open(save_path, 'w') as f:
            json.dump(aux_data, f, indent=2)

        return {"status": 1, "message": "辅助数据收集成功", "path": save_path}

    except Exception as e:
        return {"status": 0, "message": f"辅助数据收集失败: {str(e)}"}


def worker(args):
    """工作线程函数，处理单个样本的数据下载"""
    sample_id, row, project_root, year, counter = args

    worker_start = time.time()
    lon = row['longitude']
    lat = row['latitude']

    # log_path = os.path.join(project_root, "log", f"sample_{sample_id}_log.csv")
    #
    # # 记录函数
    # def log_result(task_type, result):
    #     with open(log_path, 'a', encoding='utf-8') as f:
    #         writer = csv.writer(f)
    #         writer.writerow([sample_id, lon, lat, task_type, result['status'], result['message']])

    # 1. 处理文本数据
    # text_dir = os.path.join(project_root, "text_data")
    # text_result = save_text_data(row, text_dir)
    # log_result("text_data", text_result)

    # 2. 处理辅助数据
    aux_dir = os.path.join(project_root, "Auxiliary data")
    aux_result = collect_auxiliary_data(sample_id, lon, lat, aux_dir)
    # log_result("auxiliary_data", aux_result)

    # 3. 处理Sentinel-2数据
    s2_dir = os.path.join(project_root, "Sentinel-2")
    months_success = 0
    months_total = 12

    for month in range(1, 13):
        s2_result = download_sentinel2(sample_id, lon, lat, year, month, s2_dir)
        # log_result(f"sentinel2_month{month}", s2_result)
        if s2_result['status'] > 0:  # 成功或文件已存在
            months_success += 1

    # 更新计数器并打印进度
    count = counter.update(1)
    elapsed = time.time() - worker_start

    print(f"完成样本 [{count}]: ID={sample_id}, 坐标=[{lon:.5f},{lat:.5f}], "
          f"S2={months_success}/{months_total}, 用时={elapsed:.2f}秒")

    # 限制API调用频率
    if elapsed < 2:
        time.sleep(2 - elapsed)

    return {
        "sample_id": sample_id,
        "success": months_success > 0 and aux_result['status'] > 0
        # "success": months_success > 0
    }


def main():
    parser = argparse.ArgumentParser(description="树种观测数据采集系统")
    parser.add_argument("--csv", type=str, default="GlobalGeoTree-10kEval-300.csv", help="输入CSV文件路径")
    # parser.add_argument("--out", type=str, default="./GGT", help="项目输出根目录")
    parser.add_argument("--out", type=str, default="GGT-Eval-300", help="项目输出根目录")
    # parser.add_argument("--out", type=str, default="D:\\GGT-test", help="项目输出根目录")
    parser.add_argument("--year", type=int, default=2020, help="Sentinel-2数据年份")
    parser.add_argument("--workers", type=int, default=20, help="并行工作线程数")
    parser.add_argument("--start", type=int, default=0, help="起始样本索引")
    parser.add_argument("--end", type=int, default=6263346, help="结束样本索引")
    parser.add_argument("--resume", action="store_true", help="从上次中断处继续")

    args = parser.parse_args()

    ee.Authenticate()
    ee.Initialize(project='ee-schafer')

    # 读取CSV数据
    df = pd.read_csv(args.csv)
    print(f"已加载{len(df)}条样本记录")

    # 创建目录结构
    project_root = args.out
    os.makedirs(os.path.join(project_root, "Sentinel-2"), exist_ok=True)
    os.makedirs(os.path.join(project_root, "Auxiliary data"), exist_ok=True)
    # os.makedirs(os.path.join(project_root, "text_data"), exist_ok=True)
    os.makedirs(os.path.join(project_root, "log"), exist_ok=True)

    # 创建主日志文件
    log_header = os.path.join(project_root, "log", "processing_log.csv")
    if not os.path.exists(log_header) or not args.resume:
        with open(log_header, 'w', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["sample_id", "lon", "lat", "processed", "success"])

    # 确定处理范围
    start_idx = args.start
    end_idx = args.end if args.end is not None else len(df)
    print(f"处理样本范围: {start_idx} 到 {end_idx - 1}")

    # 检查是否从上次中断处继续
    processed_samples = set()
    if args.resume:
        try:
            log_df = pd.read_csv(log_header)
            processed_samples = set(log_df['sample_id'].tolist())
            print(f"从上次中断处继续，已处理 {len(processed_samples)} 个样本")
        except:
            print("未找到有效的处理日志，将从头开始处理")

    # 创建工作任务列表
    counter = Counter(0)
    tasks = []
    for idx, row in df.iloc[start_idx:end_idx].iterrows():
        sample_id = row['sample_id']
        if args.resume and sample_id in processed_samples:
            continue
        tasks.append((sample_id, row, project_root, args.year, counter))

    print(f"待处理样本数: {len(tasks)}")
    if len(tasks) == 0:
        print("没有需要处理的样本，退出程序")
        return

    # 开始处理
    start_time = time.time()
    results = []

    if args.workers <= 1:
        # 单线程处理
        for task in tasks:
            result = worker(task)
            results.append(result)

            # 记录处理结果
            with open(log_header, 'a', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    result['sample_id'],
                    df.loc[df['sample_id'] == result['sample_id']]['longitude'].values[0],
                    df.loc[df['sample_id'] == result['sample_id']]['latitude'].values[0],
                    1,  # processed
                    int(result['success'])
                ])
    else:
        # 多线程处理
        with Pool(processes=args.workers) as p:
            for result in p.imap_unordered(worker, tasks):
                results.append(result)

                # 记录处理结果
                with open(log_header, 'a', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        result['sample_id'],
                        df.loc[df['sample_id'] == result['sample_id']]['longitude'].values[0],
                        df.loc[df['sample_id'] == result['sample_id']]['latitude'].values[0],
                        1,  # processed
                        int(result['success'])
                    ])

    # 统计结果
    success_count = sum(1 for r in results if r['success'])
    total_time = time.time() - start_time

    print(f"\n处理完成!")
    print(f"总样本数: {len(results)}")
    print(f"成功样本数: {success_count} ({success_count / len(results) * 100:.1f}%)")
    print(f"总用时: {total_time:.2f}秒, 平均每样本: {total_time / len(results):.2f}秒")


if __name__ == "__main__":
    main()











