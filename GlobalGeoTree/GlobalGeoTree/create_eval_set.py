import pandas as pd
import numpy as np
from sklearn.utils import shuffle


def sample_species(df, species_list, samples_per_species, category):
    """从每个物种中采样指定数量的样本"""
    sampled_rows = []

    for species in species_list:
        species_df = df[df['level3_species'] == species]
        n_samples = min(len(species_df), samples_per_species)
        sampled = species_df.sample(n=n_samples)
        # 添加类别标签
        sampled['Category'] = category
        sampled_rows.append(sampled)

    return pd.concat(sampled_rows)


def create_evaluation_set(input_csv, output_csv, rare_threshold, frequent_threshold, size):
    """创建评估数据集"""
    # 读取数据
    df = pd.read_csv(input_csv)

    # 获取物种分类
    species_counts = df['level3_species'].value_counts()

    # 分类物种
    rare_species = species_counts[(species_counts > 20) & (species_counts <= rare_threshold)].index
    frequent_species = species_counts[species_counts >= frequent_threshold].index
    common_species = species_counts[(species_counts > rare_threshold) &
                                    (species_counts < frequent_threshold)].index

    # 从每个类别随机选择300个物种
    selected_rare = np.random.choice(rare_species, size=size, replace=False)
    selected_common = np.random.choice(common_species, size=size, replace=False)
    selected_frequent = np.random.choice(frequent_species, size=size, replace=False)

    # 计算每个类别的采样比例
    total_samples = 10000
    rare_ratio = 0.12
    common_ratio = 0.33
    frequent_ratio = 0.54

    # 计算每个物种的采样数量
    samples_per_rare = int((total_samples * rare_ratio) / size)
    samples_per_common = int((total_samples * common_ratio) / size)
    samples_per_frequent = int((total_samples * frequent_ratio) / size) + 1

    # 采样数据并添加类别标签
    rare_samples = sample_species(df, selected_rare, samples_per_rare, 'Rare')
    common_samples = sample_species(df, selected_common, samples_per_common, 'Common')
    frequent_samples = sample_species(df, selected_frequent, samples_per_frequent, 'Frequent')

    # 合并数据
    eval_set = pd.concat([rare_samples, common_samples, frequent_samples])
    eval_set = shuffle(eval_set)

    # 保存评估集
    eval_set.to_csv(output_csv, index=False)

    # 创建预训练集（排除评估集的样本）
    # pretrain_set = df[~df['sample_id'].isin(eval_set['sample_id'])]
    # pretrain_set.to_csv('GlobalGeoTree-6M.csv', index=False)

    # 打印统计信息
    print("\nEvaluation Set Statistics:")
    print(f"Total samples: {len(eval_set)}")
    print(f"Rare species samples: {len(rare_samples)}")
    print(f"Common species samples: {len(common_samples)}")
    print(f"Frequent species samples: {len(frequent_samples)}")


def generate_eval_set_overview(eval_set_path):
    """生成评估集概览表格"""
    df = pd.read_csv(eval_set_path)
    
    # 创建概览DataFrame
    overview = pd.DataFrame(columns=['Category', 'Species', 'Samples', 'Avg_Samples_per_species'])
    
    # 为每个类别生成统计信息
    for category in ['Rare', 'Common', 'Frequent']:
        # 获取该类别的数据
        category_data = df[df['Category'] == category]
        species_samples = category_data.groupby('level3_species').size()
        
        # 选择前5个物种作为示例
        example_species = species_samples.head()
        
        # 添加示例物种
        print(f"\n{category} Species Examples:")
        for species, count in example_species.items():
            print(f"{species}: {count} samples")
            overview = overview.append({
                'Category': f"{category} - {species}",
                'Species': species,
                'Samples': count,
                'Avg_Samples_per_species': count
            }, ignore_index=True)
        
        # 添加类别总计
        overview = overview.append({
            'Category': f'{category} (Total)',
            'Species': len(category_data['level3_species'].unique()),
            'Samples': len(category_data),
            'Avg_Samples_per_species': f"{len(category_data)/len(category_data['level3_species'].unique()):.1f}"
        }, ignore_index=True)
    
    # 添加总计
    overview = overview.append({
        'Category': 'Total',
        'Species': len(df['level3_species'].unique()),
        'Samples': len(df),
        'Avg_Samples_per_species': f"{len(df)/len(df['level3_species'].unique()):.1f}"
    }, ignore_index=True)
    
    # 保存概览表
    overview.to_csv('evaluation_set_overview.csv', index=False)
    
    # 打印表格
    print("\nEvaluation Set Overview:")
    print(overview.to_string())
    
    return overview


if __name__ == "__main__":
    # 创建评估集
    create_evaluation_set(
        input_csv='GGT.csv',
        output_csv='GlobalGeoTree-10kEval-90.csv',
        rare_threshold=100,
        frequent_threshold=1500,
        size=300
    )

    # 生成概览
    generate_eval_set_overview('GlobalGeoTree-10kEval-300.csv')
