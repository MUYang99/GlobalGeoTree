import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModel, CLIPVisionConfig
from transformers import logging as hf_logging
import logging
hf_logging.set_verbosity_error()


class CLIPViTFeatureExtractor(nn.Module):
    """使用CLIP-ViT提取单张图片特征"""
    def __init__(self, clip_model_name="openai/clip-vit-base-patch16"):
        super().__init__()
        # 添加ignore_mismatched_sizes=True来抑制警告
        self.model = CLIPVisionModel.from_pretrained(
            clip_model_name,
            ignore_mismatched_sizes=True
        )
        
        # 修改第一层以适应10通道输入
        original_embedding = self.model.vision_model.embeddings.patch_embedding.weight
        original_shape = original_embedding.shape
        
        # 创建新权重 - 从3通道扩展到10通道
        new_embedding = torch.zeros(original_shape[0], 10, original_shape[2], original_shape[3],
                                  device=original_embedding.device)
        
        # 使用现有权重的平均值为新通道赋值
        channel_mean = original_embedding.mean(dim=1, keepdim=True)
        for i in range(10):
            if i < 3:
                new_embedding[:, i, :, :] = original_embedding[:, i, :, :]
            else:
                new_embedding[:, i, :, :] = channel_mean.squeeze(1)
        
        # 替换权重
        self.model.vision_model.embeddings.patch_embedding.weight = nn.Parameter(new_embedding)
        
    def forward(self, x):
        # x: (B, C, H, W)
        # 上采样到224x224
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        
        # 获取特征
        outputs = self.model(x)
        features = outputs.last_hidden_state[:, 0, :]  # [B, 768]
        return features


class CLIPViT16(nn.Module):
    """使用CLIP-ViT提取单张图片特征，支持10通道输入，patch size=16，输入自动上采样到16x16"""

    def __init__(self, clip_model_name="openai/clip-vit-base-patch16"):
        super().__init__()
        # 1. 创建配置
        config = CLIPVisionConfig.from_pretrained(clip_model_name)
        config.image_size = 16
        config.patch_size = 16
        config.hidden_size = 768

        # 2. 用新配置加载模型
        self.model = CLIPVisionModel(config)

        # 3. 修改patch embedding层
        original_embedding = self.model.vision_model.embeddings.patch_embedding.weight
        original_shape = original_embedding.shape  # [768, 3, 16, 16]
        new_embedding = torch.zeros(original_shape[0], 10, original_shape[2], original_shape[3],
                                   device=original_embedding.device)
        channel_mean = original_embedding.mean(dim=1, keepdim=True)
        for i in range(10):
            if i < 3:
                new_embedding[:, i, :, :] = original_embedding[:, i, :, :]
            else:
                new_embedding[:, i, :, :] = channel_mean.squeeze(1)
        self.model.vision_model.embeddings.patch_embedding = nn.Conv2d(
            10, 768, kernel_size=(16, 16), stride=(16, 16), bias=False
        )
        self.model.vision_model.embeddings.patch_embedding.weight = nn.Parameter(new_embedding)

        # 4. 修改position embedding长度
        num_positions = 2
        original_position_embedding = self.model.vision_model.embeddings.position_embedding.weight
        new_position_embedding = nn.Embedding(num_positions, 768)
        new_position_embedding.weight.data = original_position_embedding[:num_positions]
        self.model.vision_model.embeddings.position_embedding = new_position_embedding

        # 5. 更新position_ids
        self.model.vision_model.embeddings.register_buffer(
            "position_ids", torch.arange(num_positions).expand((1, -1))
        )

    def forward(self, x):
        # x: (B, 10, H, W)
        # 上采样到16x16
        x = F.interpolate(x, size=(16, 16), mode='bilinear', align_corners=False)
        # 获取特征
        outputs = self.model(x)
        features = outputs.last_hidden_state[:, 0, :]  # [B, 768]
        return features
    

class SelfAttentionFeatureExtractor(nn.Module):
    def __init__(self, input_dim=768, attn_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=attn_heads, batch_first=True)
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # 池化到(1, C)

    def forward(self, x):
        # x: (B, T, C)
        attn_out, _ = self.attn(x, x, x)  # (B, T, C)
        x = attn_out.transpose(1,2)       # (B, C, T)
        feat = self.global_pool(x).squeeze(-1)  # (B, C)
        return feat


class TemporalFeatureExtractor(nn.Module):
    """时序特征提取器，结合CLIP-ViT和时序注意力"""
    def __init__(self):
        super().__init__()
        self.clip_vit = CLIPViT16()
        self.temporal_attention = SelfAttentionFeatureExtractor(input_dim=768, attn_heads=4)
        
    def forward(self, x, mask=None):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        
        # 重塑张量以进行批处理
        # 将时间维度合并到批次维度中
        x_reshaped = x.reshape(B * T, C, H, W)
        
        # 批量处理所有时间步
        features = self.clip_vit(x_reshaped)
        
        # 恢复原始形状
        features = features.reshape(B, T, -1)
        
        # 应用掩码
        if mask is not None:
            features = features * mask.unsqueeze(-1)
        
        # 使用时序注意力
        temporal_features = self.temporal_attention(features)
        
        return temporal_features


class AuxiliaryEncoder(nn.Module):
    """辅助数据编码器，处理地理位置、地形、土壤湿度和生物气候变量"""
    def __init__(self, hidden_dim=256, out_dim=256):
        super().__init__()
        
        # 特征分组编码器
        self.geo_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim),  # longitude, latitude, elevation
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        self.terrain_encoder = nn.Sequential(
            nn.Linear(2, hidden_dim),  # slope, aspect
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        self.soil_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim),  # soil_moisture_0_5cm, soil_moisture_5_15cm, soil_moisture_15_30cm
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        self.bio_encoder = nn.Sequential(
            nn.Linear(19, hidden_dim),  # bio01-bio19
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 4, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU()
        )
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
    def forward(self, x):
        """
        Args:
            x: 归一化后的特征张量 [B, num_features]
        Returns:
            features: 编码后的特征向量 [B, out_dim]
        """
        
        # 分割特征
        geo_features = x[:, :3]  # 地理特征
        terrain_features = x[:, 3:5]  # 地形特征
        soil_features = x[:, 5:8]  # 土壤湿度特征
        bio_features = x[:, 8:]  # 生物气候特征
        
        # 编码各个特征组
        geo_encoded = self.geo_encoder(geo_features)
        terrain_encoded = self.terrain_encoder(terrain_features)
        soil_encoded = self.soil_encoder(soil_features)
        bio_encoded = self.bio_encoder(bio_features)
        
        # 特征融合
        combined = torch.cat([
            geo_encoded,
            terrain_encoded,
            soil_encoded,
            bio_encoded
        ], dim=1)
        
        features = self.fusion(combined)
        return features


class MixedTextEncoder(nn.Module):
    """
    混合文本编码器：将所有层级文本拼接为一句，送入CLIP文本编码器
    """
    def __init__(self, clip_model_name="openai/clip-vit-base-patch16", sep_token=" / "):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
        self.text_encoder = CLIPTextModel.from_pretrained(clip_model_name)
        self.sep_token = sep_token

    def forward(self, text_data):
        """
        Args:
            text_data: 字典列表，每个元素包含多个层级文本
                [{'level0': '...', 'level1_family': '...', 'level2_genus': '...', 'level3_species': '...'}]
        Returns:
            features: [B, 512]，每个样本的文本特征
        """
        # 拼接各层级文本
        mixed_texts = [
            self.sep_token.join([
                item['level0'],
                item['level1_family'],
                item['level2_genus'],
                item['level3_species']
            ]) for item in text_data
        ]
        device = next(self.parameters()).device
        inputs = self.tokenizer(mixed_texts, padding=True, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = self.text_encoder(**inputs)
        return outputs.pooler_output  # [B, 512]


class GeoTreeClip(nn.Module):
    """
    基于CLIP的树种识别模型，支持多层级分类和对比学习
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temporal_extractor = TemporalFeatureExtractor()
        self.auxiliary_encoder = AuxiliaryEncoder()
        self.text_encoder = MixedTextEncoder()
        
        # 图像特征投影层，将768维投影到512维
        self.image_projection = nn.Sequential(
            nn.Linear(768, 512),
            nn.LayerNorm(512),
            nn.ReLU()
        )
        
        # 辅助特征投影层，将256维投影到512维
        self.aux_projection = nn.Sequential(
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.ReLU()
        )
        
        # 特征融合层
        self.fusion = nn.Sequential(
            nn.Linear(512 + 512, 512),  # 融合图像和辅助特征
            nn.LayerNorm(512),
            nn.ReLU()
        )
        
        # 归一化层
        self.image_normalizer = CLIPNormalizer()
        self.auxiliary_normalizer = AuxiliaryNormalizer()
        
        self.temperature = temperature
        
    def forward(self, images, text_data, auxiliary_data, image_mask, aux_mask):
        # 归一化图像数据
        images = self.image_normalizer(images, image_mask)
        
        # 归一化辅助数据
        auxiliary_features = self.auxiliary_normalizer(auxiliary_data, aux_mask)
        
        # 提取时间序列特征
        image_features = self.temporal_extractor(images, image_mask)
        image_features = self.image_projection(image_features)  # [B, 512]
        
        # 提取辅助数据特征
        aux_features = self.auxiliary_encoder(auxiliary_features)
        aux_features = self.aux_projection(aux_features)  # [B, 512]
        
        # 特征融合
        combined_features = self.fusion(
            torch.cat([image_features, aux_features], dim=1)
        )
        
        # 提取文本特征
        text_features = self.text_encoder(text_data)  # 每个层级都是 [B, 512]
        
        return combined_features, text_features


class CLIPContrastiveLoss(nn.Module):
    """
    标准CLIP对比损失（对称InfoNCE），适用于单一文本特征
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, image_features, text_features):
        """
        Args:
            image_features: [B, D]
            text_features: [B, D]
        Returns:
            loss: 标量
        """
        # L2归一化
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        # 相似度矩阵 [B, B]
        logits = torch.matmul(image_features, text_features.t()) / self.temperature
        labels = torch.arange(logits.size(0), device=logits.device)

        # 图像到文本
        loss_i2t = F.cross_entropy(logits, labels)
        # 文本到图像
        loss_t2i = F.cross_entropy(logits.t(), labels)

        return (loss_i2t + loss_t2i) / 2


class CLIPNormalizer(nn.Module):
    """
    简单的归一化层，将数据归一化到(-1,1)范围
    与CLIP保持一致，使用固定的归一化方式
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, x, mask=None):
        """
        前向传播
        Args:
            x: 输入张量 [B, T, C, H, W]
            mask: 可选，用于标记有效值 [B, T]
        Returns:
            normalized: 归一化后的张量，维度与输入相同
        """
        # 计算每个通道的最小值和最大值
        if mask is not None:
            # 扩展mask到与x相同的维度 [B, T, 1, 1, 1]
            valid_mask = mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            x_masked = torch.where(valid_mask, x, torch.zeros_like(x))
            # 在batch和time维度上计算最小最大值 [C]
            min_val = x_masked.reshape(-1, x.shape[2], x.shape[3], x.shape[4]).min(dim=0)[0].min(dim=1)[0].min(dim=1)[0]
            max_val = x_masked.reshape(-1, x.shape[2], x.shape[3], x.shape[4]).max(dim=0)[0].max(dim=1)[0].max(dim=1)[0]
        else:
            # 在batch和time维度上计算最小最大值 [C]
            min_val = x.reshape(-1, x.shape[2], x.shape[3], x.shape[4]).min(dim=0)[0].min(dim=1)[0].min(dim=1)[0]
            max_val = x.reshape(-1, x.shape[2], x.shape[3], x.shape[4]).max(dim=0)[0].max(dim=1)[0].max(dim=1)[0]
        
        # 归一化到(0,1)，保持输入维度
        normalized = (x - min_val[None, None, :, None, None]) / \
                    (max_val[None, None, :, None, None] - min_val[None, None, :, None, None] + 1e-6)
        
        # 归一化到(-1,1)
        normalized = 2.0 * normalized - 1.0
        
        # 处理无效值
        if mask is not None:
            normalized = torch.where(valid_mask, normalized, torch.zeros_like(normalized))
        
        return normalized


class AuxiliaryNormalizer(nn.Module):
    """
    辅助数据归一化层，将辅助数据归一化到(-1,1)范围
    使用dataloader提供的aux_mask处理缺失值
    """
    def __init__(self):
        super().__init__()
        # 定义特征顺序，确保一致性
        self.geo_fields = ['longitude', 'latitude', 'elevation']
        self.terrain_fields = ['slope', 'aspect']
        self.soil_fields = ['soil_moisture_0_5cm', 'soil_moisture_5_15cm', 'soil_moisture_15_30cm']
        self.bio_fields = [f'bio{i:02d}' for i in range(1, 20)]
        self.all_fields = self.geo_fields + self.terrain_fields + self.soil_fields + self.bio_fields
        
    def forward(self, x, mask):
        """
        前向传播
        Args:
            x: 辅助数据字典列表，每个字典包含数值型特征
            mask: 辅助数据掩码 [B, num_features]，来自dataloader的aux_mask
        Returns:
            normalized: 归一化后的特征张量 [B, num_features]
        """
        batch_size = len(x)
        device = mask.device
        
        # 将字典数据转换为张量，处理None值
        features = []
        for sample in x:
            sample_features = []
            for field in self.all_fields:
                value = sample[field]
                # 如果值为None或mask对应位置为False，使用0
                if value is None:
                    sample_features.append(0.0)
                else:
                    try:
                        sample_features.append(float(value))
                    except (ValueError, TypeError):
                        # 如果转换失败，使用0并记录警告
                        logging.warning(f"无法转换字段 {field} 的值 {value} 为浮点数，使用0替代")
                        sample_features.append(0.0)
            features.append(sample_features)
        
        # 转换为张量 [B, num_features]
        features = torch.tensor(features, dtype=torch.float32, device=device)
        
        # 使用mask来处理缺失值，将缺失值位置设为0
        features = features * mask
        
        # 对每个特征分别计算统计量，只使用有效值
        min_vals = []
        max_vals = []
        eps = 1e-6
        
        for i in range(features.shape[1]):
            feature_col = features[:, i]
            mask_col = mask[:, i]
            valid_values = feature_col[mask_col]
            
            if len(valid_values) > 0:
                min_val = valid_values.min()
                max_val = valid_values.max()
                
                # 处理最大值等于最小值的情况
                if abs(max_val - min_val) < eps:
                    max_val = min_val + 1.0
            else:
                # 如果该特征完全没有有效值，使用安全的默认值
                min_val = 0.0
                max_val = 1.0
            
            min_vals.append(min_val)
            max_vals.append(max_val)
        
        min_vals = torch.tensor(min_vals, dtype=torch.float32, device=device)
        max_vals = torch.tensor(max_vals, dtype=torch.float32, device=device)
        
        # 归一化到(-1,1)范围
        normalized = 2.0 * ((features - min_vals) / (max_vals - min_vals + eps)) - 1.0
        
        # 确保缺失值位置为0
        normalized = normalized * mask
        
        return normalized


# 测试代码
if __name__ == "__main__":
    # 创建测试数据
    batch_size = 2
    x = torch.randn(batch_size, 12, 10, 5, 5)  # [B, T, C, H, W]
    mask = torch.ones(batch_size, 12, dtype=torch.bool)  # 所有时间步都有效

    # 创建归一化器
    normalizer = CLIPNormalizer()

    # 归一化
    normalized_x = normalizer(x, mask)
    print(normalized_x.shape)

    # 创建模型
    model = TemporalFeatureExtractor()

    # 前向传播
    features = model(normalized_x, mask)
    print("输入形状:", normalized_x.shape)
    print("输出特征形状:", features.shape)  # 应该是 [batch_size, 768]

    # 创建测试数据 - 模拟真实数据格式
    auxiliary_data = []
    for i in range(batch_size):
        sample = {
            'longitude': -65.0 + i,  # 模拟不同的经度
            'latitude': 18.0 + i,    # 模拟不同的纬度
            'elevation': 410 + i * 10,
            'slope': 14 + i,
            'aspect': 68 + i,
            'soil_moisture_0_5cm': 309 + i,
            'soil_moisture_5_15cm': 295 + i,
            'soil_moisture_15_30cm': 284 + i
        }
        # 添加生物气候特征
        for j in range(1, 20):
            sample[f'bio{j:02d}'] = 200 + j + i
        auxiliary_data.append(sample)

    # 创建掩码 - 27个特征 (3 geo + 2 terrain + 3 soil + 19 bio)
    mask_aux = torch.ones(batch_size, 27, dtype=torch.bool)
    # 随机将一些特征标记为缺失
    mask_aux[0, [1, 5, 10]] = False  # 第一个样本的一些特征缺失
    mask_aux[1, [0, 6, 15]] = False  # 第二个样本的一些特征缺失

    # 创建归一化器
    normalizer = AuxiliaryNormalizer()

    # 归一化
    normalized_aux = normalizer(auxiliary_data, mask_aux)
    print(normalized_aux.shape)

    # 创建模型
    model = AuxiliaryEncoder()

    # 前向传播
    features = model(normalized_aux)
    print("\n输出特征形状:", features.shape)  # 应该是 [batch_size, 256]

    text_data = [{
        'level0': 'Deciduous Broadleaf',
        'level1_family': 'Phytolaccaceae',
        'level2_genus': 'Phytolacca',
        'level3_species': 'Phytolacca icosandra',
        'category': 'Frequent'
    }] * 2

    model = MixedTextEncoder()

    # 前向传播
    features = model(text_data)
    print("输入文本数据:")
    print(text_data)
    print("\n输出特征形状:", features.shape)  # 应该是 [2, 512]


    # 创建模型
    model = GeoTreeClip()
    image_features, text_features = model(x, text_data, auxiliary_data, mask, mask_aux)
    print("图像特征形状:", image_features.shape)  # [batch_size, 512]
    print("文本特征形状:", text_features.shape)
