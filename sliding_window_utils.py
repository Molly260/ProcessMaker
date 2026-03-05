import torch
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import cdist
import cv2
from PIL import Image
import torchvision.transforms as transforms

# CLIP and DINO imports
try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    print("Warning: CLIP not available. Install with: pip install git+https://github.com/openai/CLIP.git")
    CLIP_AVAILABLE = False

try:
    import torch.hub
    # DINOv2 will be loaded via torch.hub
    DINO_AVAILABLE = True
except ImportError:
    print("Warning: DINOv2 not available. Make sure torch is installed properly.")
    DINO_AVAILABLE = False


# Global model cache to avoid reloading
_clip_model = None
_clip_preprocess = None
_dino_model = None

def load_clip_model(device="cuda", model_name="ViT-B/32"):
    """
    Load CLIP model
    Args:
        device: torch device
        model_name: CLIP model name
    Returns:
        model: CLIP model
        preprocess: CLIP preprocess function
    """
    global _clip_model, _clip_preprocess
    
    if _clip_model is None and CLIP_AVAILABLE:
        print(f"Loading CLIP model: {model_name}")
        _clip_model, _clip_preprocess = clip.load(model_name, device=device)
        _clip_model.eval()
    
    return _clip_model, _clip_preprocess

def load_dino_model(device="cuda", model_name="dinov2_vitb14"):
    """
    Load DINOv2 model
    Args:
        device: torch device  
        model_name: DINOv2 model name
    Returns:
        model: DINOv2 model
    """
    global _dino_model
    
    if _dino_model is None and DINO_AVAILABLE:
        print(f"Loading DINOv2 model: {model_name}")
        _dino_model = torch.hub.load('facebookresearch/dinov2', model_name)
        _dino_model = _dino_model.to(device)
        _dino_model.eval()
    
    return _dino_model

def multimodal_fusion(visual_features, text_features, device="cuda"):
    """
    Args:
        visual_features: torch.Tensor [B, visual_dim]
        text_features: torch.Tensor [B, text_dim]
        device: torch device
    Returns:
        fused_features: torch.Tensor [B, visual_dim]
    """
    # 简化的Cross-Attention实现
    # 实际可以使用更复杂的Transformer架构
    
    # 确保维度匹配
    if visual_features.shape[-1] != text_features.shape[-1]:
        # 线性投影到相同维度
        proj_dim = visual_features.shape[-1]
        text_proj = torch.nn.Linear(text_features.shape[-1], proj_dim).to(device)
        text_features = text_proj(text_features)
    
    # Cross-Attention: text attend to visual
    attention_scores = torch.matmul(text_features, visual_features.transpose(-2, -1))
    attention_weights = F.softmax(attention_scores, dim=-1)
    
    # 加权融合
    text_attended = torch.matmul(attention_weights, visual_features)
    
    # 残差连接 + 层归一化
    fused = visual_features + 0.3 * text_attended
    fused = F.layer_norm(fused, fused.shape[-1:])
    
    return fused


def compute_text_semantic_difference(text_i, text_j, device="cuda"):
    """
    使用CLIP text encoder计算文本语义差异
    Args:
        text_i, text_j: str, 文本prompt
        device: torch device
    Returns:
        text_diff: float
    """
    # 简化实现：基于词汇重叠度
    # 实际可以使用sentence-transformers或CLIP text encoder
    
    if text_i is None or text_j is None:
        return 0.0
    
    if text_i == text_j:
        return 0.0
    
    if not CLIP_AVAILABLE:
        # Fallback to Jaccard distance if CLIP not available
        words_i = set(text_i.lower().split())
        words_j = set(text_j.lower().split())
        
        if len(words_i) == 0 and len(words_j) == 0:
            return 0.0
        
        intersection = len(words_i & words_j)
        union = len(words_i | words_j)
        
        jaccard_sim = intersection / union if union > 0 else 0.0
        return 1.0 - jaccard_sim
    
    # Use CLIP text encoder
    model, _ = load_clip_model(device)
    
    with torch.no_grad():
        # Tokenize and encode texts
        tokens_i = clip.tokenize([text_i]).to(device)
        tokens_j = clip.tokenize([text_j]).to(device)
        
        # Get text features
        text_features_i = model.encode_text(tokens_i)  # [1, 512]
        text_features_j = model.encode_text(tokens_j)  # [1, 512]
        
        # Normalize features
        text_features_i = F.normalize(text_features_i, p=2, dim=1)
        text_features_j = F.normalize(text_features_j, p=2, dim=1)
        
        # Compute cosine similarity
        cosine_sim = F.cosine_similarity(text_features_i, text_features_j, dim=1).item()
        
        # Convert to difference (0 = identical, 1 = completely different)
        text_diff = 1.0 - cosine_sim
    
    return text_diff


def extract_visual_features(images, texts=None, model_type="clip", device="cuda"):
    """

    Args:
        images: List of PIL Images or torch.Tensor [B, C, H, W]
        texts: List of strings (prompts corresponding to images), optional
        model_type: "clip" or "dino"
        device: torch device
    Returns:
        features: torch.Tensor [B, feature_dim]
        patch_features: torch.Tensor [B, num_patches, patch_dim] (for local features)
    """
    if model_type == "clip" and CLIP_AVAILABLE:
        model, preprocess = load_clip_model(device)
        
        # Process images using CLIP preprocessing
        if isinstance(images, list):
            processed_images = torch.stack([preprocess(img) for img in images]).to(device)
        else:
            # Assume images are already preprocessed tensors
            processed_images = images.to(device)
        
        with torch.no_grad():
            # Extract visual features using CLIP
            visual_features = model.encode_image(processed_images)  # [B, 512]
            visual_features = F.normalize(visual_features, p=2, dim=1)
            
            # For patch features, we'll use the visual transformer's patch embeddings
            # This is a simplified approach - in practice you might want to extract from intermediate layers
            patch_features = visual_features.unsqueeze(1).repeat(1, 196, 1)  # [B, 196, 512]
            
            # Handle text feature fusion
            if texts is not None:
                # Encode texts using CLIP
                tokens = clip.tokenize(texts).to(device)
                text_features = model.encode_text(tokens)  # [B, 512]
                text_features = F.normalize(text_features, p=2, dim=1)
                
                # Multimodal fusion
                fused_features = multimodal_fusion(visual_features, text_features, device=device)
            else:
                fused_features = visual_features
            
        return fused_features, patch_features
    
    elif model_type == "dino" and DINO_AVAILABLE:
        dino_model = load_dino_model(device)
        
        # DINOv2 preprocessing
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        if isinstance(images, list):
            batch = torch.stack([transform(img) for img in images]).to(device)
        else:
            batch = images.to(device)
            
        with torch.no_grad():
            # Extract features using DINOv2
            visual_features = dino_model(batch)  # [B, 768] for dinov2_vitb14
            
            # Get patch features from the model's patch embeddings
            # This requires accessing intermediate layers - simplified here
            patch_features = visual_features.unsqueeze(1).repeat(1, 256, 1)  # [B, 256, 768]
            
            # DINO doesn't handle text natively, so we use simple fusion if texts provided
            if texts is not None:
                # Use CLIP for text encoding if available
                if CLIP_AVAILABLE:
                    clip_model, _ = load_clip_model(device)
                    tokens = clip.tokenize(texts).to(device)
                    text_features = clip_model.encode_text(tokens)
                    text_features = F.normalize(text_features, p=2, dim=1)
                    
                    # Project text features to match DINO dimension
                    if text_features.shape[-1] != visual_features.shape[-1]:
                        proj = torch.nn.Linear(text_features.shape[-1], visual_features.shape[-1]).to(device)
                        text_features = proj(text_features)
                    
                    # Simple weighted fusion
                    fused_features = 0.7 * visual_features + 0.3 * text_features
                else:
                    # Fallback: just use visual features
                    fused_features = visual_features
            else:
                fused_features = visual_features
            
        return fused_features, patch_features
    
    else:
        # Fallback to mock implementation if models not available
        print(f"Warning: {model_type.upper()} not available, using mock features")
        
        if isinstance(images, list):
            batch_size = len(images)
        else:
            batch_size = images.shape[0]
        
        if model_type == "clip":
            feature_dim = 512
            patch_dim = 512
            num_patches = 196
        else:  # dino
            feature_dim = 768
            patch_dim = 768
            num_patches = 256
        
        visual_features = torch.randn(batch_size, feature_dim, device=device)
        patch_features = torch.randn(batch_size, num_patches, patch_dim, device=device)
        
        if texts is not None:
            text_features = torch.randn(batch_size, feature_dim, device=device)
            fused_features = multimodal_fusion(visual_features, text_features, device=device)
        else:
            fused_features = visual_features
            
        return fused_features, patch_features


def compute_semantic_difference(vi, vj):
    """
    计算语义差异 Δ_sem = 1 - cosine(vi, vj)
    Args:
        vi, vj: torch.Tensor [feature_dim]
    Returns:
        semantic_diff: float
    """
    # 确保特征向量是归一化的
    vi_norm = F.normalize(vi.unsqueeze(0), p=2, dim=1)
    vj_norm = F.normalize(vj.unsqueeze(0), p=2, dim=1)
    
    # 计算余弦相似度
    cosine_sim = F.cosine_similarity(vi_norm, vj_norm, dim=1).item()
    
    # 转换为差异度
    semantic_diff = 1.0 - cosine_sim
    
    return semantic_diff


def compute_emd_distance(patches_i, patches_j):
    """
    计算两组patch特征之间的Earth Mover Distance (EMD)
    Args:
        patches_i, patches_j: torch.Tensor [num_patches, patch_dim]
    Returns:
        emd_distance: float
    """
    # 转换为numpy进行EMD计算
    patches_i_np = patches_i.cpu().numpy()
    patches_j_np = patches_j.cpu().numpy()
    
    # 计算patch间的距离矩阵
    cost_matrix = cdist(patches_i_np, patches_j_np, metric='euclidean')
    
    # 简化的EMD计算 (实际可以使用更精确的EMD算法如POT库)
    # 这里使用最小匹配距离的平均值作为近似
    min_distances = np.min(cost_matrix, axis=1)
    emd_distance = np.mean(min_distances)
    
    return float(emd_distance)


def compute_local_difference(patches_i, patches_j, method="emd"):
    """
    计算局部细节差异 Δ_loc
    Args:
        patches_i, patches_j: torch.Tensor [num_patches, patch_dim]
        method: "emd" or "chamfer"
    Returns:
        local_diff: float
    """
    if method == "emd":
        return compute_emd_distance(patches_i, patches_j)
    
    elif method == "chamfer":
        # Chamfer距离计算
        patches_i_expanded = patches_i.unsqueeze(1)  # [N, 1, D]
        patches_j_expanded = patches_j.unsqueeze(0)  # [1, M, D]
        
        # 计算所有patch对之间的距离
        distances = torch.norm(patches_i_expanded - patches_j_expanded, dim=2)  # [N, M]
        
        # Chamfer距离：每个点到最近点的距离的平均
        chamfer_i_to_j = torch.mean(torch.min(distances, dim=1)[0])
        chamfer_j_to_i = torch.mean(torch.min(distances, dim=0)[0])
        
        chamfer_distance = (chamfer_i_to_j + chamfer_j_to_i) / 2.0
        
        return chamfer_distance.item()
    
    else:
        raise ValueError(f"Unknown method: {method}")


def compute_window_difference(frame_i, frame_j, text_i=None, text_j=None, alpha=0.6, beta=0.2, device="cuda"):
    """
    计算窗口内两帧的综合差异度 Δ̂ᵢⱼ (包含文本信息)
    Args:
        frame_i, frame_j: PIL Images or torch.Tensor
        text_i, text_j: str, 对应的文本prompt
        alpha: 视觉语义差异权重 (0-1)
        beta: 文本语义差异权重 (0-1)
        device: torch device
    Returns:
        total_diff: float, 综合差异度
        semantic_diff: float, 视觉语义差异
        local_diff: float, 局部差异
        text_diff: float, 文本差异
    """
    # 提取视觉特征 (融合文本信息)
    texts = [text_i, text_j] if text_i is not None and text_j is not None else None
    global_features, patch_features = extract_visual_features(
        [frame_i, frame_j], texts=texts, model_type="clip", device=device
    )
    
    vi_global, vj_global = global_features[0], global_features[1]
    vi_patches, vj_patches = patch_features[0], patch_features[1]
    
    # 计算视觉语义差异
    visual_semantic_diff = compute_semantic_difference(vi_global, vj_global)
    
    # 计算局部差异
    local_diff = compute_local_difference(vi_patches, vj_patches, method="chamfer")
    
    # 计算文本语义差异
    text_diff = compute_text_semantic_difference(text_i, text_j, device=device)
    
    # 归一化局部差异 (可根据实际数据范围调整)
    local_diff_normalized = min(local_diff / 10.0, 1.0)  # 简单归一化到[0,1]
    
    # 综合差异度：视觉 + 局部 + 文本
    remaining_weight = 1.0 - alpha - beta
    total_diff = (alpha * visual_semantic_diff + 
                  remaining_weight * local_diff_normalized + 
                  beta * text_diff)
    
    return total_diff, visual_semantic_diff, local_diff_normalized, text_diff


def compute_adaptive_threshold(differences, k=0.8):
    """
    计算自适应阈值 τ = μ + k·σ
    Args:
        differences: list of float, 所有差异度值
        k: 阈值系数
    Returns:
        threshold: float
    """
    if len(differences) == 0:
        return 0.5  # 默认阈值
    
    mean_diff = np.mean(differences)
    std_diff = np.std(differences)
    
    threshold = mean_diff + k * std_diff
    
    return threshold


def sliding_window_detection(frames, texts=None, alpha=0.6, beta=0.2, k=0.8, device="cuda"):
    """
    完整的滑动窗口检测流程 (支持文本输入)
    Args:
        frames: List of PIL Images
        texts: List of strings (prompts), 可选
        alpha: 视觉语义差异权重
        beta: 文本语义差异权重
        k: 阈值系数
        device: torch device
    Returns:
        need_interpolation_pairs: List of tuples (i, j, diff_score)
        all_differences: List of all computed differences
        threshold: float, 自适应阈值
    """
    n = len(frames)
    all_differences = []
    candidate_pairs = []
    
    # 确保texts长度匹配frames
    if texts is not None and len(texts) != n:
        texts = None  # 如果长度不匹配，忽略文本
    
    # 第一轮：stride=1 (相邻帧)
    for i in range(n - 1):
        text_i = texts[i] if texts is not None else None
        text_j = texts[i + 1] if texts is not None else None
        
        diff, sem_diff, loc_diff, txt_diff = compute_window_difference(
            frames[i], frames[i + 1], text_i, text_j, alpha=alpha, beta=beta, device=device
        )
        all_differences.append(diff)
        candidate_pairs.append((i, i + 1, diff, "adjacent"))
    
    # 第二轮：stride=2 (跳一帧)
    if n > 3:
        for i in range(n - 2):
            text_i = texts[i] if texts is not None else None
            text_j = texts[i + 2] if texts is not None else None
            
            diff, sem_diff, loc_diff, txt_diff = compute_window_difference(
                frames[i], frames[i + 2], text_i, text_j, alpha=alpha, beta=beta, device=device
            )
            all_differences.append(diff)
            candidate_pairs.append((i, i + 2, diff, "skip_one"))
    
    # 第三轮：stride=⌊n/2⌋ (首尾跨度)
    if n > 4:
        stride = n // 2
        for i in range(n - stride):
            text_i = texts[i] if texts is not None else None
            text_j = texts[i + stride] if texts is not None else None
            
            diff, sem_diff, loc_diff, txt_diff = compute_window_difference(
                frames[i], frames[i + stride], text_i, text_j, alpha=alpha, beta=beta, device=device
            )
            all_differences.append(diff)
            candidate_pairs.append((i, i + stride, diff, "long_range"))
    
    # 计算自适应阈值
    threshold = compute_adaptive_threshold(all_differences, k=k)
    
    # 筛选需要插帧的对
    need_interpolation_pairs = []
    for i, j, diff, scan_type in candidate_pairs:
        if diff >= threshold:
            need_interpolation_pairs.append((i, j, diff))
    
    # 按差异度降序排列
    need_interpolation_pairs.sort(key=lambda x: x[2], reverse=True)
    
    return need_interpolation_pairs, all_differences, threshold


# 使用示例
if __name__ == "__main__":
    # 模拟测试
    from PIL import Image
    import numpy as np
    
    # 创建模拟图像和文本
    frames = []
    texts = []
    for i in range(4):
        # 创建渐变变化的图像
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img_array = img_array + i * 30  # 增大变化幅度
        img_array = np.clip(img_array, 0, 255)
        frames.append(Image.fromarray(img_array))
        
        # 模拟文本prompt
        texts.append(f"step {i+1}: cooking process with ingredients")
    
    # 测试设备 (如果有GPU则使用GPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 测试CLIP模型
    if CLIP_AVAILABLE:
        print("\n=== 使用CLIP模型测试 ===")
        pairs_clip, diffs_clip, threshold_clip = sliding_window_detection(
            frames, texts=texts, alpha=0.5, beta=0.3, device=device
        )
        print(f"CLIP检测结果:")
        print(f"需要插帧的对: {pairs_clip}")
        print(f"差异度分数: {[f'{d:.4f}' for d in diffs_clip]}")
        print(f"自适应阈值: {threshold_clip:.4f}")
    
    # 测试DINO模型
    if DINO_AVAILABLE:
        print("\n=== 使用DINO模型测试 ===")
        pairs_dino, diffs_dino, threshold_dino = sliding_window_detection(
            frames, texts=texts, alpha=0.6, beta=0.2, device=device
        )
        print(f"DINO检测结果:")
        print(f"需要插帧的对: {pairs_dino}")
        print(f"差异度分数: {[f'{d:.4f}' for d in diffs_dino]}")
        print(f"自适应阈值: {threshold_dino:.4f}")
    
    # 测试不含文本的情况
    print("\n=== 不含文本的检测测试 ===")
    model_type = "clip" if CLIP_AVAILABLE else "dino" if DINO_AVAILABLE else "clip"
    pairs_no_text, diffs_no_text, threshold_no_text = sliding_window_detection(
        frames, texts=None, alpha=0.8, beta=0.0, device=device
    )
    
    print(f"不含文本检测结果:")
    print(f"需要插帧的对: {pairs_no_text}")
    print(f"差异度分数: {[f'{d:.4f}' for d in diffs_no_text]}")
    print(f"自适应阈值: {threshold_no_text:.4f}")
    
    # 推荐最佳插帧对
    if pairs_no_text:
        best_pair = pairs_no_text[0]
        print(f"\n推荐的最佳插帧对: 帧{best_pair[0]} -> 帧{best_pair[1]} (差异度: {best_pair[2]:.4f})")
    else:
        print(f"\n未检测到需要插帧的区间")
        
    print("\n=== 测试完成 ===")
