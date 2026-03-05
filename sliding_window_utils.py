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
    
    if visual_features.shape[-1] != text_features.shape[-1]:
        
        proj_dim = visual_features.shape[-1]
        text_proj = torch.nn.Linear(text_features.shape[-1], proj_dim).to(device)
        text_features = text_proj(text_features)
    
    # Cross-Attention: text attend to visual
    attention_scores = torch.matmul(text_features, visual_features.transpose(-2, -1))
    attention_weights = F.softmax(attention_scores, dim=-1)
    
 
    text_attended = torch.matmul(attention_weights, visual_features)
    
    fused = visual_features + 0.3 * text_attended
    fused = F.layer_norm(fused, fused.shape[-1:])
    
    return fused


def compute_text_semantic_difference(text_i, text_j, device="cuda"):
    """
    Args:
        text_i, text_j: str, 文本prompt
        device: torch device
    Returns:
        text_diff: float
    """
    
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
     Δ_sem = 1 - cosine(vi, vj)
    Args:
        vi, vj: torch.Tensor [feature_dim]
    Returns:
        semantic_diff: float
    """
   
    vi_norm = F.normalize(vi.unsqueeze(0), p=2, dim=1)
    vj_norm = F.normalize(vj.unsqueeze(0), p=2, dim=1)
    

    cosine_sim = F.cosine_similarity(vi_norm, vj_norm, dim=1).item()
    semantic_diff = 1.0 - cosine_sim
    
    return semantic_diff


def compute_emd_distance(patches_i, patches_j):
    """
    Args:
        patches_i, patches_j: torch.Tensor [num_patches, patch_dim]
    Returns:
        emd_distance: float
    """
    patches_i_np = patches_i.cpu().numpy()
    patches_j_np = patches_j.cpu().numpy()
    
    cost_matrix = cdist(patches_i_np, patches_j_np, metric='euclidean')
    
    min_distances = np.min(cost_matrix, axis=1)
    emd_distance = np.mean(min_distances)
    
    return float(emd_distance)


def compute_local_difference(patches_i, patches_j, method="emd"):
    """
    Δ_loc
    Args:
        patches_i, patches_j: torch.Tensor [num_patches, patch_dim]
        method: "emd" or "chamfer"
    Returns:
        local_diff: float
    """
    if method == "emd":
        return compute_emd_distance(patches_i, patches_j)
    
    elif method == "chamfer":
        patches_i_expanded = patches_i.unsqueeze(1)  # [N, 1, D]
        patches_j_expanded = patches_j.unsqueeze(0)  # [1, M, D]
        
        distances = torch.norm(patches_i_expanded - patches_j_expanded, dim=2)  # [N, M]
        
        chamfer_i_to_j = torch.mean(torch.min(distances, dim=1)[0])
        chamfer_j_to_i = torch.mean(torch.min(distances, dim=0)[0])
        
        chamfer_distance = (chamfer_i_to_j + chamfer_j_to_i) / 2.0
        
        return chamfer_distance.item()
    
    else:
        raise ValueError(f"Unknown method: {method}")


def compute_window_difference(frame_i, frame_j, text_i=None, text_j=None, alpha=0.6, beta=0.2, device="cuda"):
    """
    Args:
        frame_i, frame_j: PIL Images or torch.Tensor
        text_i, text_j: str, 对应的文本prompt
        alpha: 0-1
        beta: 0-1
        device: torch device
    Returns:
        total_diff: float
        semantic_diff: float
        local_diff: float
        text_diff: float
    """

    texts = [text_i, text_j] if text_i is not None and text_j is not None else None
    global_features, patch_features = extract_visual_features(
        [frame_i, frame_j], texts=texts, model_type="clip", device=device
    )
    
    vi_global, vj_global = global_features[0], global_features[1]
    vi_patches, vj_patches = patch_features[0], patch_features[1]
    
    visual_semantic_diff = compute_semantic_difference(vi_global, vj_global)
    
    local_diff = compute_local_difference(vi_patches, vj_patches, method="chamfer")
    
    text_diff = compute_text_semantic_difference(text_i, text_j, device=device)
    
    local_diff_normalized = min(local_diff / 10.0, 1.0)  
    
    remaining_weight = 1.0 - alpha - beta
    total_diff = (alpha * visual_semantic_diff + 
                  remaining_weight * local_diff_normalized + 
                  beta * text_diff)
    
    return total_diff, visual_semantic_diff, local_diff_normalized, text_diff


def compute_adaptive_threshold(differences, k=0.8):
    """
    Args:
        differences: list of float
        k: 
    Returns:
        threshold: float
    """
    if len(differences) == 0:
        return 0.5  
    
    mean_diff = np.mean(differences)
    std_diff = np.std(differences)
    
    threshold = mean_diff + k * std_diff
    
    return threshold


def sliding_window_detection(frames, texts=None, alpha=0.6, beta=0.2, k=0.8, device="cuda"):
    """
    Args:
        frames: List of PIL Images
        texts: List of strings (prompts),
        alpha: 
        beta: 
        k: 
        device: torch device
    Returns:
        need_interpolation_pairs: List of tuples (i, j, diff_score)
        all_differences: List of all computed differences
        threshold: float, 
    """
    n = len(frames)
    all_differences = []
    candidate_pairs = []
    

    if texts is not None and len(texts) != n:
        texts = None  
    
    for i in range(n - 1):
        text_i = texts[i] if texts is not None else None
        text_j = texts[i + 1] if texts is not None else None
        
        diff, sem_diff, loc_diff, txt_diff = compute_window_difference(
            frames[i], frames[i + 1], text_i, text_j, alpha=alpha, beta=beta, device=device
        )
        all_differences.append(diff)
        candidate_pairs.append((i, i + 1, diff, "adjacent"))
    
    if n > 3:
        for i in range(n - 2):
            text_i = texts[i] if texts is not None else None
            text_j = texts[i + 2] if texts is not None else None
            
            diff, sem_diff, loc_diff, txt_diff = compute_window_difference(
                frames[i], frames[i + 2], text_i, text_j, alpha=alpha, beta=beta, device=device
            )
            all_differences.append(diff)
            candidate_pairs.append((i, i + 2, diff, "skip_one"))
    
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
    

    threshold = compute_adaptive_threshold(all_differences, k=k)
    
    need_interpolation_pairs = []
    for i, j, diff, scan_type in candidate_pairs:
        if diff >= threshold:
            need_interpolation_pairs.append((i, j, diff))
    
    need_interpolation_pairs.sort(key=lambda x: x[2], reverse=True)
    
    return need_interpolation_pairs, all_differences, threshold



if __name__ == "__main__":

    from PIL import Image
    import numpy as np
    
    frames = []
    texts = []
    for i in range(4):
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img_array = img_array + i * 30  
        img_array = np.clip(img_array, 0, 255)
        frames.append(Image.fromarray(img_array))
        
        texts.append(f"step {i+1}: cooking process with ingredients")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if CLIP_AVAILABLE:
        pairs_clip, diffs_clip, threshold_clip = sliding_window_detection(
            frames, texts=texts, alpha=0.5, beta=0.3, device=device
        )

    if DINO_AVAILABLE:
        pairs_dino, diffs_dino, threshold_dino = sliding_window_detection(
            frames, texts=texts, alpha=0.6, beta=0.2, device=device
        )
    
    model_type = "clip" if CLIP_AVAILABLE else "dino" if DINO_AVAILABLE else "clip"
    pairs_no_text, diffs_no_text, threshold_no_text = sliding_window_detection(
        frames, texts=None, alpha=0.8, beta=0.0, device=device
    )
    
   
    if pairs_no_text:
        best_pair = pairs_no_text[0]
    else:
        print(f"\n Fail")

