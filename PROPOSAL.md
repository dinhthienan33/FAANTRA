# Research Proposal: Optimizing FAANTRA for SoccerNet Ball Action Anticipation Challenge

> **Target:** CVSports Workshop @ CVPR 2026 — Codabench Deadline: April 24, 2026
> **Baseline:** FAANTRA (Avg mAP = 24.08, Ta=5s) | Upper Bound: T-DEED (Avg mAP = 63.85)
> **Goal:** Thu hẹp khoảng cách 39.77 điểm mAP giữa baseline và upper bound

---

## 1. Bối cảnh và Phân tích Vấn đề

### 1.1 Mô tả Task

Ball Action Anticipation (BAA) yêu cầu: cho **30 giây video quan sát** (observed window), dự đoán **class** và **thời điểm chính xác** của các ball-related actions trong **5–10 giây tương lai** (anticipation window). Đây là bài toán cực kỳ khó vì:

- **Tính bất định cao**: Bóng đá là môn thể thao đối kháng — cùng một tình huống có thể dẫn đến nhiều hành động khác nhau
- **Độ chính xác thời gian**: mAP@1 (tolerance 1 giây) chỉ đạt 9.74, cho thấy model rất yếu ở temporal precision
- **Class imbalance nghiêm trọng**: Pass (55.50 mAP) vs Successful Tackle (8.02 mAP) — chênh lệch gần 7x

### 1.2 Kiến trúc Baseline Hiện tại

```
Video Frames [B, S, C, H, W]
    → [RegNetY-004 + GSF] Feature extraction (timm, pretrained ImageNet)
    → [SGP-Mixer] Temporal encoding (ed_sgp_mixer, 2 layers, ks=9, r=4)
    → [Linear + ReLU] Projection: feat_dim → 512
    → [Learnable Positional Embedding]
    → [Transformer Encoder] 2 layers, 8 heads, windowed attention (k=15)
    → [Transformer Decoder] 2 layers, 8 queries, windowed attention (k=19)
    → [Output Heads]
        ├── Class Head: Linear(512, n_class) — action classification
        ├── Offset Head: Linear(512, 1) — temporal localization
        ├── Actionness Head: Linear(512, 1) — binary action confidence
        └── Segmentation Head: Linear(512, n_class) — auxiliary task trên encoder output
```

**10 action classes:** PASS, DRIVE, HEADER, HIGH PASS, THROW IN, CROSS, SHOT, OUT, BALL PLAYER BLOCK, PLAYER SUCCESSFUL TACKLE

### 1.3 Phân tích Bottleneck

| Bottleneck | Bằng chứng | Mức độ ảnh hưởng |
|---|---|---|
| **Feature extractor yếu** | RegNetY-400MF chỉ có ~4M params; tăng resolution 224→448 cho +5 mAP | **Rất cao** |
| **Temporal precision kém** | mAP@1 = 9.74 (gap 49.72 so với upper bound) | **Rất cao** |
| **Class imbalance** | Tackle: 8.02, Block: 9.16, Shot: 10.08 vs Pass: 55.50 | **Cao** |
| **Data hạn chế** | Chỉ 11.4 giờ video labeled; thêm SN-AS cho +3.78 mAP | **Cao** |
| **Single modality** | Chỉ dùng visual — bỏ qua audio/commentary cues | **Trung bình** |
| **Deterministic prediction** | Model dự đoán 1 output duy nhất cho tương lai bất định | **Trung bình** |

### 1.4 Cấu hình Training Hiện tại

```json
{
    "batch_size": 8, "clip_len": 64, "obs_perc": 0.5, "pred_perc": 0.5,
    "feature_arch": "rny004_gsf", "temporal_arch": "ed_sgp_mixer",
    "n_query": 8, "hidden_dim": 512, "n_encoder_layer": 2, "n_decoder_layer": 2,
    "learning_rate": 1e-4, "num_epochs": 20, "warm_up_epochs": 3,
    "class_weights": [0.03, 0.15, 0.2, 1, 1, 2, 3, 4.5, 1.4, 3.4, 10],
    "offset_loss_weight": 10, "eos_weight": 0.1
}
```

---

## 2. Chiến lược Nghiên cứu Đề xuất

### Tổng quan Roadmap

```
Phase 1 (Tuần 1–3):  Quick Wins — Hyperparameter tuning, data scaling, loss engineering
Phase 2 (Tuần 3–6):  Feature Upgrade — Backbone mạnh hơn, higher resolution
Phase 3 (Tuần 6–10): Architecture Innovation — Multi-scale temporal, query design
Phase 4 (Tuần 10–14): Advanced Methods — Multi-modal, generative prediction
Phase 5 (Tuần 14–16): Ensemble & Submission — Model ensemble, final optimization
```

---

## 3. Phase 1: Quick Wins (Tuần 1–3)

> **Mục tiêu:** Tăng 3–8 điểm mAP mà không thay đổi kiến trúc model

### 3.1 Tối ưu Class Imbalance

**Vấn đề:** Class weights hiện tại `[0.03, 0.15, 0.2, 1, 1, 2, 3, 4.5, 1.4, 3.4, 10]` được set thủ công. Các rare classes (Shot, Block, Tackle) có performance rất thấp.

**Giải pháp A — Adaptive Class Weights:**

```python
# Thay vì fixed weights, dùng effective number of samples (Cui et al., 2019)
# Class-Balanced Loss dựa trên effective number: E_n = (1 - β^n) / (1 - β)
import numpy as np

def compute_effective_weights(class_counts, beta=0.9999):
    """Effective Number of Samples based class balancing."""
    effective_num = 1.0 - np.power(beta, class_counts)
    weights = (1.0 - beta) / effective_num
    weights = weights / np.sum(weights) * len(weights)  # normalize
    return weights

# Thêm scheduled rebalancing: tăng dần weight của rare classes qua epochs
def get_epoch_weights(base_weights, epoch, max_epoch, gamma=2.0):
    """Progressive rebalancing: bắt đầu balanced, dần dần tăng rare class weight."""
    ratio = min(epoch / max_epoch, 1.0)
    uniform = np.ones_like(base_weights)
    return uniform * (1 - ratio**gamma) + base_weights * (ratio**gamma)
```

**Giải pháp B — Focal Loss thay thế CE:**

```python
class FocalLoss(nn.Module):
    """Focal Loss giảm loss contribution từ easy examples (frequent classes)."""
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha  # class weights
        self.gamma = gamma  # focusing parameter

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()
```

**Giải pháp C — Oversampling rare action clips:**

```python
# Trong dataset sampler: oversample clips chứa rare actions
class BalancedActionSampler(torch.utils.data.Sampler):
    """Oversample clips chứa Shot, Tackle, Block."""
    def __init__(self, dataset, rare_classes=[6, 8, 9], oversample_factor=3):
        self.indices = list(range(len(dataset)))
        self.rare_indices = [i for i in self.indices
                            if dataset.has_rare_action(i, rare_classes)]
        # Lặp lại rare clips
        self.expanded = self.indices + self.rare_indices * oversample_factor
```

**Expected Impact:** +2–4 mAP trên rare classes, +1–2 Avg mAP

### 3.2 Data Scaling với SoccerNet Action Spotting

**Bằng chứng:** Joint training với SN-AS đã cho +3.78 mAP (20.30 → 24.08). Hiện tại config Joint dùng `temporal_arch: "none"` — thiếu SGP-Mixer.

**Hành động:**

1. **Fix Joint config**: Bật `temporal_arch: "ed_sgp_mixer"` trong `Base-Config-Joint.json`
2. **Mapping classes**: Map 17 SN-AS classes ↔ 10 BAA classes (Pass↔Pass, Shot↔Shot-on-target/Shot-off-target, etc.)
3. **Curriculum learning**: Train trên SN-AS trước (pre-train), rồi fine-tune trên BAA
4. **Tăng SN-AS data**: Sử dụng toàn bộ 500 games thay vì subset

```python
# Curriculum strategy trong main.py
if args.curriculum:
    # Phase 1: Pre-train trên SN-AS (500 games, 17 classes)
    pretrain_epochs = args.pretrain_epochs  # e.g., 10
    train(model, train_loader_AS, val_loader_AS, pretrain_epochs, ...)

    # Phase 2: Fine-tune trên BAA (target task)
    # Giảm LR, freeze backbone partially
    for param in model.features.parameters():
        param.requires_grad = False  # freeze backbone
    finetune_epochs = args.finetune_epochs  # e.g., 20
    train(model, train_loader_BAA, val_loader_BAA, finetune_epochs, ...)
```

**Expected Impact:** +2–4 Avg mAP (cộng dồn từ baseline 24.08)

### 3.3 Training Recipe Optimization

| Hyperparameter | Hiện tại | Đề xuất | Lý do |
|---|---|---|---|
| `num_epochs` | 20 | 50–80 | Model chưa converge đủ |
| `learning_rate` | 1e-4 | 3e-4 với cosine decay | Faster convergence |
| `batch_size` | 8 | 16–32 (gradient accumulation) | Gradient ổn định hơn |
| `warm_up_epochs` | 3 | 5 | Stabilize training đầu |
| `weight_decay` | 5e-3 | 1e-2 | Regularization mạnh hơn |
| `label_smoothing` | 0 | 0.1 | Giảm overconfidence |
| `radi_smoothing` | 4 | Per-class adaptive | Action duration varies |
| `eos_weight` | 0.1 | 0.05 | Giảm background dominance |
| `offset_loss_weight` | 10 | 20 | Tăng temporal precision |

**Data Augmentation mở rộng:**

```python
# Thêm vào augmentation pipeline hiện có (flip, blur, color jitter)
augmentation_extended = {
    'temporal_crop': RandomTemporalCrop(min_ratio=0.8),  # Random crop temporal
    'mixup': TemporalMixup(alpha=0.2),                   # Mix 2 clips
    'cutmix_temporal': TemporalCutMix(ratio=0.3),        # Cut-paste segments
    'speed_perturbation': SpeedPerturbation(range=[0.8, 1.2]),  # Variable speed
    'frame_dropout': RandomFrameDropout(p=0.1),           # Random frame masking
}
```

**Expected Impact:** +1–3 Avg mAP

---

## 4. Phase 2: Feature Extraction Upgrade (Tuần 3–6)

> **Mục tiêu:** Thay thế bottleneck chính — backbone RegNetY quá yếu

### 4.1 Strategy A — VideoMAE Pre-training (Ưu tiên cao nhất)

**Tại sao VideoMAE:**
- Data-efficient: hoạt động tốt trên small datasets (3k–4k videos) — phù hợp với 11.4h của BAA
- Self-supervised: không cần labels, có thể pre-train trên hàng nghìn giờ football video unlabeled
- ViT backbone mạnh hơn RegNetY rất nhiều

**Implementation Plan:**

```python
# 1. Collect unlabeled football video data
#    - SoccerNet raw videos (550+ games, ~1000 hours)
#    - YouTube highlights (public domain)
#    Target: 500–1000 hours of football broadcast video

# 2. Pre-train VideoMAE trên football data
from transformers import VideoMAEForPreTraining, VideoMAEConfig

config = VideoMAEConfig(
    image_size=224,
    patch_size=16,
    num_channels=3,
    num_frames=16,
    tubelet_size=2,
    hidden_size=768,        # ViT-Base
    num_hidden_layers=12,
    num_attention_heads=12,
    decoder_hidden_size=384,
    decoder_num_hidden_layers=4,
    decoder_num_attention_heads=6,
    mask_ratio=0.9,         # VideoMAE dùng aggressive masking 90%
)

model_pretrain = VideoMAEForPreTraining(config)

# 3. Fine-tune feature extractor cho BAA
# Thay thế RegNetY trong FUTR:
class FUTR_VideoMAE(FUTR):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Thay self.features = timm RegNetY
        self.features = VideoMAEModel.from_pretrained("path/to/football-videomae")
        self.feat_dim = 768  # ViT-Base output
        self.input_embed = nn.Linear(768, self.hidden_dim)
```

**Modification cần thiết trong `model/futr.py`:**

```python
# Hiện tại (line ~50-70 trong futr.py):
# self.features = timm.create_model(feature_arch, pretrained=True)
# self.feat_dim = self.features.num_features

# Thay bằng:
if args.feature_arch.startswith('videomae'):
    from transformers import VideoMAEModel
    self.features = VideoMAEModel.from_pretrained(args.videomae_path)
    self.feat_dim = self.features.config.hidden_size  # 768 for base
    # VideoMAE xử lý cả batch frames → output [B, num_patches, hidden_size]
    # Cần pooling temporal: mean pool over patch tokens per frame
elif args.feature_arch.startswith('rny'):
    self.features = timm.create_model(...)  # giữ nguyên
```

**Expected Impact:** +5–10 Avg mAP (backbone upgrade là yếu tố ảnh hưởng lớn nhất)

### 4.2 Strategy B — InternVideo2 Feature Extraction

**Approach:** Dùng InternVideo2 (6B params) ở chế độ frozen feature extractor — extract features offline, rồi train FAANTRA transformer trên features đã extract.

```python
# Offline feature extraction
# InternVideo2 quá lớn để end-to-end training → extract features trước

import torch
from internvideo2 import InternVideo2

extractor = InternVideo2.from_pretrained("InternVideo2-6B", device="cuda")
extractor.eval()

def extract_features(video_frames, window_size=16, stride=4):
    """Extract features cho mỗi temporal window."""
    features = []
    for i in range(0, len(video_frames) - window_size, stride):
        window = video_frames[i:i+window_size]  # [16, C, H, W]
        with torch.no_grad():
            feat = extractor.encode_video(window.unsqueeze(0))  # [1, D]
        features.append(feat)
    return torch.cat(features, dim=0)  # [T', D]

# Lưu features ra disk, load trong dataset
# Modification trong dataset/datasets.py:
# Thay vì load raw frames → load pre-extracted features
```

**Trade-off:** InternVideo2 cho features chất lượng cao nhất nhưng yêu cầu GPU lớn (A100 80GB) cho extraction. VideoMAE nhẹ hơn và cho phép end-to-end fine-tuning.

**Expected Impact:** +8–12 Avg mAP (nếu features đủ tốt)

### 4.3 Strategy C — Higher Resolution với Efficient Attention

**Bằng chứng:** 224→448 resolution đã cho +5 mAP. Ball-related actions chiếm diện tích nhỏ trong broadcast frame.

```python
# Approach 1: Tăng resolution lên 448×796
# Vấn đề: memory tăng 4x → cần gradient checkpointing
config_highres = {
    "resolution": [448, 796],
    "gradient_checkpointing": True,
    "batch_size": 2,  # giảm batch size
    "gradient_accumulation_steps": 8,  # compensate
}

# Approach 2: Adaptive Region of Interest (RoI)
# Detect ball position → crop region xung quanh ball → higher effective resolution
class BallCentricCrop:
    """Crop vùng 224×224 xung quanh vị trí bóng, giữ context."""
    def __init__(self, crop_size=224, context_ratio=0.3):
        self.crop_size = crop_size
        self.context_ratio = context_ratio

    def __call__(self, frame, ball_pos):
        # ball_pos từ ball detection model hoặc SoccerNet tracking annotations
        cx, cy = ball_pos
        # Crop centered on ball with some context
        x1 = max(0, cx - self.crop_size // 2)
        y1 = max(0, cy - self.crop_size // 2)
        crop = frame[y1:y1+self.crop_size, x1:x1+self.crop_size]
        return crop
```

**Expected Impact:** +3–5 Avg mAP

---

## 5. Phase 3: Architecture Innovation (Tuần 6–10)

> **Mục tiêu:** Cải thiện temporal reasoning và prediction quality

### 5.1 Multi-Scale Temporal Encoding

**Vấn đề:** SGP-Mixer hiện tại dùng single scale (ks=9). Các actions có temporal dynamics rất khác nhau:
- Shot: xảy ra nhanh (~0.5s), cần fine-grained temporal features
- Drive/Pass: build-up chậm (2–5s), cần coarse temporal context

```python
class MultiScaleSGPMixer(nn.Module):
    """Multi-scale temporal encoding: multiple kernel sizes song song."""
    def __init__(self, feat_dim, scales=[3, 7, 15, 31]):
        super().__init__()
        self.branches = nn.ModuleList([
            EDSGPMIXERLayers(feat_dim, n_layers=2, ks=ks, r=4)
            for ks in scales
        ])
        self.fusion = nn.Linear(feat_dim * len(scales), feat_dim)

    def forward(self, x):
        # x: [B, T, C]
        multi_scale = [branch(x) for branch in self.branches]
        concat = torch.cat(multi_scale, dim=-1)  # [B, T, C*4]
        return self.fusion(concat)  # [B, T, C]
```

**Thêm Hierarchical Attention trong Encoder:**

```python
# Layer 1: Local attention (k=7) — fine-grained temporal patterns
# Layer 2: Medium attention (k=15) — medium-range dependencies
# Layer 3: Global attention — full context
# → Thay vì fixed window cho tất cả layers

def build_hierarchical_masks(seq_len, n_layers, device):
    """Progressive widening attention masks."""
    masks = []
    windows = [7, 15, seq_len]  # local → medium → global
    for i in range(n_layers):
        w = windows[min(i, len(windows)-1)]
        if w >= seq_len:
            masks.append(None)  # global attention
        else:
            mask = create_windowed_mask(seq_len, w, device)
            masks.append(mask)
    return masks
```

**Expected Impact:** +2–4 Avg mAP

### 5.2 Content-Aware Query Initialization

**Vấn đề:** Decoder queries hiện tại là purely learned embeddings — không có thông tin từ observed context.

```python
class ContentAwareQueryInit(nn.Module):
    """Initialize decoder queries dựa trên encoder output."""
    def __init__(self, hidden_dim, n_query):
        super().__init__()
        self.n_query = n_query
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.learned_queries = nn.Embedding(n_query, hidden_dim)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )

    def forward(self, encoder_output):
        # encoder_output: [B, T, C]
        # Global summary of observed context
        context = encoder_output.mean(dim=1)  # [B, C]
        context_queries = self.query_proj(context).unsqueeze(1)  # [B, 1, C]
        context_queries = context_queries.expand(-1, self.n_query, -1)

        learned = self.learned_queries.weight.unsqueeze(0).expand(
            encoder_output.size(0), -1, -1)  # [B, n_query, C]

        # Gated fusion: blend learned queries với context
        gate = self.gate(torch.cat([learned, context_queries], dim=-1))
        queries = gate * learned + (1 - gate) * context_queries
        return queries
```

**Expected Impact:** +1–2 Avg mAP

### 5.3 Gaussian Temporal Prediction

**Vấn đề:** Offset head dự đoán single point estimate → không capture uncertainty.

```python
class GaussianTemporalHead(nn.Module):
    """Predict temporal distribution (mean + variance) thay vì point estimate."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.fc_mean = nn.Linear(hidden_dim, 1)
        self.fc_logvar = nn.Linear(hidden_dim, 1)  # log variance

    def forward(self, x):
        mean = torch.sigmoid(self.fc_mean(x))  # [0, 1] normalized offset
        logvar = self.fc_logvar(x)
        std = torch.exp(0.5 * logvar)
        return mean, std

    def loss(self, pred_mean, pred_std, target):
        """NLL loss cho Gaussian prediction."""
        # Negative log-likelihood of target under predicted Gaussian
        nll = 0.5 * (torch.log(pred_std**2 + 1e-6) +
                      (target - pred_mean)**2 / (pred_std**2 + 1e-6))
        return nll.mean()

# Inference: dùng mean làm prediction, std cho confidence weighting
# Có thể reject predictions với std quá cao (uncertain)
```

**Expected Impact:** +1–3 Avg mAP (đặc biệt cải thiện mAP@1 và mAP@2)

### 5.4 Tăng Model Capacity

| Parameter | Hiện tại | Đề xuất | Lý do |
|---|---|---|---|
| `hidden_dim` | 512 | 768 | Match ViT-Base dimension |
| `n_encoder_layer` | 2 | 4 | Deeper temporal reasoning |
| `n_decoder_layer` | 2 | 4 | Better query refinement |
| `n_query` | 8 | 12 | Accommodate more predictions |
| `n_head` | 8 | 12 | Finer attention patterns |
| Dropout | 0.0 (implicit) | 0.1 | Regularization cho larger model |

**Expected Impact:** +1–3 Avg mAP (kết hợp với stronger backbone)

---

## 6. Phase 4: Advanced Methods (Tuần 10–14)

> **Mục tiêu:** Áp dụng kỹ thuật tiên tiến để tackle fundamental challenges

### 6.1 DiffAnt — Diffusion-Based Action Anticipation

**Motivation:** Action anticipation là inherently stochastic. DiffAnt reformulate bài toán thành conditional generation — iteratively refine predictions từ Gaussian noise.

```python
class DiffusionAnticipation(nn.Module):
    """
    Diffusion model cho action anticipation.
    Thay vì predict trực tiếp, model học denoise:
    given noisy future action representation + observed context → clean prediction.
    """
    def __init__(self, hidden_dim, n_class, n_steps=100):
        super().__init__()
        self.n_steps = n_steps
        self.noise_schedule = cosine_noise_schedule(n_steps)

        # Denoising network: conditioned on encoder output
        self.denoiser = TransformerDecoder(
            hidden_dim=hidden_dim,
            n_layers=4,
            n_heads=8,
            cross_attn=True  # attend to encoder memory
        )

        # Output heads
        self.class_head = nn.Linear(hidden_dim, n_class)
        self.offset_head = nn.Linear(hidden_dim, 1)
        self.step_embed = nn.Embedding(n_steps, hidden_dim)

    def forward_train(self, encoder_memory, gt_actions):
        """Training: add noise to GT, predict noise."""
        t = torch.randint(0, self.n_steps, (encoder_memory.size(0),))
        noise = torch.randn_like(gt_actions)
        noisy = self.add_noise(gt_actions, noise, t)

        # Condition on: noisy input + timestep + encoder memory
        step_emb = self.step_embed(t)
        pred_noise = self.denoiser(noisy + step_emb, encoder_memory)
        return F.mse_loss(pred_noise, noise)

    def forward_inference(self, encoder_memory, n_queries=8):
        """Inference: iterative denoising from pure noise."""
        x = torch.randn(encoder_memory.size(0), n_queries, self.hidden_dim)
        for t in reversed(range(self.n_steps)):
            step_emb = self.step_embed(torch.tensor([t]))
            pred_noise = self.denoiser(x + step_emb, encoder_memory)
            x = self.remove_noise(x, pred_noise, t)

        classes = self.class_head(x)
        offsets = self.offset_head(x)
        return classes, offsets
```

**Integration với FAANTRA:**

```
Observed Frames → [Backbone] → [SGP-Mixer] → [Transformer Encoder] → encoder_memory
                                                                          ↓
                                                    [Diffusion Decoder] ← noise
                                                          ↓ (iterative refinement)
                                                    predicted actions + offsets
```

**Expected Impact:** +3–6 Avg mAP (tốt nhất trên rare/spontaneous actions)

### 6.2 Audio Modality Integration

**Motivation:** Broadcast audio chứa anticipatory cues:
- Commentator thường mô tả tình huống *trước* action xảy ra ("đang dẫn bóng vào vòng cấm...")
- Crowd noise tăng intensity trước goals/shots
- Referee whistle precedes certain actions

```python
class AudioVisualFusion(nn.Module):
    """
    Multi-modal fusion theo AFFT (Anticipative Feature Fusion Transformer).
    Mid-level fusion: visual + audio features qua cross-attention.
    """
    def __init__(self, visual_dim, audio_dim, hidden_dim, n_heads=8):
        super().__init__()
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)

        # Cross-attention: visual attends to audio và ngược lại
        self.v2a_attn = nn.MultiheadAttention(hidden_dim, n_heads)
        self.a2v_attn = nn.MultiheadAttention(hidden_dim, n_heads)

        # Fusion gate
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )

    def forward(self, visual_feat, audio_feat):
        v = self.visual_proj(visual_feat)  # [B, T_v, C]
        a = self.audio_proj(audio_feat)    # [B, T_a, C]

        # Cross-modal attention
        v_enhanced, _ = self.v2a_attn(v, a, a)  # visual query, audio key/value
        a_enhanced, _ = self.a2v_attn(a, v, v)  # audio query, visual key/value

        # Gated fusion
        gate = self.gate(torch.cat([v_enhanced, a_enhanced[:, :v.size(1)]], dim=-1))
        fused = gate * v_enhanced + (1 - gate) * a_enhanced[:, :v.size(1)]
        return fused

# Audio feature extraction:
# - Whisper encoder hoặc wav2vec2 cho audio features
# - Mel spectrogram → CNN cho lightweight option
# - Resample audio to match video frame rate
```

**Expected Impact:** +2–4 Avg mAP

### 6.3 LLM-Augmented Tactical Priors (AntGPT-inspired)

**Motivation:** LLMs encode strong prior knowledge về action sequences trong football.

```python
# Two-stage approach:
# Stage 1: Fine-tune LLM trên football action sequences
# Stage 2: Distill LLM knowledge vào compact network

# Stage 1: Data preparation
def create_action_sequences(annotations):
    """Convert temporal annotations thành text sequences cho LLM."""
    sequences = []
    for game in annotations:
        seq = []
        for action in game['actions']:
            seq.append(f"{action['class']} at {action['time']:.1f}s")
        sequences.append(" → ".join(seq))
    return sequences

# Ví dụ sequence:
# "PASS at 0.0s → DRIVE at 2.1s → PASS at 4.3s → CROSS at 7.8s → HEADER at 9.2s"

# Stage 2: Distillation
class TacticalPriorNetwork(nn.Module):
    """Compact network (distilled from LLM) predicting action transition probs."""
    def __init__(self, n_class, hidden_dim=256, n_context=5):
        super().__init__()
        self.action_embed = nn.Embedding(n_class + 1, hidden_dim)  # +1 for padding
        self.temporal_embed = nn.Linear(1, hidden_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_dim, 4, hidden_dim*2),
            num_layers=3
        )
        self.pred_head = nn.Linear(hidden_dim, n_class)

    def forward(self, past_actions, past_times):
        """Given observed actions → predict probability of next action class."""
        action_emb = self.action_embed(past_actions)
        time_emb = self.temporal_embed(past_times.unsqueeze(-1))
        x = action_emb + time_emb
        x = self.transformer(x)
        return self.pred_head(x[:, -1])  # predict next action

# Integration: dùng tactical prior output làm auxiliary signal
# Combine với visual model prediction qua learned weighting
```

**Expected Impact:** +1–3 Avg mAP

---

## 7. Phase 5: Ensemble & Final Optimization (Tuần 14–16)

### 7.1 Model Ensemble Strategy

```python
class EnsemblePredictor:
    """Ensemble multiple models cho robust prediction."""
    def __init__(self, models, weights=None):
        self.models = models
        self.weights = weights or [1.0/len(models)] * len(models)

    def predict(self, inputs):
        all_preds = []
        for model, weight in zip(self.models, self.weights):
            pred = model(inputs)
            all_preds.append({
                'action': pred['action'] * weight,
                'offset': pred['offset'] * weight,
                'actionness': pred['actionness'] * weight
            })

        # Weighted average
        ensemble = {
            'action': sum(p['action'] for p in all_preds),
            'offset': sum(p['offset'] for p in all_preds),
            'actionness': sum(p['actionness'] for p in all_preds)
        }
        return ensemble

# Ensemble candidates:
# 1. FAANTRA + VideoMAE backbone (best single model)
# 2. FAANTRA + InternVideo2 features (different representation)
# 3. FAANTRA + Diffusion decoder (handles uncertainty)
# 4. FAANTRA + Audio-Visual (multi-modal)
```

### 7.2 Test-Time Augmentation (TTA)

```python
def test_time_augmentation(model, video_frames, n_augments=5):
    """Multiple passes với different augmentations, aggregate results."""
    all_preds = []

    # Original
    all_preds.append(model(video_frames))

    # Horizontal flip
    all_preds.append(model(video_frames.flip(dims=[-1])))

    # Temporal jitter: slight shifts in observation window
    for shift in [-2, -1, 1, 2]:
        shifted = temporal_shift(video_frames, shift)
        all_preds.append(model(shifted))

    # Multi-scale: different resolutions
    for scale in [0.9, 1.1]:
        scaled = F.interpolate(video_frames, scale_factor=scale)
        all_preds.append(model(scaled))

    return aggregate_predictions(all_preds)
```

### 7.3 Post-Processing

```python
class ActionNMS:
    """Non-Maximum Suppression cho temporal action predictions."""
    def __init__(self, iou_threshold=0.3, class_specific=True):
        self.iou_threshold = iou_threshold
        self.class_specific = class_specific

    def __call__(self, predictions):
        """Remove overlapping predictions of same class."""
        # Sort by confidence (actionness * class_prob)
        sorted_preds = sorted(predictions, key=lambda p: p['score'], reverse=True)
        kept = []
        for pred in sorted_preds:
            overlap = False
            for kept_pred in kept:
                if self.class_specific and pred['class'] != kept_pred['class']:
                    continue
                temporal_iou = compute_temporal_iou(pred['offset'], kept_pred['offset'])
                if temporal_iou > self.iou_threshold:
                    overlap = True
                    break
            if not overlap:
                kept.append(pred)
        return kept
```

---

## 8. Experiment Plan

### 8.1 Ablation Study Design

| Experiment ID | Modification | Baseline | Expected Δ mAP |
|---|---|---|---|
| E1.1 | Focal Loss (γ=2) | 24.08 | +1–2 |
| E1.2 | Effective class weights (β=0.9999) | 24.08 | +1–2 |
| E1.3 | Oversampling rare classes (3x) | 24.08 | +1–2 |
| E1.4 | Extended training (50 epochs) | 24.08 | +1–2 |
| E1.5 | Label smoothing (0.1) | 24.08 | +0.5–1 |
| E1.6 | Increased offset_loss_weight (20) | 24.08 | +0.5–1 |
| E2.1 | VideoMAE-Base backbone | Best of E1 | +5–10 |
| E2.2 | InternVideo2 frozen features | Best of E1 | +8–12 |
| E2.3 | Resolution 448×796 | Best of E1 | +3–5 |
| E3.1 | Multi-scale SGP-Mixer | Best of E2 | +2–4 |
| E3.2 | Content-aware query init | Best of E2 | +1–2 |
| E3.3 | Gaussian temporal head | Best of E2 | +1–3 |
| E3.4 | Hierarchical attention | Best of E2 | +1–2 |
| E3.5 | Increased model capacity (768d, 4L) | Best of E2 | +1–3 |
| E4.1 | Diffusion decoder | Best of E3 | +3–6 |
| E4.2 | Audio fusion (AFFT-style) | Best of E3 | +2–4 |
| E4.3 | Tactical prior network | Best of E3 | +1–3 |
| E5.1 | Ensemble (top 3 models) | Best of E4 | +2–4 |
| E5.2 | TTA + NMS post-processing | Best of E5.1 | +1–2 |

### 8.2 Evaluation Protocol

Mỗi experiment:
1. Train trên **train split** của BAA (+ SN-AS nếu joint training)
2. Validate trên **val split** — chọn best checkpoint theo `a_mAP_stable`
3. Report: mAP@1, mAP@2, mAP@3, mAP@4, mAP@5, mAP@∞, Avg mAP
4. Per-class breakdown cho 10 classes (đặc biệt theo dõi rare classes)
5. Tất cả experiments tracked trên WandB

### 8.3 Compute Requirements

| Phase | GPU | Thời gian ước tính | Ghi chú |
|---|---|---|---|
| Phase 1 | 1× RTX 3090/4090 | 3–5 ngày | Quick experiments |
| Phase 2 (VideoMAE pretrain) | 4× A100 40GB | 3–5 ngày | Self-supervised pre-training |
| Phase 2 (InternVideo2 extract) | 1× A100 80GB | 1–2 ngày | Feature extraction offline |
| Phase 3 | 1–2× A100 40GB | 5–7 ngày | Architecture experiments |
| Phase 4 (Diffusion) | 2× A100 40GB | 5–7 ngày | Training diffusion model |
| Phase 4 (Audio) | 1× A100 40GB | 2–3 ngày | Audio feature extraction + training |
| Phase 5 | 1× RTX 3090 | 2–3 ngày | Ensemble evaluation |

**Tổng:** ~25–35 ngày GPU time (có thể parallelize)

---

## 9. Implementation Roadmap

### 9.1 File Modifications Required

| File | Modification | Phase |
|---|---|---|
| `model/futr.py` | Thêm backbone options (VideoMAE, InternVideo2) | P2 |
| `model/futr.py` | Multi-scale SGP-Mixer, content-aware queries | P3 |
| `model/futr.py` | Gaussian temporal head | P3 |
| `model/futr.py` | Audio-visual fusion module | P4 |
| `model/diffusion.py` | **New file** — DiffusionAnticipation module | P4 |
| `model/tactical.py` | **New file** — TacticalPriorNetwork | P4 |
| `model/extras/transformer.py` | Hierarchical attention masks | P3 |
| `train.py` | Focal loss, progressive rebalancing, new losses | P1 |
| `train.py` | Diffusion training loop | P4 |
| `dataset/datasets.py` | Oversampling, audio loading, pre-extracted features | P1–P4 |
| `dataset/frame.py` | Higher resolution, ball-centric crop | P2 |
| `dataset/augmentation.py` | **New file** — Extended augmentations | P1 |
| `opts.py` | New config parameters cho tất cả modifications | P1–P4 |
| `config/SoccerNetBall/` | New config files cho mỗi experiment | P1–P5 |
| `eval.py` / `eval_BAA.py` | TTA, NMS post-processing | P5 |
| `ensemble.py` | **New file** — Ensemble prediction | P5 |

### 9.2 New Config Parameters

```json
{
    "_comment": "Phase 1 additions",
    "loss_func": "focal",
    "focal_gamma": 2.0,
    "class_weight_strategy": "effective_number",
    "class_weight_beta": 0.9999,
    "progressive_rebalancing": true,
    "label_smoothing": 0.1,
    "oversample_rare": true,
    "oversample_factor": 3,
    "rare_classes": [6, 8, 9],

    "_comment": "Phase 2 additions",
    "feature_arch": "videomae_base",
    "videomae_path": "/path/to/football-videomae",
    "resolution": [448, 796],
    "gradient_checkpointing": true,
    "gradient_accumulation_steps": 4,
    "use_preextracted_features": false,
    "preextracted_feature_dir": "/path/to/internvideo2_features",

    "_comment": "Phase 3 additions",
    "multiscale_sgp": true,
    "sgp_scales": [3, 7, 15, 31],
    "content_aware_queries": true,
    "gaussian_temporal": true,
    "hierarchical_attention": true,
    "hidden_dim": 768,
    "n_encoder_layer": 4,
    "n_decoder_layer": 4,
    "dropout": 0.1,

    "_comment": "Phase 4 additions",
    "use_diffusion": false,
    "diffusion_steps": 100,
    "use_audio": false,
    "audio_feature_dir": "/path/to/audio_features",
    "audio_model": "whisper_base",
    "use_tactical_prior": false,
    "tactical_model_path": "/path/to/tactical_model"
}
```

---

## 10. Risk Analysis và Mitigation

| Risk | Xác suất | Impact | Mitigation |
|---|---|---|---|
| VideoMAE pre-training không đủ data | Trung bình | Cao | Fallback: dùng public VideoMAE checkpoint + fine-tune |
| Diffusion model training không ổn định | Trung bình | Cao | Dùng proven DiffAnt codebase; start từ simple DDPM |
| Audio features không informative | Thấp | TB | Test audio quality trước; fallback: chỉ crowd noise features |
| GPU resources không đủ | Trung bình | Cao | Ưu tiên Phase 1+2; dùng gradient accumulation; cloud compute |
| Overfitting do data nhỏ | Cao | Cao | Aggressive regularization: dropout, augmentation, early stopping |
| Challenge deadline pressure | Thấp | Cao | Phase 1–2 đủ cho competitive submission; P3–5 là bonus |

---

## 11. Expected Performance Trajectory

```
Baseline (FAANTRA):                          Avg mAP = 24.08
+ Phase 1 (Quick Wins):                     Avg mAP ≈ 28–32  (+4–8)
+ Phase 2 (Backbone Upgrade):               Avg mAP ≈ 36–44  (+8–12)
+ Phase 3 (Architecture):                   Avg mAP ≈ 40–50  (+4–6)
+ Phase 4 (Advanced Methods):               Avg mAP ≈ 45–56  (+5–6)
+ Phase 5 (Ensemble + Post-processing):     Avg mAP ≈ 48–60  (+3–4)
```

**Target cuối cùng: Avg mAP ≈ 48–60** (thu hẹp gap từ 39.77 xuống còn 4–16 điểm)

---

## 12. Deliverables

1. **Technical Report** — Paper-ready mô tả tất cả improvements, ablations, results
2. **Codebase** — Clean, modular code với tất cả modifications, configs, và scripts
3. **Trained Models** — Best checkpoints cho mỗi phase
4. **Challenge Submission** — JSON predictions trên test set cho Codabench
5. **WandB Dashboard** — Complete experiment tracking với visualizations

---

## 13. References

### Core Papers
- [FAANTRA] Dalal et al., "Action Anticipation from SoccerNet Football Video Broadcasts", CVPRW 2025
- [FUTR] Gong et al., "Future Transformer for Long-term Action Anticipation", ECCV 2022
- [T-DEED] Xarles et al., "Temporal-Discriminability Enhancer Encoder-Decoder", CVPRW 2024

### Feature Extraction
- [VideoMAE] Tong et al., "Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training", NeurIPS 2022
- [InternVideo2] Wang et al., "Scaling Foundation Models for Multimodal Video Understanding", 2024

### Anticipation Methods
- [DiffAnt] Guo et al., "Diffusion Models for Action Anticipation", 2023
- [AntGPT] Huang et al., "Can Large Language Models Help Long-term Action Anticipation?", ICLR 2024
- [AFFT] Zhong et al., "Anticipative Feature Fusion Transformer for Multi-Modal Action Anticipation", WACV 2023
- [m&m-Ant] Kim et al., "Multi-level and Multi-modal Action Anticipation", ICIP 2025

### Class Imbalance
- [Focal Loss] Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
- [Effective Number] Cui et al., "Class-Balanced Loss Based on Effective Number of Samples", CVPR 2019
- [BRL] "Balanced Representation Learning for Long-tailed Skeleton-based Action Recognition", 2023

### Data Augmentation
- [DynaAugment] "Exploring Temporally Dynamic Data Augmentation for Video Understanding", 2022
