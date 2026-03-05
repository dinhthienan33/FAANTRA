# Phase 1: Quick Wins — Implementation Report

> **Branch:** `phase1/quick-wins`
> **Baseline:** FAANTRA (Avg mAP = 24.08, Ta=5s)
> **Upper Bound:** T-DEED Action Spotting (Avg mAP = 63.85)
> **Target Phase 1:** +4–8 điểm mAP mà không thay đổi kiến trúc model

---

## 1. Tổng quan

Phase 1 tập trung vào **"Quick Wins"** — các cải tiến về loss function, training recipe, và data augmentation có thể tăng performance mà **không cần thay đổi kiến trúc model FUTR**. Tất cả thay đổi đều backward compatible: config gốc `Base-Config-BAA.json` chạy bình thường vì mọi parameter mới đều có default khớp với behavior cũ.

### Tóm tắt thay đổi

| File | Dòng thêm/xóa | Mô tả |
|---|---|---|
| `utils.py` | +80 / -5 | Focal Loss, Effective Number Weights, Label Smoothing |
| `train.py` | +31 / -13 | Gradient Accumulation, Gradient Clipping, truyền loss params |
| `main.py` | +4 / -1 | Scheduler tính đúng steps cho gradient accumulation |
| `opts.py` | +18 / -1 | 5 config parameters mới |
| `model/futr.py` | +11 / -1 | Extended augmentation pipeline |
| **Tổng** | **+144 / -21** | |

### Config files mới (4 files)

| Config | Mục đích |
|---|---|
| `Phase1-Config-BAA.json` | Full Phase 1: tất cả improvements kết hợp |
| `Phase1-Ablation-FocalOnly.json` | Ablation: chỉ Focal Loss |
| `Phase1-Ablation-SmoothingOnly.json` | Ablation: chỉ Label Smoothing |
| `Phase1-Ablation-ExtendedTrain.json` | Ablation: training recipe (epochs, LR, WD, accum, aug) |

---

## 2. Chi tiết Implementation

### 2.1 Focal Loss (`utils.py`)

**Vấn đề giải quyết:** Class imbalance nghiêm trọng — Pass đạt 55.50 mAP trong khi Successful Tackle chỉ 8.02 mAP (chênh lệch gần 7x). CE loss chuẩn treat mọi sample ngang nhau, khiến model bị dominate bởi frequent classes.

**Giải pháp:** Focal Loss (Lin et al., ICCV 2017) thêm modulating factor `(1 - p_t)^γ` vào CE loss:

```
FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
```

- Khi model predict đúng dễ dàng (p_t cao) → `(1 - p_t)^γ` rất nhỏ → loss gần 0
- Khi model predict sai (p_t thấp) → `(1 - p_t)^γ` gần 1 → loss giữ nguyên
- Hiệu quả: model tập trung học các hard examples (thường là rare classes)

**Code mới trong `utils.py`:**

```python
def focal_loss(pred, gold, trg_pad_idx, class_weights=None, gamma=2.0, label_smoothing=0.0):
    non_pad_mask = gold.ne(trg_pad_idx)
    if non_pad_mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)

    pred_valid = pred[non_pad_mask]
    gold_valid = gold[non_pad_mask]

    ce_loss = F.cross_entropy(
        pred_valid, gold_valid, weight=class_weights,
        reduction='none', label_smoothing=label_smoothing
    )

    log_pt = -ce_loss
    pt = torch.exp(log_pt)
    focal_weight = (1.0 - pt) ** gamma
    loss = focal_weight * ce_loss

    return loss.mean()
```

**Tích hợp:** Thêm `loss_func="focal"` vào hàm `cal_loss()` — khi config set `"loss_func": "focal"`, loss tự động chuyển sang Focal Loss. Hàm `cal_performance()` nhận thêm `focal_gamma` và `label_smoothing` rồi truyền xuống `cal_loss()`.

**Config parameter:** `"focal_gamma": 2.0` (γ = 0 tương đương CE chuẩn, γ = 2 là default khuyến nghị)

---

### 2.2 Effective Number of Samples Class Weights (`utils.py`)

**Vấn đề giải quyết:** Class weights hiện tại `[0.03, 0.15, 0.2, 1, 1, 2, 3, 4.5, 1.4, 3.4, 10]` được set thủ công — không optimal và khó tune.

**Giải pháp:** Effective Number of Samples (Cui et al., CVPR 2019):

```
E_n = (1 - β^n) / (1 - β)
weight_i = (1 - β) / E_n_i
```

Trong đó `n` là số samples của class, `β ∈ [0, 1)` là hyperparameter. β càng cao → rebalancing càng mạnh.

**Code mới trong `utils.py`:**

```python
def compute_effective_weights(class_counts, beta=0.9999):
    class_counts = np.array(class_counts, dtype=np.float64)
    class_counts = np.maximum(class_counts, 1.0)
    effective_num = 1.0 - np.power(beta, class_counts)
    weights = (1.0 - beta) / effective_num
    weights = weights / np.sum(weights) * len(weights)
    return weights
```

**Ví dụ kết quả** với BAA class distribution ước tính:

```
Input counts:  [5000, 3000, 500, 200, 180, 100, 50, 300, 80, 30]
Output weights: [0.027, 0.042, 0.221, 0.544, 0.604, 1.083, 2.161, 0.365, 1.353, 3.599]
```

Rare classes (Shot=2.161, Tackle=3.599) nhận weight cao hơn nhiều so với frequent classes (Pass=0.027).

**Ghi chú:** Hàm này được export sẵn để dùng — user có thể tính weights từ actual class counts rồi đưa vào config `class_weights`. Chưa auto-integrate vào training loop vì cần class counts từ dataset.

---

### 2.3 Label Smoothing (`utils.py`, `train.py`)

**Vấn đề giải quyết:** Model có thể overfit vào one-hot labels, đặc biệt trên dataset nhỏ (BAA chỉ 11.4 giờ). Overconfident predictions dẫn đến poor generalization.

**Giải pháp:** Label Smoothing phân phối nhỏ probability cho non-target classes:

```
y_smooth = y * (1 - ε) + ε / K
```

Trong đó `ε` là smoothing factor, `K` là số classes. Với `ε = 0.1`, target class nhận 0.9 thay vì 1.0, và mỗi non-target class nhận `0.1/K`.

**Implementation:** Sử dụng trực tiếp PyTorch built-in `label_smoothing` parameter trong `F.cross_entropy()` — không cần code thủ công:

```python
# Trong cal_loss():
loss = F.cross_entropy(pred, gold, ignore_index=trg_pad_idx,
                       weight=class_weights, label_smoothing=label_smoothing)
```

Label smoothing cũng được áp dụng cho Focal Loss vì Focal Loss xây dựng trên CE loss.

**Áp dụng:** Label smoothing được truyền vào **cả 4 vị trí** gọi `cal_performance()` trong `train.py`:
1. Training segmentation loss (line 86-87)
2. Training anticipation loss (line 136-140)
3. Validation segmentation loss (line 288-289)
4. Validation anticipation loss (line 338-342)

**Config parameter:** `"label_smoothing": 0.1` (0.0 = tắt, 0.1 là giá trị khuyến nghị)

---

### 2.4 Gradient Accumulation (`train.py`, `main.py`)

**Vấn đề giải quyết:** Batch size 8 nhỏ → gradient estimates noisy → training không ổn định. Tăng batch size trực tiếp yêu cầu thêm GPU memory.

**Giải pháp:** Gradient Accumulation tích lũy gradients qua `N` forward passes trước khi gọi `optimizer.step()`, simulate effective batch size = `batch_size × N` mà không cần thêm memory.

**Implementation trong `train.py`:**

```python
# Đọc config
accum_steps = getattr(args, 'gradient_accumulation_steps', 1)

# Chỉ zero gradients tại accumulation boundaries
if i % accum_steps == 0:
    optimizer.zero_grad()

# Scale loss để tổng loss qua N steps = loss trung bình
scaled_loss = losses / accum_steps
scaled_loss.backward()

# Chỉ step optimizer tại accumulation boundaries (hoặc cuối epoch)
if (i + 1) % accum_steps == 0 or (i + 1) == len(train_loader):
    if grad_clip_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
    optimizer.step()
    scheduler.step()
```

**Scheduler adjustment trong `main.py`:** Scheduler cần biết số *effective* steps per epoch (không phải raw steps):

```python
accum_steps = getattr(args, 'gradient_accumulation_steps', 1)
effective_steps_per_epoch = (num_steps_per_epoch + accum_steps - 1) // accum_steps
scheduler = get_lr_scheduler(args, optimizer, effective_steps_per_epoch)
```

Lưu ý: `scheduler.step()` cũ ở cuối training loop đã được **xóa** để tránh trùng lặp — giờ chỉ gọi bên trong block accumulation.

**Config parameters:**
- `"gradient_accumulation_steps": 2` → effective batch size = 8 × 2 = 16
- `"grad_clip_norm": 1.0` → clip gradient norm tối đa 1.0 (0 = tắt)

---

### 2.5 Gradient Clipping (`train.py`)

**Vấn đề giải quyết:** Gradient explosion có thể xảy ra, đặc biệt ở đầu training hoặc khi dùng loss functions mới (Focal Loss).

**Giải pháp:** `torch.nn.utils.clip_grad_norm_()` giới hạn tổng L2 norm của tất cả gradients. Nếu norm > `max_norm`, tất cả gradients được scale xuống proportionally.

**Integration:** Gradient clipping chạy **sau** accumulation hoàn thành, **trước** `optimizer.step()`:

```python
if grad_clip_norm > 0:
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
optimizer.step()
```

**Config parameter:** `"grad_clip_norm": 1.0` (0.0 = tắt)

---

### 2.6 Extended Data Augmentation (`model/futr.py`)

**Vấn đề giải quyết:** Augmentation pipeline gốc chỉ có ColorJitter, GaussianBlur, HorizontalFlip — khá basic, dễ overfit trên dataset nhỏ.

**Giải pháp:** Thêm 2 augmentations mới khi `extended_augmentation=True`:

```python
base_augs = [
    # ... (6 augmentations gốc giữ nguyên)
]
if getattr(args, 'extended_augmentation', False):
    base_augs.extend([
        T.RandomApply([T.RandomErasing(p=1.0, scale=(0.02, 0.15), ratio=(0.3, 3.3))], p=0.2),
        T.RandomApply([T.RandomPerspective(distortion_scale=0.1)], p=0.15),
    ])
self.augmentation = T.Compose(base_augs)
```

| Augmentation | Mô tả | Probability |
|---|---|---|
| **RandomErasing** | Xóa random rectangle trong frame → buộc model không rely vào single spatial region | 20% |
| **RandomPerspective** | Biến dạng perspective nhẹ → simulate camera angle variations | 15% |

**Config parameter:** `"extended_augmentation": true/false`

---

### 2.7 Config Parameters mới (`opts.py`)

Tất cả 5 parameters mới được thêm vào cuối `update_args()` với **defaults backward compatible:**

```python
# Phase 1: Quick Wins — new config parameters
args.focal_gamma = config.get("focal_gamma", 2.0)
args.label_smoothing = config.get("label_smoothing", 0.0)
args.gradient_accumulation_steps = config.get("gradient_accumulation_steps", 1)
args.grad_clip_norm = config.get("grad_clip_norm", 0.0)
args.extended_augmentation = config.get("extended_augmentation", False)
```

| Parameter | Default | Giải thích |
|---|---|---|
| `focal_gamma` | `2.0` | Focusing parameter cho Focal Loss (chỉ có tác dụng khi `loss_func="focal"`) |
| `label_smoothing` | `0.0` | Label smoothing factor (0.0 = tắt) |
| `gradient_accumulation_steps` | `1` | Số steps tích lũy gradient (1 = không accumulation) |
| `grad_clip_norm` | `0.0` | Max gradient norm cho clipping (0.0 = tắt) |
| `extended_augmentation` | `false` | Bật augmentations mở rộng (RandomErasing, RandomPerspective) |

---

## 3. Config Files cho Experiments

### 3.1 Full Phase 1 (`Phase1-Config-BAA.json`)

Kết hợp **tất cả** Phase 1 improvements + tuned hyperparameters:

| Parameter | Baseline | Phase 1 | Lý do |
|---|---|---|---|
| `loss_func` | `"CE"` | `"focal"` | Focal Loss giải quyết class imbalance |
| `focal_gamma` | — | `2.0` | γ=2 là default khuyến nghị từ paper gốc |
| `label_smoothing` | `0.0` | `0.1` | Giảm overconfidence, cải thiện generalization |
| `gradient_accumulation_steps` | `1` | `2` | Effective batch size 16 thay vì 8 |
| `grad_clip_norm` | `0.0` | `1.0` | Ngăn gradient explosion |
| `extended_augmentation` | `false` | `true` | Stronger regularization |
| `num_epochs` | `20` | `50` | Training lâu hơn để converge tốt hơn |
| `learning_rate` | `0.0001` | `0.0002` | LR cao hơn + cosine decay dài hơn |
| `warm_up_epochs` | `3` | `5` | Warmup lâu hơn cho training ổn định |
| `weight_decay` | `5e-3` | `0.01` | Regularization mạnh hơn |
| `eos_weight` | `0.1` | `0.05` | Giảm background dominance |
| `offset_loss_weight` | `10` | `15` | Tăng temporal precision |
| `start_map_epoch` | `10` | `15` | Cho model warm up trước khi eval |
| `temporal_arch` | `"none"` | `"ed_sgp_mixer"` | Bật SGP-Mixer (baseline BAA config không có) |

### 3.2 Ablation: Focal Loss Only (`Phase1-Ablation-FocalOnly.json`)

Chỉ thay đổi `"loss_func": "focal"` — giữ nguyên mọi thứ khác giống baseline. Isolate hiệu quả riêng của Focal Loss.

### 3.3 Ablation: Label Smoothing Only (`Phase1-Ablation-SmoothingOnly.json`)

Chỉ thay đổi `"label_smoothing": 0.1` — giữ nguyên `loss_func="CE"`. Isolate hiệu quả riêng của Label Smoothing.

### 3.4 Ablation: Extended Training (`Phase1-Ablation-ExtendedTrain.json`)

Thay đổi training recipe mà **không đổi loss function**: 50 epochs, LR 2e-4, WD 0.01, gradient accumulation 2x, gradient clipping 1.0, extended augmentation. Giữ nguyên CE loss. Isolate hiệu quả của training recipe improvements.

---

## 4. Cách chạy Experiments

```bash
# Full Phase 1 (tất cả improvements)
python main.py config/SoccerNetBall/Phase1-Config-BAA.json faantra_phase1

# Ablation: chỉ Focal Loss
python main.py config/SoccerNetBall/Phase1-Ablation-FocalOnly.json faantra_focal

# Ablation: chỉ Label Smoothing
python main.py config/SoccerNetBall/Phase1-Ablation-SmoothingOnly.json faantra_smooth

# Ablation: chỉ Extended Training recipe
python main.py config/SoccerNetBall/Phase1-Ablation-ExtendedTrain.json faantra_extended

# Baseline (để so sánh — không cần thay đổi gì)
python main.py config/SoccerNetBall/Base-Config-BAA.json faantra_baseline
```

---

## 5. Backward Compatibility

Tất cả thay đổi hoàn toàn backward compatible:

- Config gốc `Base-Config-BAA.json` **chạy bình thường** — không cần thêm bất kỳ parameter mới nào
- Mọi parameter mới dùng `config.get(key, default)` với defaults giữ nguyên behavior cũ:
  - `focal_gamma=2.0` — chỉ có tác dụng khi `loss_func="focal"`, CE vẫn dùng CE
  - `label_smoothing=0.0` — bằng 0 = hoàn toàn không smoothing
  - `gradient_accumulation_steps=1` — bằng 1 = mỗi step đều update weights (behavior cũ)
  - `grad_clip_norm=0.0` — bằng 0 = không clip (behavior cũ)
  - `extended_augmentation=false` — tắt = augmentation pipeline giữ nguyên
- Trong `train.py`, dùng `getattr(args, 'param', default)` thay vì truy cập trực tiếp → an toàn nếu args từ checkpoint cũ không có param mới
- Checkpoint format **không thay đổi** — có thể resume training từ checkpoint cũ

---

## 6. Luồng Dữ liệu Kỹ thuật

### 6.1 Loss Function Flow (Anticipation task)

```
train.py                          utils.py
────────                          ────────
focal_gamma = args.focal_gamma
label_smoothing = args.label_smoothing
        │
        ▼
cal_performance(                  cal_performance(pred, gold, ...,
  output, target, pad_idx,  ──►     focal_gamma, label_smoothing)
  loss_func="focal",                    │
  focal_gamma=2.0,                      ▼
  label_smoothing=0.1)            cal_loss(pred, gold, ...,
                                     loss_func, focal_gamma,
                                     label_smoothing)
                                        │
                            ┌───────────┼───────────┐
                            ▼           ▼           ▼
                        "CE"        "focal"      "BCE"
                            │           │           │
                            ▼           ▼           ▼
                     F.cross_entropy  focal_loss  F.bce_with_logits
                     (label_smoothing) (gamma,     (unchanged)
                                       label_smoothing)
```

### 6.2 Gradient Accumulation Flow

```
Step 0:  zero_grad() → forward → loss/N → backward
Step 1:               → forward → loss/N → backward
         ──── gradients accumulated ────
         clip_grad_norm → optimizer.step() → scheduler.step()

Step 2:  zero_grad() → forward → loss/N → backward
Step 3:               → forward → loss/N → backward
         ──── gradients accumulated ────
         clip_grad_norm → optimizer.step() → scheduler.step()
...
```

### 6.3 Scheduler Steps Calculation

```
main.py:
  num_steps_per_epoch = len(train_loader)     # e.g., 976 steps
  accum_steps = args.gradient_accumulation_steps  # e.g., 2
  effective_steps_per_epoch = ceil(976 / 2) = 488

  scheduler = LinearWarmup(5 × 488 steps) + CosineAnnealing(45 × 488 steps)
```

---

## 7. Ablation Study Design

### Expected Results Matrix

| Experiment | loss_func | smoothing | accum | epochs | LR | Extended Aug | Expected Δ mAP |
|---|---|---|---|---|---|---|---|
| Baseline | CE | 0.0 | 1 | 20 | 1e-4 | No | 0 (baseline) |
| E1: Focal Only | focal γ=2 | 0.0 | 1 | 20 | 1e-4 | No | +1–2 |
| E2: Smoothing Only | CE | 0.1 | 1 | 20 | 1e-4 | No | +0.5–1 |
| E3: Extended Train | CE | 0.0 | 2 | 50 | 2e-4 | Yes | +1–3 |
| E4: Full Phase 1 | focal γ=2 | 0.1 | 2 | 50 | 2e-4 | Yes | +4–8 |

### Metrics theo dõi

Với mỗi experiment, báo cáo:
1. **Avg mAP** = mean(mAP@1, mAP@2, mAP@3, mAP@4, mAP@5, mAP@∞) — metric chính
2. **mAP@1** — temporal precision (cần cải thiện nhiều nhất, hiện 9.74)
3. **Per-class mAP** — đặc biệt rare classes: SHOT, BALL PLAYER BLOCK, PLAYER SUCCESSFUL TACKLE
4. **Training curves** trên WandB: loss convergence, offset accuracy, class-wise stats

---

## 8. Phân tích Rủi ro

| Rủi ro | Khả năng | Mitigation |
|---|---|---|
| Focal Loss làm giảm performance trên frequent classes (Pass, Drive) | Thấp | Ablation FocalOnly cho phép isolate; có thể mix CE+Focal |
| Label Smoothing + Focal Loss confict | Thấp | Smoothing đã được tích hợp vào Focal qua CE base; ablation verify |
| Gradient accumulation thay đổi training dynamics | Rất thấp | Mathematically equivalent với larger batch; scheduler đã adjust |
| Extended augmentation quá mạnh | Thấp | Probabilities (20% erasing, 15% perspective) khá conservative |
| 50 epochs overfitting | Trung bình | Weight decay tăng 2x, augmentation mở rộng, label smoothing bù đắp |

---

## 9. Tiếp theo: Phase 2

Phase 1 tập trung vào training methodology — **không thay đổi model architecture**. Phase 2 sẽ tackle bottleneck lớn nhất:

1. **Backbone upgrade**: RegNetY-006 (~6M params) → VideoMAE-Base (~87M params) hoặc InternVideo2 features
2. **Higher resolution**: 224p → 448p+ với gradient checkpointing
3. **SGP-Mixer bật lại**: Phase 1 config đã set `temporal_arch: "ed_sgp_mixer"` (baseline BAA config dùng `"none"`)

Expected cumulative impact: Phase 1 (+4–8) + Phase 2 (+8–12) = **+12–20 Avg mAP** so với baseline 24.08.
