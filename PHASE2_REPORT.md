# Phase 2: Feature Extraction Upgrade — Implementation Report

> **Mục tiêu Phase 2 (theo `PROPOSAL.md`)**: nâng backbone/feature extractor (VideoMAE/InternVideo2 hoặc backbone mạnh hơn) + hỗ trợ train ổn định ở cấu hình nặng (higher-res, freeze/unfreeze, gradient checkpointing, LR tách riêng).
>
> **Trạng thái**: các phần **Backbone upgrade + recipe hỗ trợ fine-tuning** đã được code và có config chạy kèm. Riêng nhánh **InternVideo2 pre-extracted features** mới có “hook” ở model/opts nhưng **chưa được nối vào dataset loader** (xem mục “Hạn chế / TODO”).

---

## 1. Tổng quan

Phase 2 tập trung xử lý bottleneck lớn nhất của baseline: **feature extractor quá yếu**. Code đã được mở rộng để:

- Dùng **backbone mạnh hơn** (RegNet lớn hơn, EfficientNetV2, Swin Tiny, VideoMAE).
- Hỗ trợ **fine-tune an toàn** với:
  - **freeze backbone → unfreeze** theo epoch,
  - **gradient checkpointing** để giảm VRAM,
  - **learning rate tách riêng** cho backbone và phần còn lại.

### Tóm tắt thay đổi chính

| File | Nội dung Phase 2 |
| --- | --- |
| `model/futr.py` | Backbone registry (timm), VideoMAE wrapper (HuggingFace), gradient checkpointing, freeze/unfreeze backbone, “pre-extracted feature mode” (stub) |
| `opts.py` | Thêm các config params Phase 2 (checkpointing, freeze/unfreeze, LR multiplier, VideoMAE params, pre-extracted params) |
| `main.py` | Tách param groups để dùng LR riêng cho backbone (`backbone_lr_multiplier`) |
| `train.py` | Gọi `maybe_unfreeze_backbone(epoch)` mỗi epoch để progressive unfreezing |
| `model/T_Deed_Modules/shift.py` | Mở rộng bắt type/attr cho GSM/GSF để tương thích nhiều biến thể backbone/timm |

### Config files Phase 2 (5 files)

| Config | Mục đích |
| --- | --- |
| `config/SoccerNetBall/Phase2-VideoMAE-BAA.json` | Backbone VideoMAE + freeze/unfreeze + checkpointing + LR backbone thấp |
| `config/SoccerNetBall/Phase2-HighRes-BAA.json` | Bật gradient checkpointing + `resolution` để resize input rõ ràng (phục vụ training nặng VRAM) |
| `config/SoccerNetBall/Phase2-LargeRegNet-BAA.json` | Dùng RegNet lớn hơn (ví dụ `rny040_gsf`) + freeze/unfreeze + LR backbone multiplier |
| `config/SoccerNetBall/Phase2-EfficientNetV2-BAA.json` | EfficientNetV2 backbone (timm) + freeze/unfreeze |
| `config/SoccerNetBall/Phase2-SwinTiny-BAA.json` | Swin Tiny backbone (timm) + freeze/unfreeze |
| `config/SoccerNetBall/Phase2-InternVideo2-PreExtracted-BAA.json` | **InternVideo2 Stage2 features offline** (`use_preextracted_features=true`) |

---

## 2. Chi tiết Implementation

### 2.1 Backbone registry (timm) và mở rộng họ backbone

Trong `model/futr.py` đã có `TIMM_BACKBONE_REGISTRY` để map string trong config → timm model name, giúp thêm backbone mới dễ dàng.

- **Đã hỗ trợ**:
  - RegNetY: từ nhỏ (`rny002/rny004/...`) đến lớn (`rny016` → `rny160`)
  - EfficientNetV2: `efficientnetv2_s/m`, `efficientnetv2_rw_s`
  - Swin: `swin_tiny/small/base/...`
- **GSF/GSM temporal shift**:
  - Chỉ áp dụng cho nhóm RegNet qua `GSF_COMPATIBLE_BACKBONES`
  - Nếu config suffix `_gsm/_gsf` nhưng backbone không hỗ trợ → warning và bỏ qua suffix

### 2.2 VideoMAE backbone (HuggingFace transformers)

`feature_arch: "videomae_base"` sẽ kích hoạt VideoMAE backbone.

- **Wrapper**: `VideoMAEBackboneWrapper` nhận input frames dạng `[B, S, C, H, W]`, chạy VideoMAE theo clip-size cố định (thường 16 frames), rồi **interpolate** về đúng S để dùng như per-frame feature extractor.
- **Config params**:
  - `videomae_model_name` (default: `MCG-NJU/videomae-base-finetuned-kinetics`)
  - `videomae_pool` (hiện dùng mean pooling theo spatial tokens)

**Phụ thuộc thêm**: cần package `transformers` khi dùng VideoMAE.

### 2.3 Gradient checkpointing cho backbone

Trong `model/futr.py`:

- Với backbone 2D (timm): checkpoint trên tensor flattened `[B*S, C, H, W]`.
- Với backbone video (VideoMAE): checkpoint trực tiếp trên `[B, S, C, H, W]`.

Config param: `gradient_checkpointing: true/false`.

### 2.4 Freeze backbone và progressive unfreezing

Trong `model/futr.py`:

- `freeze_backbone: true` → set `requires_grad=False` cho backbone.
- `unfreeze_backbone_epoch: N` → gọi `maybe_unfreeze_backbone(epoch)` để mở khóa backbone từ epoch N.

Trong `train.py`, mỗi epoch đều gọi `maybe_unfreeze_backbone(epoch)` (tương thích `DataParallel`).

### 2.5 Learning rate tách riêng cho backbone

Trong `main.py`:

- Nếu `backbone_lr_multiplier != 1.0` thì chia param groups:
  - backbone params: `lr = args.lr * backbone_lr_multiplier`
  - phần còn lại: `lr = args.lr`

Mục tiêu: backbone pretrained update chậm hơn, tránh “catastrophic forgetting”.

### 2.6 Pre-extracted features mode (InternVideo2) — hiện mới ở mức “hook”

Trong `opts.py` và `model/futr.py` đã có:

- `use_preextracted_features`
- `preextracted_feat_dim`
- `preextracted_feature_dir`

Và trong `model/futr.py` đã có `PreextractedBackbone` (pass-through + optional projection), giả định dataset trả về `src` là feature tensor `[B, S, D]` thay vì frames.

Tuy nhiên hiện tại **dataset pipeline chưa đọc `preextracted_feature_dir`** nên mode này chưa chạy end-to-end (xem mục 5).

---

## 3. Cách chạy Phase 2

### 3.1 VideoMAE

```bash
python main.py config/SoccerNetBall/Phase2-VideoMAE-BAA.json faantra_phase2_videomae
```

Gợi ý môi trường (khi chưa có `transformers`):

```bash
pip install transformers
```

### 3.2 Các backbone timm (RegNet lớn / EfficientNetV2 / Swin)

```bash
python main.py config/SoccerNetBall/Phase2-LargeRegNet-BAA.json faantra_phase2_rny040
python main.py config/SoccerNetBall/Phase2-EfficientNetV2-BAA.json faantra_phase2_effnetv2
python main.py config/SoccerNetBall/Phase2-SwinTiny-BAA.json faantra_phase2_swin_tiny
```

### 3.3 Checkpointing/high-res recipe

```bash
python main.py config/SoccerNetBall/Phase2-HighRes-BAA.json faantra_phase2_highres
```

Lưu ý: “high-res” bây giờ có thể điều khiển rõ qua config `resolution: [H, W]` (dataset sẽ resize khi đọc frame). Bạn vẫn có thể trỏ `frame_dir` tới bộ frames gốc (720p) và resize xuống 448×796 khi train.

### 3.4 InternVideo2 pre-extracted features (offline)

Có **2 preset** sẵn (dùng `--preset` khi extract):

| Option | Preset | Model | Feat dim | VRAM | Config |
|--------|--------|--------|----------|------|--------|
| **1** | `stage2_6b` | InternVideo2-Stage2_6B-224p-f4 | 1024 | ~24GB+ | `Phase2-InternVideo2-Stage2-6B-PreExtracted-BAA.json` |
| **2** | `stage2_1b` | InternVideo2-CLIP-1B-224p-f8 | 768 | ~12GB | `Phase2-InternVideo2-Stage2-1B-PreExtracted-BAA.json` |

**Option 1 — Stage2-6B (chất lượng cao, GPU mạnh):**

1. Extract features:

```bash
python scripts/extract_internvideo2_features.py \
  --frame_dir /workspace/FAANTRA/data/soccernetball/720p \
  --output_dir /workspace/FAANTRA/features/internvideo2_stage2_6b_224p_f4 \
  --preset stage2_6b
```

2. Train:

```bash
python main.py config/SoccerNetBall/Phase2-InternVideo2-Stage2-6B-PreExtracted-BAA.json faantra_phase2_internvideo2_6b
```

**Option 2 — Stage2-1B (nhẹ hơn, VRAM thấp):**

1. Extract features:

```bash
python scripts/extract_internvideo2_features.py \
  --frame_dir /workspace/FAANTRA/data/soccernetball/720p \
  --output_dir /workspace/FAANTRA/features/internvideo2_stage2_1b_224p_f8 \
  --preset stage2_1b
```

2. Train:

```bash
python main.py config/SoccerNetBall/Phase2-InternVideo2-Stage2-1B-PreExtracted-BAA.json faantra_phase2_internvideo2_1b
```

Ghi chú:

- `preextracted_feat_dim` trong config đã khớp với từng preset (1024 / 768).
- Có thể bỏ `--preset` và dùng `--model_id`, `--fnum`, `--image_size` thủ công; khi đó cần chỉnh `preextracted_feat_dim` trong config cho đúng với model.
- Layout: `preextracted_feature_dir/<relative_path_from_frame_dir>/features.pt`.

---

## 4. Backward compatibility

Phase 2 được implement theo kiểu “opt-in”:

- Các tham số Phase 2 trong `opts.py` đều dùng `config.get(key, default)` → config cũ vẫn chạy.
- Nếu không bật `gradient_checkpointing`/`freeze_backbone`/`backbone_lr_multiplier` thì behavior tương đương trước Phase 2.

---

## 5. Hoàn thiện các phần còn thiếu (đã implement)

### 5.1 InternVideo2 / offline pre-extracted features (end-to-end)

Dataset train/val (`ActionSpotDataset`) đã hỗ trợ `use_preextracted_features`. Khi bật, dataset sẽ **không đọc JPEG frames**, mà đọc feature tensor từ disk và trả về `frames` dạng `[S, D]` để `model/futr.py` đi vào nhánh `use_preextracted_features`.

- **Config keys**:
  - `use_preextracted_features`: `true`
  - `preextracted_feature_dir`: thư mục gốc chứa features
  - `preextracted_feat_dim`: ví dụ `768`
- **Quy ước đường dẫn file feature** (mặc định):
  - `preextracted_feature_dir/<relative_path_from_frame_dir>/features.pt`
  - hoặc `features.npy` (fallback: `feat.pt`, `feat.npy`)

Ví dụ: nếu `frame_dir=/data/720p` và clip frames ở `/data/720p/<video>/clip_1/`, thì feature file kỳ vọng ở:

- `preextracted_feature_dir/<video>/clip_1/features.pt`

### 5.2 Resolution control qua config (resize khi đọc frame)

Đã thêm tham số:

- `resolution: [H, W]`

Dataset sẽ resize ngay tại `FrameReader`/`FrameReaderVideo`, áp dụng cho cả training và evaluation.

---

## 6. Tiếp theo: Phase 3 (Architecture Innovation)

Theo roadmap trong `PROPOSAL.md`, Phase 3 sẽ tập trung vào:

- Multi-scale temporal encoding
- Content-aware query initialization
- (Tuỳ chọn) Gaussian temporal prediction / hierarchical attention
