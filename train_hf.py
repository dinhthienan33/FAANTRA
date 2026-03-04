import os
import argparse

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torchvision.transforms as T

from opts import update_args
from util.io import load_json
from util.dataset import load_classes
from dataset.datasets import get_datasets
from model.hf_video_models import HFVideoMAEClassifier, HFTimesformerClassifier


class SoccerNetBallHFWrapper(Dataset):
    """VideoMAE expects 16 frames, TimeSformer expects 8. Subsamples from clip_len (e.g. 64)."""

    def __init__(
        self,
        base_dataset,
        num_classes: int,
        label_pad_idx: int,
        image_size: int = 224,
        num_frames: int = 16,
    ) -> None:
        super().__init__()
        self.base_dataset = base_dataset
        self.num_classes = num_classes
        self.label_pad_idx = label_pad_idx
        self.num_frames = num_frames

        self.resize = T.Resize((image_size, image_size))
        self.normalize = T.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )

    def __len__(self) -> int:
        return len(self.base_dataset)

    def _sample_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """Subsample to num_frames (VideoMAE=16, TimeSformer=8) to match pretrained."""
        T = frames.size(0)
        if T <= self.num_frames:
            return frames
        indices = torch.linspace(0, T - 1, self.num_frames, dtype=torch.long)
        return frames[indices]

    def _frames_to_pixel_values(self, frames: torch.Tensor) -> torch.Tensor:
        frames = frames.float() / 255.0
        processed = []
        for f in frames:
            f_resized = self.resize(f)
            f_norm = self.normalize(f_resized)
            processed.append(f_norm)
        return torch.stack(processed, dim=0)

    def _targets_to_label(self, future_target: torch.Tensor) -> int:

        ft = future_target.long()
        mask = (ft != self.label_pad_idx) & (ft != 0)
        valid = ft[mask]
        if valid.numel() == 0:
            return 0
        label = int(valid[0].item())
        # Đảm bảo label nằm trong [0, num_classes-1]
        label = max(0, min(self.num_classes - 1, label))
        return label

    def __getitem__(self, idx):
        item = self.base_dataset[idx]

        frames = item["frames"]  # [T, C, H, W]
        frames = self._sample_frames(frames)  # subsample to num_frames (16 or 8)
        future_target = item["future_target"]  # [n_query]

        pixel_values = self._frames_to_pixel_values(frames)
        label = self._targets_to_label(future_target)

        return {
            "pixel_values": pixel_values,
            "labels": torch.tensor(label, dtype=torch.long),
        }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune HuggingFace video models (VideoMAE / TimeSformer) "
        "trên dataset SoccerNet Ball (data/soccernetball).",
    )
    parser.add_argument("config", type=str, help="Đường dẫn tới file config JSON (vd: config/SoccerNetBall/Base-Config-BAS.json)")
    parser.add_argument("model", type=str, help="Tên model run (dùng để đặt thư mục save)")

    parser.add_argument(
        "--hf-backbone",
        type=str,
        default="videomae",
        choices=["videomae", "timesformer"],
        help="Backbone HuggingFace để fine-tune.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Số epoch train (mặc định dùng num_epochs trong config).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (mặc định dùng learning_rate trong config).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (mặc định dùng batch_size trong config).",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Chạy trên CPU thay vì GPU.",
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="Chỉ định index GPU để dùng (mặc định 0, tương ứng với `cuda:0`).",
    )

    return parser.parse_args()


def train(args, model, train_loader, val_loader, optimizer, scheduler, criterion,
          model_save_path, pad_idx, device, num_pred_frames, n_class, class_dict, n_query,
          start_epoch=0, offset_loss_weight=1.0, use_actionness=False, use_anchors=False,
          loss_func="CE", best_mAP=0, best_model_path=""):
    hf_backbone = getattr(args, "hf_backbone", "videomae")
    config = load_json(args.config)

    # Dataset gốc (clips + labels) giống pipeline FAANTRA
    classes, _, train_base, val_base, _ = get_datasets(args, pad_idx, n_class)
    if args.store_mode == "store":
        print("Datasets đang ở chế độ 'store'. Hãy chạy main.py một lần với store_mode='store' "
              "hoặc chỉnh config sang 'load' sau khi đã store xong.")
        return model, ""

    num_frames = 16 if hf_backbone == "videomae" else 8
    train_ds = SoccerNetBallHFWrapper(
        train_base, n_class, pad_idx, image_size=224, num_frames=num_frames
    )
    val_ds = SoccerNetBallHFWrapper(
        val_base, n_class, pad_idx, image_size=224, num_frames=num_frames
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    save_dir = os.path.dirname(model_save_path)
    os.makedirs(save_dir, exist_ok=True)
    best_path = os.path.join(save_dir, "best_hf.ckpt")

    best_val_acc = 0.0
    for epoch in range(start_epoch, args.epochs):
        ####################
        # Train
        ####################
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        train_loop = tqdm(train_loader)
        for i, batch in enumerate(train_loop):
            pixel_values = batch["pixel_values"].to(device)  # [B, T, C, H, W]
            labels = batch["labels"].to(device)              # [B]

            optimizer.zero_grad()
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * pixel_values.size(0)
            preds = outputs.logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            train_loop.set_description(f"Epoch [{epoch+1}/{args.epochs}]")
            train_loop.set_postfix(loss=loss.item(), acc=correct/max(1,total))
        scheduler.step()
        train_loss = running_loss / max(1, total)
        train_acc = correct / max(1, total)

        ####################
        # Validation
        ####################
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0
        val_loop = tqdm(val_loader)
        with torch.no_grad():
            for j, batch in enumerate(val_loop):
                pixel_values = batch["pixel_values"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss

                val_loss_sum += loss.item() * pixel_values.size(0)
                preds = outputs.logits.argmax(dim=-1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                val_loop.set_description(f"Epoch [{epoch+1}/{args.epochs}]")
                val_loop.set_postfix(loss=loss.item(), acc=val_correct/max(1,val_total))
        val_loss = val_loss_sum / max(1, val_total)
        val_acc = val_correct / max(1, val_total)

        print(
            f"Epoch [{epoch + 1}/{args.epochs}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_model_path = best_path
            state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
            torch.save(state, best_path)

    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Best checkpoint saved to: {best_model_path}")
    return model, best_model_path


def main():
    cli_args = parse_args()
    from argparse import Namespace
    base_args = Namespace(
        config=cli_args.config,
        model=cli_args.model,
        seed=42,
        cpu=cli_args.cpu,
        checkpoint_path=None,
        wandb_new_id=False,
    )
    args = update_args(base_args, load_json(cli_args.config))
    args.hf_backbone = cli_args.hf_backbone
    if cli_args.batch_size is not None:
        args.batch_size = cli_args.batch_size
    if cli_args.epochs is not None:
        args.epochs = cli_args.epochs
    if cli_args.lr is not None:
        args.learning_rate = cli_args.lr
    device = torch.device("cpu") if (args.cpu or not torch.cuda.is_available()) else torch.device(f"cuda:{cli_args.gpu_id}")
    actions_dict = load_classes(os.path.join("data", args.dataset, "class.txt"))
    n_class = len(actions_dict) - len(args.excluded_classes)
    pad_idx = 255
    _, _, train_base, val_base, _ = get_datasets(args, pad_idx, n_class)
    num_frames = 16 if cli_args.hf_backbone == "videomae" else 8
    train_ds = SoccerNetBallHFWrapper(train_base, n_class, pad_idx, image_size=224, num_frames=num_frames)
    val_ds = SoccerNetBallHFWrapper(val_base, n_class, pad_idx, image_size=224, num_frames=num_frames)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    model = HFVideoMAEClassifier(num_classes=n_class) if cli_args.hf_backbone == "videomae" else HFTimesformerClassifier(num_classes=n_class)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=getattr(args, "learning_rate", args.lr), weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = None
    model_save_path = os.path.join(args.save_dir, "model", "transformer")
    train(args, model, train_loader, val_loader, optimizer, scheduler, criterion,
          model_save_path, pad_idx, device, int(args.pred_perc*args.clip_len),
          n_class, actions_dict, args.n_query)


if __name__ == "__main__":
    main()