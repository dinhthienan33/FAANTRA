import os
import argparse

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T

from opts import update_args
from util.io import load_json
from util.dataset import load_classes
from dataset.datasets import get_datasets
from model.hf_video_models import HFVideoMAEClassifier, HFTimesformerClassifier


class SoccerNetBallHFWrapper(Dataset):

    def __init__(
        self,
        base_dataset,
        num_classes: int,
        label_pad_idx: int,
        image_size: int = 224,
    ) -> None:
        super().__init__()
        self.base_dataset = base_dataset
        self.num_classes = num_classes
        self.label_pad_idx = label_pad_idx

        self.resize = T.Resize((image_size, image_size))
        self.normalize = T.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )

    def __len__(self) -> int:
        return len(self.base_dataset)

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

        item = self.base_dataset[0]

        frames = item["frames"]  # [T, C, H, W]
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


def main():
    cli_args = parse_args()

    config = load_json(cli_args.config)

    from argparse import Namespace

    base_args = Namespace(
        config=cli_args.config,
        model=cli_args.model,
        seed=42,
        cpu=cli_args.cpu,
        checkpoint_path=None,
        wandb_new_id=False,
    )
    args = update_args(base_args, config)

    if cli_args.batch_size is not None:
        args.batch_size = cli_args.batch_size
    if cli_args.epochs is not None:
        args.epochs = cli_args.epochs
    if cli_args.lr is not None:
        args.learning_rate = cli_args.lr

    if args.cpu or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{cli_args.gpu_id}")
        torch.cuda.set_device(device)
    print(f"Using device: {device}")

    # Classes và số lớp (như main.py)
    actions_dict = load_classes(os.path.join("data", args.dataset, "class.txt"))
    n_class = len(actions_dict) - len(args.excluded_classes)
    print("Number of classes (including BACKGROUND=0):", n_class)
    pad_idx = 255

    # Dataset gốc (clips + labels) giống pipeline FAANTRA
    classes, _, train_base, val_base, _ = get_datasets(args, pad_idx, n_class)
    if args.store_mode == "store":
        print("Datasets đang ở chế độ 'store'. Hãy chạy main.py một lần với store_mode='store' "
              "hoặc chỉnh config sang 'load' sau khi đã store xong.")
        return

    train_ds = SoccerNetBallHFWrapper(train_base, n_class, pad_idx, image_size=224)
    val_ds = SoccerNetBallHFWrapper(val_base, n_class, pad_idx, image_size=224)

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

    # Chọn backbone HF
    if cli_args.hf_backbone == "videomae":
        model = HFVideoMAEClassifier(num_classes=n_class)
    else:
        model = HFTimesformerClassifier(num_classes=n_class)
    model.to(device)

    lr = getattr(args, "learning_rate", 1e-4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)

    # Đơn giản: giảm LR theo cosine theo số epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    save_dir = os.path.join(args.save_dir, args.model, f"hf_{cli_args.hf_backbone}")
    os.makedirs(save_dir, exist_ok=True)
    best_val_acc = 0.0
    best_path = os.path.join(save_dir, "best.ckpt")

    for epoch in range(args.epochs):
        ####################
        # Train
        ####################
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch in train_loader:
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
        with torch.no_grad():
            for batch in val_loader:
                pixel_values = batch["pixel_values"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss

                val_loss_sum += loss.item() * pixel_values.size(0)
                preds = outputs.logits.argmax(dim=-1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss = val_loss_sum / max(1, val_total)
        val_acc = val_correct / max(1, val_total)

        print(
            f"Epoch [{epoch + 1}/{args.epochs}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        # Lưu best checkpoint
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "val_acc": val_acc,
                    "config": config,
                    "hf_backbone": cli_args.hf_backbone,
                },
                best_path,
            )

    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Best checkpoint saved to: {best_path}")


if __name__ == "__main__":
    main()