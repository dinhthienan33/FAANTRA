import torch
from torch import nn

try:
    from transformers import (
        VideoMAEForVideoClassification,
        TimesformerForVideoClassification,
    )
except ImportError as e:
    raise ImportError(
        "transformers is required for HFVideoMAEClassifier and HFTimesformerClassifier. "
        "Please install it with `pip install transformers`."
    ) from e


class HFVideoMAEClassifier(nn.Module):

    def __init__(
        self,
        num_classes: int,
        pretrained_model_name: str = "MCG-NJU/videomae-base-finetuned-kinetics",
    ) -> None:
        super().__init__()
        self.model = VideoMAEForVideoClassification.from_pretrained(
            pretrained_model_name
        )
        hidden_size = self.model.config.hidden_size
        self.model.classifier = nn.Linear(hidden_size, num_classes)

    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: torch.Tensor | None = None,
    ):
        """
        Forward pass.

        Args:
            pixel_values: Tensor[B, T, C, H, W]
            labels: optional Tensor[B] with class indices

        Returns:
            transformers.modeling_outputs.VideoClassifierOutput
            (includes `logits` and optionally `loss` if labels are provided).
        """
        return self.model(pixel_values=pixel_values, labels=labels)


class HFTimesformerClassifier(nn.Module):
    """
    Wrapper around `facebook/timesformer-base-finetuned-k400` for fine-tuning
    on custom video classification datasets.

    Expected input:
        pixel_values: Tensor[B, T, C, H, W], float, preprocessed to match
        TimeSformer expectations (e.g. resized to 224x224 and normalized).

    The classifier head is replaced so that the number of output classes
    matches `num_classes`.
    """

    def __init__(
        self,
        num_classes: int,
        pretrained_model_name: str = "facebook/timesformer-base-finetuned-k400",
    ) -> None:
        super().__init__()
        self.model = TimesformerForVideoClassification.from_pretrained(
            pretrained_model_name
        )
        hidden_size = self.model.config.hidden_size
        self.model.classifier = nn.Linear(hidden_size, num_classes)

    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: torch.Tensor | None = None,
    ):
        """
        Forward pass.

        Args:
            pixel_values: Tensor[B, T, C, H, W]
            labels: optional Tensor[B] with class indices

        Returns:
            transformers.modeling_outputs.VideoClassifierOutput
            (includes `logits` and optionally `loss` if labels are provided).
        """
        return self.model(pixel_values=pixel_values, labels=labels)

