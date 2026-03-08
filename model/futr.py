import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math
import os
import sys
import pdb
import torchvision.transforms as T
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from einops import repeat, rearrange
from model.extras.transformer import Transformer
from model.extras.position import PositionalEncoding
import timm
from model.T_Deed_Modules.modules import EDSGPMIXERLayers
from model.T_Deed_Modules.shift import make_temporal_shift

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))


# ====================================================
# Phase 2: Backbone registry — maps config string to
# (timm_model_name, head_attr_path) for easy extension
# ====================================================
TIMM_BACKBONE_REGISTRY = {
    # Original RegNetY family
    'rny002': 'regnety_002',
    'rny004': 'regnety_004',
    'rny006': 'regnety_006',
    'rny008': 'regnety_008',
    # Phase 2: Larger RegNetY
    'rny016': 'regnety_016',
    'rny032': 'regnety_032',
    'rny040': 'regnety_040',
    'rny064': 'regnety_064',
    'rny080': 'regnety_080',
    'rny120': 'regnety_120',
    'rny160': 'regnety_160',
    # Phase 2: EfficientNetV2 family
    'efficientnetv2_s': 'efficientnetv2_s',
    'efficientnetv2_m': 'efficientnetv2_m',
    'efficientnetv2_rw_s': 'efficientnetv2_rw_s',
    # Phase 2: Swin Transformer (2D, used per-frame)
    'swin_tiny': 'swin_tiny_patch4_window7_224',
    'swin_small': 'swin_small_patch4_window7_224',
    'swin_base': 'swin_base_patch4_window7_224',
    'swin_base_22k': 'swin_base_patch4_window7_224_in22k',
}

# Backbones that support GSM/GSF temporal shift (only RegNet)
GSF_COMPATIBLE_BACKBONES = {
    'rny002', 'rny004', 'rny006', 'rny008',
    'rny016', 'rny032', 'rny040', 'rny064', 'rny080', 'rny120', 'rny160',
}


def _get_base_arch(feature_arch):
    """Strip _gsm/_gsf suffix to get base architecture name."""
    for suffix in ('_gsm', '_gsf'):
        if feature_arch.endswith(suffix):
            return feature_arch[:-len(suffix)]
    return feature_arch


def _build_timm_backbone(model_name, pretrained=True):
    """Create a timm model and remove its classification head.

    Returns:
        (nn.Module, int): backbone module and its output feature dimension
    """
    model = timm.create_model(model_name, pretrained=pretrained)
    feat_dim = model.num_features
    # Reset classifier to Identity — timm's universal API
    model.reset_classifier(0)
    return model, feat_dim


def _build_videomae_backbone(args):
    """Build a VideoMAE backbone from HuggingFace transformers.

    VideoMAE processes clips of frames (not single frames), so we wrap it
    to output per-frame features via temporal interpolation.

    Returns:
        (nn.Module, int): wrapper module and its output feature dimension
    """
    from transformers import VideoMAEModel

    videomae_model_name = getattr(args, 'videomae_model_name',
                                   'MCG-NJU/videomae-base-finetuned-kinetics')
    videomae_pool = getattr(args, 'videomae_pool', 'mean')
    model = VideoMAEModel.from_pretrained(videomae_model_name)
    feat_dim = model.config.hidden_size  # 768 for base, 1024 for large
    num_frames = model.config.num_frames  # typically 16
    wrapper = VideoMAEBackboneWrapper(model, num_frames=num_frames, pool=videomae_pool)
    return wrapper, feat_dim


class VideoMAEBackboneWrapper(nn.Module):
    """Wraps HuggingFace VideoMAEModel to work as a per-frame feature extractor.

    VideoMAE expects input [B, T_vmae, C, H, W] where T_vmae is a fixed number
    of frames (typically 16). The observation window S may differ, so we:
    1. Sample S frames into chunks of T_vmae with overlap/stride
    2. Run VideoMAE on each chunk
    3. Interpolate back to S frames

    For simplicity, when S <= T_vmae, we pad; when S > T_vmae, we use
    sliding window with stride and interpolate.
    """

    def __init__(self, model, num_frames=16, pool='mean'):
        super().__init__()
        self.model = model
        self.num_frames = num_frames
        self.pool = pool  # 'mean' or 'cls'
        self.hidden_size = model.config.hidden_size

    def _extract_single_clip(self, pixel_values):
        """Run VideoMAE on a single clip [B, T, C, H, W] -> [B, T, D]."""
        outputs = self.model(pixel_values=pixel_values)
        # last_hidden_state: [B, num_patches, D]
        # For ViT, num_patches = (T/tubelet_size) * (H/patch_size) * (W/patch_size)
        hidden = outputs.last_hidden_state  # [B, N, D]

        tubelet_size = self.model.config.tubelet_size
        t_tokens = self.num_frames // tubelet_size
        spatial_tokens = hidden.shape[1] // t_tokens

        # Reshape to [B, t_tokens, spatial_tokens, D]
        hidden = hidden.reshape(hidden.shape[0], t_tokens, spatial_tokens, self.hidden_size)

        if self.pool == 'mean':
            # Mean pool over spatial dimension -> [B, t_tokens, D]
            frame_features = hidden.mean(dim=2)
        else:  # 'cls' — but VideoMAE doesn't have CLS, so fallback to mean
            frame_features = hidden.mean(dim=2)

        return frame_features  # [B, t_tokens, D]

    def forward(self, frames):
        """Process observation frames through VideoMAE.

        Args:
            frames: [B, S, C, H, W] — all observation frames for one batch

        Returns:
            [B, S, D] — per-frame features interpolated to match input length
        """
        B, S, C, H, W = frames.shape
        T = self.num_frames

        if S <= T:
            # Pad to T frames by repeating last frame
            if S < T:
                pad = frames[:, -1:].expand(-1, T - S, -1, -1, -1)
                clip = torch.cat([frames, pad], dim=1)
            else:
                clip = frames
            feat = self._extract_single_clip(clip)  # [B, t_tokens, D]
            # Interpolate to S frames
            feat = feat.permute(0, 2, 1)  # [B, D, t_tokens]
            feat = F.interpolate(feat, size=S, mode='linear', align_corners=False)
            feat = feat.permute(0, 2, 1)  # [B, S, D]
            return feat
        else:
            # Sliding window: stride = T//2 for overlap
            stride = max(T // 2, 1)
            all_feats = []
            positions = []
            for start in range(0, S, stride):
                end = min(start + T, S)
                if end - start < T:
                    start = max(0, end - T)
                clip = frames[:, start:start + T]
                feat = self._extract_single_clip(clip)  # [B, t_tokens, D]
                # Interpolate chunk features to T positions
                feat = feat.permute(0, 2, 1)
                feat = F.interpolate(feat, size=T, mode='linear', align_corners=False)
                feat = feat.permute(0, 2, 1)  # [B, T, D]
                all_feats.append(feat)
                positions.append((start, start + T))
                if start + T >= S:
                    break

            # Merge overlapping windows by averaging
            output = torch.zeros(B, S, self.hidden_size, device=frames.device, dtype=feat.dtype)
            counts = torch.zeros(B, S, 1, device=frames.device, dtype=feat.dtype)
            for feat, (start, end) in zip(all_feats, positions):
                length = end - start
                output[:, start:end] += feat[:, :length]
                counts[:, start:end] += 1.0
            output = output / counts.clamp(min=1.0)
            return output  # [B, S, D]


class PreextractedBackbone(nn.Module):
    """Dummy backbone for pre-extracted features mode.

    When using pre-extracted features (e.g., from InternVideo2), the dataset
    returns features directly instead of raw frames. This module is a
    pass-through that optionally projects the feature dimension.
    """

    def __init__(self, input_feat_dim, output_feat_dim=None):
        super().__init__()
        self.input_feat_dim = input_feat_dim
        self.output_feat_dim = output_feat_dim or input_feat_dim
        if self.input_feat_dim != self.output_feat_dim:
            self.proj = nn.Linear(input_feat_dim, output_feat_dim)
        else:
            self.proj = nn.Identity()

    def forward(self, x):
        """x: [B*S, D] or [B, S, D] -> same shape with optional projection."""
        return self.proj(x)


class FUTR(nn.Module):

    def __init__(self, n_class, hidden_dim, src_pad_idx, device, args, n_query=8, n_head=8,
                 num_encoder_layers=6, num_decoder_layers=6, src_attn_mask=None, tgt_attn_mask=None):
        super().__init__()
        if num_decoder_layers < 1 and n_query > 1:
            raise ValueError(f"n_query must be 1 if no decoder is to be used\nGiven values are: {n_query} and {num_decoder_layers} respectively")
        self.encoder_only = num_decoder_layers == 0
        self.src_pad_idx = src_pad_idx
        self.device = device
        self.hidden_dim = hidden_dim
        self.feature_arch = args.feature_arch
        self.temp_arch = args.temporal_arch
        self.src_attn_mask = src_attn_mask
        self.tgt_attn_mask = tgt_attn_mask
        self.jointtrain_available = args.jointtrain is not None

        # Phase 2: Gradient checkpointing flag
        self.use_gradient_checkpointing = getattr(args, 'gradient_checkpointing', False)

        # Phase 2: Pre-extracted features mode
        self.use_preextracted = getattr(args, 'use_preextracted_features', False)

        # Phase 2: Backbone freezing
        self.freeze_backbone = getattr(args, 'freeze_backbone', False)
        self.unfreeze_backbone_epoch = getattr(args, 'unfreeze_backbone_epoch', -1)
        # Track whether backbone is video-based (processes clips) vs image-based (per-frame)
        self.is_video_backbone = False

        # ====================================================
        # Phase 2: Expanded backbone initialization
        # ====================================================
        base_arch = _get_base_arch(self.feature_arch)

        if self.use_preextracted:
            # Pre-extracted features: no visual backbone needed
            preextracted_dim = getattr(args, 'preextracted_feat_dim', 768)
            self.features = PreextractedBackbone(preextracted_dim)
            self.input_dim = preextracted_dim
            print(f'=> Using pre-extracted features (dim={preextracted_dim})')

        elif base_arch.startswith('videomae'):
            # Phase 2: VideoMAE backbone (HuggingFace)
            self.features, feat_dim = _build_videomae_backbone(args)
            self.input_dim = feat_dim
            self.is_video_backbone = True
            print(f'=> Using VideoMAE backbone (dim={feat_dim})')

        elif base_arch in TIMM_BACKBONE_REGISTRY:
            # Timm-based backbones (RegNet, ConvNeXt, EfficientNetV2, Swin, etc.)
            timm_name = TIMM_BACKBONE_REGISTRY[base_arch]
            self.features, feat_dim = _build_timm_backbone(timm_name, pretrained=True)
            self.input_dim = feat_dim
            print(f'=> Using timm backbone: {timm_name} (dim={feat_dim})')
        else:
            raise NotImplementedError(
                f"Unknown feature_arch: '{args.feature_arch}'. "
                f"Supported: {list(TIMM_BACKBONE_REGISTRY.keys())} + videomae_* + preextracted"
            )

        # Add Temporal Shift Modules (only for compatible backbones)
        # NOTE: NEED TO CHANGE 2ND ARGUMENT FOR CHEATING DATASET
        max_obs_len = int(args.clip_len*args.cheating_range[1])-int(args.clip_len*args.cheating_range[0]) if args.cheating_dataset else int(args.clip_len*max(args.obs_perc))
        if base_arch in GSF_COMPATIBLE_BACKBONES:
            if self.feature_arch.endswith('_gsm'):
                make_temporal_shift(self.features, max_obs_len, mode='gsm')
            elif self.feature_arch.endswith('_gsf'):
                make_temporal_shift(self.features, max_obs_len, mode='gsf')
        elif self.feature_arch.endswith(('_gsm', '_gsf')):
            print(f'WARNING: GSM/GSF not supported for {base_arch}, ignoring temporal shift suffix')

        # Phase 2: Apply backbone freezing if requested
        if self.freeze_backbone and not self.use_preextracted:
            self._freeze_backbone()
            print(f'=> Backbone frozen (will unfreeze at epoch {self.unfreeze_backbone_epoch})')

        if self.temp_arch == 'ed_sgp_mixer':
            #Positional encoding
            self.temp_enc = nn.Parameter(torch.normal(mean = 0, std = 1 / max_obs_len, size = (max_obs_len, self.input_dim)))
            self.temp_fine = EDSGPMIXERLayers(self.input_dim, max_obs_len, num_layers=args.n_layers, ks = args.sgp_ks, k = args.sgp_r, concat = True)

        self.input_embed = nn.Linear(self.input_dim, hidden_dim)
        self.transformer = Transformer(hidden_dim, n_head, num_encoder_layers, num_decoder_layers,
                                        hidden_dim*4, normalize_before=False)
        self.n_query = n_query
        self.args = args
        nn.init.xavier_uniform_(self.input_embed.weight)
        self.query_embed = nn.Embedding(self.n_query, hidden_dim)


        if args.seg :
            self.fc_seg = nn.Linear(hidden_dim, n_class)
            nn.init.xavier_uniform_(self.fc_seg.weight)
            if self.jointtrain_available:
                self.fc_seg_jointtrain = nn.Linear(hidden_dim, args.jointtrain['num_classes'] + 1)  # +1 for background class
                nn.init.xavier_uniform_(self.fc_seg_jointtrain.weight)

        if args.anticipate :
            # Anticipation head has the capacity to predict background class despite it not being anywhere in anticipation
            # To avoid this I will make the EOS token the same number as the background class
            self.fc = nn.Linear(hidden_dim, n_class - 1*args.actionness)
            nn.init.xavier_uniform_(self.fc.weight)
            self.fc_len = nn.Linear(hidden_dim, 1)
            nn.init.xavier_uniform_(self.fc_len.weight)
            if self.jointtrain_available:
                self.fc_jointtrain = nn.Linear(hidden_dim, args.jointtrain['num_classes'] + 1 - 1*args.actionness)
                nn.init.xavier_uniform_(self.fc_jointtrain.weight)
                self.fc_len_jointtrain = nn.Linear(hidden_dim, 1)
                nn.init.xavier_uniform_(self.fc_len_jointtrain.weight)
        
        if args.actionness :
            self.fc_actionness = nn.Linear(hidden_dim, 1)
            nn.init.xavier_uniform_(self.fc_actionness.weight)
            if self.jointtrain_available:
                self.fc_actionness_jointtrain = nn.Linear(hidden_dim, 1)
                nn.init.xavier_uniform_(self.fc_actionness_jointtrain.weight)

        if args.pos_emb:
            #pos embedding
            max_seq_len = args.max_pos_len
            self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, hidden_dim))
            nn.init.xavier_uniform_(self.pos_embedding)
            # Sinusoidal position encoding
            self.pos_enc = PositionalEncoding(hidden_dim)

        # Preprocessing
        # Augmentations
        base_augs = [
            T.RandomApply([T.ColorJitter(hue = 0.2)], p = 0.25),
            T.RandomApply([T.ColorJitter(saturation = (0.7, 1.2))], p = 0.25),
            T.RandomApply([T.ColorJitter(brightness = (0.7, 1.2))], p = 0.25),
            T.RandomApply([T.ColorJitter(contrast = (0.7, 1.2))], p = 0.25),
            T.RandomApply([T.GaussianBlur(5)], p = 0.25),
            T.RandomHorizontalFlip(),
        ]
        # Phase 1: Extended augmentations for stronger regularization
        if getattr(args, 'extended_augmentation', False):
            base_augs.extend([
                T.RandomApply([T.RandomErasing(p=1.0, scale=(0.02, 0.15), ratio=(0.3, 3.3))], p=0.2),
                T.RandomApply([T.RandomPerspective(distortion_scale=0.1)], p=0.15),
            ])
        self.augmentation = T.Compose(base_augs)
        #Standarization
        self.standarization = T.Compose([
            T.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)) #Imagenet mean and std
        ])
        
    # ====================================================
    # Phase 2: Backbone freeze/unfreeze for fine-tuning
    # ====================================================
    def _freeze_backbone(self):
        """Freeze all backbone parameters."""
        for param in self.features.parameters():
            param.requires_grad = False

    def _unfreeze_backbone(self):
        """Unfreeze all backbone parameters."""
        for param in self.features.parameters():
            param.requires_grad = True

    def maybe_unfreeze_backbone(self, epoch):
        """Called at the start of each epoch to check if backbone should be unfrozen.

        Implements progressive unfreezing: backbone starts frozen, then gets
        unfrozen at a specified epoch so the rest of the model can warm up first.
        """
        if self.freeze_backbone and self.unfreeze_backbone_epoch >= 0 and epoch >= self.unfreeze_backbone_epoch:
            self._unfreeze_backbone()
            self.freeze_backbone = False
            print(f'=> Backbone unfrozen at epoch {epoch}')
            return True
        return False

    def augment(self, x):
            for i in range(x.shape[0]):
                x[i] = self.augmentation(x[i])
            return x

    def standarize(self, x):
        for i in range(x.shape[0]):
            x[i] = self.standarization(x[i])
        return x

    def _extract_features_image_backbone(self, src):
        """Extract features using a per-frame (2D image) backbone.

        Args:
            src: [B, S, C, H, W] preprocessed frames
        Returns:
            [B, S, input_dim] features
        """
        B, S, C, H, W = src.size()
        flat = src.view(-1, C, H, W)  # [B*S, C, H, W]
        if self.use_gradient_checkpointing and self.training:
            feat = grad_checkpoint(self.features, flat, use_reentrant=False)
        else:
            feat = self.features(flat)
        return feat.reshape(B, S, self.input_dim)

    def _extract_features_video_backbone(self, src):
        """Extract features using a video backbone (VideoMAE).

        Args:
            src: [B, S, C, H, W] preprocessed frames
        Returns:
            [B, S, input_dim] features
        """
        if self.use_gradient_checkpointing and self.training:
            return grad_checkpoint(self.features, src, use_reentrant=False)
        else:
            return self.features(src)

    def forward(self, inputs, mode='train'):
        if mode == 'train' :
            src, src_label = inputs
            tgt_key_padding_mask = None
            src_key_padding_mask = get_pad_mask(src_label, self.src_pad_idx).to(self.device)
            memory_key_padding_mask = src_key_padding_mask.clone().to(self.device)
        else :
            src = inputs
            src_key_padding_mask = None
            memory_key_padding_mask = None
            tgt_key_padding_mask = None

        src_mask = self.src_attn_mask
        tgt_mask = self.tgt_attn_mask

        # ====================================================
        # Phase 2: Handle both raw frames and pre-extracted features
        # ====================================================
        if self.use_preextracted:
            # Pre-extracted features: src is [B, S, D], skip visual processing
            B, S = src.shape[0], src.shape[1]
            src = self.features(src)  # optional projection
        else:
            B, S, C, H, W = src.size()
            src = src/255.0         # Normalize
            if mode == "train":
                src = self.augment(src) #augmentation per-batch
            src = self.standarize(src) #standarization imagenet stats
            # Phase 2: Route to appropriate backbone type
            if self.is_video_backbone:
                src = self._extract_features_video_backbone(src)
            else:
                src = self._extract_features_image_backbone(src)

        if self.temp_arch == 'ed_sgp_mixer':
            src = src + self.temp_enc.expand(B, -1, -1)
            src = self.temp_fine(src)
        src = self.input_embed(src) #[B, S, C]
        src = F.relu(src)

        # action query embedding
        action_query = self.query_embed.weight
        action_query = action_query.unsqueeze(0).repeat(B, 1, 1)
        tgt = torch.zeros_like(action_query)

        # pos embedding
        if self.encoder_only:
            pos = self.pos_embedding[:, :S+1,].repeat(B, 1, 1)
            if src_key_padding_mask is not None:
                false_append = torch.tensor([False], device=src_key_padding_mask.device).expand((src_key_padding_mask.shape[0], 1))
                src_key_padding_mask = torch.cat((src_key_padding_mask, false_append),dim=1)
        else:
            pos = self.pos_embedding[:, :S,].repeat(B, 1, 1)
        src = rearrange(src, 'b t c -> t b c')
        tgt = rearrange(tgt, 'b t c -> t b c')
        pos = rearrange(pos, 'b t c -> t b c')
        action_query = rearrange(action_query, 'b t c -> t b c')
        src, tgt = self.transformer(src, tgt, src_key_padding_mask, src_mask, tgt_mask, None, action_query, pos, None)

        tgt = rearrange(tgt, 't b c -> b t c')
        src = rearrange(src, 't b c -> b t c')

        output = dict()
        if self.args.anticipate :
            # action anticipation
            output_class = self.fc(tgt) #[T, B, C]  Note: I actually think this is [B, T, C]
            offset = self.fc_len(tgt) #[B, T, 1]
            offset = offset.squeeze(2) #[B, T]
            if self.jointtrain_available:
                output_class_jointtrain = self.fc_jointtrain(tgt)
                offset_jointtrain = self.fc_len_jointtrain(tgt)
                offset_jointtrain = offset_jointtrain.squeeze(2)
                output_class = torch.cat([output_class, output_class_jointtrain], dim = 2)
                offset = torch.cat([offset, offset_jointtrain], dim = 1)
            output['offset'] = offset
            output['action'] = output_class

        if self.args.seg :
            # action segmentation
            tgt_seg = self.fc_seg(src)
            if self.jointtrain_available:
                tgt_seg_jointtrain = self.fc_seg_jointtrain(src)
                tgt_seg = torch.cat([tgt_seg, tgt_seg_jointtrain], dim = 2)
            output['seg'] = tgt_seg
        
        if self.args.actionness :
            # actionness
            actionness = self.fc_actionness(tgt)    #[B, T, 1]
            actionness = actionness.squeeze(2)      #[B, T]
            if self.jointtrain_available:
                actionness_jointtrain = self.fc_actionness_jointtrain(tgt)
                actionness_jointtrain = actionness_jointtrain.squeeze(2)      #[B, T]
                actionness = torch.cat([actionness, actionness_jointtrain], dim = 1)
            output['actionness'] = actionness

        return output


def get_pad_mask(seq, pad_idx):
    return (seq ==pad_idx)