"""
models.py — Arhitecturi fidele articolului MELECON 2026

1) VideoViViT — Video Vision Transformer (ViViT)
   - Tubelet embedding (spatio-temporal patches)
   - CLS token → clasificare
   - 16 Transformer encoder layers, 12 attention heads
   - Input: (B, C, T, H, W) = (B, 3, 64, 224, 224)

2) AudioTransformer — Transformer pe MFCC + Δ + ΔΔ
   - Linear Projection → Positional Encodings → Transformer Encoder → Mean Pooling
   - 8 layers, 8 attention heads
   - Input: (B, T, 120)

3) WeightedLateFusion — Fuziune cu ponderi learnable α_video, α_audio
"""
import math
import torch
import torch.nn as nn
from transformers import VivitModel, VivitConfig

from config import *


# ══════════════════════════════════════════════════════════════════════════════
# 1) VIDEO — ViViT (Video Vision Transformer)
# ══════════════════════════════════════════════════════════════════════════════

class VideoViViT(nn.Module):
    """
    ViViT fidel articolului (Sectiunea III.A, Fig. 1):

      Video frames → Tubelet Embedding → [CLS token] + Positional Embeddings
      → Transformer Encoder (16 layers) → CLS token → MLP Head → Softmax

    Foloseste HuggingFace VivitModel cu configuratie custom:
      - num_frames=64 (NF=64 din Tabelul I)
      - num_hidden_layers=16 (HL=16 din Tabelul I)
      - tubelet_size=[2, 16, 16] (temporal x spatial)
      - hidden_size=768, num_attention_heads=12
      - dropout=0.2

    Input: (batch, 3, 64, 224, 224) — format (C, T, H, W)
    Output: logits (batch, num_classes)
    """

    def __init__(self, num_classes=NUM_CLASSES, embed_dim=VIDEO_EMBED_DIM,
                 num_frames=VIDEO_NUM_FRAMES, num_layers=VIDEO_NUM_LAYERS,
                 hidden_size=VIDEO_HIDDEN_SIZE, num_heads=VIDEO_NUM_HEADS,
                 tubelet_size=None, dropout=VIDEO_DROPOUT,
                 pretrained=True):
        super().__init__()

        if tubelet_size is None:
            tubelet_size = VIDEO_TUBELET_SIZE

        # Configuratie ViViT custom (adaptat din articol)
        vivit_config = VivitConfig(
            image_size=VIDEO_RESIZE,              # 224
            num_frames=num_frames,                # 64
            tubelet_size=tubelet_size,            # [2, 16, 16]
            num_channels=3,
            hidden_size=hidden_size,              # 768
            num_hidden_layers=num_layers,         # 16
            num_attention_heads=num_heads,        # 12
            intermediate_size=hidden_size * 4,    # 3072
            hidden_dropout_prob=dropout,          # 0.2
            attention_probs_dropout_prob=dropout,  # 0.2
            initializer_range=0.02,
            qkv_bias=True,
        )

        if pretrained:
            # Incarca ViViT preantrenat (Kinetics-400), apoi rescrie config
            # Modelul preantrenat are 12 layere — adaugam 4 layere in plus
            try:
                print("[ViViT] Incarcarea modelului preantrenat google/vivit-b-16x2-kinetics400...")
                self.vivit = VivitModel.from_pretrained(
                    "google/vivit-b-16x2-kinetics400",
                    ignore_mismatched_sizes=True,
                )
                # Rescrie configuratia pentru 64 cadre si 16 layere
                # Nota: modelul preantrenat are 12 layere, 32 cadre
                # Vom reinitializa embeddings si layerele extra
                self._adapt_pretrained(vivit_config)
                print("[ViViT] Model preantrenat adaptat cu succes")
            except Exception as e:
                print(f"[ViViT] Warning: Nu pot incarca preantrenat ({e})")
                print("[ViViT] Initializare de la zero cu configuratia din articol")
                self.vivit = VivitModel(vivit_config)
        else:
            self.vivit = VivitModel(vivit_config)

        # Embedding projection (CLS token → embedding fix pentru fuziune)
        self.embed_proj = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # MLP Classification head (CLS token → clase)
        # Din articol: "output of encoder → MLP head → softmax"
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

        self._init_head_weights()

    def _adapt_pretrained(self, target_config):
        """
        Adapteaza modelul preantrenat (12 layere, 32 cadre)
        la configuratia din articol (16 layere, 64 cadre).
        """
        current_config = self.vivit.config

        # Daca numarul de layere e diferit, adaugam layere noi
        current_layers = len(self.vivit.encoder.layer)
        target_layers = target_config.num_hidden_layers

        if target_layers > current_layers:
            # Duplicam ultimele layere (transfer learning)
            import copy
            for i in range(target_layers - current_layers):
                new_layer = copy.deepcopy(self.vivit.encoder.layer[-1])
                self.vivit.encoder.layer.append(new_layer)
            print(f"  Adaugate {target_layers - current_layers} layere extra "
                  f"({current_layers} → {target_layers})")

        # Actualizam configuratia
        self.vivit.config.num_hidden_layers = target_layers
        self.vivit.config.num_frames = target_config.num_frames
        self.vivit.config.hidden_dropout_prob = target_config.hidden_dropout_prob
        self.vivit.config.attention_probs_dropout_prob = target_config.attention_probs_dropout_prob

        # Reinitializeaza positional embeddings pentru 64 cadre
        # Numarul de tubelets se schimba cu num_frames
        # ViViT: num_patches = (T/t_t) * (H/t_h) * (W/t_w)
        # Cu tubelet [2,16,16]: (64/2) * (224/16) * (224/16) = 32 * 14 * 14 = 6272
        # + 1 CLS token = 6273
        t_t, t_h, t_w = target_config.tubelet_size
        num_temporal = target_config.num_frames // t_t
        num_spatial_h = target_config.image_size // t_h
        num_spatial_w = target_config.image_size // t_w
        num_patches = num_temporal * num_spatial_h * num_spatial_w
        new_seq_len = num_patches + 1  # +1 pentru CLS token

        old_pos_emb = self.vivit.embeddings.position_embeddings
        old_seq_len = old_pos_emb.shape[1]

        if new_seq_len != old_seq_len:
            print(f"  Resize positional embeddings: {old_seq_len} → {new_seq_len}")
            # Interpolare a positional embeddings
            new_pos_emb = nn.Parameter(torch.zeros(1, new_seq_len, target_config.hidden_size))
            nn.init.trunc_normal_(new_pos_emb, std=0.02)

            # Copiaza CLS token embedding
            new_pos_emb.data[:, 0, :] = old_pos_emb.data[:, 0, :]

            # Interpoleaza restul
            if old_seq_len > 1 and new_seq_len > 1:
                old_patch_emb = old_pos_emb.data[:, 1:, :].permute(0, 2, 1)
                new_patch_emb = nn.functional.interpolate(
                    old_patch_emb, size=new_seq_len - 1, mode='linear', align_corners=False
                ).permute(0, 2, 1)
                new_pos_emb.data[:, 1:, :] = new_patch_emb

            self.vivit.embeddings.position_embeddings = new_pos_emb

        # Reinitializeaza tube embedding daca numarul de cadre e diferit
        # (tubelet conv3d adapteaza automat la orice numar de cadre)

    def _init_head_weights(self):
        for m in [self.embed_proj, self.classifier]:
            for layer in m.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def freeze_backbone(self):
        for p in self.vivit.parameters():
            p.requires_grad = False
        print("[ViViT] Backbone FROZEN")

    def unfreeze_backbone(self):
        for p in self.vivit.parameters():
            p.requires_grad = True
        print("[ViViT] Backbone UNFROZEN")

    def get_embedding(self, pixel_values):
        """
        Extrage embedding-ul CLS token (pentru fuziune).
        pixel_values: (batch, C, T, H, W) = (batch, 3, 64, 224, 224)
        Returneaza: (batch, embed_dim)
        """
        # HuggingFace VivitModel asteapta (batch, T, C, H, W)
        # Convertim din (B, C, T, H, W) → (B, T, C, H, W)
        x = pixel_values.permute(0, 2, 1, 3, 4)

        outputs = self.vivit(pixel_values=x)

        # CLS token e la pozitia 0 din last_hidden_state
        # Din articol: "By the final encoder layer, the CLS token carries
        # a compact but information-rich representation of the entire video"
        cls_output = outputs.last_hidden_state[:, 0, :]  # (batch, hidden_size)

        return self.embed_proj(cls_output)  # (batch, embed_dim)

    def forward(self, pixel_values):
        """
        pixel_values: (batch, C, T, H, W) = (batch, 3, 64, 224, 224)
        Returneaza: logits (batch, num_classes)
        """
        emb = self.get_embedding(pixel_values)
        return self.classifier(emb)


# ══════════════════════════════════════════════════════════════════════════════
# 2) AUDIO — Transformer pe MFCC + Δ + ΔΔ (Fig. 2 din articol)
# ══════════════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (Vaswani et al.)."""
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1), :])


class AudioTransformer(nn.Module):
    """
    Transformer encoder pentru clasificare audio (Fig. 2 din articol).

    Pipeline:
      MFCC + Δ + ΔΔ (120 dim) → Linear Projection → Positional Encodings
      → Transformer Encoder (8 layers, 8 heads) → Mean Pooling → MLP → Softmax

    Din articol: "Instead of treating the spectrogram as an image, we structured
    the features as a sequence of tabular vectors, each corresponding to a time
    frame. This allowed us to apply a Transformer directly on the temporal sequence."

    Input: (batch, T, 120)
    Output: logits (batch, num_classes)
    """

    def __init__(self, input_dim=120, num_classes=NUM_CLASSES,
                 d_model=AUDIO_D_MODEL, nhead=AUDIO_NHEAD,
                 num_layers=AUDIO_NUM_LAYERS, dim_ff=AUDIO_DIM_FF,
                 dropout=AUDIO_DROPOUT, embed_dim=AUDIO_EMBED_DIM):
        super().__init__()

        # Linear Projection (Fig. 2: "Linear Projection")
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Positional Encodings (Fig. 2)
        self.pos_encoder = PositionalEncoding(d_model, max_len=512, dropout=dropout)

        # Transformer Encoder (Fig. 2: multi-head self-attention + FFN)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.post_norm = nn.LayerNorm(d_model)

        # Embedding projection (pentru fuziune)
        self.embed_proj = nn.Sequential(
            nn.Linear(d_model, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # MLP Classification Head (Fig. 2: "MLP → Softmax")
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def get_embedding(self, x):
        """
        x: (batch, T, input_dim)
        Returneaza: (batch, embed_dim)
        """
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = self.post_norm(x)
        # Mean pooling (din articol: "aggregated into a single sequence-level
        # representation via mean pooling")
        x = x.mean(dim=1)
        return self.embed_proj(x)

    def forward(self, x):
        return self.classifier(self.get_embedding(x))


# ══════════════════════════════════════════════════════════════════════════════
# 3) FUSION — Weighted Late Fusion (Fig. 3 din articol)
# ══════════════════════════════════════════════════════════════════════════════

class WeightedLateFusion(nn.Module):
    """
    Weighted Late Fusion (Sectiunea III.A, Fig. 3):

      "two learnable scalar weights are trained alongside the model,
       one for the audio embedding and one for the video embedding.
       The weighted audio and video embeddings are then added together
       to create a single combined representation."

      combined = α_v * video_emb + α_a * audio_emb → MLP → Softmax
    """

    def __init__(self, video_embed_dim=VIDEO_EMBED_DIM,
                 audio_embed_dim=AUDIO_EMBED_DIM,
                 hidden_dim=FUSION_HIDDEN_DIM,
                 num_classes=NUM_CLASSES,
                 dropout=FUSION_DROPOUT):
        super().__init__()

        # Ponderi learnable (initializate la 1.0)
        self.alpha_video = nn.Parameter(torch.tensor(1.0))
        self.alpha_audio = nn.Parameter(torch.tensor(1.0))

        # Proiectie daca dimensiunile difera
        combined_dim = video_embed_dim
        self.video_proj = nn.Identity()
        self.audio_proj = nn.Identity()
        if video_embed_dim != audio_embed_dim:
            combined_dim = hidden_dim
            self.video_proj = nn.Linear(video_embed_dim, hidden_dim)
            self.audio_proj = nn.Linear(audio_embed_dim, hidden_dim)

        # MLP classification head cu softmax
        self.classifier = nn.Sequential(
            nn.LayerNorm(combined_dim),
            nn.Linear(combined_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, video_emb, audio_emb):
        v = self.video_proj(video_emb)
        a = self.audio_proj(audio_emb)
        w_v = torch.sigmoid(self.alpha_video)
        w_a = torch.sigmoid(self.alpha_audio)
        combined = w_v * v + w_a * a
        return self.classifier(combined)

    def get_weights(self):
        w_v = torch.sigmoid(self.alpha_video).item()
        w_a = torch.sigmoid(self.alpha_audio).item()
        total = w_v + w_a
        return {
            'video_weight': w_v / total,
            'audio_weight': w_a / total,
            'raw_video': w_v,
            'raw_audio': w_a,
        }


