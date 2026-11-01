
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultimodalDocumentRouter(nn.Module):
    """
    Visual Document Router with position-aware attention pooling.
    
    Compatible with frozen vision foundation encoders (e.g. DINOv3 ViT-L/16).
    """
    def __init__(
        self,
        visual_dim: int = 1024,
        text_dim: int = 2048,
        pdf_dim: int = 14,
        use_visual: bool = True,
        use_text: bool = False,
        use_pdf: bool = False,
        pooling_mode: str = "hybrid",
        dropout: float = 0.25,
        is_regression: bool = False
    ):
        super().__init__()
        self.use_visual = use_visual
        self.use_text = use_text
        self.use_pdf = use_pdf
        self.pooling_mode = pooling_mode
        self.is_regression = is_regression
        
        total_feature_dim = 0
        
        # 1. Visual Page Sequence Processing
        if use_visual:
            self.pos_proj = nn.Sequential(
                nn.Linear(4, 32),
                nn.GELU()
            )
            self.page_proj = nn.Sequential(
                nn.Linear(visual_dim + 32, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            self.attn_net = nn.Sequential(
                nn.Linear(256, 128),
                nn.Tanh(),
                nn.Linear(128, 1, bias=False)
            )
            total_feature_dim += (256 * 3 if pooling_mode == "hybrid" else 256)
            
        # 2. Text Representation Processing (Optional)
        if use_text:
            self.text_norm = nn.LayerNorm(text_dim)
            self.text_proj = nn.Sequential(
                nn.Linear(text_dim, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            total_feature_dim += 256
            
        # 3. Structural PDF Features (Optional)
        if use_pdf:
            self.pdf_norm = nn.BatchNorm1d(pdf_dim)
            self.pdf_proj = nn.Sequential(
                nn.Linear(pdf_dim, 64),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            total_feature_dim += 64
            
        # 4. Classification / Routing Head
        out_dim = 1
        self.classifier = nn.Sequential(
            nn.Linear(total_feature_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, out_dim)
        )

    def forward(
        self,
        pages: Optional[torch.Tensor] = None,       # [B, N, visual_dim]
        positions: Optional[torch.Tensor] = None,   # [B, N, 4]
        mask: Optional[torch.Tensor] = None,        # [B, N] bool
        text_emb: Optional[torch.Tensor] = None,    # [B, text_dim]
        pdf_feat: Optional[torch.Tensor] = None     # [B, pdf_dim]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        Forward pass.
        
        Args:
            pages: Batch of page feature sequences [B, N, visual_dim]
            positions: Page positional features [B, N, 4]
            mask: Valid page mask [B, N] (True for valid pages, False for padding)
            text_emb: Optional document text embeddings [B, text_dim]
            pdf_feat: Optional document structural features [B, pdf_dim]
            
        Returns:
            out: Logits tensor [B, 1]
            attn_weights: Attention distribution across pages [B, N]
            entropy: Attention entropy regularization scalar
        """
        reps = []
        attn_weights = None
        entropy = torch.tensor(0.0, device=pages.device if pages is not None else "cpu")
        
        if self.use_visual and pages is not None:
            B, N, _ = pages.size()
            if mask is None:
                mask = torch.ones((B, N), dtype=torch.bool, device=pages.device)
            if positions is None:
                positions = torch.zeros((B, N, 4), dtype=torch.float32, device=pages.device)
                
            pos_emb = self.pos_proj(positions)
            page_inputs = torch.cat([pages, pos_emb], dim=-1)
            projected = self.page_proj(page_inputs)
            
            scores = self.attn_net(projected).squeeze(-1)
            scores = scores.masked_fill(~mask, -1e9)
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
            
            v_attn = torch.bmm(attn_weights.unsqueeze(1), projected).squeeze(1)
            
            if self.pooling_mode == "hybrid":
                mask_exp = mask.unsqueeze(-1).float()
                v_mean = (projected * mask_exp).sum(dim=1) / mask.sum(dim=-1, keepdim=True).clamp(min=1).float()
                v_max = projected.masked_fill(~mask.unsqueeze(-1), -1e9).max(dim=1)[0]
                doc_vis = torch.cat([v_attn, v_mean, v_max], dim=-1)
            else:
                doc_vis = v_attn
                
            reps.append(doc_vis)
            eps = 1e-8
            entropy = - (attn_weights * torch.log(attn_weights + eps)).sum(dim=-1).mean()
            
        if self.use_text and text_emb is not None:
            normed_text = self.text_norm(text_emb)
            doc_text = self.text_proj(normed_text)
            reps.append(doc_text)
            
        if self.use_pdf and pdf_feat is not None:
            normed_pdf = self.pdf_norm(pdf_feat)
            doc_pdf = self.pdf_proj(normed_pdf)
            reps.append(doc_pdf)
            
        combined = torch.cat(reps, dim=-1)
        out = self.classifier(combined)
        return out, attn_weights, entropy
