"""
Eve-2-MoE — Custom Mixture of Experts Language Model
=====================================================
Architecture: DeepSeek-V3 style Shared Expert + Top-K Routed Experts + RoPE
Author: Anthony Maio / Making Minds AI Research
License: MIT

Usage (HuggingFace):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        "anthonym21/Eve-2-MoE-272M", trust_remote_code=True
    )

Usage (standalone):
    from modeling_eve import ModelConfig, DeepSeekMoE
    model = DeepSeekMoE(ModelConfig())
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import OrderedDict
from dataclasses import dataclass


# ============================================================
#  Standalone config (no transformers dependency)
# ============================================================

@dataclass
class ModelConfig:
    """Configuration for Eve-2-MoE (standalone, no HF dependency)."""

    # Model dimensions
    vocab_size: int = 50304
    n_layer: int = 12
    n_embd: int = 512
    n_head: int = 8
    head_dim: int = 64
    block_size: int = 2048

    # MoE settings
    num_experts: int = 8
    top_k: int = 2
    expert_intermediate_size: int = 1408
    shared_expert_intermediate_size: int = 1408
    router_aux_loss_coef: float = 0.01

    # Training settings
    use_checkpointing: bool = False  # Gradient checkpointing (saves VRAM, costs speed)

    # RoPE settings
    rope_theta: float = 10000.0


# ============================================================
#  Utility: strip torch.compile prefix from state dicts
# ============================================================

def _strip_orig_mod_prefix(state_dict):
    """Remove '_orig_mod.' prefix from keys saved by torch.compile'd models."""
    cleaned = OrderedDict()
    for k, v in state_dict.items():
        cleaned[k.replace("_orig_mod.", "")] = v
    return cleaned


# ============================================================
#  Building blocks (shared by standalone and HF models)
# ============================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def precompute_rope_freqs(head_dim: int, max_seq_len: int, theta: float = 10000.0,
                          device: torch.device = None) -> torch.Tensor:
    """Precompute the complex exponential frequencies for RoPE.

    Returns a (max_seq_len, head_dim // 2) complex tensor.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(max_seq_len, device=device).float()
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embeddings to input tensor.

    Args:
        x: (B, n_head, T, head_dim)
        freqs_cis: (T, head_dim // 2) complex
    Returns:
        (B, n_head, T, head_dim) with rotary embeddings applied
    """
    # Reshape x to complex: (B, n_head, T, head_dim//2, 2) -> complex
    B, H, T, D = x.shape
    x_complex = torch.view_as_complex(x.float().reshape(B, H, T, D // 2, 2))
    # Broadcast freqs_cis: (1, 1, T, head_dim//2)
    freqs_cis = freqs_cis[:T].unsqueeze(0).unsqueeze(0)
    x_rotated = x_complex * freqs_cis
    # Back to real: (B, H, T, head_dim)
    return torch.view_as_real(x_rotated).reshape(B, H, T, D).type_as(x)


class MLP(nn.Module):
    """Feed-forward network with SwiGLU activation."""

    def __init__(self, config, intermediate_size: int = None):
        super().__init__()
        hidden_dim = intermediate_size or config.expert_intermediate_size
        self.w1 = nn.Linear(config.n_embd, hidden_dim, bias=False)  # Gate
        self.w2 = nn.Linear(config.n_embd, hidden_dim, bias=False)  # Up
        self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=False)  # Down

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(F.silu(self.w1(x)) * self.w2(x))


class SharedMoE(nn.Module):
    """Mixture of Experts with one shared expert and K routed experts.

    DeepSeek-V3 style: a shared expert processes all tokens while a top-k
    router selects from a pool of specialized experts per token.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.top_k = config.top_k

        # Shared expert (always active)
        self.shared_expert = MLP(config, config.shared_expert_intermediate_size)

        # Routed experts
        self.experts = nn.ModuleList([MLP(config) for _ in range(config.num_experts)])
        self.router = nn.Linear(config.n_embd, config.num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, C = x.shape

        # Shared path
        shared_out = self.shared_expert(x)

        # Router
        logits = self.router(x)
        probs = F.softmax(logits, dim=-1)

        # Top-K selection with normalized weights
        top_k_weights, top_k_indices = torch.topk(probs, self.top_k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

        # Load balancing auxiliary loss
        flat_probs = probs.view(-1, self.config.num_experts)
        expert_usage = flat_probs.mean(dim=0)
        aux_loss = torch.sum(expert_usage * expert_usage) * self.config.num_experts

        # Route tokens to experts
        routed_out = torch.zeros_like(x)
        flat_x = x.view(-1, C)
        flat_indices = top_k_indices.view(-1, self.top_k)
        flat_weights = top_k_weights.view(-1, self.top_k)

        for i, expert in enumerate(self.experts):
            mask = flat_indices == i
            batch_idx, rank_idx = torch.where(mask)

            expert_input = flat_x[batch_idx]
            expert_output = expert(expert_input)
            weight = flat_weights[batch_idx, rank_idx].unsqueeze(-1)
            routed_out.view(-1, C).index_add_(0, batch_idx, expert_output * weight)

        return shared_out + routed_out, aux_loss


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with Rotary Position Embeddings."""

    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.n_embd = config.n_embd

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Apply RoPE to Q and K
        q = apply_rope(q, freqs_cis)
        k = apply_rope(k, freqs_cis)

        # Flash Attention (auto-dispatches to cuDNN/FlashAttn kernels)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class Block(nn.Module):
    """Transformer block: RMSNorm -> Attention -> RMSNorm -> MoE."""

    def __init__(self, config):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = RMSNorm(config.n_embd)
        self.mlp = SharedMoE(config)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x + self.attn(self.ln_1(x), freqs_cis)
        mlp_out, aux_loss = self.mlp(self.ln_2(x))
        x = x + mlp_out
        return x, aux_loss


# ============================================================
#  Standalone model (backward compatible, no HF dependency)
# ============================================================

class DeepSeekMoE(nn.Module):
    """Eve-2-MoE: DeepSeek-V3 style Mixture of Experts language model.

    Standalone nn.Module — works without the transformers library.
    For HuggingFace integration, use EveMoEForCausalLM instead.

    Architecture:
        - Token embeddings (no learned position embeddings — uses RoPE)
        - N transformer blocks with RoPE attention + shared MoE FFN
        - RMSNorm + tied linear head
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=RMSNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying
        self.transformer.wte.weight = self.lm_head.weight

        # Precompute RoPE frequencies (registered as buffer so they move with .to(device))
        freqs_cis = precompute_rope_freqs(config.head_dim, config.block_size, config.rope_theta)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        B, T = idx.shape
        assert T <= self.config.block_size, f"Sequence length {T} exceeds block_size {self.config.block_size}"

        x = self.transformer.wte(idx)

        total_aux_loss = 0.0
        for block in self.transformer.h:
            if self.config.use_checkpointing and self.training:
                x, aux_loss = torch.utils.checkpoint.checkpoint(
                    block, x, self.freqs_cis, use_reentrant=False
                )
            else:
                x, aux_loss = block(x, self.freqs_cis)
            total_aux_loss += aux_loss

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss = loss + self.config.router_aux_loss_coef * total_aux_loss

        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int,
                 temperature: float = 0.8, top_k: int = 50) -> torch.Tensor:
        """Autoregressive generation with temperature and top-k sampling."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


# ============================================================
#  HuggingFace PreTrainedModel integration
#  (only available when transformers is installed)
# ============================================================

try:
    from transformers import PreTrainedModel
    from transformers.modeling_outputs import CausalLMOutputWithPast

    try:
        from .configuration_eve import EveConfig
    except ImportError:
        from configuration_eve import EveConfig

    class EveMoEPreTrainedModel(PreTrainedModel):
        """Base class for Eve-2-MoE HuggingFace models."""

        config_class = EveConfig
        base_model_prefix = "transformer"
        supports_gradient_checkpointing = True
        _no_split_modules = ["Block"]

        def _init_weights(self, module):
            std = 0.02
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=std)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=std)

    class EveMoEForCausalLM(EveMoEPreTrainedModel):
        """Eve-2-MoE for causal language modeling (HuggingFace compatible).

        This model has the same weights and architecture as DeepSeekMoE but
        follows HuggingFace conventions for from_pretrained() and generate().

        Usage:
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(
                "anthonym21/Eve-2-MoE-272M", trust_remote_code=True
            )
            output = model.generate(input_ids, max_new_tokens=100)
        """

        _tied_weights_keys = ["lm_head.weight"]

        def __init__(self, config: EveConfig):
            super().__init__(config)

            self.transformer = nn.ModuleDict(dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=RMSNorm(config.n_embd),
            ))
            self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

            # Precompute RoPE frequencies
            freqs_cis = precompute_rope_freqs(config.head_dim, config.block_size, config.rope_theta)
            self.register_buffer("freqs_cis", freqs_cis, persistent=False)

            # Initialize weights and apply final processing
            self.post_init()

        def get_input_embeddings(self):
            return self.transformer.wte

        def set_input_embeddings(self, value):
            self.transformer.wte = value

        def get_output_embeddings(self):
            return self.lm_head

        def set_output_embeddings(self, new_embeddings):
            self.lm_head = new_embeddings

        def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: torch.Tensor = None,
            labels: torch.LongTensor = None,
            return_dict: bool = None,
            **kwargs,
        ):
            """
            Args:
                input_ids: Token IDs, shape (batch, seq_len).
                attention_mask: Ignored (model uses causal mask via Flash Attention).
                    Accepted for pipeline/generate() compatibility.
                labels: Language modeling labels. Same shape as input_ids.
                    The loss is computed with internal shift (labels[..., 1:] predicted
                    from input[..., :-1]), following HuggingFace convention.
                return_dict: Whether to return a CausalLMOutputWithPast or a tuple.
            """
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            B, T = input_ids.shape
            assert T <= self.config.block_size, \
                f"Sequence length {T} exceeds block_size {self.config.block_size}"

            x = self.transformer.wte(input_ids)

            total_aux_loss = 0.0
            for block in self.transformer.h:
                if self.config.use_checkpointing and self.training:
                    x, aux_loss = torch.utils.checkpoint.checkpoint(
                        block, x, self.freqs_cis, use_reentrant=False
                    )
                else:
                    x, aux_loss = block(x, self.freqs_cis)
                total_aux_loss += aux_loss

            x = self.transformer.ln_f(x)
            logits = self.lm_head(x)

            loss = None
            if labels is not None:
                # Shift so that tokens < n predict n (HF convention)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, self.config.vocab_size),
                    shift_labels.view(-1),
                )
                loss = loss + self.config.router_aux_loss_coef * total_aux_loss

            if not return_dict:
                output = (logits,)
                return (loss,) + output if loss is not None else output

            return CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
            )

        def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **kwargs):
            # Truncate to block_size for models without KV cache
            if input_ids.shape[1] > self.config.block_size:
                input_ids = input_ids[:, -self.config.block_size:]
                if attention_mask is not None:
                    attention_mask = attention_mask[:, -self.config.block_size:]

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }

        def load_state_dict(self, state_dict, *args, **kwargs):
            """Override to handle weights saved from torch.compile'd models."""
            # Strip _orig_mod. prefix if present (torch.compile artifact)
            if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
                state_dict = _strip_orig_mod_prefix(state_dict)
            return super().load_state_dict(state_dict, *args, **kwargs)

except ImportError:
    # transformers not installed — standalone usage only (DeepSeekMoE + ModelConfig)
    pass
