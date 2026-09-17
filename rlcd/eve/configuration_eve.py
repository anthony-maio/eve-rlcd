"""
Eve-2-MoE Configuration
========================
HuggingFace-compatible configuration for the Eve-2-MoE architecture.

Usage:
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained("anthonym21/Eve-2-MoE-272M", trust_remote_code=True)
"""

from transformers import PretrainedConfig


class EveConfig(PretrainedConfig):
    """Configuration for the Eve-2-MoE model.

    This is a DeepSeek-V3 style Mixture of Experts architecture with a shared
    expert, top-k routed experts, RoPE positional encoding, and SwiGLU activations.

    Args:
        vocab_size: Vocabulary size (padded for efficiency). Default: 50304.
        n_layer: Number of transformer blocks. Default: 12.
        n_embd: Hidden dimension / embedding size. Default: 512.
        n_head: Number of attention heads. Default: 8.
        head_dim: Dimension per attention head. Default: 64.
        block_size: Maximum sequence length (context window). Default: 2048.
        num_experts: Number of routed MoE experts. Default: 8.
        top_k: Number of experts activated per token. Default: 2.
        expert_intermediate_size: FFN hidden dim for each expert (SwiGLU). Default: 1408.
        shared_expert_intermediate_size: FFN hidden dim for the shared expert. Default: 1408.
        router_aux_loss_coef: Weight of the load-balancing auxiliary loss. Default: 0.01.
        rope_theta: Base frequency for RoPE. Default: 10000.0.
        use_checkpointing: Enable gradient checkpointing to save VRAM. Default: False.
    """

    model_type = "eve-moe"

    def __init__(
        self,
        vocab_size: int = 50304,
        n_layer: int = 12,
        n_embd: int = 512,
        n_head: int = 8,
        head_dim: int = 64,
        block_size: int = 2048,
        num_experts: int = 8,
        top_k: int = 2,
        expert_intermediate_size: int = 1408,
        shared_expert_intermediate_size: int = 1408,
        router_aux_loss_coef: float = 0.01,
        rope_theta: float = 10000.0,
        use_checkpointing: bool = False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = head_dim
        self.block_size = block_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.expert_intermediate_size = expert_intermediate_size
        self.shared_expert_intermediate_size = shared_expert_intermediate_size
        self.router_aux_loss_coef = router_aux_loss_coef
        self.rope_theta = rope_theta
        self.use_checkpointing = use_checkpointing

        # Default tie_word_embeddings to True (Eve-2 ties embedding + lm_head)
        kwargs.setdefault("tie_word_embeddings", True)

        super().__init__(**kwargs)
