"""Backend-agnostic decision policies: one forward pass in, 26 letter logits out.

Three backends share one interface. "eve" wraps the functions in rlcd.policy unchanged, so
Eve results stay reproducible. "hf-decoder" drives any Hugging Face causal LM and reads the
letters at the last prompt token. "hf-mlm" drives a masked LM and reads them at a mask token
placed right after the prompt.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Protocol

import torch

from rlcd.loop import save_checkpoint
from rlcd.policy import EVE_ID, decision_logits, load_eve, questions_to_batch
from rlcd.schema import MAX_CHOICES, NEG, Question, letter_token_ids, render_prompt

BACKENDS = ("eve", "hf-decoder", "hf-mlm")
POLICY_FILE = "policy.json"
LORA = dict(r=32, lora_alpha=64, lora_dropout=0.0, target_modules="all-linear")



@dataclass(frozen=True)
class PinnedCode:
    """The exact commit of a hub repo whose custom code was read, and the sha256 of every code
    file at that commit. A hub load pins the commit; a local checkpoint of that repo carries a
    copy of the code, which transformers imports as is, so the copy is checked against the hash."""
    revision: str
    files: dict[str, str]


# Hub repos whose custom modeling code has been read and judged safe, each pinned to the exact
# commit that was read. trust_remote_code is never enabled for anything that is not listed here.
#
# LiquidAI/LFM2.5-Encoder-350M, modeling_lfm2_bidirectional.py at the commit below: imports only
# torch and transformers, touches neither the network nor the file system. It does patch the
# transformers lfm2 module for the whole process at import time (create_causal_mask becomes a
# padding-only mask, Lfm2ShortConv becomes a symmetric non-causal convolution). So never load it
# in a process that also runs a causal LFM2 decoder: that decoder would silently stop being causal.
PINNED_REMOTE_CODE: dict[str, PinnedCode] = {
    "LiquidAI/LFM2.5-Encoder-350M": PinnedCode(
        "b886781f7c6f10ca9b7096e21b83e30a073c2f39",
        {"modeling_lfm2_bidirectional.py": "f171f518be2a07da48b17fdea5655cad0a2452ab548e90e8ae903143686647e2"}),
}

# Set once any pinned custom code has been imported into this process. The encoder's code patches
# the lfm2 module, so no causal LFM2 decoder may be loaded afterwards.
_ENCODER_CODE_LOADED = False


def _check_lfm2_unpatched(path_or_id: str) -> None:
    """Refuse to run a causal LFM2 decoder in a process where the encoder's patches are, or may
    be, in place: the flag says the pinned code was imported, and the two identity checks catch
    the patch itself, whoever installed it."""
    import transformers.masking_utils as masking
    import transformers.models.lfm2.modeling_lfm2 as lfm2
    if _ENCODER_CODE_LOADED:
        raise RuntimeError(f"cannot load the LFM2 decoder {path_or_id}: the LiquidAI/LFM2.5-Encoder "
                           "custom code has been imported into this process and has patched the "
                           "transformers lfm2 module to be non-causal; use a fresh process")
    if lfm2.create_causal_mask is not masking.create_causal_mask:
        raise RuntimeError(f"cannot load the LFM2 decoder {path_or_id}: "
                           "transformers.models.lfm2.modeling_lfm2.create_causal_mask has been replaced")
    if lfm2.Lfm2ShortConv.slow_forward.__module__ != "transformers.models.lfm2.modeling_lfm2":
        raise RuntimeError(f"cannot load the LFM2 decoder {path_or_id}: Lfm2ShortConv.slow_forward "
                           f"comes from {lfm2.Lfm2ShortConv.slow_forward.__module__}")


class Policy(Protocol):
    name: str
    model: torch.nn.Module
    tok: Any
    letters: list[int]

    def decision_logits(self, questions: list[Question], max_len: int,
                        device: str) -> tuple[torch.Tensor, torch.Tensor]:
        """fp32 logits (B, 26), positions >= k filled with NEG, plus a scalar aux loss tensor."""

    def trainable_parameters(self) -> Iterable[torch.nn.Parameter]: ...

    def aux_coef(self) -> float: ...

    def save(self, out_dir: str, meta: dict) -> None: ...

    def train(self, mode: bool = True) -> "Policy": ...

    def eval(self) -> "Policy": ...


def _read_json(path: str) -> dict:
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _local_code_references(directory: str) -> list[str]:
    """Every custom-code reference a checkpoint directory makes: the auto_map values of its
    config.json and tokenizer_config.json, flattened (tokenizer entries are [slow, fast] pairs)."""
    refs: list[str] = []
    for name in ("config.json", "tokenizer_config.json"):
        for value in (_read_json(os.path.join(directory, name)).get("auto_map") or {}).values():
            for ref in value if isinstance(value, (list, tuple)) else [value]:
                if ref is not None:
                    refs.append(ref)
    return refs


def _verify_local_code(directory: str, origin: str, pin: PinnedCode) -> bool:
    """True when the directory names custom code, all of it matching the reviewed files of its
    origin; False when it names none. Raises on anything else. save_pretrained copies the modeling
    file into the directory and names it in auto_map without a repo prefix, and transformers then
    imports that local copy, ignoring code_revision. A reference that keeps a repo prefix is
    fetched from that repo instead, at code_revision, so it only has to point back at the origin."""
    refs = _local_code_references(directory)
    for ref in refs:
        if "--" in ref:
            repo, ref = ref.split("--", 1)
            if repo != origin:
                raise RuntimeError(f"{directory} pulls custom code from {repo}, which is not its "
                                   f"pinned origin {origin}")
            continue
        module_file = ref.split(".")[0] + ".py"
        path = os.path.join(directory, module_file)
        if module_file not in pin.files:
            raise RuntimeError(f"{directory} names custom code in {module_file}, which is not among "
                               f"the reviewed files of {origin}: {sorted(pin.files)}")
        if not os.path.isfile(path):
            raise RuntimeError(f"{directory} names custom code in {module_file}, which is missing")
        with open(path, "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()
        if digest != pin.files[module_file]:
            raise RuntimeError(f"{path} does not match the reviewed {module_file} of {origin} at "
                               f"{pin.revision}: sha256 {digest}, expected {pin.files[module_file]}")
    return bool(refs)


def remote_code_kwargs(path_or_id: str, origin: str | None = None) -> dict:
    """from_pretrained kwargs that enable custom code, only for a pinned repo. A local checkpoint
    whose origin is pinned is trusted only after every code file it names hashes to the reviewed
    one; a directory that names no custom code needs no trust at all."""
    path_or_id = os.fspath(path_or_id)
    if path_or_id in PINNED_REMOTE_CODE:
        return {"trust_remote_code": True, "revision": PINNED_REMOTE_CODE[path_or_id].revision}
    if origin in PINNED_REMOTE_CODE and os.path.isdir(path_or_id):
        pin = PINNED_REMOTE_CODE[origin]
        if _verify_local_code(path_or_id, origin, pin):
            return {"trust_remote_code": True, "code_revision": pin.revision}
    return {}


def _mask_beyond_k(logits: torch.Tensor, questions: list[Question]) -> torch.Tensor:
    return mask_beyond_k(logits, [q.k for q in questions])


def mask_beyond_k(logits: torch.Tensor, ks) -> torch.Tensor:
    """NEG at every letter position at or beyond each row's option count."""
    k = torch.as_tensor(ks, dtype=torch.long, device=logits.device)
    positions = torch.arange(MAX_CHOICES, device=logits.device)[None, :]
    return logits.masked_fill(positions >= k[:, None], NEG)


class _PolicyBase:
    name = ""
    model: torch.nn.Module

    def train(self, mode: bool = True):
        self.model.train(mode)
        return self

    def eval(self):
        return self.train(False)

    @property
    def training(self) -> bool:
        return self.model.training

    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        return [p for p in self.model.parameters() if p.requires_grad]

    def aux_coef(self) -> float:
        return 0.0

    def _write_record(self, out_dir: str, record: dict) -> None:
        (Path(out_dir) / POLICY_FILE).write_text(json.dumps(record, indent=2))


class EvePolicy(_PolicyBase):
    name = "eve"

    def __init__(self, model, tok, letters: list[int] | None = None):
        self.model = model
        self.tok = tok
        self.letters = letter_token_ids(tok) if letters is None else list(letters)

    def decision_logits(self, questions, max_len, device):
        ids, last, k = questions_to_batch(self.tok, questions, max_len, device)
        return decision_logits(self.model, ids, last, self.letters, k)

    def aux_coef(self) -> float:
        return self.model.config.router_aux_loss_coef

    def save(self, out_dir: str, meta: dict) -> None:
        save_checkpoint(self.model, self.tok, out_dir, meta)
        self._write_record(out_dir, {"backend": self.name})


def _adds_bos(tok) -> bool:
    """True when the tokenizer prepends its BOS token by default. Such models never saw a
    sequence without it, so the policy keeps it in front of the (possibly truncated) prompt."""
    bos = getattr(tok, "bos_token_id", None)
    if bos is None:
        return False
    ids = tok.encode("x", add_special_tokens=True)
    return len(ids) > 0 and ids[0] == bos


class _HFPolicy(_PolicyBase):
    """Shared by the two Hugging Face backends: right padding with an explicit attention mask,
    left truncation that keeps the tail, and an fp32 readout at one position per row."""

    def __init__(self, model, tok, letters: list[int] | None = None, base: str | None = None,
                 lora: bool = False, origin: str | None = None, revision: str | None = None,
                 grad_checkpointing: bool = False):
        """base: for a LoRA policy, the path or hub id of the weights the adapter sits on, and
        None for a full fine-tune. origin: the hub repo the lineage started from, used only to
        decide about remote code. revision: the hub commit the lineage started from (of base
        for a LoRA policy, of the weights first loaded for a full fine-tune), or None."""
        self.model = model
        self.tok = tok
        self.letters = letter_token_ids(tok) if letters is None else list(letters)
        self.base = base
        self.origin = origin
        self.revision = revision
        self.lora = lora
        self.grad_checkpointing = grad_checkpointing
        self.prepend_bos = _adds_bos(tok)
        pad = getattr(tok, "pad_token_id", None)
        self.pad_id = pad if pad is not None else getattr(tok, "eos_token_id", None)
        if self.pad_id is None:
            raise ValueError("the tokenizer has neither a pad token nor an eos token to pad with")

    def _suffix(self) -> list[int]:
        return []

    def _hf_model(self):
        """The transformers model, from under the peft wrapper when there is one."""
        return self.model.get_base_model() if self.lora else self.model

    def encode(self, questions: list[Question], max_len: int, device: str):
        """Right-padded ids, the attention mask, and each row's readout position: the last real
        token. The prompt is truncated from the left; BOS and the suffix always survive."""
        prefix = [self.tok.bos_token_id] if self.prepend_bos else []
        suffix = self._suffix()
        room = max_len - len(prefix) - len(suffix)
        if room <= 0:
            raise ValueError(f"max_len {max_len} leaves no room for the prompt")
        seqs = [prefix + self.tok.encode(render_prompt(q), add_special_tokens=False)[-room:] + suffix
                for q in questions]
        width = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), width), self.pad_id, dtype=torch.long)
        mask = torch.zeros((len(seqs), width), dtype=torch.long)
        last = torch.zeros(len(seqs), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
            mask[i, : len(s)] = 1
            last[i] = len(s) - 1
        return ids.to(device), mask.to(device), last.to(device)

    def _readout_rows(self, questions, max_len, device) -> torch.Tensor:
        """The body's final hidden state at each row's readout position, (B, d)."""
        ids, mask, last = self.encode(questions, max_len, device)
        hidden = self._body()(input_ids=ids, attention_mask=mask, use_cache=False)[0]
        return hidden[torch.arange(hidden.size(0), device=hidden.device), last]

    def _body(self):
        raise NotImplementedError

    def save(self, out_dir: str, meta: dict) -> None:
        """Full fine-tune: the whole model. LoRA: the adapter only, with policy.json naming the
        base it belongs to (and its hub commit). The tokenizer goes along either way."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(out, safe_serialization=True)
        self.tok.save_pretrained(out)
        (out / "meta.json").write_text(json.dumps(meta, indent=2))
        self._write_record(out_dir, {"backend": self.name, "lora": self.lora, "base": self.base,
                                     "origin": self.origin, "revision": self.revision,
                                     "prepend_bos": self.prepend_bos,
                                     "grad_checkpointing": self.grad_checkpointing})


class HFDecoderPolicy(_HFPolicy):
    name = "hf-decoder"

    def _body(self):
        model = self._hf_model()
        get = getattr(model, "get_decoder", None)
        body = get() if callable(get) else None
        if body is None or body is model:
            body = getattr(model, "model", None)
        if body is None or body is model:
            body = model.base_model
        if body is model:
            raise ValueError(f"cannot find the decoder body of {type(model).__name__}")
        return body

    def letter_head(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        """The 26 letter rows of the output embedding, (26, d), and their bias if the head has
        one. Everything the decision needs from the vocabulary projection."""
        head = self._hf_model().get_output_embeddings()
        index = torch.as_tensor(self.letters, device=head.weight.device)
        bias = getattr(head, "bias", None)
        return head.weight[index], None if bias is None else bias[index]

    def logits_from_hidden(self, rows: torch.Tensor, ks) -> torch.Tensor:
        """fp32 decision logits (B, 26) from the body's final hidden state at each row's readout
        position, masked at and beyond each row's option count. The matmul runs in fp32 with
        autocast off: autocast would downcast it and quantize the decision logits."""
        with torch.autocast(device_type=rows.device.type, enabled=False):
            weight, bias = self.letter_head()
            logits = rows.float() @ weight.float().t()
            if bias is not None:
                logits = logits + bias.float()
            return mask_beyond_k(logits, ks)

    def decision_logits(self, questions, max_len, device):
        """Runs the body only and multiplies the decision rows by the 26 letter rows of the
        output embedding, so full-vocabulary logits are never materialized."""
        rows = self._readout_rows(questions, max_len, device)
        aux = torch.zeros((), device=rows.device, dtype=torch.float32)
        return self.logits_from_hidden(rows, [q.k for q in questions]), aux


def _mlm_head(model):
    """The module chain that maps hidden states to vocabulary logits in a masked LM."""
    for names in (("cls",), ("lm_head",), ("head", "decoder")):
        if all(isinstance(getattr(model, n, None), torch.nn.Module) for n in names):
            return [getattr(model, n) for n in names]
    raise ValueError(f"cannot find the masked-LM head of {type(model).__name__}")


class HFMaskedLMPolicy(_HFPolicy):
    """The same prompt followed by one mask token; the letters are read at the mask."""
    name = "hf-mlm"

    def __init__(self, model, tok, letters=None, base=None, lora=False, origin=None, revision=None,
                 grad_checkpointing=False):
        if getattr(tok, "mask_token_id", None) is None:
            raise ValueError("the hf-mlm backend needs a tokenizer with a mask token")
        super().__init__(model, tok, letters, base, lora, origin, revision, grad_checkpointing)

    def _suffix(self) -> list[int]:
        return [self.tok.mask_token_id]

    def _body(self):
        return self._hf_model().base_model

    def _readout_rows(self, questions, max_len, device) -> torch.Tensor:
        """Rows go through the body grouped by exact length, never padded. The LFM2 encoder's
        symmetric convolution reads the token to the right of the mask, and the LFM2 layers of
        transformers 4.x do not zero padding states (the conv layers are handed the 4D attention
        mask, which apply_mask_to_padding_states ignores). A causal decoder never looks right,
        so only this backend pays for it."""
        ids, mask, last = self.encode(questions, max_len, "cpu")
        lengths = mask.sum(1)
        rows: list = [None] * len(questions)
        body = self._body()
        for n in sorted(set(lengths.tolist())):
            index = (lengths == n).nonzero().flatten().tolist()
            chunk = ids[index, :n].to(device)
            hidden = body(input_ids=chunk, attention_mask=torch.ones_like(chunk), use_cache=False)[0]
            for j, i in enumerate(index):
                rows[i] = hidden[j, n - 1]
        return torch.stack(rows)

    def decision_logits(self, questions, max_len, device):
        """A masked-LM head is more than a matmul (dense, activation, norm, then the decoder),
        so it runs whole, but only on the B mask rows, in fp32."""
        rows = self._readout_rows(questions, max_len, device)
        with torch.autocast(device_type=rows.device.type, enabled=False):
            x = rows.float()[:, None, :]
            for module in _mlm_head(self._hf_model()):
                x = module(x)
            index = torch.as_tensor(self.letters, device=rows.device)
            logits = x[:, 0, :].float()[:, index]
            aux = torch.zeros((), device=rows.device, dtype=torch.float32)
            return _mask_beyond_k(logits, questions), aux


def _read_record(path_or_id: str) -> dict:
    return _read_json(os.path.join(path_or_id, POLICY_FILE))


def _config_dict(path_or_id: str) -> dict:
    """config.json of a local directory, or of a pinned hub repo at its pinned commit (the only
    hub repos whose custom-code auto_map matters, and cached after the first fetch). Anything
    else is {} rather than a network round trip."""
    if os.path.isdir(path_or_id):
        return _read_json(os.path.join(path_or_id, "config.json"))
    if path_or_id in PINNED_REMOTE_CODE:
        from huggingface_hub import hf_hub_download
        return _read_json(hf_hub_download(path_or_id, "config.json",
                                          revision=PINNED_REMOTE_CODE[path_or_id].revision))
    return {}


def resolve_backend(path_or_id: str, backend: str = "auto") -> str:
    """auto: what a local policy.json records, else eve for the Eve hub id or an Eve config.json,
    else hf-mlm for a config whose custom-code auto_map offers a masked LM and no causal LM,
    else hf-decoder."""
    path_or_id = os.fspath(path_or_id)
    if backend != "auto":
        if backend not in BACKENDS:
            raise ValueError(f"unknown backend {backend!r}; expected auto or one of {BACKENDS}")
        return backend
    record = _read_record(path_or_id)
    if record:
        return resolve_backend(path_or_id, record["backend"])
    if path_or_id == EVE_ID:
        return "eve"
    config = _config_dict(path_or_id)
    if config.get("model_type") == "eve-moe":
        return "eve"
    auto_map = config.get("auto_map") or {}
    if "AutoModelForMaskedLM" in auto_map and "AutoModelForCausalLM" not in auto_map:
        return "hf-mlm"
    return "hf-decoder"


def _load_tokenizer(path_or_id: str, **kwargs):
    """AutoTokenizer, falling back to PreTrainedTokenizerFast for repos saved by transformers 5,
    whose tokenizer_config.json names a class ("TokenizersBackend") that 4.x does not have.
    Such a repo ships a self-contained tokenizer.json, which the fast class reads as is."""
    from transformers import AutoTokenizer, PreTrainedTokenizerFast
    try:
        return AutoTokenizer.from_pretrained(path_or_id, **kwargs)
    except ValueError as e:
        if "Tokenizer class" not in str(e):
            raise
        kwargs.pop("trust_remote_code", None)
        print(f"note: {e} Falling back to PreTrainedTokenizerFast on tokenizer.json.")
        return PreTrainedTokenizerFast.from_pretrained(path_or_id, **kwargs)


def _check_rope(config) -> None:
    """A config.json written by transformers 5 keeps rope_theta inside rope_parameters, where
    4.x does not look. Refuse to run on a silently defaulted value."""
    theta = (getattr(config, "rope_parameters", None) or {}).get("rope_theta")
    if theta is not None and getattr(config, "rope_theta", theta) != theta:
        raise RuntimeError(f"config.rope_theta {config.rope_theta} differs from "
                           f"rope_parameters.rope_theta {theta}")


def _is_tied_lm_head(model, key: str) -> bool:
    """True for lm_head.weight when it shares storage with the input embedding: a checkpoint of
    a tied model stores the embedding only, so the head is not missing at all."""
    if key != "lm_head.weight":
        return False
    head, embed = model.get_output_embeddings(), model.get_input_embeddings()
    return head is not None and embed is not None and head.weight.data_ptr() == embed.weight.data_ptr()


def _from_pretrained_strict(auto_cls, weights: str, **kwargs):
    """from_pretrained that refuses a checkpoint whose tensors do not match the model exactly,
    where transformers would only warn and leave the gaps randomly initialized."""
    model, info = auto_cls.from_pretrained(weights, torch_dtype=torch.float32, output_loading_info=True,
                                           **kwargs)
    missing = [k for k in info["missing_keys"] if not _is_tied_lm_head(model, k)]
    mismatched = [k[0] if isinstance(k, (tuple, list)) else k for k in info.get("mismatched_keys", [])]
    if missing or info["unexpected_keys"] or mismatched:
        raise RuntimeError(f"{weights} does not match {type(model).__name__}: missing {missing}, "
                           f"unexpected {info['unexpected_keys']}, mismatched {mismatched}")
    return model


def _load_adapter_strict(model, adapter_dir: str):
    """PeftModel.from_pretrained, but refusing an adapter file with missing or stray tensors,
    where peft would only warn."""
    from peft import MODEL_TYPE_TO_PEFT_MODEL_MAPPING, PeftConfig, PeftModel
    config = PeftConfig.from_pretrained(adapter_dir)
    config.inference_mode = False
    cls = MODEL_TYPE_TO_PEFT_MODEL_MAPPING.get(config.task_type, PeftModel)
    peft_model = cls(model, config, adapter_name="default")
    result = peft_model.load_adapter(adapter_dir, "default", is_trainable=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"{adapter_dir} does not match its adapter config: missing "
                           f"{list(result.missing_keys)}, unexpected {list(result.unexpected_keys)}")
    return peft_model


def _lineage(path_or_id: str, record: dict) -> tuple[str, str | None, str | None]:
    """(weights, origin, revision) for a load. A saved adapter's weights are its recorded base at
    the recorded revision; anything else loads its own weights. The origin is the hub repo the
    lineage started from: recorded, or the id itself for a fresh hub load. Records written before
    origin existed kept the lineage in base."""
    if record.get("lora"):
        return record["base"], record.get("origin", record["base"]), record.get("revision")
    origin = record.get("origin", record.get("base"))
    if origin is None and not os.path.isdir(path_or_id):
        origin = path_or_id
    return path_or_id, origin, None


def _load_hf(policy_cls, auto_cls, path_or_id: str, device: str, lora: bool, grad_checkpointing: bool):
    global _ENCODER_CODE_LOADED
    record = _read_record(path_or_id)
    saved_adapter = bool(record.get("lora"))
    weights, origin, revision = _lineage(path_or_id, record)
    # Custom code is only ever enabled for the pinned encoder, so only the hf-mlm backend may
    # ask for it; a decoder from the same lineage would run its stock transformers class.
    custom = policy_cls is HFMaskedLMPolicy
    tok_kwargs = remote_code_kwargs(path_or_id, origin) if custom else {}
    kwargs = remote_code_kwargs(weights, origin) if custom else {}
    if revision is not None:
        if kwargs.get("revision", revision) != revision:
            raise RuntimeError(f"{path_or_id} was trained on {weights} at {revision}, but the pinned "
                               f"remote-code revision is {kwargs['revision']}")
        kwargs["revision"] = revision
    tok = _load_tokenizer(path_or_id, **tok_kwargs)
    recorded_bos = record.get("prepend_bos")
    if recorded_bos is not None and recorded_bos != _adds_bos(tok):
        raise RuntimeError(f"{path_or_id} was trained with prepend_bos={recorded_bos}, but its tokenizer "
                           f"now gives {_adds_bos(tok)}; the prompts would not match the training ones")
    model = _from_pretrained_strict(auto_cls, weights, **kwargs)
    if tok_kwargs.get("trust_remote_code") or kwargs.get("trust_remote_code"):
        _ENCODER_CODE_LOADED = True
    if policy_cls is HFDecoderPolicy and model.config.model_type == "lfm2":
        _check_lfm2_unpatched(path_or_id)
    _check_rope(model.config)
    if not os.path.isdir(weights):
        # A remote-code model comes back without config._commit_hash; its load was pinned to
        # kwargs["revision"], which is then the commit that was loaded.
        revision = getattr(model.config, "_commit_hash", None) or kwargs.get("revision") or revision
    elif not saved_adapter:
        # A full fine-tune directory keeps the hub commit its lineage started from, as
        # information only: its weights are its own and were loaded from the directory.
        revision = record.get("revision")
    if grad_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False
    if saved_adapter:
        model = _load_adapter_strict(model, path_or_id)
    elif lora:
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(**LORA))
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise RuntimeError("master weights must stay fp32")
    model.to(device)
    lora = saved_adapter or lora
    return policy_cls(model, tok, base=weights if lora else None, lora=lora, origin=origin,
                      revision=revision, grad_checkpointing=grad_checkpointing)


def load_policy(path_or_id: str, device: str = "cuda", backend: str = "auto", lora: bool = False,
                grad_checkpointing: bool = False) -> Policy:
    """A policy from a hub id or a checkpoint directory written by Policy.save, fp32 on device
    and in train mode; call .eval() for inference."""
    path_or_id = os.fspath(path_or_id)
    backend = resolve_backend(path_or_id, backend)
    if backend == "eve":
        if lora or grad_checkpointing:
            raise ValueError("the eve backend supports neither LoRA nor gradient checkpointing")
        model, tok = load_eve(path_or_id, device=device)
        return EvePolicy(model, tok).train()
    if backend == "hf-decoder":
        from transformers import AutoModelForCausalLM
        return _load_hf(HFDecoderPolicy, AutoModelForCausalLM, path_or_id, device, lora,
                        grad_checkpointing).train()
    from transformers import AutoModelForMaskedLM
    return _load_hf(HFMaskedLMPolicy, AutoModelForMaskedLM, path_or_id, device, lora,
                    grad_checkpointing).train()
