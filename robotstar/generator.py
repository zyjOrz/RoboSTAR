from __future__ import annotations

from pathlib import Path
from typing import Sequence
import warnings

import torch
import torch.nn.functional as F
from torch import nn

from .checkpoints import load_safetensors_directory, read_json, resolve_model_root
from .config import GeneratorConfig
from .vocabulary import RobotSTARVocabulary


def resolve_attention_implementation(requested: str | None, model_type: str | None) -> str:
    """Return a Transformers attention backend that is valid for the model family.

    The private RobotSTAR trainer requested SDPA but deliberately caught the
    MT5 unsupported-backend error and reloaded without SDPA.  In the audited
    Transformers runtime that means eager attention.  Keep that effective
    behavior explicit for both pretrained and config-only construction.
    """
    value = str(requested or "eager").strip().lower() or "eager"
    if str(model_type or "").strip().lower() == "mt5" and value != "eager":
        warnings.warn(
            f"MT5 does not support attn_implementation={value!r} in this "
            "Transformers runtime; using 'eager', matching the training fallback.",
            RuntimeWarning,
            stacklevel=2,
        )
        return "eager"
    return value


def _set_config_attention(config: object, implementation: str) -> None:
    # Transformers versions expose different public/private config attributes.
    # Setting all available forms keeps AutoModel.from_config deterministic.
    for attribute in (
        "_attn_implementation_internal",
        "_attn_implementation",
        "attn_implementation",
    ):
        try:
            setattr(config, attribute, implementation)
        except Exception:
            pass


def shift_tokens_right(labels: torch.Tensor, pad_id: int) -> torch.Tensor:
    shifted = labels.new_full(labels.shape, int(pad_id))
    valid = labels.ne(-100)
    counts = valid.long().sum(dim=1).clamp_min(1)
    start = labels.gather(1, (counts - 1).unsqueeze(1)).squeeze(1)
    start = torch.where(start.eq(-100), torch.full_like(start, int(pad_id)), start)
    shifted[:, 0] = start
    if labels.shape[1] > 1:
        previous = labels[:, :-1].clone()
        previous.masked_fill_(previous.eq(-100), int(pad_id))
        shifted[:, 1:] = previous
    return shifted


class RobotSTARGenerator(nn.Module):
    def __init__(
        self,
        cfg: GeneratorConfig,
        vocab: RobotSTARVocabulary,
        backbone_init: str = "pretrained",
    ) -> None:
        super().__init__()
        from transformers import AutoConfig, AutoModelForSeq2SeqLM

        self.cfg = cfg
        self.vocab = vocab
        base_config = AutoConfig.from_pretrained(cfg.model_path)
        self.requested_attn_implementation = str(cfg.attn_implementation or "eager")
        self.effective_attn_implementation = resolve_attention_implementation(
            self.requested_attn_implementation,
            getattr(base_config, "model_type", cfg.lm_family),
        )
        # Persist the effective behavior, not the unsupported historical request.
        self.cfg.attn_implementation = self.effective_attn_implementation
        _set_config_attention(base_config, self.effective_attn_implementation)

        if backbone_init == "pretrained":
            try:
                self.main_lm = AutoModelForSeq2SeqLM.from_pretrained(
                    cfg.model_path,
                    config=base_config,
                    attn_implementation=self.effective_attn_implementation,
                )
            except TypeError:
                # Older Transformers versions may not accept the keyword.
                self.main_lm = AutoModelForSeq2SeqLM.from_pretrained(
                    cfg.model_path, config=base_config
                )
            except ValueError as error:
                message = str(error).lower()
                if "attention implementation" not in message and "scaled_dot_product_attention" not in message:
                    raise
                _set_config_attention(base_config, "eager")
                self.effective_attn_implementation = "eager"
                self.cfg.attn_implementation = "eager"
                self.main_lm = AutoModelForSeq2SeqLM.from_pretrained(
                    cfg.model_path, config=base_config
                )
        elif backbone_init == "config_only":
            # No base weights are loaded here; public safetensors are loaded below.
            # MT5 must be constructed with eager attention in Transformers 5.12.1.
            self.main_lm = AutoModelForSeq2SeqLM.from_config(base_config)
        else:
            raise ValueError(f"Unknown backbone_init={backbone_init!r}")

        current_vocab = int(self.main_lm.get_input_embeddings().num_embeddings)
        if current_vocab != vocab.vocab_size:
            try:
                self.main_lm.resize_token_embeddings(vocab.vocab_size, mean_resizing=False)
            except TypeError:
                self.main_lm.resize_token_embeddings(vocab.vocab_size)
        if getattr(self.main_lm.config, "decoder_start_token_id", None) is None:
            self.main_lm.config.decoder_start_token_id = vocab.en_asl_id
        if getattr(self.main_lm.config, "pad_token_id", None) is None:
            self.main_lm.config.pad_token_id = vocab.pad_id
        if getattr(self.main_lm.config, "eos_token_id", None) is None:
            self.main_lm.config.eos_token_id = vocab.eos_id

        dimension = int(self.main_lm.config.d_model)
        stages = len(cfg.stage_divisors)
        self.stage_embedding = nn.Embedding(stages, dimension)
        self.coarse_seed = nn.Parameter(torch.zeros(stages, dimension))
        self.context_projection = nn.ModuleList([nn.Linear(dimension, dimension, bias=False) for _ in range(stages)])
        self.cross_scale_gates = nn.Parameter(torch.zeros(stages, stages))
        self.length_pool = nn.Sequential(nn.LayerNorm(dimension), nn.Linear(dimension, dimension), nn.GELU())
        self.length_head = nn.Linear(dimension, cfg.max_motion_tokens)
        self.text_contrastive = nn.Linear(dimension, dimension, bias=False)
        self.motion_contrastive = nn.Linear(dimension, dimension, bias=False)

        self.register_buffer("body_code_ids", torch.tensor(vocab.body_code_to_id, dtype=torch.long), persistent=False)
        self.register_buffer("left_code_ids", torch.tensor(vocab.left_code_to_id, dtype=torch.long), persistent=False)
        self.register_buffer("right_code_ids", torch.tensor(vocab.right_code_to_id, dtype=torch.long), persistent=False)
        self.register_buffer("compact_to_body", self._compact_lookup(vocab.body_id_to_code), persistent=False)
        self.register_buffer("compact_to_left", self._compact_lookup(vocab.left_id_to_code), persistent=False)
        self.register_buffer("compact_to_right", self._compact_lookup(vocab.right_id_to_code), persistent=False)

        body_allowed = self._allowed_ids(vocab.body_code_to_id)
        left_allowed = self._allowed_ids(vocab.left_code_to_id)
        right_allowed = self._allowed_ids(vocab.right_code_to_id)
        self.register_buffer("body_allowed_ids", body_allowed, persistent=False)
        self.register_buffer("left_allowed_ids", left_allowed, persistent=False)
        self.register_buffer("right_allowed_ids", right_allowed, persistent=False)
        self.register_buffer("body_label_map", self._label_map(body_allowed), persistent=False)
        self.register_buffer("left_label_map", self._label_map(left_allowed), persistent=False)
        self.register_buffer("right_label_map", self._label_map(right_allowed), persistent=False)
        self.register_buffer("mask_body", self._part_mask(vocab.body_code_to_id), persistent=False)
        self.register_buffer("mask_left", self._part_mask(vocab.left_code_to_id), persistent=False)
        self.register_buffer("mask_right", self._part_mask(vocab.right_code_to_id), persistent=False)
        self.register_buffer("code_mask_body", self._code_only_mask(vocab.body_code_to_id), persistent=False)
        self.register_buffer("code_mask_left", self._code_only_mask(vocab.left_code_to_id), persistent=False)
        self.register_buffer("code_mask_right", self._code_only_mask(vocab.right_code_to_id), persistent=False)

    @property
    def nstages(self) -> int:
        return len(self.cfg.stage_divisors)

    def shared(self) -> nn.Embedding:
        return self.main_lm.get_input_embeddings()

    def _compact_lookup(self, mapping: dict[int, int]) -> torch.Tensor:
        result = torch.full((self.vocab.vocab_size,), -1, dtype=torch.long)
        for token_id, code in mapping.items():
            result[int(token_id)] = int(code)
        return result

    def _part_mask(self, token_ids: Sequence[int]) -> torch.Tensor:
        result = torch.full((self.vocab.vocab_size,), float("-inf"), dtype=torch.float32)
        result[list(map(int, token_ids)) + [self.vocab.eos_id, self.vocab.en_asl_id]] = 0.0
        return result

    def _code_only_mask(self, token_ids: Sequence[int]) -> torch.Tensor:
        result = torch.full((self.vocab.vocab_size,), float("-inf"), dtype=torch.float32)
        result[list(map(int, token_ids))] = 0.0
        return result

    def _allowed_ids(self, token_ids: Sequence[int]) -> torch.Tensor:
        return torch.tensor(list(map(int, token_ids)) + [self.vocab.eos_id, self.vocab.en_asl_id], dtype=torch.long)

    def _label_map(self, allowed_ids: torch.Tensor) -> torch.Tensor:
        result = torch.full((self.vocab.vocab_size,), -100, dtype=torch.long)
        result[allowed_ids.long()] = torch.arange(len(allowed_ids), dtype=torch.long)
        return result

    def encode_source(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        encoded = self.main_lm.get_encoder()(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        valid = attention_mask.to(encoded.last_hidden_state.dtype)
        pooled = (encoded.last_hidden_state * valid.unsqueeze(-1)).sum(dim=1) / valid.sum(dim=1, keepdim=True).clamp_min(1)
        return encoded, pooled

    def _code_to_token_id(self, codes: torch.Tensor, part: str) -> torch.Tensor:
        table = {"body": self.body_code_ids, "left": self.left_code_ids, "right": self.right_code_ids}[part]
        return table[codes.clamp_min(0)]

    def _mixed_code_embeddings(self, codes: dict[str, torch.Tensor]) -> torch.Tensor:
        embedding = self.shared()
        body = embedding(self._code_to_token_id(codes["body"], "body"))
        left = embedding(self._code_to_token_id(codes["left"], "left"))
        right = embedding(self._code_to_token_id(codes["right"], "right"))
        mixed = (1.0 - 2.0 * self.cfg.alpha_hand) * body + self.cfg.alpha_hand * left + self.cfg.alpha_hand * right
        return mixed * codes["body"].ge(0).unsqueeze(-1).to(mixed.dtype)

    def _align_context(
        self,
        codes: dict[str, torch.Tensor],
        source_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
        target_max: int,
    ) -> torch.Tensor:
        embeddings = self._mixed_code_embeddings(codes)
        _, source_max, dimension = embeddings.shape
        positions = torch.arange(target_max, device=embeddings.device).unsqueeze(0)
        source_lengths = source_lengths.unsqueeze(1).clamp_min(1)
        target_lengths_column = target_lengths.unsqueeze(1).clamp_min(1)
        indices = torch.div(positions * source_lengths, target_lengths_column, rounding_mode="floor")
        indices = indices.clamp(max=max(source_max - 1, 0))
        gathered = embeddings.gather(1, indices.unsqueeze(-1).expand(-1, -1, dimension))
        valid = positions < target_lengths.unsqueeze(1)
        return gathered * valid.unsqueeze(-1).to(gathered.dtype)

    def _cross_scale_context(
        self,
        stage_idx: int,
        prior_stages: Sequence[dict[str, torch.Tensor]],
        all_stage_lengths: torch.Tensor,
        current_lengths: torch.Tensor,
        current_max: int,
    ) -> torch.Tensor:
        batch = len(current_lengths)
        dimension = int(self.main_lm.config.d_model)
        if stage_idx == 0 or not prior_stages:
            seed = self.coarse_seed[stage_idx].view(1, 1, -1).expand(batch, current_max, dimension)
            return self.context_projection[stage_idx](seed)
        accumulated = torch.zeros(
            (batch, current_max, dimension),
            device=current_lengths.device,
            dtype=self.shared().weight.dtype,
        )
        denominator = accumulated.new_zeros(())
        for prior_idx, codes in enumerate(prior_stages):
            aligned = self._align_context(codes, all_stage_lengths[:, prior_idx], current_lengths, current_max)
            gate = torch.sigmoid(self.cross_scale_gates[stage_idx, prior_idx])
            accumulated = accumulated + gate * aligned
            denominator = denominator + gate
        return self.context_projection[stage_idx](accumulated / denominator.clamp_min(1e-4))

    def _decoder_context_full(
        self,
        code_context: torch.Tensor,
        code_lengths: torch.Tensor,
        label_length: int,
    ) -> torch.Tensor:
        batch, code_max, dimension = code_context.shape
        result = code_context.new_zeros((batch, label_length, dimension))
        result[:, :code_max] = code_context
        if code_max:
            final_indices = (code_lengths - 1).clamp(0, code_max - 1)
            final_context = code_context.gather(1, final_indices[:, None, None].expand(-1, 1, dimension)).squeeze(1)
            rows = torch.arange(batch, device=result.device)
            result[rows, code_lengths.clamp(max=label_length - 1)] = final_context
            result[rows, (code_lengths + 1).clamp(max=label_length - 1)] = final_context
        return result

    def _scale_for_output_projection(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.cfg.scale_decoder_outputs:
            return hidden * (self.shared().embedding_dim ** -0.5)
        return hidden

    def _allowed_logits(self, hidden: torch.Tensor, allowed_ids: torch.Tensor) -> torch.Tensor:
        hidden = self._scale_for_output_projection(hidden)
        ids = allowed_ids.to(hidden.device)
        weight = self.main_lm.lm_head.weight.index_select(0, ids)
        logits = hidden @ weight.t()
        bias = getattr(self.main_lm.lm_head, "bias", None)
        if bias is not None:
            logits = logits + bias.index_select(0, ids)
        return logits

    def _part_logits(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self._allowed_logits(hidden, self.body_allowed_ids),
            self._allowed_logits(hidden, self.left_allowed_ids),
            self._allowed_logits(hidden, self.right_allowed_ids),
        )

    def _code_logits(self, hidden: torch.Tensor, part: str) -> torch.Tensor:
        ids = {"body": self.body_code_ids, "left": self.left_code_ids, "right": self.right_code_ids}[part]
        return self._allowed_logits(hidden, ids)

    def _labels_to_local(self, labels: torch.Tensor, part: str) -> torch.Tensor:
        table = {"body": self.body_label_map, "left": self.left_label_map, "right": self.right_label_map}[part]
        local = table.to(labels.device).gather(0, labels.clamp_min(0).reshape(-1)).reshape_as(labels)
        return local.masked_fill(labels.eq(-100), -100)

    def _stage_train(
        self,
        encoder_hidden: torch.Tensor,
        encoder_mask: torch.Tensor,
        labels: dict[str, torch.Tensor],
        code_lengths: torch.Tensor,
        prior_stages: Sequence[dict[str, torch.Tensor]],
        all_stage_lengths: torch.Tensor,
        stage_idx: int,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        body_input = shift_tokens_right(labels["body"], self.vocab.pad_id)
        left_input = shift_tokens_right(labels["left"], self.vocab.pad_id)
        right_input = shift_tokens_right(labels["right"], self.vocab.pad_id)
        embedding = self.shared()
        decoder_embeddings = (
            (1.0 - 2.0 * self.cfg.alpha_hand) * embedding(body_input)
            + self.cfg.alpha_hand * embedding(left_input)
            + self.cfg.alpha_hand * embedding(right_input)
        )
        code_max = int(labels["body"].shape[1] - 2)
        context = self._cross_scale_context(stage_idx, prior_stages, all_stage_lengths, code_lengths, code_max)
        decoder_embeddings = decoder_embeddings + self._decoder_context_full(context, code_lengths, decoder_embeddings.shape[1])
        decoder_embeddings = decoder_embeddings + self.stage_embedding.weight[stage_idx]
        decoder_mask = labels["body"].ne(-100)
        decoded = self.main_lm.decoder(
            inputs_embeds=decoder_embeddings,
            attention_mask=decoder_mask,
            encoder_hidden_states=encoder_hidden,
            encoder_attention_mask=encoder_mask,
            use_cache=False,
            return_dict=True,
        )
        body_logits, left_logits, right_logits = self._part_logits(decoded.last_hidden_state)
        logits_by_part = {"body": body_logits, "left": left_logits, "right": right_logits}
        local_labels = {part: self._labels_to_local(labels[part], part) for part in ("body", "left", "right")}
        losses = {
            part: F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                local_labels[part].reshape(-1),
                ignore_index=-100,
                label_smoothing=self.cfg.label_smoothing,
            )
            for part, logits in logits_by_part.items()
        }
        positions = torch.arange(code_max, device=body_logits.device).unsqueeze(0)
        valid = positions < code_lengths.unsqueeze(1)
        predictions: dict[str, torch.Tensor] = {}
        accuracy: dict[str, torch.Tensor] = {}
        for part, logits in logits_by_part.items():
            size = self.cfg.body_codes if part == "body" else self.cfg.hand_codes
            predicted = logits[:, :code_max, :size].argmax(dim=-1)
            predictions[part] = predicted.masked_fill(~valid, -100)
            target = local_labels[part][:, :code_max]
            accuracy[part] = ((predicted == target) & valid).sum() / valid.sum().clamp_min(1)
        return {
            "loss_body": losses["body"],
            "loss_left": losses["left"],
            "loss_right": losses["right"],
            "acc_body": accuracy["body"],
            "acc_left": accuracy["left"],
            "acc_right": accuracy["right"],
            "predictions": predictions,
        }

    def _corrupt_context(self, codes: dict[str, torch.Tensor], probability: float) -> dict[str, torch.Tensor]:
        if not self.training or probability <= 0:
            return codes
        result: dict[str, torch.Tensor] = {}
        for part, values in codes.items():
            valid = values.ge(0)
            replace = valid & (torch.rand(values.shape, device=values.device) < float(probability))
            upper = self.cfg.body_codes if part == "body" else self.cfg.hand_codes
            random_values = torch.randint(0, upper, values.shape, device=values.device)
            result[part] = torch.where(replace, random_values, values)
        return result

    def _contrastive_loss(
        self,
        text_pool: torch.Tensor,
        final_codes: dict[str, torch.Tensor],
        final_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        motion = self._mixed_code_embeddings(final_codes)
        positions = torch.arange(motion.shape[1], device=motion.device).unsqueeze(0)
        valid = positions < final_lengths.unsqueeze(1)
        motion_pool = (motion * valid.unsqueeze(-1).to(motion.dtype)).sum(dim=1) / valid.sum(dim=1, keepdim=True).clamp_min(1)
        text_features = F.normalize(self.text_contrastive(text_pool), dim=-1)
        motion_features = F.normalize(self.motion_contrastive(motion_pool), dim=-1)
        logits = text_features @ motion_features.t() / max(self.cfg.contrastive_temperature, 1e-5)
        target = torch.arange(len(text_features), device=text_features.device)
        loss = 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.t(), target))
        return loss, logits.argmax(dim=-1).eq(target).float().mean()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        full_lengths: torch.Tensor,
        length_classes: torch.Tensor,
        stage_lengths: torch.Tensor,
        stage_codes: Sequence[dict[str, torch.Tensor]],
        stage_labels: Sequence[dict[str, torch.Tensor]],
        context_corruption_probability: float = 0.0,
        self_condition_probability: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        encoded, text_pool = self.encode_source(input_ids, attention_mask)
        length_logits = self.length_head(self.length_pool(text_pool))
        length_loss = F.cross_entropy(length_logits, length_classes, label_smoothing=0.02)
        length_accuracy = length_logits.argmax(dim=-1).eq(length_classes).float().mean()
        length_mae = (length_logits.argmax(dim=-1) + 1 - full_lengths).abs().float().mean()
        weighted_loss = text_pool.sum() * 0
        total_weight = 0.0
        metrics: dict[str, torch.Tensor] = {}
        contexts: list[dict[str, torch.Tensor]] = []
        for stage_idx, (codes, labels) in enumerate(zip(stage_codes, stage_labels)):
            output = self._stage_train(
                encoded.last_hidden_state,
                attention_mask,
                labels,
                stage_lengths[:, stage_idx],
                contexts,
                stage_lengths,
                stage_idx,
            )
            part_loss = output["loss_body"] + self.cfg.hand_loss_weight * 0.5 * (output["loss_left"] + output["loss_right"])
            weight = float(self.cfg.stage_loss_weights[stage_idx])
            weighted_loss = weighted_loss + weight * part_loss
            total_weight += weight
            metrics[f"stage_{stage_idx + 1}_loss"] = part_loss.detach()
            for part in ("body", "left", "right"):
                metrics[f"stage_{stage_idx + 1}_acc_{part}"] = output[f"acc_{part}"].detach()
            metrics[f"stage_{stage_idx + 1}_acc"] = (
                output["acc_body"] + output["acc_left"] + output["acc_right"]
            ).detach() / 3.0
            if self.training and self_condition_probability > 0:
                choose = torch.rand(len(input_ids), device=input_ids.device) < float(self_condition_probability)
                context = {
                    part: torch.where(choose.unsqueeze(1), output["predictions"][part], codes[part])
                    for part in ("body", "left", "right")
                }
            else:
                context = {part: codes[part] for part in ("body", "left", "right")}
            contexts.append(self._corrupt_context(context, context_corruption_probability))
        token_loss = weighted_loss / max(total_weight, 1e-6)
        contrastive_loss, contrastive_r1 = self._contrastive_loss(text_pool, stage_codes[-1], stage_lengths[:, -1])
        total = token_loss + self.cfg.length_loss_weight * length_loss + self.cfg.contrastive_loss_weight * contrastive_loss
        return {
            "loss": total,
            "token_loss": token_loss.detach(),
            "length_loss": length_loss.detach(),
            "length_acc": length_accuracy.detach(),
            "length_mae": length_mae.detach(),
            "contrastive_loss": contrastive_loss.detach(),
            "contrastive_r1": contrastive_r1.detach(),
            **metrics,
        }

    @staticmethod
    def _sample_local_code(logits: torch.Tensor, temperature: float, top_k: int, do_sample: bool) -> torch.Tensor:
        logits = logits / max(float(temperature), 1e-5)
        if top_k > 0:
            values, _ = torch.topk(logits, min(int(top_k), logits.shape[-1]), dim=-1)
            logits = logits.masked_fill(logits < values[..., -1:], float("-inf"))
        if do_sample:
            return torch.multinomial(F.softmax(logits, dim=-1), 1)
        return logits.argmax(dim=-1, keepdim=True)

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        explicit_full_lengths: torch.Tensor | None = None,
        temperature: float = 1.0,
        top_k: int = 0,
        do_sample: bool = False,
    ) -> dict[str, torch.Tensor | list[dict[str, torch.Tensor]] | dict[str, torch.Tensor]]:
        encoded, text_pool = self.encode_source(input_ids, attention_mask)
        if explicit_full_lengths is None:
            full_lengths = self.length_head(self.length_pool(text_pool)).argmax(dim=-1) + 1
        else:
            full_lengths = explicit_full_lengths.long()
        full_lengths = full_lengths.clamp(1, self.cfg.max_motion_tokens)
        stage_lengths = torch.stack(
            [
                torch.div(full_lengths + int(divisor) - 1, int(divisor), rounding_mode="floor")
                for divisor in self.cfg.stage_divisors
            ],
            dim=1,
        )
        stages: list[dict[str, torch.Tensor]] = []
        for stage_idx, _ in enumerate(self.cfg.stage_divisors):
            lengths = stage_lengths[:, stage_idx]
            max_length = int(lengths.max().item())
            context = self._cross_scale_context(stage_idx, stages, stage_lengths, lengths, max_length)
            context = context + self.stage_embedding.weight[stage_idx]
            batch = len(input_ids)
            body_sequence = input_ids.new_full((batch, 1), self.vocab.en_asl_id)
            left_sequence = body_sequence.clone()
            right_sequence = body_sequence.clone()
            codes = {part: input_ids.new_full((batch, max_length), -100) for part in ("body", "left", "right")}
            past = None
            for time_index in range(max_length):
                if past is None:
                    body_embeddings = self.shared()(body_sequence)
                    left_embeddings = self.shared()(left_sequence)
                    right_embeddings = self.shared()(right_sequence)
                else:
                    body_embeddings = self.shared()(body_sequence[:, -1:])
                    left_embeddings = self.shared()(left_sequence[:, -1:])
                    right_embeddings = self.shared()(right_sequence[:, -1:])
                decoder_embeddings = (
                    (1.0 - 2.0 * self.cfg.alpha_hand) * body_embeddings
                    + self.cfg.alpha_hand * left_embeddings
                    + self.cfg.alpha_hand * right_embeddings
                    + context[:, time_index : time_index + 1]
                )
                try:
                    decoded = self.main_lm.decoder(
                        inputs_embeds=decoder_embeddings,
                        encoder_hidden_states=encoded.last_hidden_state,
                        encoder_attention_mask=attention_mask,
                        past_key_values=past,
                        use_cache=True,
                        return_dict=True,
                    )
                    past = decoded.past_key_values
                    hidden = decoded.last_hidden_state[:, -1]
                except Exception:
                    past = None
                    body_full = self.shared()(body_sequence)
                    left_full = self.shared()(left_sequence)
                    right_full = self.shared()(right_sequence)
                    full_embeddings = (
                        (1.0 - 2.0 * self.cfg.alpha_hand) * body_full
                        + self.cfg.alpha_hand * left_full
                        + self.cfg.alpha_hand * right_full
                        + context[:, : body_sequence.shape[1]]
                    )
                    decoded = self.main_lm.decoder(
                        inputs_embeds=full_embeddings,
                        encoder_hidden_states=encoded.last_hidden_state,
                        encoder_attention_mask=attention_mask,
                        use_cache=False,
                        return_dict=True,
                    )
                    hidden = decoded.last_hidden_state[:, -1]
                local_body = self._sample_local_code(self._code_logits(hidden, "body"), temperature, top_k, do_sample)
                local_left = self._sample_local_code(self._code_logits(hidden, "left"), temperature, top_k, do_sample)
                local_right = self._sample_local_code(self._code_logits(hidden, "right"), temperature, top_k, do_sample)
                active = time_index < lengths
                codes["body"][:, time_index] = local_body[:, 0].masked_fill(~active, -100)
                codes["left"][:, time_index] = local_left[:, 0].masked_fill(~active, -100)
                codes["right"][:, time_index] = local_right[:, 0].masked_fill(~active, -100)
                body_token = self.body_code_ids.to(input_ids.device).gather(0, local_body[:, 0].clamp_min(0)).unsqueeze(1)
                left_token = self.left_code_ids.to(input_ids.device).gather(0, local_left[:, 0].clamp_min(0)).unsqueeze(1)
                right_token = self.right_code_ids.to(input_ids.device).gather(0, local_right[:, 0].clamp_min(0)).unsqueeze(1)
                body_token = torch.where(active.unsqueeze(1), body_token, self.body_code_ids[0].expand_as(body_token))
                left_token = torch.where(active.unsqueeze(1), left_token, self.left_code_ids[0].expand_as(left_token))
                right_token = torch.where(active.unsqueeze(1), right_token, self.right_code_ids[0].expand_as(right_token))
                body_sequence = torch.cat((body_sequence, body_token), dim=1)
                left_sequence = torch.cat((left_sequence, left_token), dim=1)
                right_sequence = torch.cat((right_sequence, right_token), dim=1)
            stages.append(codes)
        return {"full_lengths": full_lengths, "stage_lengths": stage_lengths, "stages": stages, "final_codes": stages[-1]}

    @classmethod
    def from_pretrained(
        cls,
        model: str | Path,
        base_model: str | Path | None = None,
        device: str | torch.device = "cpu",
        strict: bool = True,
    ) -> tuple["RobotSTARGenerator", RobotSTARVocabulary]:
        root = resolve_model_root(model, allow_patterns=["generator/*", "README.md", "manifest.json"])
        folder = root / "generator" if (root / "generator").is_dir() else root
        values = read_json(folder / "config.json")
        if base_model is not None:
            values["model_path"] = str(base_model)
        values.setdefault("scale_decoder_outputs", True)
        cfg = GeneratorConfig.from_mapping(values)
        vocab = RobotSTARVocabulary(
            cfg.model_path,
            cfg.body_codes,
            cfg.hand_codes,
            cfg.use_compact_map,
        )
        vocab_file = folder / "vocab.json"
        if vocab_file.is_file():
            expected = read_json(vocab_file)
            for key, actual in (("vocab_size", vocab.vocab_size), ("en_asl_id", vocab.en_asl_id)):
                if key in expected and int(expected[key]) != int(actual):
                    raise RuntimeError(f"Vocabulary contract mismatch for {key}: expected {expected[key]}, got {actual}")
        instance = cls(cfg, vocab, backbone_init="config_only")
        state = load_safetensors_directory(folder, device="cpu")
        missing, unexpected = instance.load_state_dict(state, strict=strict)
        if strict and (missing or unexpected):
            raise RuntimeError(f"Generator state mismatch: missing={missing}, unexpected={unexpected}")
        return instance.to(device), vocab
