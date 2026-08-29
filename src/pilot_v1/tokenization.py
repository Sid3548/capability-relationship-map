from __future__ import annotations

from dataclasses import dataclass

import torch

from .data import PilotItem


@dataclass
class EncodedItem:
    item: PilotItem
    input_ids: list[int]
    answer_token_indices: list[int]
    answer_token_ids: list[int]
    offsets: list[tuple[int, int]]


def encode_joint(tokenizer, item: PilotItem) -> EncodedItem:
    full_text = item.prompt + item.gold
    boundary = len(item.prompt)
    encoded = tokenizer(full_text, add_special_tokens=True, return_offsets_mapping=True)
    ids = list(encoded["input_ids"])
    offsets = [tuple(map(int, pair)) for pair in encoded["offset_mapping"]]
    answer_indices = [i for i, (start, end) in enumerate(offsets) if end > boundary and end > start]
    if not answer_indices:
        raise ValueError(f"{item.id}: joint tokenization selected no answer tokens")
    first_start = offsets[answer_indices[0]][0]
    if first_start != boundary:
        raise ValueError(
            f"{item.id}: answer boundary falls inside a token (boundary={boundary}, token_offset={offsets[answer_indices[0]]}); "
            "edit prompt/gold whitespace so the boundary is tokenizer-aligned"
        )
    if answer_indices[0] == 0:
        raise ValueError(f"{item.id}: first answer token has no causal predecessor")
    if answer_indices != list(range(answer_indices[0], answer_indices[-1] + 1)):
        raise ValueError(f"{item.id}: answer token indices are not contiguous")
    return EncodedItem(item, ids, answer_indices, [ids[i] for i in answer_indices], offsets)


def collate_right_padded(encoded: list[EncodedItem], pad_token_id: int, device: str):
    if not encoded:
        raise ValueError("cannot collate an empty batch")
    max_len = max(len(x.input_ids) for x in encoded)
    batch = torch.full((len(encoded), max_len), pad_token_id, dtype=torch.long)
    attention = torch.zeros((len(encoded), max_len), dtype=torch.long)
    answer_mask = torch.zeros((len(encoded), max_len), dtype=torch.bool)
    for row, value in enumerate(encoded):
        n = len(value.input_ids)
        batch[row, :n] = torch.tensor(value.input_ids, dtype=torch.long)
        attention[row, :n] = 1
        answer_mask[row, value.answer_token_indices] = True
    return batch.to(device), attention.to(device), answer_mask.to(device)


def gold_token_nll(logits: torch.Tensor, input_ids: torch.Tensor, answer_mask: torch.Tensor):
    shift_logits = logits[:, :-1, :].float()
    shift_labels = input_ids[:, 1:]
    shift_mask = answer_mask[:, 1:]
    log_probs = torch.log_softmax(shift_logits, dim=-1)
    token_nll = -log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
    per_item: list[list[float]] = []
    for row in range(token_nll.shape[0]):
        selected = token_nll[row][shift_mask[row]]
        if selected.numel() == 0:
            raise RuntimeError(f"batch row {row}: no answer tokens selected")
        per_item.append(selected.detach().cpu().tolist())
    return per_item

