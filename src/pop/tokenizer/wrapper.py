"""HF-compatible SentencePiece tokenizer wrapper for the pop T5 stack.

Mirrors the notebook's ``SentencePieceTokenizer`` (cell 13): fixed special ids
``pad=0``, ``eos=1``, ``unk=2``; 100 T5-style sentinels ``<extra_id_0..99>``
excluded from decoded output by default.
"""

from __future__ import annotations

from pathlib import Path

import sentencepiece as spm

NUM_SENTINELS = 100
PAD_ID = 0
EOS_ID = 1
UNK_ID = 2


class PopTokenizer:
    """Thin, HF-generation-compatible wrapper around a SentencePiece model."""

    def __init__(self, sp_model_path: str | Path):
        self.sp_model_path = Path(sp_model_path)
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(str(self.sp_model_path))

        self.pad_id = PAD_ID
        self.eos_id = EOS_ID
        self.unk_id = UNK_ID
        self.bos_id = PAD_ID  # notebook parity: no distinct bos, aliased to pad
        self.vocab_size = self.sp.GetPieceSize()

        self._sentinel_ids = [self.sp.PieceToId(f"<extra_id_{i}>") for i in range(NUM_SENTINELS)]
        self._sentinel_id_set = set(self._sentinel_ids)

    def __len__(self) -> int:
        return self.vocab_size

    @classmethod
    def load(cls, path: str | Path) -> PopTokenizer:
        return cls(path)

    def encode(self, text: str, max_length: int | None = None) -> list[int]:
        """Encode text to a list of token ids (no special tokens appended)."""
        ids = self.sp.EncodeAsIds(text)
        if max_length is not None:
            ids = ids[:max_length]
        return ids

    def batch_encode(self, texts: list[str], max_length: int | None = None) -> list[list[int]]:
        return [self.encode(text, max_length=max_length) for text in texts]

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        ids = list(token_ids)
        if skip_special_tokens:
            special = {self.pad_id, self.eos_id, self.unk_id, *self._sentinel_id_set}
            ids = [tid for tid in ids if tid not in special]
        if not ids:
            return ""
        return self.sp.DecodeIds(ids)

    def get_vocab(self) -> dict[str, int]:
        return {self.sp.IdToPiece(i): i for i in range(self.sp.GetPieceSize())}

    def sentinel_id(self, index: int) -> int:
        """Return the vocabulary id of ``<extra_id_{index}>``."""
        if not 0 <= index < NUM_SENTINELS:
            raise ValueError(f"sentinel index must be in [0, {NUM_SENTINELS}), got {index}")
        return self._sentinel_ids[index]

    def is_sentinel_id(self, token_id: int) -> bool:
        return token_id in self._sentinel_id_set
