"""multilingual-e5-small embeddings via ONNX Runtime on CPU."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

Vector = NDArray[np.float32]


@runtime_checkable
class Embedder(Protocol):
    dim: int

    def embed(self, texts: Sequence[str]) -> Vector:
        """Return L2-normalized embeddings, shape (len(texts), dim)."""
        ...


class E5OnnxEmbedder:
    """intfloat/multilingual-e5-small, ONNX, CPU. Texts are embedded as "query: ..."."""

    dim = 384

    def __init__(
        self,
        repo_id: str = "intfloat/multilingual-e5-small",
        onnx_file: str = "onnx/model.onnx",
        max_length: int = 128,
        threads: int = 2,
    ) -> None:
        import onnxruntime as ort  # type: ignore[import-untyped]
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        model_path = hf_hub_download(repo_id, onnx_file)
        tok_path = hf_hub_download(repo_id, "tokenizer.json")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        self._session = ort.InferenceSession(
            model_path, sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self._input_names = {i.name for i in self._session.get_inputs()}
        self._tok = Tokenizer.from_file(tok_path)
        self._tok.enable_truncation(max_length=max_length)
        self._tok.enable_padding(pad_id=1, pad_token="<pad>")  # XLM-R pad id

    def embed(self, texts: Sequence[str]) -> Vector:
        enc = self._tok.encode_batch([f"query: {t}" for t in texts])
        ids = np.array([e.ids for e in enc], dtype=np.int64)
        mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
        feeds: dict[str, NDArray[np.int64]] = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self._input_names:
            feeds["token_type_ids"] = np.zeros_like(ids)
        hidden = self._session.run(None, feeds)[0]  # (B, T, D)
        m = mask[..., None].astype(np.float32)
        pooled = (hidden * m).sum(axis=1) / np.clip(m.sum(axis=1), 1e-9, None)
        norm = np.linalg.norm(pooled, axis=1, keepdims=True)
        return np.asarray(pooled / np.clip(norm, 1e-12, None), dtype=np.float32)
