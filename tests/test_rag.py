"""Tests for pop.rag (retrievers, prompt building, generation dispatch)."""

from __future__ import annotations

import json
import sys

import numpy as np
import pytest

from pop.rag.generate import build_generator, generate_with_resume
from pop.rag.prompt import build_messages, extract_fix, render_prompt
from pop.rag.retrievers import BM25Retriever, CodeBERTRetriever

# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

KB_PAIRS = [
    {
        "buggy": "public int add(int a, int b) { return a - b; }",
        "fixed": "public int add(int a, int b) { return a + b; }",
    },
    {
        "buggy": "public String greet(String name) { return null; }",
        "fixed": 'public String greet(String name) { return "Hello " + name; }',
    },
    {
        "buggy": "public boolean isEmpty(List list) { return false; }",
        "fixed": "public boolean isEmpty(List list) { return list.size() == 0; }",
    },
    {
        "buggy": "public int max(int a, int b) { return a; }",
        "fixed": "public int max(int a, int b) { return a > b ? a : b; }",
    },
    {
        "buggy": "public void close(Connection conn) { }",
        "fixed": "public void close(Connection conn) { conn.close(); }",
    },
    {
        "buggy": "public int size(int[] arr) { return 0; }",
        "fixed": "public int size(int[] arr) { return arr.length; }",
    },
    {
        "buggy": "public double average(int[] arr) { return 0.0; }",
        "fixed": "public double average(int[] arr) { return sum(arr) / arr.length; }",
    },
    {
        "buggy": "public String trim(String s) { return s; }",
        "fixed": "public String trim(String s) { return s.trim(); }",
    },
    {
        "buggy": "public boolean contains(List list, Object o) { return false; }",
        "fixed": "public boolean contains(List list, Object o) { return list.indexOf(o) >= 0; }",
    },
    {
        "buggy": "public int subtract(int a, int b) { return a + b; }",
        "fixed": "public int subtract(int a, int b) { return a - b; }",
    },
]


class _StubTokenizer:
    """Hand-rolled stub matching PreTrainedTokenizerBase.apply_chat_template's
    contract, using a tiny local Jinja chat template (no Qwen download)."""

    CHAT_TEMPLATE = (
        "{% for message in messages %}"
        "<|{{ message['role'] }}|>\n{{ message['content'] }}\n"
        "{% endfor %}"
        "{% if add_generation_prompt %}<|assistant|>\n{% endif %}"
    )

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kwargs):
        from jinja2 import Template

        assert tokenize is False
        template = Template(self.CHAT_TEMPLATE)
        return template.render(messages=messages, add_generation_prompt=add_generation_prompt)


# ---------------------------------------------------------------------------
# BM25Retriever
# ---------------------------------------------------------------------------


def test_bm25_retrieves_on_topic_doc():
    retriever = BM25Retriever()
    retriever.index(KB_PAIRS)

    results = retriever.retrieve("public int add(int x, int y) { return x - y; }", k=1)

    assert len(results) == 1
    assert results[0] in KB_PAIRS
    assert "add" in results[0]["buggy"]


def test_bm25_retrieve_respects_k():
    retriever = BM25Retriever()
    retriever.index(KB_PAIRS)

    results = retriever.retrieve("public int add(int a, int b) { return a - b; }", k=3)
    assert len(results) == 3


def test_bm25_retrieve_k_larger_than_kb_size():
    retriever = BM25Retriever()
    retriever.index(KB_PAIRS[:2])
    results = retriever.retrieve("add", k=5)
    assert len(results) == 2


def test_bm25_retrieve_k_zero_returns_empty():
    retriever = BM25Retriever()
    retriever.index(KB_PAIRS)
    assert retriever.retrieve("add", k=0) == []


def test_bm25_retrieve_before_index_raises():
    retriever = BM25Retriever()
    with pytest.raises(RuntimeError):
        retriever.retrieve("add", k=1)


# ---------------------------------------------------------------------------
# CodeBERTRetriever (stubbed encoder -- no model download)
# ---------------------------------------------------------------------------


def _fake_encode_fn(texts: list[str]) -> np.ndarray:
    """Deterministic hash-based fake embedding, no network/model."""
    dim = 16
    vecs = np.zeros((len(texts), dim), dtype="float32")
    for i, text in enumerate(texts):
        for word in text.lower().split():
            vecs[i, hash(word) % dim] += 1.0
    return vecs


def test_codebert_retriever_uses_injected_encode_fn():
    retriever = CodeBERTRetriever(encode_fn=_fake_encode_fn)
    retriever.index(KB_PAIRS)

    results = retriever.retrieve("public int add(int a, int b) { return a - b; }", k=1)

    assert len(results) == 1
    assert results[0] in KB_PAIRS


def test_codebert_retriever_respects_k():
    retriever = CodeBERTRetriever(encode_fn=_fake_encode_fn)
    retriever.index(KB_PAIRS)
    results = retriever.retrieve("add", k=4)
    assert len(results) == 4


def test_codebert_retriever_before_index_raises():
    retriever = CodeBERTRetriever(encode_fn=_fake_encode_fn)
    with pytest.raises(RuntimeError):
        retriever.retrieve("add", k=1)


def test_codebert_retriever_does_not_import_torch_without_default_encode(monkeypatch):
    # Constructing/using with an injected encode_fn must never need torch/transformers.
    import sys

    for mod in ("torch", "transformers"):
        monkeypatch.setitem(sys.modules, mod, None)  # poison import
    retriever = CodeBERTRetriever(encode_fn=_fake_encode_fn)
    retriever.index(KB_PAIRS)
    retriever.retrieve("add", k=1)


def test_codebert_index_batches_encode_and_never_passes_whole_kb_at_once():
    # Regression: index() once handed the entire ~52k-pair train split to encode()
    # in a single call, building one [N, seq, hidden] tensor that OOM-killed the
    # process (SIGKILL / exit -9). It must chunk to batch_size instead.
    call_sizes: list[int] = []

    def recording_encode(texts: list[str]) -> np.ndarray:
        call_sizes.append(len(texts))
        return _fake_encode_fn(texts)

    retriever = CodeBERTRetriever(encode_fn=recording_encode, batch_size=3)
    retriever.index(KB_PAIRS)  # KB_PAIRS has >3 entries -> must be >1 chunk

    assert len(call_sizes) > 1, "index() did not batch -- whole KB passed in one encode() call"
    assert all(n <= 3 for n in call_sizes), f"a batch exceeded batch_size: {call_sizes}"
    assert sum(call_sizes) == len(KB_PAIRS), "every KB pair must be encoded exactly once"

    # Chunked embeddings must be concatenated in order, so retrieval still works.
    results = retriever.retrieve("public int add(int a, int b) { return a - b; }", k=1)
    assert len(results) == 1
    assert results[0] in KB_PAIRS


# ---------------------------------------------------------------------------
# build_messages
# ---------------------------------------------------------------------------


def test_build_messages_zero_shot_has_no_exemplars_section():
    messages = build_messages("public int add(int a, int b) { return a - b; }", [])
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "Example" not in messages[1]["content"]
    assert "add" in messages[1]["content"]


def test_build_messages_preserves_long_exemplar_verbatim():
    # Regression test vs the old-notebook bug: exemplars were truncated at 200
    # chars with a literal "..." appended. The rebuilt prompt must never do that.
    long_buggy = "public void method() {\n" + ("    int x = 1;\n" * 50) + "}"
    long_fixed = "public void method() {\n" + ("    int x = 2;\n" * 50) + "}"
    assert len(long_buggy) > 200
    assert len(long_fixed) > 200

    exemplars = [{"buggy": long_buggy, "fixed": long_fixed}]
    messages = build_messages("public int add(int a, int b) { return a - b; }", exemplars)
    user_content = messages[1]["content"]

    assert long_buggy in user_content
    assert long_fixed in user_content
    assert "..." not in user_content


def test_build_messages_labels_buggy_and_fixed():
    exemplars = [{"buggy": "BUGGY_MARKER", "fixed": "FIXED_MARKER"}]
    messages = build_messages("query buggy code", exemplars)
    content = messages[1]["content"]
    assert "BUGGY_MARKER" in content
    assert "FIXED_MARKER" in content
    assert "Buggy" in content
    assert "Fixed" in content


def test_build_messages_instructs_output_only_fixed_code():
    messages = build_messages("buggy", [])
    full_text = messages[0]["content"] + messages[1]["content"]
    assert "only" in full_text.lower()


# ---------------------------------------------------------------------------
# render_prompt
# ---------------------------------------------------------------------------


def test_render_prompt_applies_chat_template_with_generation_prompt():
    tokenizer = _StubTokenizer()
    messages = build_messages("public int add(int a, int b) { return a - b; }", [])
    rendered = render_prompt(tokenizer, messages)

    assert "<|system|>" in rendered
    assert "<|user|>" in rendered
    assert rendered.rstrip().endswith("<|assistant|>")


# ---------------------------------------------------------------------------
# extract_fix
# ---------------------------------------------------------------------------


def test_extract_fix_fenced_with_language_tag():
    text = "Here you go:\n```java\npublic int add(int a, int b) { return a + b; }\n```\nDone."
    assert extract_fix(text) == "public int add(int a, int b) { return a + b; }"


def test_extract_fix_fenced_without_language_tag():
    text = "```\npublic int add(int a, int b) { return a + b; }\n```"
    assert extract_fix(text) == "public int add(int a, int b) { return a + b; }"


def test_extract_fix_bare_code():
    text = "public int add(int a, int b) { return a + b; }"
    assert extract_fix(text) == "public int add(int a, int b) { return a + b; }"


def test_extract_fix_chatty_preamble():
    text = "Here is the fixed code:\n\npublic int add(int a, int b) { return a + b; }"
    result = extract_fix(text)
    assert result == "public int add(int a, int b) { return a + b; }"
    assert "Here is" not in result


def test_extract_fix_chatty_preamble_variant():
    text = "Sure, here's the corrected method:\npublic int add(int a, int b) { return a + b; }"
    result = extract_fix(text)
    assert "Sure" not in result
    assert "public int add" in result


def test_extract_fix_trailing_explanation():
    text = (
        "public int add(int a, int b) {\n"
        "    return a + b;\n"
        "}\n\n"
        "This method now correctly returns the sum of the two arguments."
    )
    result = extract_fix(text)
    assert result == "public int add(int a, int b) {\n    return a + b;\n}"
    assert "correctly" not in result


def test_extract_fix_multiple_trailing_sentences():
    text = (
        "public int add(int a, int b) {\n"
        "    return a + b;\n"
        "}\n"
        "This fixes the bug where subtraction was used instead of addition.\n"
        "The method now passes all provided test cases."
    )
    result = extract_fix(text)
    assert result == "public int add(int a, int b) {\n    return a + b;\n}"
    assert "fixes the bug" not in result
    assert "test cases" not in result


def test_extract_fix_leading_and_trailing_chatter():
    text = (
        "Let me walk through the fix for this method.\n"
        "public int add(int a, int b) {\n"
        "    return a + b;\n"
        "}\n"
        "This should resolve the issue described above."
    )
    result = extract_fix(text)
    assert result == "public int add(int a, int b) {\n    return a + b;\n}"
    assert "walk through" not in result
    assert "resolve the issue" not in result


def test_extract_fix_pure_code_multiline_unchanged():
    text = "public int add(int a, int b) {\n    int sum = a + b;\n    return sum;\n}"
    assert extract_fix(text) == text


def test_extract_fix_keeps_sentence_like_line_comment():
    text = (
        "public int add(int a, int b) {\n"
        "    // Use addition instead of subtraction.\n"
        "    return a + b;\n"
        "}"
    )
    assert extract_fix(text) == text


def test_extract_fix_keeps_javadoc_comment():
    text = (
        "/**\n"
        " * Adds two integers together.\n"
        " * This method replaces the buggy subtraction.\n"
        " */\n"
        "public int add(int a, int b) {\n"
        "    return a + b;\n"
        "}"
    )
    assert extract_fix(text) == text


def test_extract_fix_javadoc_method_with_trailing_chatter():
    text = (
        "/**\n"
        " * Adds two integers together.\n"
        " */\n"
        "public int add(int a, int b) {\n"
        "    // Sum the operands.\n"
        "    return a + b;\n"
        "}\n"
        "This should resolve the reported issue."
    )
    result = extract_fix(text)
    assert result == (
        "/**\n"
        " * Adds two integers together.\n"
        " */\n"
        "public int add(int a, int b) {\n"
        "    // Sum the operands.\n"
        "    return a + b;\n"
        "}"
    )


def test_extract_fix_fenced_preferred_over_trailing_chatter():
    text = (
        "```java\n"
        "public int add(int a, int b) { return a + b; }\n"
        "```\n"
        "This is the fixed version of the method."
    )
    assert extract_fix(text) == "public int add(int a, int b) { return a + b; }"


# ---------------------------------------------------------------------------
# build_generator dispatch / fallback / kwarg-normalization
# (no model download -- generator factories injected as fakes)
# ---------------------------------------------------------------------------


def test_build_generator_uses_vllm_when_importable(monkeypatch):
    import pop.rag.generate as gen_mod

    monkeypatch.setattr(gen_mod, "_vllm_importable", lambda: True)
    made: dict = {}

    def fake_vllm_factory(model_name, sampling):
        made["backend"] = "vllm"
        made["sampling"] = sampling
        return lambda prompts: [f"v:{p}" for p in prompts]

    gen = gen_mod.build_generator("m", vllm_generator_factory=fake_vllm_factory, max_new_tokens=123)

    assert made["backend"] == "vllm"
    # vLLM's SamplingParams uses `max_tokens`, and temperature 0 == greedy.
    assert made["sampling"]["max_tokens"] == 123
    assert made["sampling"]["temperature"] == 0.0
    assert gen(["p1", "p2"]) == ["v:p1", "v:p2"]


def test_build_generator_uses_transformers_when_vllm_not_importable(monkeypatch):
    import pop.rag.generate as gen_mod

    monkeypatch.setattr(gen_mod, "_vllm_importable", lambda: False)
    made: dict = {}

    def fake_tf_factory(model_name, gen_kwargs):
        made["gen_kwargs"] = gen_kwargs
        return lambda prompts: [f"t:{p}" for p in prompts]

    gen = gen_mod.build_generator(
        "m", transformers_generator_factory=fake_tf_factory, max_new_tokens=77
    )

    assert made["gen_kwargs"]["max_new_tokens"] == 77
    assert made["gen_kwargs"]["do_sample"] is False
    assert made["gen_kwargs"]["batch_size"] == 16  # batched, not one-at-a-time
    assert gen(["p"]) == ["t:p"]


def test_build_generator_falls_back_to_transformers_when_vllm_init_fails(monkeypatch):
    import pop.rag.generate as gen_mod

    monkeypatch.setattr(gen_mod, "_vllm_importable", lambda: True)
    made: dict = {}

    def boom_factory(model_name, sampling):
        raise RuntimeError("libcudart.so.13: cannot open shared object file")

    def fake_tf_factory(model_name, gen_kwargs):
        made["hit"] = True
        return lambda prompts: list(prompts)

    gen = gen_mod.build_generator(
        "m", vllm_generator_factory=boom_factory, transformers_generator_factory=fake_tf_factory
    )

    assert made.get("hit") is True  # fell back after vLLM engine-init failure
    assert gen(["p"]) == ["p"]


def test_build_generator_forced_vllm_does_not_fall_back():
    def boom_factory(model_name, sampling):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        build_generator("m", backend="vllm", vllm_generator_factory=boom_factory)


def test_build_generator_explicit_gen_kwargs_override_greedy_default(monkeypatch):
    import pop.rag.generate as gen_mod

    monkeypatch.setattr(gen_mod, "_vllm_importable", lambda: True)
    made: dict = {}

    def fake_vllm_factory(model_name, sampling):
        made["sampling"] = sampling
        return lambda prompts: list(prompts)

    gen_mod.build_generator("m", vllm_generator_factory=fake_vllm_factory, temperature=0.7)
    assert made["sampling"]["temperature"] == 0.7  # caller override wins over greedy default


# ---------------------------------------------------------------------------
# the REAL transformers closure (_default_transformers_generator), driven with a
# fake `pipeline` -- no model download. This is the function that turns model
# output into arm-C predictions, so it gets executed, not just dispatched to.
# ---------------------------------------------------------------------------

_FAKE_COMPLETION = "public int add(int a, int b) { return a + b; }"


class _FakeTokenizer:
    pad_token_id = None
    pad_token = None
    eos_token = "<|endoftext|>"
    padding_side = "right"
    clean_up_tokenization_spaces = True


class _FakeGenerationConfig:
    max_length = 20


class _FakeModel:
    def __init__(self) -> None:
        self.generation_config = _FakeGenerationConfig()


class _FakePipeline:
    """Mimics `transformers.pipeline("text-generation")`'s return_full_text contract.

    With `return_full_text=False` it yields the completion alone. Otherwise (the
    library default) it yields *re-decoded prompt* + completion -- and the re-decoded
    prompt is deliberately NOT byte-identical to the input string here (a trailing
    newline is lost), which is exactly the round-trip drift that made the old
    prefix-strip fall through and return the whole prompt.
    """

    def __init__(self) -> None:
        self.tokenizer = _FakeTokenizer()
        self.model = _FakeModel()
        self.calls: list[dict] = []

    def __call__(self, prompts, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("return_full_text") is False:
            return [[{"generated_text": _FAKE_COMPLETION}] for _ in prompts]
        return [[{"generated_text": p.strip() + _FAKE_COMPLETION}] for p in prompts]


@pytest.fixture
def fake_pipeline(monkeypatch):
    """Swap `transformers.pipeline` for the fake above, so the closure runs with no model.

    The attribute is *read* before it is patched, on purpose. `transformers` installs a
    lazy module, and resolving one of its lazy attributes for the first time replaces the
    object registered in ``sys.modules["transformers"]``. Patching the pre-read object
    would therefore write to an orphan: `_default_transformers_generator` re-resolves
    `from transformers import pipeline` at call time against `sys.modules`, get the real
    loader, and try to download the model (observed: an outbound request to
    huggingface.co). Read first, then patch whatever `sys.modules` ended up holding.
    """
    import transformers

    _ = transformers.pipeline  # materialize the lazy module before choosing the target
    pipe = _FakePipeline()
    monkeypatch.setattr(sys.modules["transformers"], "pipeline", lambda *a, **k: pipe)
    return pipe


def test_transformers_generator_never_returns_the_prompt(fake_pipeline):
    """A prompt the pipeline does not echo back byte-identically must not leak into output.

    The old implementation recovered the completion with
    ``full_text[len(prompt):] if full_text.startswith(prompt) else full_text`` -- so when
    the re-decoded prompt drifted by a single character the ELSE branch returned prompt +
    completion, and `extract_fix` would then pick a retrieved exemplar's *fixed* method
    out of the KB block and emit it as the model's prediction.
    """
    import pop.rag.generate as gen_mod

    exemplar = 'public String greet(String n) { return "Hello " + n; }'
    prompt = f"### Example fixed method:\n{exemplar}\n### Fix this:\npublic int add() {{}}\n"

    gen = gen_mod._default_transformers_generator("m", {"max_new_tokens": 8})
    out = gen([prompt])

    assert out == [_FAKE_COMPLETION]
    assert exemplar not in out[0]  # a retrieved exemplar must never become the prediction
    assert prompt not in out[0]


def test_transformers_generator_asks_the_pipeline_to_strip_the_prompt(fake_pipeline):
    """The prompt is stripped by the tokenizer-aware pipeline, not by string surgery.

    Same contract as the LoRA arm's twin (`pop.train.lora._default_lora_generator`), so
    the two arms cannot drift apart on how a completion is recovered.
    """
    import pop.rag.generate as gen_mod

    gen = gen_mod._default_transformers_generator("m", {"max_new_tokens": 8, "batch_size": 4})
    gen(["p1", "p2"])

    assert fake_pipeline.calls[0]["return_full_text"] is False
    assert fake_pipeline.calls[0]["max_new_tokens"] == 8  # caller gen_kwargs still forwarded


def test_transformers_generator_configures_the_tokenizer_for_batched_decoding(fake_pipeline):
    """Decoder-only batching needs a pad token and left padding (Qwen ships neither)."""
    import pop.rag.generate as gen_mod

    gen_mod._default_transformers_generator("m", {})

    assert fake_pipeline.tokenizer.pad_token == fake_pipeline.tokenizer.eos_token
    assert fake_pipeline.tokenizer.padding_side == "left"
    assert fake_pipeline.model.generation_config.max_length is None


def test_transformers_generator_handles_a_bare_dict_result(monkeypatch):
    """Some pipeline versions return a bare dict per prompt rather than a 1-element list."""
    import transformers

    import pop.rag.generate as gen_mod

    class _FlatPipeline(_FakePipeline):
        def __call__(self, prompts, **kwargs):
            return [{"generated_text": _FAKE_COMPLETION} for _ in prompts]

    _ = transformers.pipeline  # see the fake_pipeline fixture for why this read comes first
    monkeypatch.setattr(sys.modules["transformers"], "pipeline", lambda *a, **k: _FlatPipeline())
    gen = gen_mod._default_transformers_generator("m", {})
    assert gen(["p1", "p2"]) == [_FAKE_COMPLETION, _FAKE_COMPLETION]


# ---------------------------------------------------------------------------
# generate_with_resume: chunked, checkpointed, atomically-finalized writing
# ---------------------------------------------------------------------------


def test_generate_with_resume_writes_all_predictions_atomically(tmp_path):
    out = tmp_path / "predictions.jsonl"
    prompts = [f"p{i}" for i in range(5)]
    refs = [f"r{i}" for i in range(5)]

    n = generate_with_resume(prompts, refs, out, lambda ps: [f"fix:{p}" for p in ps], chunk_size=2)

    assert n == 5
    assert not (tmp_path / "predictions.jsonl.partial").exists()  # partial cleaned up
    lines = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines()]
    assert lines[0] == {"prediction": "fix:p0", "reference": "r0"}
    assert lines[4] == {"prediction": "fix:p4", "reference": "r4"}


def test_generate_with_resume_continues_after_interruption(tmp_path):
    out = tmp_path / "predictions.jsonl"
    partial = tmp_path / "predictions.jsonl.partial"
    prompts = [f"p{i}" for i in range(5)]
    refs = [f"r{i}" for i in range(5)]

    calls: list[list[str]] = []

    def flaky(ps):
        calls.append(list(ps))
        if len(calls) == 2:  # blow up on the second chunk
            raise RuntimeError("simulated disconnect")
        return [f"fix:{p}" for p in ps]

    with pytest.raises(RuntimeError):
        generate_with_resume(prompts, refs, out, flaky, chunk_size=2)

    assert not out.exists()  # never finalized
    # 1st chunk saved
    assert partial.exists()
    assert len(partial.read_text(encoding="utf-8").splitlines()) == 2

    seen: list[str] = []

    def ok(ps):
        seen.extend(ps)
        return [f"fix:{p}" for p in ps]

    n = generate_with_resume(prompts, refs, out, ok, chunk_size=2)

    assert n == 5
    assert seen == ["p2", "p3", "p4"]  # already-done prompts are NOT regenerated
    assert not partial.exists()
    preds = [json.loads(x)["prediction"] for x in out.read_text(encoding="utf-8").splitlines()]
    assert preds == [f"fix:p{i}" for i in range(5)]


def test_generate_with_resume_empty_prompts_writes_empty_file(tmp_path):
    out = tmp_path / "predictions.jsonl"
    n = generate_with_resume([], [], out, lambda ps: [], chunk_size=2)
    assert n == 0
    assert out.exists() and out.read_text(encoding="utf-8") == ""


def test_generate_with_resume_rejects_length_mismatch(tmp_path):
    out = tmp_path / "predictions.jsonl"
    with pytest.raises(ValueError):
        generate_with_resume(["p0", "p1"], ["r0"], out, lambda ps: list(ps))


# --- resume from a torn checkpoint -------------------------------------------------------


def test_resume_discards_a_torn_trailing_write(tmp_path):
    """A process killed mid-`write` leaves a partial JSON object with no newline. It still
    counted as a line, so the next append concatenated onto it and produced one unparseable
    record in the middle of the finished file."""
    out = tmp_path / "predictions.jsonl"
    partial = tmp_path / "predictions.jsonl.partial"
    partial.write_text(
        '{"prediction": "fix:p0", "reference": "r0"}\n'
        '{"prediction": "fix:p1", "reference": "r1"}\n'
        '{"prediction": "fix:p2", "refere',  # <- torn, no trailing newline
        encoding="utf-8",
    )

    prompts = [f"p{i}" for i in range(4)]
    refs = [f"r{i}" for i in range(4)]
    seen: list[str] = []

    def generate(chunk):
        seen.extend(chunk)
        return [f"fix:{p}" for p in chunk]

    n = generate_with_resume(prompts, refs, out, generate, chunk_size=2)

    assert n == 4
    assert seen == ["p2", "p3"], "the torn record must be regenerated, not kept"
    lines = out.read_text(encoding="utf-8").splitlines()
    assert [json.loads(x)["prediction"] for x in lines] == [f"fix:p{i}" for i in range(4)]


def test_resume_discards_a_trailing_line_that_is_not_json(tmp_path):
    out = tmp_path / "predictions.jsonl"
    partial = tmp_path / "predictions.jsonl.partial"
    partial.write_text(
        '{"prediction": "fix:p0", "reference": "r0"}\n{"predi\n',
        encoding="utf-8",
    )

    n = generate_with_resume(
        ["p0", "p1"], ["r0", "r1"], out, lambda c: [f"fix:{p}" for p in c], chunk_size=2
    )

    assert n == 2
    lines = out.read_text(encoding="utf-8").splitlines()
    assert [json.loads(x)["prediction"] for x in lines] == ["fix:p0", "fix:p1"]


def test_intact_partial_is_left_alone(tmp_path):
    out = tmp_path / "predictions.jsonl"
    partial = tmp_path / "predictions.jsonl.partial"
    partial.write_text('{"prediction": "fix:p0", "reference": "r0"}\n', encoding="utf-8")

    seen: list[str] = []

    def generate(chunk):
        seen.extend(chunk)
        return [f"fix:{p}" for p in chunk]

    generate_with_resume(["p0", "p1"], ["r0", "r1"], out, generate, chunk_size=2)
    assert seen == ["p1"], "a complete record must not be regenerated"
