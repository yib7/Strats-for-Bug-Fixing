"""Prompt construction, chat-template rendering, and fix extraction.

Two easy-to-hit pitfalls this module avoids by design:

1. Truncating retrieved exemplars (e.g. to 200 chars with a literal ``"..."``)
   destroys most of the retrieved context. ``build_messages`` never truncates.
2. Feeding an instruction-tuned model (Qwen2.5-Coder-Instruct) raw concatenated
   text instead of its chat format gives it malformed input. ``render_prompt``
   always calls ``tokenizer.apply_chat_template(..., add_generation_prompt=True)``.
"""

from __future__ import annotations

import re
from typing import Protocol

SYSTEM_PROMPT = (
    "You are an expert Java software engineer specializing in automated program "
    "repair. You will be given a buggy Java method, optionally along with "
    "examples of similar bugs and their fixes. Respond with ONLY the complete "
    "fixed Java method -- no explanation, no markdown code fences, no commentary."
)


class ChatTemplateTokenizer(Protocol):
    def apply_chat_template(
        self, messages: list[dict], tokenize: bool, add_generation_prompt: bool
    ) -> str: ...


def build_messages(buggy: str, exemplars: list[dict]) -> list[dict]:
    """Build a system+user chat message list for the RAG prompt.

    `exemplars` are full, untruncated ``{"buggy": str, "fixed": str}`` dicts
    (retrieved from the train-split KB); each is labeled Buggy/Fixed in full.
    An empty `exemplars` list produces a zero-shot prompt (k=0) with no
    exemplars section.
    """
    sections: list[str] = []

    if exemplars:
        sections.append(
            "Here are examples of similar Java bugs and their fixes. Study how "
            "each buggy method was corrected."
        )
        for i, exemplar in enumerate(exemplars, start=1):
            sections.append(
                f"Example {i}:\nBuggy:\n{exemplar['buggy']}\nFixed:\n{exemplar['fixed']}"
            )

    sections.append(
        "Now fix the following buggy Java method. Output only the complete "
        "fixed method and nothing else."
    )
    sections.append(f"Buggy:\n{buggy}")

    user_content = "\n\n".join(sections)

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def render_prompt(tokenizer: ChatTemplateTokenizer, messages: list[dict]) -> str:
    """Render `messages` through the tokenizer's chat template, ready to generate."""
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


_FENCE_RE = re.compile(r"```[a-zA-Z0-9_+-]*\n?(.*?)```", re.DOTALL)

_PREAMBLE_RE = re.compile(
    r"^\s*"
    r"(here('s| is)|sure|certainly|of course)"
    r"[^\n:]*:\s*\n*",
    re.IGNORECASE,
)

# Punctuation that strongly signals a line is code, not prose.
_CODE_PUNCT = "{}();"

_CODE_KEYWORDS = frozenset(
    {
        "public",
        "private",
        "protected",
        "static",
        "final",
        "abstract",
        "class",
        "interface",
        "void",
        "return",
        "import",
        "package",
        "throw",
        "throws",
        "try",
        "catch",
        "finally",
        "if",
        "else",
        "for",
        "while",
        "switch",
        "case",
        "break",
        "continue",
        "new",
        "int",
        "long",
        "short",
        "byte",
        "char",
        "boolean",
        "double",
        "float",
        "String",
        "@Override",
    }
)


def _score_line(line: str) -> int:
    """Score a single line for code-likeness (positive) vs. prose (negative).

    Comment lines (``//``, ``/*``, ``*``, ``*/`` prefixes) are code context,
    never chatter -- they score positive before any prose/sentence check so
    that sentence-like comments and Javadoc bodies are never stripped.
    """
    stripped = line.strip()
    if not stripped:
        return 0

    if stripped.startswith(("//", "/*", "*")):
        # Line or block comment (incl. Javadoc body/`*/` close): part of the
        # code even when it reads like a natural-language sentence.
        return 1

    has_code_punct = any(ch in stripped for ch in _CODE_PUNCT)
    first_word = stripped.split(None, 1)[0].strip("(){};,") if stripped.split() else ""
    starts_with_keyword = first_word in _CODE_KEYWORDS

    if has_code_punct or starts_with_keyword:
        return 2

    words = stripped.split()
    ends_like_sentence = stripped.endswith((".", "!", "?"))
    if ends_like_sentence and len(words) >= 3:
        # Natural-language sentence: no code punctuation, several words,
        # trailing sentence-ending punctuation.
        return -3

    if line[:1] in (" ", "\t"):
        # Indented but otherwise ambiguous (e.g. a lone identifier) -- weak
        # code signal.
        return 1

    return 0


def _largest_code_block(text: str) -> str:
    """Return the largest contiguous code-like region of `text`.

    Scores each line by code-likeness vs. prose-likeness, then finds the
    contiguous run of lines with the highest total score (Kadane's
    algorithm). This strips leading and/or trailing prose commentary while
    leaving pure code untouched.
    """
    lines = text.splitlines()
    if not lines:
        return text

    scores = [_score_line(line) for line in lines]
    if not any(score > 0 for score in scores):
        # No line looks like code -- nothing to trim, return as-is.
        return text.strip()

    best_sum = scores[0]
    best_start = best_end = 0
    cur_sum = scores[0]
    cur_start = 0
    for i in range(1, len(scores)):
        if cur_sum < 0:
            cur_sum = scores[i]
            cur_start = i
        else:
            cur_sum += scores[i]
        if cur_sum > best_sum:
            best_sum = cur_sum
            best_start, best_end = cur_start, i

    return "\n".join(lines[best_start : best_end + 1]).strip()


def extract_fix(text: str) -> str:
    """Extract the fixed-code answer from raw model output.

    Preference order: the first fenced code block (``` or ```java); otherwise
    strip a common chatty preamble ("Here is the fixed code:", "Sure, ...:")
    and return the largest contiguous code-like region of what remains, which
    handles both leading chatter and trailing explanatory prose. Comment lines
    (``//``, ``/* ... */``, Javadoc) count as code, so commented methods are
    kept intact rather than split at sentence-like comment lines.
    """
    if not text:
        return ""

    fence_match = _FENCE_RE.search(text)
    if fence_match:
        return fence_match.group(1).strip()

    stripped = text.strip()
    without_preamble = _PREAMBLE_RE.sub("", stripped, count=1).strip()
    return _largest_code_block(without_preamble)
