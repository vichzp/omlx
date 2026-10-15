# SPDX-License-Identifier: Apache-2.0
"""
Thinking/reasoning content parser for separating <think>...</think> blocks.

Provides both streaming (ThinkingParser) and non-streaming (extract_thinking)
interfaces for separating reasoning content from regular response content.

Used by reasoning models like DeepSeek R1, Qwen3/3.5, MiniMax that wrap
their chain-of-thought reasoning in <think>...</think> tags.
"""

import codecs
import re
import weakref
from collections.abc import Callable, Sequence
from typing import Any, List, Optional, Tuple

# Tags used for thinking blocks
_OPEN_TAG = "<think>"
_CLOSE_TAG = "</think>"
_OPEN_LEN = len(_OPEN_TAG)   # 7
_CLOSE_LEN = len(_CLOSE_TAG)  # 8
_MINIMAX_OPEN_TAG = "<mm:think>"
_MINIMAX_CLOSE_TAG = "</mm:think>"
_HY3_OPEN_TAG = "<think:opensource>"
_HY3_CLOSE_TAG = "</think:opensource>"

# Regex for non-streaming extraction (complete text)
_THINKING_PATTERN = re.compile(r'<think>(.*?)</think>', re.DOTALL)
# Handle case where <think> is missing but </think> is present
# (scheduler prepends <think>\n but the tag may be split)
_THINKING_TAIL_PATTERN = re.compile(r'^(.*?)</think>', re.DOTALL)


def _gpt2_byte_decoder() -> dict[str, int]:
    """Inverse of GPT-2's ``bytes_to_unicode`` table (byte-level BPE)."""
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\u00a1"), ord("\u00ac") + 1))
        + list(range(ord("\u00ae"), ord("\u00ff") + 1))
    )
    codepoints = list(printable)
    extra = 0
    for byte in range(256):
        if byte not in printable:
            printable.append(byte)
            codepoints.append(256 + extra)
            extra += 1
    return {chr(cp): byte for byte, cp in zip(printable, codepoints)}


_GPT2_BYTE_DECODER = _gpt2_byte_decoder()

# Ceiling for the ``thinking_budget_finish_sentence`` switch: once the budget
# is exhausted mid-sentence, the model may sample at most this many extra
# tokens looking for a sentence or line end before the close is forced.
THINKING_BUDGET_FINISH_SENTENCE_MAX_TOKENS = 40

# A Western ``.`` ``!`` ``?`` only ends a sentence when the next token
# starts with whitespace ("3.5", "e.g", "file.py" do not). After such a
# token the grace window closes at the next position when the model puts
# more than this probability mass on whitespace-led tokens.
THINKING_BUDGET_FINISH_SENTENCE_WHITESPACE_THRESHOLD = 0.5

# Reasoning boundary for the thinking-budget grace window, judged on a piece
# once trailing whitespace, closing quotes/brackets and markdown emphasis are
# stripped. A newline anywhere in the piece, or a CJK full stop, is a hard
# boundary; a Western sentence end is a soft one (see the threshold above).
_HARD_SENTENCE_END_CHARS = ("\u3002", "\uff01", "\uff1f")
_SOFT_SENTENCE_END_CHARS = (".", "!", "?")
_SENTENCE_TRAILING_CHARS = " \t\"'*)]}\u201d\u2019\u300d\u300f\uff09"

_BOUNDARY_NONE = 0
_BOUNDARY_HARD = 1
_BOUNDARY_SOFT = 2

_BYTE_FALLBACK_PIECE = re.compile(r"<0x([0-9A-Fa-f]{2})>")
_SENTENCEPIECE_SPACE = "\u2581"


def _piece_starts_with_whitespace(piece) -> bool:
    """True when a vocabulary piece decodes to text led by whitespace.

    Handles SentencePiece (``\u2581`` word marker, ``<0x0A>`` byte
    fallback), GPT-2 byte-level BPE (``\u0120`` space, ``\u010a`` newline)
    and plain pieces. A zero-width space does not count.
    """
    if not piece:
        return False
    if isinstance(piece, bytes):
        text = piece.decode("utf-8", errors="ignore")
        return bool(text) and text[0].isspace()
    first = piece[0]
    if first == _SENTENCEPIECE_SPACE:
        return True
    fallback = _BYTE_FALLBACK_PIECE.match(piece)
    if fallback is not None:
        return chr(int(fallback.group(1), 16)).isspace()
    byte = _GPT2_BYTE_DECODER.get(first)
    if byte is not None:
        return byte < 0x80 and chr(byte).isspace()
    return first.isspace() and first != "\u200b"


# tokenizer -> {logits vocab size: mask or None}
_WHITESPACE_MASKS: "weakref.WeakKeyDictionary[Any, dict]" = weakref.WeakKeyDictionary()


def leading_whitespace_mask(tokenizer, vocab_size: int):
    """Boolean ``(vocab_size,)`` mask of token ids whose piece starts with
    whitespace, cached per tokenizer and logits width.

    Ids past the tokenizer's vocabulary (padded embedding rows) are False.
    Returns ``None`` when the vocabulary cannot be read or is wider than the
    logits (e.g. a reduced draft vocabulary), so callers can fall back.
    """
    try:
        cache = _WHITESPACE_MASKS.setdefault(tokenizer, {})
    except TypeError:
        cache = {}
    if vocab_size in cache:
        return cache[vocab_size]
    mask = None
    try:
        import mlx.core as mx

        try:
            size = len(tokenizer)
        except TypeError:
            size = int(tokenizer.vocab_size)
        if 0 < size <= vocab_size:
            pieces = tokenizer.convert_ids_to_tokens(list(range(size)))
            flags = [_piece_starts_with_whitespace(p) for p in pieces]
            if len(flags) == size:
                flags.extend([False] * (vocab_size - size))
                mask = mx.array(flags, dtype=mx.bool_)
    except Exception:
        mask = None
    cache[vocab_size] = mask
    return mask


def _safe_tokenizer_attr(tokenizer, attr: str, default=None):
    if tokenizer is None:
        return default
    try:
        return getattr(tokenizer, attr, default)
    except (AttributeError, TypeError, ValueError):
        return default


def _single_token_id(value) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _convert_token_to_id(tokenizer, token: str) -> int | None:
    convert = _safe_tokenizer_attr(tokenizer, "convert_tokens_to_ids")
    if not callable(convert):
        return None
    try:
        token_id = convert(token)
    except (AttributeError, KeyError, TypeError, ValueError):
        return None
    if token_id == _safe_tokenizer_attr(tokenizer, "unk_token_id"):
        return None
    return _single_token_id(token_id)


def _encode_prompt_ids(tokenizer, prompt: str) -> list[int] | None:
    encode = _safe_tokenizer_attr(tokenizer, "encode")
    if not callable(encode):
        return None
    try:
        return list(encode(prompt, add_special_tokens=False))
    except TypeError:
        try:
            return list(encode(prompt))
        except Exception:
            return None
    except Exception:
        return None


def _think_end_token_ids(tokenizer) -> list[int] | None:
    think_end_id = _single_token_id(_safe_tokenizer_attr(tokenizer, "think_end_id"))
    if think_end_id is not None:
        return [think_end_id]

    think_end_tag = _safe_tokenizer_attr(tokenizer, "think_end", _CLOSE_TAG)
    encoded = _encode_prompt_ids(tokenizer, think_end_tag or _CLOSE_TAG)
    if encoded:
        return encoded

    token_id = _convert_token_to_id(tokenizer, _CLOSE_TAG)
    if token_id is not None:
        return [token_id]
    return None


def prompt_opens_thinking(
    tokenizer,
    prompt: str,
    prompt_token_ids: Sequence[int] | None = None,
) -> tuple[bool, str]:
    """Return whether a raw prompt would make the engine prepend ``<think>``.

    Presentation-layer stripping must mirror the engine/scheduler decision, not
    just the raw text suffix. Some prompts can contain a literal ``<think>``
    without tokenizing to the model's think-start id, and templates can leave
    the think-start token in the final token tail without the raw string ending
    in the visible tag. When the caller already has prompt ids from the same
    tokenizer path as the scheduler, those ids are authoritative.
    """
    think_tag = (
        _safe_tokenizer_attr(tokenizer, "think_start", _OPEN_TAG) or _OPEN_TAG
    )
    if tokenizer is None:
        return prompt.rstrip().endswith(think_tag), think_tag

    think_start_id = _single_token_id(
        _safe_tokenizer_attr(tokenizer, "think_start_id")
    )
    if think_start_id is None:
        think_start_id = _convert_token_to_id(tokenizer, think_tag)
    if think_start_id is None:
        return False, think_tag

    if prompt_token_ids is None:
        prompt_ids = _encode_prompt_ids(tokenizer, prompt)
    else:
        prompt_ids = list(prompt_token_ids)
    if not prompt_ids or not think_start_id:
        return False, think_tag

    last_tokens = list(prompt_ids[-3:])
    if think_start_id not in last_tokens:
        return False, think_tag

    last_idx = len(last_tokens) - 1 - last_tokens[::-1].index(think_start_id)
    after_start = last_tokens[last_idx + 1 :]

    if after_start:
        think_end_ids = _think_end_token_ids(tokenizer)
        if think_end_ids and think_end_ids[0] in after_start:
            return False, think_tag

    return True, think_tag


def extract_thinking(
    text: str, *, truncated: bool = False, start_in_thinking: bool = False
) -> Tuple[str, str]:
    """Extract thinking and content from complete text.

    Handles:
    - Normal: ``<think>reasoning</think>answer`` → ``("reasoning", "answer")``
    - No thinking: ``just answer`` → ``("", "just answer")``
    - Partial (no open tag): ``reasoning</think>answer`` → ``("reasoning", "answer")``
    - Empty think: ``<think></think>answer`` → ``("", "answer")``
    - Think only: ``<think>reasoning</think>`` → ``("reasoning", "")``
    - Malformed (open with no close): ``<think>everything…`` →
      ``("", "everything…")`` — recovery for V4-style models that
      occasionally skip the ``</think>`` boundary token. Without this
      fallback the entire body would be classified as thinking and the
      visible answer would be empty.

    With ``truncated=True``, unfinished thinking stays in the thinking channel,
    including a block the prompt opened (``start_in_thinking``).

    Other tag-free text is always classified as content. Mirrors
    ``ThinkingParser.finish()`` recovery semantics (`_content_emitted`
    fallback): when the model emits no thinking markers, surface the body
    as the answer so the response is never empty.

    Args:
        text: Complete model output text.
        truncated: Keep unfinished thinking in its channel on length termination.
        start_in_thinking: The prompt opened a think block before ``text``.

    Returns:
        Tuple of (thinking_content, regular_content).
    """
    if not text:
        return ("", "")

    text = (
        text.replace(_MINIMAX_OPEN_TAG, _OPEN_TAG)
        .replace(_MINIMAX_CLOSE_TAG, _CLOSE_TAG)
        .replace(_HY3_OPEN_TAG, _OPEN_TAG)
        .replace(_HY3_CLOSE_TAG, _CLOSE_TAG)
    )

    if (
        truncated
        and start_in_thinking
        and _OPEN_TAG not in text
        and _CLOSE_TAG not in text
    ):
        return (text.strip(), "")

    thinking_parts = []
    remaining = text

    # Extract all <think>...</think> blocks
    while True:
        match = _THINKING_PATTERN.search(remaining)
        if not match:
            break
        thinking_parts.append(match.group(1))
        remaining = remaining[:match.start()] + remaining[match.end():]

    if truncated and _OPEN_TAG in remaining:
        before, after = remaining.split(_OPEN_TAG, 1)
        thinking_parts.append(after)
        remaining = before

    if thinking_parts:
        thinking = "\n".join(thinking_parts).strip()
        return (thinking, remaining.strip())

    # Handle partial: content before </think> without <think> tag
    if '</think>' in text and '<think>' not in text:
        match = _THINKING_TAIL_PATTERN.match(text)
        if match:
            thinking = match.group(1).strip()
            remaining = text[match.end():].strip()
            return (thinking, remaining)

    # Malformed: <think> opened but never closed. Drop the open tag and
    # treat the remainder as content so the answer body is not empty.
    if '<think>' in text and '</think>' not in text:
        idx = text.index('<think>')
        before = text[:idx]
        after = text[idx + _OPEN_LEN:]
        return ("", (before + after).strip())

    return ("", text)


class ThinkingParser:
    """Stateful streaming parser for separating <think>...</think> from content.

    Handles streaming chunks where tags may span multiple chunks.
    Returns (thinking_delta, content_delta) tuples for each feed() call.

    Example::

        parser = ThinkingParser()

        # Chunk 1: "<think>Let me"
        t, c = parser.feed("<think>Let me")
        # t = "Let me", c = ""

        # Chunk 2: " think</think>Answer"
        t, c = parser.feed(" think</think>Answer")
        # t = " think", c = "Answer"

        # Flush remaining
        t, c = parser.finish()
    """

    def __init__(self, start_in_thinking: bool = False):
        self._in_thinking: bool = start_in_thinking
        self._buffer: str = ""  # Buffer for potential partial tags
        # Recovery state for malformed thinking: when the prompt prepends
        # ``<think>`` and the model never emits ``</think>`` before EOS,
        # everything we streamed went out as thinking. The streamed events
        # cannot be retracted, so finish() emits the accumulated thinking
        # text once more as content — the client will show both panels but
        # the answer body is no longer empty.
        self._close_seen: bool = False
        self._thinking_accumulated: List[str] = []
        self._content_emitted: bool = False

    def feed(self, text: str) -> Tuple[str, str]:
        """Feed a text chunk, return (thinking_delta, content_delta).

        Args:
            text: New text chunk from model output.

        Returns:
            Tuple of (thinking_text, content_text) extracted from this chunk.
        """
        if not text:
            return ("", "")

        # Prepend any buffered partial tag content
        text = self._buffer + text
        self._buffer = ""

        thinking_out = []
        content_out = []

        i = 0
        while i < len(text):
            if text[i] == '<':
                # Check if this could be a tag start
                remaining = text[i:]

                # Try to match <think>
                if remaining.startswith(_OPEN_TAG):
                    self._in_thinking = True
                    i += _OPEN_LEN
                    continue

                if remaining.startswith(_HY3_OPEN_TAG):
                    self._in_thinking = True
                    i += len(_HY3_OPEN_TAG)
                    continue

                # Try to match </think>
                if remaining.startswith(_CLOSE_TAG):
                    self._in_thinking = False
                    self._close_seen = True
                    i += _CLOSE_LEN
                    continue

                if remaining.startswith(_HY3_CLOSE_TAG):
                    self._in_thinking = False
                    self._close_seen = True
                    i += len(_HY3_CLOSE_TAG)
                    continue

                # Check if it could be a partial tag (not enough chars yet)
                if self._could_be_tag(remaining):
                    # Buffer the rest and wait for more data
                    self._buffer = remaining
                    break

                # Not a tag, emit the '<' as regular content
                if self._in_thinking:
                    thinking_out.append('<')
                else:
                    content_out.append('<')
                i += 1
            else:
                if self._in_thinking:
                    thinking_out.append(text[i])
                else:
                    content_out.append(text[i])
                i += 1

        thinking_delta = "".join(thinking_out)
        content_delta = "".join(content_out)
        if thinking_delta:
            self._thinking_accumulated.append(thinking_delta)
        if content_delta:
            self._content_emitted = True
        return (thinking_delta, content_delta)

    def finish(
        self, *, truncated: bool = False, has_tool_calls: bool = False
    ) -> Tuple[str, str]:
        """Flush any remaining buffered content.

        Should be called when the stream is complete to emit any
        buffered characters that were waiting for potential tag completion.
        Unless truncated or ended by tool calls, recover an unclosed thinking block as content when no answer was emitted.

        Returns:
            Tuple of (thinking_text, content_text) from remaining buffer
            (plus recovered content if applicable).
        """
        partial = self._buffer
        self._buffer = ""

        # Recovery: prompt opened a thinking block (or model echoed
        # ``<think>`` itself), the close tag never arrived, and nothing
        # ever streamed as content. Re-emit the accumulated thinking text
        # as content so the answer body is not empty. The thinking events
        # already streamed live cannot be retracted, so the client sees
        # the same text twice — once in the thinking panel, once as the
        # answer. UX trade-off documented in the chat template plan.
        # A tool-call turn is not empty, so its reasoning stays in thinking.
        if (
            not truncated
            and not has_tool_calls
            and self._in_thinking
            and not self._close_seen
            and not self._content_emitted
            and self._thinking_accumulated
        ):
            recovered = "".join(self._thinking_accumulated) + partial
            self._content_emitted = True
            return ("", recovered)

        if not partial:
            return ("", "")

        # Partial tag never completed — emit it as-is in the current mode.
        if self._in_thinking:
            self._thinking_accumulated.append(partial)
            return (partial, "")
        else:
            self._content_emitted = True
            return ("", partial)

    @staticmethod
    def _could_be_tag(text: str) -> bool:
        """Check if text could be the start of a <think> or </think> tag.

        Returns True if text is a proper prefix of either tag but not
        yet a complete match.
        """
        length = len(text)
        if length >= len(_HY3_CLOSE_TAG):
            # Long enough to determine - not a partial tag
            return False

        # Check against all recognised tags
        for tag in (_OPEN_TAG, _CLOSE_TAG, _HY3_OPEN_TAG, _HY3_CLOSE_TAG):
            if length < len(tag) and tag[:length] == text:
                return True

        return False


class ThinkingBudgetProcessor:
    """Logits processor that enforces a thinking token budget.

    Counts tokens generated while in thinking mode.  When the budget is
    exceeded, forces the close-think token(s) one at a time, then becomes
    a no-op for the rest of generation.

    Handles both single-token and multi-token close-think sequences, and
    supports alternative think markers (e.g. ``<longcat_think>``).

    Args:
        think_end_token_ids: Token ID(s) for the close-think tag.
        budget: Maximum number of thinking tokens before forcing close.
        think_start_token_id: Token ID for the open-think tag (re-entry detection).
        start_in_thinking: False when the model, not the prompt, opens
            thinking. Counting then starts at ``think_start_token_id``.
        grace_tokens: Optional boundary grace window. When the budget is
            reached mid-sentence, keep sampling normally until an accepted
            token ends a line or a sentence, then force the close sequence;
            force it anyway after this many extra tokens. ``None``/``0``
            closes immediately at the budget.
        whitespace_mask: ``vocab_size -> mask | None`` lookup (see
            :func:`leading_whitespace_mask`). With it, a Western ``.!?``
            only closes the grace window when the next token is likely
            whitespace-led; without it, such a token closes at once.
    """

    def __init__(
        self,
        think_end_token_ids: List[int],
        budget: int,
        think_start_token_id: Optional[int] = None,
        leading_token_ids: Optional[List[int]] = None,
        trailing_token_ids: Optional[List[int]] = None,
        token_to_piece: Optional[Callable[[int], str | bytes | None]] = None,
        start_in_thinking: bool = True,
        grace_tokens: int | None = None,
        whitespace_mask: Callable[[int], Any] | None = None,
    ):
        self._think_end_ids = think_end_token_ids
        # Full force sequence: \n + </think> + \n\n (matches training pattern)
        self._force_sequence = (
            (leading_token_ids or [])
            + list(think_end_token_ids)
            + (trailing_token_ids or [])
        )
        self._budget = budget
        self._grace = max(0, int(grace_tokens or 0))
        self._whitespace_mask = whitespace_mask
        self._think_start_id = think_start_token_id
        self._token_to_piece = token_to_piece

        # State
        self._thinking_tokens: int = 0
        self._in_thinking: bool = start_in_thinking
        self._forcing: bool = False
        self._waiting_utf8: bool = False
        self._force_idx: int = 0
        self._done: bool = False
        self._first_call: bool = True
        # Sliding window for multi-token end detection
        self._recent_tokens: List[int] = []
        self._last_token_utf8_complete: bool = True
        self._pending_utf8: bytes = b""
        # Boundary grace window (only touched when ``self._grace > 0``)
        self._waiting_boundary: bool = False
        self._grace_used: int = 0
        self._last_token_boundary: int = _BOUNDARY_NONE
        self._boundary_tail: bytes = b""
        # A soft boundary was accepted: the next position peeks at the
        # whitespace mass and lazily selects the forced close.
        self._peek_pending: bool = False

    def __call__(self, tokens, logits):
        """mlx-lm logits processor: (tokens, logits) -> logits."""
        import mlx.core as mx

        # In new mlx-lm API, tokens is the full history list.
        # Accept each genuinely generated token exactly once (see grammar.py).
        n = len(tokens)
        is_new_position = n > getattr(self, "_accepted_up_to", -1)
        if not hasattr(self, "_accepted_up_to"):
            self._accepted_up_to = n  # skip prompt tokens
        elif n > self._accepted_up_to:
            for i in range(self._accepted_up_to, n):
                self._update_state(int(tokens[i]))
            self._accepted_up_to = n

        # If state changed by _update_state, handle immediately
        if self._done:
            return logits

        if self._forcing:
            return self._force_next_token(logits, mx)

        if self._waiting_utf8:
            return logits

        if self._waiting_boundary:
            if self._peek_pending:
                return self._peek_close(logits, mx)
            return logits

        # Resuming a dropped decode step repeats the same sampling position.
        if self._in_thinking and is_new_position:
            self._thinking_tokens += 1
            if self._thinking_tokens >= self._budget:
                # Grace: let the model finish its sentence or line first.
                # Skipped when no thinking token was accepted yet (nothing
                # to finish) or when the last one already ends a boundary.
                if (
                    self._grace
                    and self._thinking_tokens > 1
                    and self._last_token_boundary != _BOUNDARY_HARD
                ):
                    self._waiting_boundary = True
                    self._grace_used = 0
                    if self._last_token_boundary == _BOUNDARY_SOFT:
                        self._peek_pending = True
                        return self._peek_close(logits, mx)
                    return logits
                if self._last_token_utf8_complete:
                    self._forcing = True
                    self._force_idx = 0
                    self._recent_tokens = []
                    return self._force_next_token(logits, mx)
                self._waiting_utf8 = True
                self._recent_tokens = []

        return logits

    def _update_state(self, token_id: int) -> None:
        """Update thinking state based on the last generated token."""
        if self._grace and self._in_thinking and not (self._done or self._forcing):
            # Track whether the latest thinking token ends a sentence/line.
            self._last_token_boundary = self._is_boundary(token_id)
            if (
                self._last_token_boundary == _BOUNDARY_SOFT
                and self._whitespace_mask is None
            ):
                self._last_token_boundary = _BOUNDARY_HARD
        self._last_token_utf8_complete = self._is_utf8_complete(token_id)

        if self._done:
            if self._think_start_id and token_id == self._think_start_id:
                self._in_thinking = True
                self._done = False
                self._thinking_tokens = 0
                self._recent_tokens = []
                self._reset_grace()
            return

        if self._forcing:
            self._force_idx += 1
            if self._force_idx >= len(self._force_sequence):
                self._in_thinking = False
                self._forcing = False
                self._done = True
            return

        if self._peek_pending:
            # Resolve the lazily selected close from the token that won.
            self._peek_pending = False
            if token_id == self._force_sequence[0]:
                self._waiting_boundary = False
                self._recent_tokens = []
                if len(self._force_sequence) == 1:
                    self._in_thinking = False
                    self._done = True
                else:
                    self._forcing = True
                    self._force_idx = 1
                return

        # Detect natural close-think via sliding window
        if len(self._think_end_ids) == 1:
            if token_id == self._think_end_ids[0]:
                self._in_thinking = False
                self._done = True
                self._reset_grace()
                return
        else:
            self._recent_tokens.append(token_id)
            if len(self._recent_tokens) > len(self._think_end_ids):
                self._recent_tokens.pop(0)
            if self._recent_tokens == self._think_end_ids:
                self._in_thinking = False
                self._done = True
                self._reset_grace()
                return

        if self._waiting_boundary:
            self._grace_used += 1
            boundary = self._last_token_boundary
            if boundary == _BOUNDARY_SOFT and self._grace_used < self._grace:
                if self._last_token_utf8_complete:
                    self._peek_pending = True
                    return
                boundary = _BOUNDARY_HARD
            if boundary or self._grace_used >= self._grace:
                self._waiting_boundary = False
                self._recent_tokens = []
                if self._last_token_utf8_complete:
                    self._forcing = True
                    self._force_idx = 0
                else:
                    self._waiting_utf8 = True
            return

        if self._waiting_utf8:
            if self._last_token_utf8_complete:
                self._waiting_utf8 = False
                self._forcing = True
                self._force_idx = 0
                self._recent_tokens = []
            return

        # Detect re-entry into thinking (rare but possible)
        if not self._in_thinking and self._think_start_id and token_id == self._think_start_id:
            self._in_thinking = True
            self._done = False
            self._thinking_tokens = 0
            self._recent_tokens = []
            self._reset_grace()

    def _reset_grace(self) -> None:
        self._waiting_boundary = False
        self._grace_used = 0
        self._last_token_boundary = _BOUNDARY_NONE
        self._boundary_tail = b""
        self._peek_pending = False

    def _peek_close(self, logits, mx):
        """Close here when the next token is likely whitespace-led.

        Stays lazy: the decision is an ``mx.where`` between the forced row
        and the model's row, and ``_update_state`` reads the outcome from the
        accepted token. Without a usable vocabulary mask the soft boundary
        closes at once.
        """
        mask = None
        if self._whitespace_mask is not None:
            try:
                mask = self._whitespace_mask(int(logits.shape[-1]))
            except Exception:
                mask = None
        if mask is None:
            self._peek_pending = False
            self._waiting_boundary = False
            self._forcing = True
            self._force_idx = 0
            self._recent_tokens = []
            return self._force_next_token(logits, mx)
        probs = mx.softmax(logits.astype(mx.float32), axis=-1)
        p_ws = mx.sum(mx.where(mask, probs, 0.0), axis=-1, keepdims=True)
        forced = mx.full(logits.shape, float("-inf"), dtype=logits.dtype)
        forced[..., self._force_sequence[0]] = 0.0
        return mx.where(
            p_ws > THINKING_BUDGET_FINISH_SENTENCE_WHITESPACE_THRESHOLD,
            forced,
            logits,
        )

    def _is_boundary(self, token_id: int) -> int:
        """Boundary kind of the accepted token (none / hard / soft).

        Byte pieces and GPT-2 byte-level BPE strings (e.g. ``'ĊĊ'`` for
        ``\\n\\n``) are decoded to text first; a multibyte character split
        across tokens is completed from the carried tail bytes.
        """
        if self._token_to_piece is None:
            return _BOUNDARY_NONE
        try:
            piece = self._token_to_piece(token_id)
        except Exception:
            return _BOUNDARY_NONE
        if piece is None:
            self._boundary_tail = b""
            return _BOUNDARY_NONE
        if isinstance(piece, str):
            try:
                data = bytes(_GPT2_BYTE_DECODER[ch] for ch in piece)
            except KeyError:
                data = None
        else:
            data = piece
        text = None
        if data is not None:
            decoder = codecs.getincrementaldecoder("utf-8")("strict")
            try:
                text = decoder.decode(self._boundary_tail + data, final=False)
                self._boundary_tail = decoder.getstate()[0]
            except UnicodeDecodeError:
                text = None
        if text is None:
            self._boundary_tail = b""
            if isinstance(piece, str):
                text = piece
            else:
                text = piece.decode("utf-8", errors="ignore")
        if "\n" in text:
            return _BOUNDARY_HARD
        text = text.rstrip(_SENTENCE_TRAILING_CHARS)
        if text.endswith(_HARD_SENTENCE_END_CHARS):
            return _BOUNDARY_HARD
        if text.endswith(_SOFT_SENTENCE_END_CHARS):
            return _BOUNDARY_SOFT
        return _BOUNDARY_NONE

    def _is_utf8_complete(self, token_id: int) -> bool:
        """Best-effort UTF-8 boundary check for accepted token bytes."""
        if self._token_to_piece is None:
            return True
        try:
            piece = self._token_to_piece(token_id)
        except Exception:
            return True
        if piece is None:
            return True
        if isinstance(piece, str):
            self._pending_utf8 = b""
            return True
        self._pending_utf8 += piece
        try:
            self._pending_utf8.decode("utf-8")
            self._pending_utf8 = b""
            return True
        except UnicodeDecodeError as exc:
            if exc.reason == "unexpected end of data" or exc.end == len(
                self._pending_utf8
            ):
                return False
            self._pending_utf8 = b""
            return True

    def _force_next_token(self, logits, mx):
        """Force the next token in the close-think + trailing sequence."""
        target_id = self._force_sequence[self._force_idx]
        forced = mx.full(logits.shape, float("-inf"))
        forced[..., target_id] = 0.0
        return forced

    # -- Speculative-decoding (vlm_mtp) support -----------------------------
    #
    # The vlm_mtp decode path applies this processor inside mlx-vlm's
    # speculative verify walk, where positions past the first draft
    # rejection are discarded and re-sampled on the next round.
    # ``snapshot_state()`` / ``restore_state()`` let the caller checkpoint
    # the processor after each position and rewind it when a draft suffix
    # is rejected, keeping the one-call-per-emitted-token contract intact.
    # See ``omlx/speculative/processing_sampler.py``.

    _SNAPSHOT_ATTRS = (
        "_thinking_tokens",
        "_in_thinking",
        "_forcing",
        "_waiting_utf8",
        "_force_idx",
        "_done",
        "_first_call",
        "_recent_tokens",
        "_last_token_utf8_complete",
        "_pending_utf8",
        "_accepted_up_to",
        "_waiting_boundary",
        "_grace_used",
        "_last_token_boundary",
        "_boundary_tail",
        "_peek_pending",
    )

    def snapshot_state(self) -> dict:
        """Checkpoint mutable state for position-keyed rewind (vlm_mtp)."""
        state: dict = {}
        for name in self._SNAPSHOT_ATTRS:
            if not hasattr(self, name):
                continue
            value = getattr(self, name)
            if isinstance(value, list):
                value = list(value)
            state[name] = value
        return state

    def restore_state(self, state: dict) -> None:
        """Restore a checkpoint produced by :meth:`snapshot_state`."""
        for name in self._SNAPSHOT_ATTRS:
            if name in state:
                value = state[name]
                if isinstance(value, list):
                    value = list(value)
                setattr(self, name, value)
            elif name == "_accepted_up_to" and hasattr(self, name):
                # Lazily-created attr absent from the snapshot: drop it so
                # the next __call__ re-baselines from the history it sees.
                delattr(self, name)
