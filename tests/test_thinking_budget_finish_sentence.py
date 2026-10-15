# SPDX-License-Identifier: Apache-2.0
"""Tests for the thinking-budget finish-sentence grace window."""

import ast
from pathlib import Path
from unittest.mock import MagicMock

import pytest

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

from omlx.adapter.output_parser import OutputParserFactory
from omlx.api.thinking import (
    THINKING_BUDGET_FINISH_SENTENCE_MAX_TOKENS,
    ThinkingBudgetProcessor,
)
from omlx.model_settings import ModelSettings
from omlx.request import Request, SamplingParams
from omlx.scheduler import Scheduler

THINK_START = 41
THINK_END = 42
PROMPT = [1]

# Fake vocabulary: words never end a boundary, punctuation/newlines do.
PIECES = {
    10: " alpha",
    11: " beta",
    12: " gamma",
    13: " delta",
    14: " epsilon",
    15: " zeta",
    16: " eta",
    17: " theta",
    60: ".",
    61: "\n",
    62: '?"',
    63: " end.**",
    64: "。",  # CJK full stop
    70: b"\xe2",  # first byte of a 3-byte char (no boundary)
    71: b"\x82",
    72: b"\xac",
    80: "Ġfine",  # GPT-2 byte-level " fine"
    81: "ĊĊ",  # GPT-2 byte-level "\n\n"
    82: "ãĢĤ",  # GPT-2 byte-level "。"
    83: "ãĢ",  # first two bytes of "。"
    84: "Ĥ",  # last byte of "。"
    THINK_START: "<think>",
    THINK_END: "</think>",
}


def _piece(token_id):
    return PIECES.get(token_id, " word")


def _make_processor(budget=3, grace=5, end_ids=None, token_to_piece=_piece):
    return ThinkingBudgetProcessor(
        think_end_token_ids=end_ids or [THINK_END],
        budget=budget,
        think_start_token_id=THINK_START,
        token_to_piece=token_to_piece,
        grace_tokens=grace,
    )


def _logits():
    return mx.zeros((1, 100))


def _forced_id(logits):
    """Token id the processor forces at this position, or None."""
    if mx.array_equal(logits, _logits()):
        return None
    assert (logits == 0.0).sum().item() == 1
    return mx.argmax(logits).item()


def _run(proc, generated):
    """Drive the processor one position at a time.

    Returns the forced token id (or None) at each sampling position, where
    position ``i`` sees ``PROMPT + generated[:i]``.
    """
    out = []
    for i in range(len(generated) + 1):
        out.append(_forced_id(proc(mx.array(PROMPT + generated[:i]), _logits())))
    return out


@pytest.mark.skipif(not HAS_MLX, reason="mlx not available")
class TestFinishSentenceProcessor:
    @pytest.mark.parametrize("grace", [0, None])
    def test_no_grace_matches_current_forcing(self, grace):
        words = [10, 11, 12, 13, 14, 15]
        baseline = ThinkingBudgetProcessor(
            think_end_token_ids=[THINK_END],
            budget=3,
            think_start_token_id=THINK_START,
            token_to_piece=_piece,
        )
        proc = _make_processor(budget=3, grace=grace)
        expected = _run(baseline, words[:2])
        assert _run(proc, words[:2]) == expected
        assert expected[-1] == THINK_END
        assert not proc._waiting_boundary

    def test_boundary_inside_window_forces_right_after_it(self):
        proc = _make_processor(budget=3, grace=5)
        # positions 0-1 free, budget hit at 2 (last token 11 is mid-sentence)
        forced = _run(proc, [10, 11, 12, 60])
        # free sampling at 2 and 3; token 60 '.' accepted -> close at 4
        assert forced == [None, None, None, None, THINK_END]
        assert proc._forcing
        assert not proc._waiting_boundary

    @pytest.mark.parametrize("boundary", [61, 62, 63, 64, 81, 82])
    def test_boundary_kinds(self, boundary):
        proc = _make_processor(budget=3, grace=5)
        forced = _run(proc, [10, 11, 12, boundary])
        assert forced[-1] == THINK_END

    def test_no_boundary_forces_after_exactly_grace_tokens(self):
        grace = 4
        proc = _make_processor(budget=3, grace=grace)
        words = [10, 11, 12, 13, 14, 15, 16, 17]
        forced = _run(proc, words[: 2 + grace])
        # budget reached at position 2; positions 2..2+grace-1 sample freely
        assert forced[: 2 + grace] == [None] * (2 + grace)
        assert forced[2 + grace] == THINK_END
        assert proc._grace_used == grace

    def test_budget_landing_on_boundary_closes_immediately(self):
        proc = _make_processor(budget=3, grace=5)
        forced = _run(proc, [10, 60])
        assert forced == [None, None, THINK_END]
        assert not proc._waiting_boundary

    def test_zero_budget_closes_immediately(self):
        proc = _make_processor(budget=0, grace=5)
        assert _run(proc, []) == [THINK_END]

    def test_natural_close_during_grace_is_respected(self):
        proc = _make_processor(budget=3, grace=5)
        forced = _run(proc, [10, 11, 12, THINK_END, 13, 14])
        assert forced == [None] * 7
        assert proc._done
        assert not proc._waiting_boundary
        assert not proc._forcing

    def test_multi_token_natural_close_during_grace(self):
        proc = _make_processor(budget=3, grace=5, end_ids=[50, 51])
        forced = _run(proc, [10, 11, 12, 50, 51, 13])
        assert forced == [None] * 7
        assert proc._done

    def test_reentry_after_close_during_grace_is_budgeted_again(self):
        proc = _make_processor(budget=3, grace=5)
        generated = [10, 11, 12, THINK_END, 13, THINK_START, 14, 15]
        forced = _run(proc, generated)
        # The second round hits the budget mid-sentence and enters grace.
        assert forced == [None] * len(forced)
        assert proc._in_thinking and proc._waiting_boundary
        forced = _run(proc, generated + [60])
        assert forced[-1] == THINK_END

    def test_incomplete_utf8_at_grace_end_waits_for_completion(self):
        proc = _make_processor(budget=3, grace=2)
        forced = _run(proc, [10, 11, 12, 70])
        # grace cap reached on an incomplete piece: no forcing yet
        assert forced[-1] is None
        assert proc._waiting_utf8 and not proc._waiting_boundary
        forced = _run(proc, [10, 11, 12, 70, 71])
        assert forced[-1] is None
        forced = _run(proc, [10, 11, 12, 70, 71, 72])
        assert forced[-1] == THINK_END

    def test_cjk_stop_split_across_byte_level_tokens(self):
        proc = _make_processor(budget=3, grace=5)
        forced = _run(proc, [10, 11, 83, 84])
        assert forced[-2] is None
        assert forced[-1] == THINK_END

    def test_gpt2_byte_level_word_is_not_a_boundary(self):
        proc = _make_processor(budget=3, grace=5)
        assert _run(proc, [10, 11, 80])[-1] is None
        assert proc._waiting_boundary

    def test_no_token_to_piece_runs_to_the_cap(self):
        proc = _make_processor(budget=3, grace=2, token_to_piece=None)
        forced = _run(proc, [10, 11, 12, 13])
        assert forced == [None, None, None, None, THINK_END]

    def test_snapshot_restore_mid_grace(self):
        proc = _make_processor(budget=3, grace=5)
        _run(proc, [10, 11, 12])
        assert proc._waiting_boundary and proc._grace_used == 1
        snap = proc.snapshot_state()

        # Draft position accepts a boundary -> forcing would start.
        assert _forced_id(proc(mx.array(PROMPT + [10, 11, 12, 60]), _logits())) == (
            THINK_END
        )
        assert proc._forcing

        # Rejected: rewind and accept a non-boundary token instead.
        proc.restore_state(snap)
        assert proc._waiting_boundary and proc._grace_used == 1
        assert not proc._forcing
        assert proc(mx.array(PROMPT + [10, 11, 12, 13]), _logits()) is not None
        assert not proc._forcing
        assert proc._grace_used == 2

    def test_snapshot_restores_split_boundary_tail(self):
        proc = _make_processor(budget=3, grace=5)
        _run(proc, [10, 11, 83])
        snap = proc.snapshot_state()
        assert snap["_boundary_tail"] == b"\xe3\x80"
        proc(mx.array(PROMPT + [10, 11, 83, 13]), _logits())
        assert proc._boundary_tail == b""
        proc.restore_state(snap)
        assert proc._boundary_tail == b"\xe3\x80"
        assert _forced_id(proc(mx.array(PROMPT + [10, 11, 83, 84]), _logits())) == (
            THINK_END
        )

    def test_snapshot_attrs_cover_grace_state(self):
        for name in (
            "_waiting_boundary",
            "_grace_used",
            "_last_token_boundary",
            "_boundary_tail",
        ):
            assert name in ThinkingBudgetProcessor._SNAPSHOT_ATTRS


# ---------------------------------------------------------------------------
# Settings, request models and resolution
# ---------------------------------------------------------------------------


class TestFinishSentenceSettings:
    def test_model_setting_defaults_off(self):
        assert ModelSettings().thinking_budget_finish_sentence is False
        assert SamplingParams().thinking_budget_finish_sentence is False

    def test_model_setting_round_trips(self):
        settings = ModelSettings.from_dict(
            {"thinking_budget_enabled": True, "thinking_budget_finish_sentence": True}
        )
        assert settings.to_dict()["thinking_budget_finish_sentence"] is True

    def test_profile_field_is_universal(self):
        from omlx.model_profiles import UNIVERSAL_PROFILE_FIELDS

        assert "thinking_budget_finish_sentence" in UNIVERSAL_PROFILE_FIELDS

    @pytest.mark.parametrize("model", ["ChatCompletionRequest", "CompletionRequest"])
    def test_request_field(self, model):
        from omlx.api import openai_models

        cls = getattr(openai_models, model)
        extra = (
            {"messages": [{"role": "user", "content": "hi"}]}
            if model == "ChatCompletionRequest"
            else {"prompt": "p"}
        )
        assert cls(model="m", **extra).thinking_budget_finish_sentence is None
        req = cls(model="m", thinking_budget_finish_sentence=True, **extra)
        assert req.thinking_budget_finish_sentence is True


class TestResolveFinishSentence:
    @pytest.fixture
    def server(self, monkeypatch):
        import omlx.server as server

        self.settings = None
        monkeypatch.setattr(
            server, "get_model_settings_for_request", lambda model_id: self.settings
        )
        return server

    def _request(self, value=None, budget=None):
        req = MagicMock(spec=[])
        req.model = "m"
        if value is not None:
            req.thinking_budget_finish_sentence = value
        if budget is not None:
            req.thinking_budget = budget
        return req

    def test_default_off(self, server):
        assert (
            server._resolve_thinking_budget_finish_sentence(self._request(), "m")
            is False
        )

    def test_model_setting_applies(self, server):
        self.settings = ModelSettings(thinking_budget_finish_sentence=True)
        assert (
            server._resolve_thinking_budget_finish_sentence(self._request(), "m")
            is True
        )

    @pytest.mark.parametrize("model_value", [True, False])
    def test_request_overrides_model_setting(self, server, model_value):
        self.settings = ModelSettings(thinking_budget_finish_sentence=model_value)
        req = self._request(value=not model_value)
        assert server._resolve_thinking_budget_finish_sentence(req, "m") is (
            not model_value
        )

    def test_forwarded_only_with_a_budget(self, server):
        self.settings = ModelSettings(thinking_budget_finish_sentence=True)
        kwargs = {}
        server._forward_thinking_budget_finish_sentence(kwargs, self._request())
        assert kwargs == {}
        kwargs = {"thinking_budget": 256}
        server._forward_thinking_budget_finish_sentence(kwargs, self._request())
        assert kwargs["thinking_budget_finish_sentence"] is True

    def test_request_off_is_not_forwarded(self, server):
        self.settings = ModelSettings(thinking_budget_finish_sentence=True)
        kwargs = {"thinking_budget": 256}
        server._forward_thinking_budget_finish_sentence(
            kwargs, self._request(value=False)
        )
        assert "thinking_budget_finish_sentence" not in kwargs

    @pytest.mark.parametrize(
        "handler",
        [
            "create_completion",
            "stream_completion",
            "create_chat_completion",
            "create_anthropic_message",
            "create_response",
        ],
    )
    def test_every_budget_handler_forwards_the_switch(self, handler):
        source = (
            Path(__file__).resolve().parents[1] / "omlx" / "server.py"
        ).read_text()
        for node in ast.walk(ast.parse(source)):
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == handler
            ):
                calls = [
                    n
                    for n in ast.walk(node)
                    if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Name)
                    and n.func.id == "_forward_thinking_budget_finish_sentence"
                ]
                assert calls, f"{handler} must forward thinking_budget_finish_sentence"
                return
        raise AssertionError(f"{handler} not found in server.py")


# ---------------------------------------------------------------------------
# Engine plumbing
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_MLX, reason="mlx not available")
class TestFinishSentenceSchedulerWiring:
    def _build(self, finish_sentence):
        factory = OutputParserFactory(
            kind="gemma4",
            create_session=MagicMock(),
            thinking_end_text="<channel|>",
        )
        scheduler = MagicMock(spec=Scheduler)
        scheduler._output_parser_factory = factory
        scheduler._xtc_special_tokens = set()
        scheduler._model_suppress_tokens = set()
        for name in (
            "_get_think_token_id",
            "_get_output_parser_thinking_end_text",
            "_encode_thinking_marker",
            "_token_piece_to_bytes",
            "_resolve_output_parser_thinking_trailing_ids",
            "_resolve_think_end_token_ids",
            "_build_sampler_and_processors",
        ):
            setattr(
                scheduler, name, getattr(Scheduler, name).__get__(scheduler, Scheduler)
            )
        scheduler._resolve_think_close_pattern = MagicMock(return_value=(None, None))
        tokenizer = MagicMock()
        tokenizer.encode.side_effect = lambda text, add_special_tokens=False: {
            "<channel|>": [101]
        }[text]
        scheduler.tokenizer = tokenizer
        request = Request(
            request_id="finish-sentence",
            prompt="test",
            sampling_params=SamplingParams(
                thinking_budget=512,
                thinking_budget_finish_sentence=finish_sentence,
            ),
            prompt_token_ids=[1, 2, 3],
            num_prompt_tokens=3,
        )
        request.needs_think_prefix = False
        _, processors = scheduler._build_sampler_and_processors(
            request.sampling_params, request
        )
        (budget,) = [p for p in processors if isinstance(p, ThinkingBudgetProcessor)]
        return budget

    def test_switch_on_uses_the_ceiling(self):
        budget = self._build(True)
        assert budget._grace == THINKING_BUDGET_FINISH_SENTENCE_MAX_TOKENS
        assert budget._whitespace_mask is not None

    def test_switch_off_closes_immediately(self):
        budget = self._build(False)
        assert budget._grace == 0
        assert budget._whitespace_mask is None


class TestFinishSentenceEnginePlumbing:
    @pytest.mark.parametrize("engine", ["batched.py", "vlm.py"])
    def test_engines_pass_switch_into_sampling_params(self, engine):
        source = (
            Path(__file__).resolve().parents[1] / "omlx" / "engine" / engine
        ).read_text()
        assert source.count("thinking_budget_finish_sentence=bool(") == 2


# ---------------------------------------------------------------------------
# Western sentence ends: close only when the next token is whitespace-led
# ---------------------------------------------------------------------------

WS_ID = 20  # whitespace-led token (" next")
DIGIT_ID = 30  # "5" in "3.5"
LEAD_ID = 99  # leading "\n" of a wrapped close sequence
TRAIL_ID = 98


def _ws_mask(vocab_size):
    flags = [False] * vocab_size
    flags[WS_ID] = True
    return mx.array(flags)


def _peaked(token_id, vocab=100):
    logits = mx.zeros((1, vocab))
    logits[0, token_id] = 10.0
    return logits


def _forced_in(logits):
    """Forced token id when the row is fully masked but one id, else None."""
    finite = mx.isfinite(logits).sum().item()
    return mx.argmax(logits).item() if finite == 1 else None


def _peek_processor(
    budget=3,
    grace=5,
    whitespace_mask=_ws_mask,
    leading=None,
    trailing=None,
):
    return ThinkingBudgetProcessor(
        think_end_token_ids=[THINK_END],
        budget=budget,
        think_start_token_id=THINK_START,
        leading_token_ids=leading,
        trailing_token_ids=trailing,
        token_to_piece=_piece,
        grace_tokens=grace,
        whitespace_mask=whitespace_mask,
    )


def _feed(proc, generated, logits_fn=lambda: _peaked(WS_ID)):
    """Call the processor at every position; return the last output."""
    out = None
    for i in range(len(generated) + 1):
        out = proc(mx.array(PROMPT + generated[:i]), logits_fn())
    return out


@pytest.mark.skipif(not HAS_MLX, reason="mlx not available")
class TestFinishSentenceWhitespacePeek:
    def test_whitespace_likely_closes_at_the_period(self):
        proc = _peek_processor()
        out = _feed(proc, [10, 11, 12, 60])
        assert proc._peek_pending
        assert _forced_in(out) == THINK_END
        # The lazily forced token is accepted: the close completes.
        proc(mx.array(PROMPT + [10, 11, 12, 60, THINK_END]), _peaked(WS_ID))
        assert proc._done and not proc._in_thinking
        assert not proc._peek_pending

    def test_digit_after_period_keeps_grace_open(self):
        proc = _peek_processor()
        out = _feed(proc, [10, 11, 12, 60], logits_fn=lambda: _peaked(DIGIT_ID))
        assert _forced_in(out) is None
        assert mx.array_equal(out, _peaked(DIGIT_ID))
        used = proc._grace_used
        out = proc(mx.array(PROMPT + [10, 11, 12, 60, DIGIT_ID]), _peaked(DIGIT_ID))
        assert _forced_in(out) is None
        assert proc._waiting_boundary and not proc._peek_pending
        assert proc._grace_used == used + 1

    def test_digit_then_cap_still_closes(self):
        proc = _peek_processor(grace=3)
        out = _feed(
            proc, [10, 11, 12, 60, DIGIT_ID], logits_fn=lambda: _peaked(DIGIT_ID)
        )
        assert _forced_in(out) == THINK_END
        assert proc._forcing

    @pytest.mark.parametrize("boundary", [61, 64, 81, 82])
    def test_newline_and_cjk_stay_immediate(self, boundary):
        proc = _peek_processor()
        out = _feed(proc, [10, 11, 12, boundary], logits_fn=lambda: _peaked(DIGIT_ID))
        assert _forced_in(out) == THINK_END
        assert proc._forcing and not proc._peek_pending

    def test_lazily_forced_leading_token_continues_the_sequence(self):
        proc = _peek_processor(leading=[LEAD_ID], trailing=[TRAIL_ID])
        out = _feed(proc, [10, 11, 12, 60])
        assert _forced_in(out) == LEAD_ID
        hist = [10, 11, 12, 60, LEAD_ID]
        out = proc(mx.array(PROMPT + hist), _peaked(WS_ID))
        assert proc._forcing and proc._force_idx == 1
        assert _forced_in(out) == THINK_END
        out = proc(mx.array(PROMPT + hist + [THINK_END]), _peaked(WS_ID))
        assert _forced_in(out) == TRAIL_ID
        proc(mx.array(PROMPT + hist + [THINK_END, TRAIL_ID]), _peaked(WS_ID))
        assert proc._done

    def test_close_tag_first_still_forces_trailing(self):
        proc = _peek_processor(trailing=[TRAIL_ID])
        out = _feed(proc, [10, 11, 12, 60])
        assert _forced_in(out) == THINK_END
        out = proc(mx.array(PROMPT + [10, 11, 12, 60, THINK_END]), _peaked(WS_ID))
        assert _forced_in(out) == TRAIL_ID

    def test_budget_landing_on_period_peeks(self):
        proc = _peek_processor()
        out = _feed(proc, [10, 60])
        assert proc._waiting_boundary and proc._peek_pending
        assert _forced_in(out) == THINK_END
        proc2 = _peek_processor()
        out = _feed(proc2, [10, 60], logits_fn=lambda: _peaked(DIGIT_ID))
        assert _forced_in(out) is None and proc2._waiting_boundary

    def test_mask_size_mismatch_closes_immediately(self):
        proc = _peek_processor(whitespace_mask=lambda vocab: None)
        out = _feed(proc, [10, 11, 12, 60], logits_fn=lambda: _peaked(DIGIT_ID))
        assert _forced_in(out) == THINK_END
        assert proc._forcing and not proc._peek_pending

    def test_raising_mask_lookup_closes_immediately(self):
        def broken(vocab):
            raise RuntimeError("no vocab")

        proc = _peek_processor(whitespace_mask=broken)
        out = _feed(proc, [10, 11, 12, 60], logits_fn=lambda: _peaked(DIGIT_ID))
        assert _forced_in(out) == THINK_END

    def test_without_mask_provider_period_is_immediate(self):
        proc = _peek_processor(whitespace_mask=None)
        out = _feed(proc, [10, 11, 12, 60], logits_fn=lambda: _peaked(DIGIT_ID))
        assert _forced_in(out) == THINK_END
        assert proc._forcing

    def test_pending_survives_snapshot_restore(self):
        proc = _peek_processor()
        _feed(proc, [10, 11, 12, 60])
        snap = proc.snapshot_state()
        assert snap["_peek_pending"] is True
        proc(mx.array(PROMPT + [10, 11, 12, 60, DIGIT_ID]), _peaked(WS_ID))
        assert not proc._peek_pending
        proc.restore_state(snap)
        assert proc._peek_pending
        proc(mx.array(PROMPT + [10, 11, 12, 60, THINK_END]), _peaked(WS_ID))
        assert proc._done

    def test_chain_verify_rejected_draft_matches_serial(self):
        """Lightning MTP verify: rows see draft prefixes, the processor is
        checkpointed per row and rewound to the last emitted row."""
        committed = [10, 11, 12, 60]
        serial = _peek_processor()
        _feed(serial, committed)
        serial(mx.array(PROMPT + committed + [THINK_END]), _peaked(WS_ID))

        proc = _peek_processor()
        _feed(proc, committed[:-1])
        drafts = [13, 14]  # drafted past the period, never a close
        prev_rows = [PROMPT + committed]
        for d in drafts:
            prev_rows.append(prev_rows[-1] + [d])
        targets, row_snaps = [], []
        for prev in prev_rows:
            out = proc(mx.array(prev), _peaked(WS_ID))
            targets.append(mx.argmax(out).item())
            row_snaps.append(proc.snapshot_state())
        # Row 0 is forced lazily; draft 13 is rejected there (m = 0).
        assert targets[0] == THINK_END
        m = next(i for i, d in enumerate(drafts) if targets[i] != d)
        assert m == 0
        proc.restore_state(row_snaps[m])
        assert proc._peek_pending and proc._waiting_boundary
        # Next cycle commits the emitted correction token.
        proc(mx.array(PROMPT + committed + [targets[m]]), _peaked(WS_ID))
        for name in ("_done", "_in_thinking", "_grace_used", "_peek_pending"):
            assert getattr(proc, name) == getattr(serial, name), name


class _FakeVocab:
    def __init__(self, pieces):
        self._pieces = pieces

    def __len__(self):
        return len(self._pieces)

    def convert_ids_to_tokens(self, ids):
        return [self._pieces[i] for i in ids]


@pytest.mark.skipif(not HAS_MLX, reason="mlx not available")
class TestLeadingWhitespaceMask:
    PIECES = [
        "Ġthe",  # GPT-2 " the"
        "Ċ",  # GPT-2 "\n"
        "3",
        "▁word",  # SentencePiece " word"
        "<0x0A>",  # byte fallback "\n"
        "​",  # zero-width space
        "a",
        " x",
        "<|im_end|>",
        b" raw",
    ]
    EXPECTED = [True, True, False, True, True, False, False, True, False, True]

    def test_flags_and_padding(self):
        from omlx.api.thinking import leading_whitespace_mask

        tok = _FakeVocab(self.PIECES)
        mask = leading_whitespace_mask(tok, 12)
        assert mask.tolist() == self.EXPECTED + [False, False]
        assert leading_whitespace_mask(tok, 12) is mask  # cached

    def test_narrower_logits_return_none(self):
        from omlx.api.thinking import leading_whitespace_mask

        assert leading_whitespace_mask(_FakeVocab(self.PIECES), 5) is None

    def test_unreadable_vocab_returns_none(self):
        from omlx.api.thinking import leading_whitespace_mask

        assert leading_whitespace_mask(object(), 16) is None
