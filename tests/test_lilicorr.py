# SPDX-License-Identifier: Apache-2.0
"""LiLiCorr drafter (Nemotron 3.5 Bolt): config detection, lattice selection
and the Nemotron-H target hooks."""

from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest

from omlx.patches.mlx_lm_mtp import drafter_target
from omlx.speculative.dflash import lilicorr
from omlx.utils.model_loading import dflash_batched_supported


@pytest.mark.parametrize(
    "config, expected",
    [
        ({"dflash_config": {"projector_type": "lilicorr"}}, True),
        ({"dflash_config": {"projector_type": "mlp"}}, False),
        ({}, False),
    ],
)
def test_is_lilicorr_config(config, expected):
    assert lilicorr.is_lilicorr_config(config) is expected


class _FixedHead:
    """Stands in for LiLiCorrHead: returns preset lattice scores."""

    def __init__(self, scores, topk):
        self.scores = scores
        self.topk = topk

    def __call__(self, *_):
        return self.scores


def test_greedy_select_walks_the_lattice():
    # One row, 3 slots, vocab 6, top-2 candidates per slot.
    logits = mx.array(
        [[[5.0, 4.0, 0, 0, 0, 0], [0, 0, 5.0, 4.0, 0, 0], [0, 0, 0, 0, 5.0, 4.0]]]
    )
    # scores[b, slot, previous, current]: slot 0 ignores ``previous``.
    scores = mx.array(
        [
            [
                [[0.0, 1.0], [0.0, 1.0]],  # slot 0 picks candidate 1 (token 1)
                [[0.0, 0.0], [1.0, 0.0]],  # after candidate 1 -> candidate 0 (token 2)
                [[0.0, 1.0], [1.0, 0.0]],  # after candidate 0 -> candidate 1 (token 5)
            ]
        ]
    )
    model = SimpleNamespace(
        lilicorr=_FixedHead(scores, topk=2), embed_tokens=lambda ids: ids
    )
    ((tokens, q),) = lilicorr.select(
        model, None, logits, mx.zeros((1, 1, 4)), [1], None, [None]
    )
    assert tokens.tolist() == [[1, 2, 5]]
    assert q == []


def test_nemotron_h_layers_are_found_under_backbone():
    layers = [nn.Linear(2, 2)]
    nemotron = SimpleNamespace(backbone=SimpleNamespace(layers=layers))
    llama = SimpleNamespace(model=SimpleNamespace(layers=layers))
    assert drafter_target._decoder_layers(nemotron) is layers
    assert drafter_target._decoder_layers(llama) is layers
    assert drafter_target._decoder_layers(SimpleNamespace()) is None


@pytest.mark.parametrize("model_type", ["nemotron_h", "nemotron_h_puzzle"])
def test_nemotron_h_takes_a_batched_block_drafter(model_type):
    assert dflash_batched_supported(model_type, "batched")
