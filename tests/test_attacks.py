"""Unit tests for the attack factories.

``attacks`` pulls in the model data stack, so these tests are skipped unless the
full environment (torch et al.) is installed.
"""

import pytest

pytest.importorskip("torch")


def test_repetitive_factory_default_and_param():
    from attacks import stress_repeat_2

    # 1-arg call uses the global probe.
    out = stress_repeat_2({"input": "X"})
    assert out["input"].startswith("X\n\n")
    # 2-arg call (SEP path) uses the per-example probe.
    out2 = stress_repeat_2({"input": "X"}, "CUSTOM_PROBE")
    assert out2["input"].count("CUSTOM_PROBE") == 2


def test_positional_inject_start_and_end():
    from attacks import inject_pos_0, inject_pos_100

    start = inject_pos_0({"input": "body text"}, "probe")
    assert start["input"].endswith("body text")
    end = inject_pos_100({"input": "body text"}, "probe")
    assert end["input"].startswith("body text")
