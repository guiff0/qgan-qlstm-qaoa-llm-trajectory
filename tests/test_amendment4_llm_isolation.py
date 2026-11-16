"""
Non-regression tests that verify changing synthetic_ratio in config does NOT
affect LLM fine-tuning.

SAFETY PROPERTY (confirmed before Amendment 4 was applied):
  nvidia_finetune.py's build_training_examples() receives synthetic_ratio as
  an explicit argument, NOT from model.config.  Therefore:
    - Changing config["synthetic_ratio"] from 0.4 to 1.0 only affects
      evaluate_poisoning_resistance() (line 128 of poisoning_resistance.py).
    - LLM fine-tuning ratios are NOT affected.

Regression: if someone refactors build_training_examples to read from config,
these tests will catch that the LLM ratio has changed unexpectedly.
"""
import importlib
import inspect
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestLLMFinetuningIsolation:

    def test_build_training_examples_has_explicit_ratio_param(self):
        """
        build_training_examples must have a real_ratio or synthetic_ratio
        PARAMETER — not reading from config.  If this signature changes the
        isolation guarantee disappears.
        """
        try:
            mod = importlib.import_module("src.llm.nvidia_finetune")
        except ImportError:
            pytest.skip("src.llm.nvidia_finetune not importable in this environment")

        func = getattr(mod, "build_training_examples", None)
        if func is None:
            pytest.skip("build_training_examples not found in nvidia_finetune")

        sig = inspect.signature(func)
        param_names = list(sig.parameters.keys())
        assert any("ratio" in p for p in param_names), (
            f"build_training_examples params={param_names}; expected a *ratio* parameter. "
            "If the function now reads from config, the LLM training ratio will change "
            "whenever synthetic_ratio changes in the config — verify this is intentional."
        )

    def test_build_training_examples_does_not_reference_model_config(self):
        """
        The source of build_training_examples must not contain
        model.config["synthetic_ratio"] or similar — if it did, changing the
        config would silently alter LLM training data composition.
        """
        try:
            mod = importlib.import_module("src.llm.nvidia_finetune")
        except ImportError:
            pytest.skip("src.llm.nvidia_finetune not importable in this environment")

        func = getattr(mod, "build_training_examples", None)
        if func is None:
            pytest.skip("build_training_examples not found in nvidia_finetune")

        try:
            src = inspect.getsource(func)
        except (OSError, TypeError):
            pytest.skip("Could not retrieve source for build_training_examples")

        suspicious = [
            '.config["synthetic_ratio"]',
            ".config['synthetic_ratio']",
            ".config.get(\"synthetic_ratio\"",
            ".config.get('synthetic_ratio'",
        ]
        for pattern in suspicious:
            assert pattern not in src, (
                f"build_training_examples references config synthetic_ratio via '{pattern}'; "
                "this means changing the config would alter LLM training data unexpectedly"
            )

    def test_poisoning_resistance_reads_from_model_config(self):
        """
        Inverse check: poisoning_resistance.py MUST read synthetic_ratio from
        model.config — that is the fix.  If someone moves it to an explicit arg,
        the config change loses effect.
        """
        try:
            mod = importlib.import_module("src.evaluation.poisoning_resistance")
        except ImportError:
            pytest.skip("poisoning_resistance not importable")

        func = getattr(mod, "evaluate_poisoning_resistance", None)
        if func is None:
            pytest.skip("evaluate_poisoning_resistance not found")

        try:
            src = inspect.getsource(func)
        except (OSError, TypeError):
            pytest.skip("Could not retrieve source for evaluate_poisoning_resistance")

        assert 'config' in src and 'synthetic_ratio' in src, (
            "evaluate_poisoning_resistance no longer reads synthetic_ratio from model.config; "
            "the Amendment 4 fix may be broken"
        )
        # The specific expression from line 128
        assert '.get("synthetic_ratio"' in src or ".get('synthetic_ratio'" in src, (
            "evaluate_poisoning_resistance: expected .get(\"synthetic_ratio\") pattern at line 128; "
            "the routing may have changed"
        )

    def test_llm_finetune_and_poisoning_resistance_use_different_ratio_sources(self):
        """
        Structural: the two functions must source the ratio differently.
        One reads from an explicit parameter, the other from model.config.
        """
        try:
            pr_mod = importlib.import_module("src.evaluation.poisoning_resistance")
            ft_mod = importlib.import_module("src.llm.nvidia_finetune")
        except ImportError:
            pytest.skip("Required modules not importable")

        pr_func = getattr(pr_mod, "evaluate_poisoning_resistance", None)
        ft_func = getattr(ft_mod, "build_training_examples", None)
        if pr_func is None or ft_func is None:
            pytest.skip("One or both functions not found")

        try:
            pr_src = inspect.getsource(pr_func)
            ft_src = inspect.getsource(ft_func)
        except (OSError, TypeError):
            pytest.skip("Could not retrieve source")

        # poisoning_resistance reads from config
        pr_reads_config = ".config" in pr_src and "synthetic_ratio" in pr_src
        # LLM fine-tuning must NOT read from config for synthetic_ratio
        ft_reads_config = (
            '.config["synthetic_ratio"]' in ft_src
            or ".config['synthetic_ratio']" in ft_src
            or '.config.get("synthetic_ratio"' in ft_src
            or ".config.get('synthetic_ratio'" in ft_src
        )

        assert pr_reads_config, (
            "poisoning_resistance.evaluate_poisoning_resistance no longer reads "
            "synthetic_ratio from model.config; the Amendment 4 fix is broken"
        )
        assert not ft_reads_config, (
            "nvidia_finetune.build_training_examples reads synthetic_ratio from "
            "model.config; changing the config now affects LLM training data — "
            "verify this is intentional before proceeding"
        )
