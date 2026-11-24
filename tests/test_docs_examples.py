"""Docs and examples are code: if they rot, these fail."""

import importlib.util
import re
from pathlib import Path

import pytest

from bitstraining.core.config import load_spec
from bitstraining.interfaces.cli import TEMPLATES, build_parser

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("cfg", sorted((ROOT / "examples" / "configs").glob("*.yaml")), ids=lambda p: p.name)
def test_example_configs_are_valid(cfg):
    spec = load_spec(cfg)
    assert spec.model.base and spec.data.train


def test_builtin_templates_are_valid():
    from bitstraining.core.config import spec_from_dict

    for name, t in TEMPLATES.items():
        assert spec_from_dict(t).name, name


def _subcommands():
    parser = build_parser()
    action = next(a for a in parser._actions if a.dest == "cmd")
    return set(action.choices)


def test_every_cli_command_in_the_readme_exists():
    text = (ROOT / "README.md").read_text()
    used = set(re.findall(r"^\s*bitstraining ([a-z]+)", text, flags=re.M))
    assert used, "README should show CLI usage"
    assert used <= _subcommands(), f"README mentions unknown commands: {used - _subcommands()}"


def test_every_readme_option_is_accepted_by_the_parser():
    """Parse each README command line (placeholders substituted) to catch renamed/removed flags."""
    text = (ROOT / "README.md").read_text()
    parser = build_parser()
    checked = 0
    for line in re.findall(r"^\s*bitstraining (.+?)(?:\s+#.*)?$", text, flags=re.M):
        if "..." in line or "<" in line:
            continue
        argv = line.replace("--backend vllm", "--backend vllm").split()
        try:
            parser.parse_args(argv)
        except SystemExit as e:  # argparse exits on bad input
            pytest.fail(f"README command no longer parses: bitstraining {line} (exit {e.code})")
        checked += 1
    assert checked >= 10


def test_custom_provider_example_runs():
    spec = importlib.util.spec_from_file_location("custom_provider_example", ROOT / "examples" / "custom_provider.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main()  # raises if the orchestrator, bundle or runner break


def test_readme_does_not_hardcode_a_test_count():
    assert not re.search(r"\b\d{3} tests\b", (ROOT / "README.md").read_text())
