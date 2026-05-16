"""
run_tests.py
------------
Minimal pure-Python test runner for environments where pytest cannot be
installed (no network access).

Discovers every function named test_* in tests/ and runs them, collecting
pass/fail counts and printing a pytest-style summary.

Usage:
    python run_tests.py

When pytest IS available (normal dev environments):
    pytest tests/ -v --cov=src --cov-report=term-missing
"""

from __future__ import annotations

import importlib
import inspect
import sys
import time
import traceback
from pathlib import Path
from typing import Callable, List, Tuple


# ── colour helpers ────────────────────────────────────────────────────────────
try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init(autoreset=True)
    GREEN  = Fore.GREEN  + Style.BRIGHT
    RED    = Fore.RED    + Style.BRIGHT
    YELLOW = Fore.YELLOW + Style.BRIGHT
    CYAN   = Fore.CYAN
    RESET  = Style.RESET_ALL
except ImportError:
    GREEN = RED = YELLOW = CYAN = RESET = ""


def _print_header(title: str) -> None:
    bar = "═" * (len(title) + 4)
    print(f"\n{CYAN}{bar}")
    print(f"  {title}")
    print(f"{bar}{RESET}\n")


# ── fixture system (minimal subset of pytest fixtures) ────────────────────────

class FixtureError(Exception):
    pass


def _resolve_fixtures(fn: Callable, fixture_registry: dict) -> dict:
    """
    Return a dict of {param_name: value} for all parameters of fn that
    are registered as fixtures.  Non-fixture params are skipped (they'll
    cause a TypeError at call time, which we catch).
    """
    sig = inspect.signature(fn)
    resolved = {}
    for name, param in sig.parameters.items():
        if name in fixture_registry:
            val = fixture_registry[name]
            resolved[name] = val() if callable(val) else val
        # else: skip — the test may have defaults or we let it fail naturally
    return resolved


# ── test discovery ────────────────────────────────────────────────────────────

def _discover_test_modules() -> List[Path]:
    tests_dir = Path(__file__).parent / "tests"
    return sorted(tests_dir.glob("test_*.py"))


def _collect_test_functions(module) -> List[Tuple[type | None, Callable]]:
    """Return (class_or_None, function) pairs for every test_ callable."""
    collected = []
    for name, obj in inspect.getmembers(module):
        if inspect.isclass(obj) and name.startswith("Test"):
            for mname, mobj in inspect.getmembers(obj):
                if mname.startswith("test_") and callable(mobj):
                    collected.append((obj, mobj))
        elif name.startswith("test_") and callable(obj):
            collected.append((None, obj))
    return collected


# ── fixture registry (mirrors conftest.py) ────────────────────────────────────

def _build_fixture_registry():
    """
    Build the shared fixtures declared in conftest.py.
    We import conftest and call each fixture function to get session-scoped values.
    """
    sys.path.insert(0, str(Path(__file__).parent))
    import tests.conftest as cf

    from src.embedding.embedder import Embedder
    from src.embedding.mock_vertex import MockGenerativeModel, MockTextEmbeddingModel
    from src.storage.vector_store import VectorStore
    from tests.conftest import MINI_CORPUS

    # Session-scoped singletons
    _embedder = Embedder()

    def _populated_store():
        s = VectorStore(embedding_dim=_embedder.embedding_dim)
        s.ingest(MINI_CORPUS, _embedder)
        return s

    import numpy as np

    def _random_vector():
        rng = np.random.default_rng(42)
        v = rng.standard_normal(_embedder.embedding_dim).astype("float32")
        return v / np.linalg.norm(v)

    return {
        "embedder":                _embedder,
        "mock_text_embedding_model": MockTextEmbeddingModel.from_pretrained("textembedding-gecko@003"),
        "mock_generative_model":   MockGenerativeModel("gemini-1.5-pro"),
        "populated_store":         _populated_store,   # callable → fresh store each time
        "empty_store":             lambda: VectorStore(embedding_dim=_embedder.embedding_dim),
        "random_vector":           _random_vector,
    }


# ── runner ────────────────────────────────────────────────────────────────────

def run() -> int:
    _print_header("RAG PROJECT — TEST SUITE")

    # Add project root to path
    root = Path(__file__).parent
    sys.path.insert(0, str(root))

    fixtures = _build_fixture_registry()
    modules_paths = _discover_test_modules()

    passed: List[str] = []
    failed: List[Tuple[str, str]] = []
    skipped: List[str] = []

    for mod_path in modules_paths:
        # Import module
        mod_name = f"tests.{mod_path.stem}"
        try:
            mod = importlib.import_module(mod_name)
        except Exception as e:
            print(f"{RED}IMPORT ERROR  {mod_path.name}: {e}{RESET}")
            continue

        tests = _collect_test_functions(mod)
        print(f"{CYAN}{'─'*60}")
        print(f"  {mod_path.name}  ({len(tests)} tests)")
        print(f"{'─'*60}{RESET}")

        for cls, fn in tests:
            # Build test id
            fn_name = fn.__name__ if cls is None else f"{cls.__name__}.{fn.__name__}"
            test_id = f"{mod_path.stem}::{fn_name}"

            # Resolve fixtures
            try:
                kwargs = _resolve_fixtures(fn, fixtures)
            except Exception as e:
                print(f"  {YELLOW}SKIP  {fn_name}  (fixture error: {e}){RESET}")
                skipped.append(test_id)
                continue

            # Instantiate class if needed
            instance = cls() if cls is not None else None
            # Call setup_method if defined
            if instance and hasattr(instance, "setup_method"):
                instance.setup_method(None)
            callable_fn = getattr(instance, fn.__name__) if instance else fn

            # Call
            t0 = time.perf_counter()
            try:
                callable_fn(**kwargs)
                elapsed = time.perf_counter() - t0
                print(f"  {GREEN}PASS{RESET}  {fn_name}  {CYAN}({elapsed*1000:.1f}ms){RESET}")
                passed.append(test_id)
            except (AssertionError, Exception) as exc:
                elapsed = time.perf_counter() - t0
                tb = traceback.format_exc()
                print(f"  {RED}FAIL{RESET}  {fn_name}  {CYAN}({elapsed*1000:.1f}ms){RESET}")
                print(f"       {RED}{exc}{RESET}")
                failed.append((test_id, tb))

        print()

    # ── summary ───────────────────────────────────────────────────────────────
    total = len(passed) + len(failed) + len(skipped)
    _print_header(f"RESULTS: {len(passed)} passed, {len(failed)} failed, {len(skipped)} skipped  /  {total} total")

    if failed:
        print(f"{RED}Failed tests:{RESET}")
        for tid, tb in failed:
            print(f"\n  {RED}✗ {tid}{RESET}")
            # Show just the relevant lines of the traceback
            lines = [l for l in tb.splitlines() if l.strip() and "File" in l or "Error" in l or "assert" in l.lower()]
            for line in lines[-4:]:
                print(f"    {line}")
        print()
        return 1

    print(f"{GREEN}All tests passed! ✓{RESET}\n")
    return 0


if __name__ == "__main__":
    sys.exit(run())
