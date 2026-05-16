"""
test_benchmarker.py
-------------------
Integration tests for src/evaluation/benchmarker.py

Tests cover:
  - Benchmarker initialises and ingests the full corpus
  - run() returns structured results for all queries
  - Strategy A and B both return top_k results per query
  - Scores are valid and descending
  - Expanded queries are non-empty and longer than originals
  - JSON and Markdown reports can be saved
  - Custom queries are respected

Note: We instantiate Benchmarker once per class via a class-level _bench
attribute to avoid re-loading the model on every single test method.
"""
from __future__ import annotations

import json
import os
import tempfile

import pytest

from src.evaluation.benchmarker import BENCHMARK_QUERIES, Benchmarker


# ---------------------------------------------------------------------------
# Shared Benchmarker — created once at module load time (not via pytest
# session fixture, so it works with both pytest and our custom runner)
# ---------------------------------------------------------------------------

def _make_bench() -> Benchmarker:
    b = Benchmarker(queries=BENCHMARK_QUERIES, top_k=3)
    b.run()
    return b


class TestBenchmarkerInit:
    def setup_method(self, _):
        if not hasattr(TestBenchmarkerInit, "_bench"):
            TestBenchmarkerInit._bench = _make_bench()
        self.bench = TestBenchmarkerInit._bench

    def test_vector_store_populated(self) -> None:
        assert len(self.bench.vector_store) == 10  # full CORPUS

    def test_retriever_a_attached(self) -> None:
        assert self.bench.retriever_a is not None

    def test_retriever_b_attached(self) -> None:
        assert self.bench.retriever_b is not None


class TestRunResults:
    def setup_method(self, _):
        if not hasattr(TestRunResults, "_bench"):
            TestRunResults._bench = _make_bench()
        self.bench = TestRunResults._bench

    def test_results_count_matches_queries(self) -> None:
        assert len(self.bench._results) == len(BENCHMARK_QUERIES)

    def test_each_result_has_required_keys(self) -> None:
        required = {"query", "expanded_query", "strategy_a", "strategy_b"}
        for entry in self.bench._results:
            assert required.issubset(entry.keys()), f"Missing: {required - entry.keys()}"

    def test_strategy_a_returns_top_k(self) -> None:
        for entry in self.bench._results:
            assert len(entry["strategy_a"]) == 3

    def test_strategy_b_returns_top_k(self) -> None:
        for entry in self.bench._results:
            assert len(entry["strategy_b"]) == 3

    def test_expanded_query_non_empty(self) -> None:
        for entry in self.bench._results:
            assert entry["expanded_query"].strip()

    def test_expanded_query_differs_from_original(self) -> None:
        for entry in self.bench._results:
            assert len(entry["expanded_query"]) > len(entry["query"])

    def test_strategy_a_scores_descending(self) -> None:
        for entry in self.bench._results:
            scores = [r["score"] for r in entry["strategy_a"]]
            assert scores == sorted(scores, reverse=True)

    def test_strategy_b_scores_descending(self) -> None:
        for entry in self.bench._results:
            scores = [r["score"] for r in entry["strategy_b"]]
            assert scores == sorted(scores, reverse=True)

    def test_result_scores_bounded(self) -> None:
        for entry in self.bench._results:
            for strategy in ("strategy_a", "strategy_b"):
                for r in entry[strategy]:
                    assert -1.01 <= r["score"] <= 1.01

    def test_result_doc_ids_exist(self) -> None:
        """All retrieved doc IDs must correspond to ingested corpus documents."""
        corpus_ids = {d.id for d in self.bench.vector_store._documents}
        for entry in self.bench._results:
            for strategy in ("strategy_a", "strategy_b"):
                for r in entry[strategy]:
                    assert r["id"] in corpus_ids


class TestCustomQueries:
    def test_single_custom_query(self) -> None:
        b = Benchmarker(queries=["How does caching reduce database load?"], top_k=2)
        results = b.run()
        assert len(results) == 1
        assert len(results[0]["strategy_a"]) == 2
        assert len(results[0]["strategy_b"]) == 2

    def test_multiple_custom_queries(self) -> None:
        queries = [
            "What is a circuit breaker?",
            "How does autoscaling work?",
        ]
        b = Benchmarker(queries=queries, top_k=1)
        results = b.run()
        assert len(results) == 2
        for entry in results:
            assert entry["query"] in queries


class TestOutputSaving:
    def setup_method(self, _):
        if not hasattr(TestOutputSaving, "_bench"):
            TestOutputSaving._bench = _make_bench()
        self.bench = TestOutputSaving._bench

    def test_save_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "results.json")
            saved = self.bench.save_json(path)
            assert os.path.exists(saved)
            with open(saved) as f:
                data = json.load(f)
            assert isinstance(data, list)
            assert len(data) == len(BENCHMARK_QUERIES)

    def test_json_is_valid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "results.json")
            self.bench.save_json(path)
            with open(path) as f:
                data = json.load(f)
            first = data[0]
            assert "query" in first
            assert "strategy_a" in first
            assert isinstance(first["strategy_a"], list)

    def test_save_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "report.md")
            saved = self.bench.save_markdown_report(path)
            assert os.path.exists(saved)
            with open(saved) as f:
                content = f.read()
            assert "Strategy A" in content
            assert "Strategy B" in content
            assert "Overlap" in content

    def test_markdown_contains_all_queries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "report.md")
            self.bench.save_markdown_report(path)
            with open(path) as f:
                content = f.read()
            for q in BENCHMARK_QUERIES:
                assert q[:40] in content, f"Query not found in report: {q[:40]!r}"
