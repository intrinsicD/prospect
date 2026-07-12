#!/usr/bin/env python3
"""Structural validator for transformational-research-ideation portfolios.

This checks formatting and calibrated language. It does not establish scientific
novelty, correctness, or publication value.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

FORBIDDEN = (
    "definitely novel",
    "no one has ever done this",
    "guaranteed publication",
    "guarantees a publication",
)

IDEA_FIELDS = (
    "central claim",
    "novelty class",
    "known foundation",
    "irreducible delta",
    "new prediction",
    "cheapest killing test",
    "prior-art threats",
    "novelty confidence",
)

PORTFOLIO_SECTIONS = (
    "frontier map",
    "functional problem signature",
    "productive recombinations",
    "exploratory candidates",
    "transformational candidates",
    "cross-domain transfers",
    "new-evidence discovery programs",
    "recommended first experiment",
    "audit limitations",
)

NOVELTY_CLASS_RE = re.compile(r"\bN(?:[0-4](?:-T)?)\b", re.IGNORECASE)
HEADING_RE = re.compile(r"^#{2,4}\s+(.+?)\s*$", re.MULTILINE)


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold()).strip()


def validate(text: str) -> list[str]:
    problems: list[str] = []
    lowered = normalize(text)

    for phrase in FORBIDDEN:
        if phrase in lowered:
            problems.append(f"forbidden certainty claim: {phrase!r}")

    for section in PORTFOLIO_SECTIONS:
        if section not in lowered:
            problems.append(f"missing portfolio section: {section}")

    if not NOVELTY_CLASS_RE.search(text):
        problems.append("no novelty class (N0–N4, optionally -T) found")

    idea_count = len(
        re.findall(
            r"^#{2,4}\s+.*(?:idea|candidate|transfer|program|hypothesis|method)",
            text,
            re.I | re.M,
        )
    )
    if idea_count < 6:
        problems.append(
            "fewer than 6 candidate-like headings detected; full portfolios should contain multiple independent ideas"
        )

    for field in IDEA_FIELDS:
        if field not in lowered:
            problems.append(f"no idea card contains required field: {field}")

    if "cutoff" not in lowered:
        problems.append("missing literature/search cutoff disclosure")
    if "null hypothesis" not in lowered:
        problems.append("missing null hypothesis in a falsification test")
    if "adoption barrier" not in lowered:
        problems.append("missing adoption barrier for cross-domain transfer")
    if "broken correspondence" not in lowered and "broken correspondences" not in lowered:
        problems.append("missing broken-correspondence analysis for cross-domain transfer")

    return problems


def self_test() -> int:
    valid = """
# Research Portfolio
Literature cutoff: 2026-07-12. Sources searched: papers and repositories.
## Frontier map
## Functional problem signature
## Productive recombinations
### Idea one
Central claim: testable. Novelty class: N1. Known foundation: A. Irreducible delta: D.
New prediction: P. Cheapest killing test: E. Null hypothesis: H0. Prior-art threats: W.
Novelty confidence: provisional.
### Idea two
### Candidate three
## Exploratory candidates
### Candidate four
## Transformational candidates
### Candidate five
Novelty class: N3.
## Cross-domain transfers
### Transfer six
Adoption barrier: previous compute. Broken correspondences: three mismatches.
## New-evidence discovery programs
### Program seven
## Recommended first experiment
## Audit limitations
"""
    invalid = "This is definitely novel and guaranteed publication."
    ok = not validate(valid) and bool(validate(invalid))
    print("self-test: PASS" if ok else "self-test: FAIL")
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("portfolio", nargs="?", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        return self_test()
    if args.portfolio is None:
        parser.error("provide a Markdown portfolio path or --self-test")
    if not args.portfolio.is_file():
        print(f"error: file not found: {args.portfolio}", file=sys.stderr)
        return 2

    problems = validate(args.portfolio.read_text(encoding="utf-8"))
    if problems:
        print("Portfolio validation: FAIL")
        for problem in problems:
            print(f"- {problem}")
        return 1

    print("Portfolio validation: PASS")
    print("Note: structural validation does not establish scientific novelty.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
