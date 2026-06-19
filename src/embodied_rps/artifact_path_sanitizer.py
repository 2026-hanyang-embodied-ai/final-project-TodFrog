"""Utilities for keeping durable artifact metadata workspace-relative."""

from __future__ import annotations

import json
from pathlib import Path

TEXT_ARTIFACT_SUFFIXES = frozenset(
    {
        ".csv",
        ".html",
        ".json",
        ".jsonl",
        ".md",
        ".txt",
        ".yaml",
        ".yml",
    }
)


def relativize_project_root_text_paths(
    *,
    project_root: Path,
    roots: list[Path],
    external_roots: dict[str, Path] | None = None,
) -> dict[str, object]:
    """Rewrite local absolute paths in text artifacts to portable path tokens."""

    project_root = project_root.resolve(strict=False)
    replacements = _root_replacements(project_root, child_prefix="", exact_replacement=".")
    for name, root in (external_roots or {}).items():
        token = f"{name}:"
        replacements.extend(
            _root_replacements(root.resolve(strict=False), child_prefix=f"{token}/", exact_replacement=token)
        )
    rewritten: list[str] = []
    scanned = 0
    for root in roots:
        root = root.resolve(strict=False)
        if not root.exists():
            continue
        root.relative_to(project_root)
        candidates = [root] if root.is_file() else sorted(path for path in root.rglob("*") if path.is_file())
        for path in candidates:
            if path.suffix.lower() not in TEXT_ARTIFACT_SUFFIXES:
                continue
            scanned += 1
            try:
                original = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            updated = _apply_replacements(original, replacements)
            if updated != original:
                path.write_text(updated, encoding="utf-8")
                rewritten.append(path.relative_to(project_root).as_posix())
    return {
        "status": "passed",
        "scanned_text_artifacts": scanned,
        "rewritten_text_artifacts": rewritten,
        "rewritten_count": len(rewritten),
        "policy": "local absolute paths are rewritten to workspace-relative paths or explicit external-root tokens in durable text metadata",
    }


def _root_replacements(root: Path, *, child_prefix: str, exact_replacement: str) -> list[tuple[str, str]]:
    posix_root = root.as_posix()
    native_root = str(root)
    variants = [posix_root, native_root, _json_string_body(posix_root), _json_string_body(native_root)]
    ordered_variants = sorted({variant for variant in variants if variant}, key=len, reverse=True)
    replacements: list[tuple[str, str]] = []
    for variant in ordered_variants:
        replacements.extend(
            [
                (f"{variant}/", child_prefix),
                (f"{variant}\\", child_prefix),
                (variant, exact_replacement),
            ]
        )
    return replacements


def _json_string_body(value: str) -> str:
    return json.dumps(value, ensure_ascii=True)[1:-1]


def _apply_replacements(text: str, replacements: list[tuple[str, str]]) -> str:
    updated = text
    for old, new in replacements:
        updated = updated.replace(old, new)
    return updated


__all__ = ["relativize_project_root_text_paths"]
