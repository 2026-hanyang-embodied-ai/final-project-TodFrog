"""Helpers for probing and sanitizing imported SCHUNK USD articulations."""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

_PRIM_LINE = re.compile(r'^\s*(?:def(?:\s+[A-Za-z_][\w:]*)?|over)\s+"([^"]+)"')
_SCHEMA_LINE = re.compile(r'apiSchemas\s*=\s*\[(?P<schemas>[^\]]*)\]')
_MIMIC_TOKENS = ("physxMimicJoint:", "newton:mimicJoint")
_ROOT_ARTICULATION_SCHEMAS = ("PhysicsArticulationRootAPI", "NewtonArticulationRootAPI", "PhysxArticulationAPI")
_PROMOTED_ROOT_SCHEMAS = ("PhysicsArticulationRootAPI", "PhysxArticulationAPI")


def candidate_roots_from_usda_text(text: str) -> list[str]:
    """Return prim paths that declare an articulation API in ASCII USDA text."""

    stack: list[str] = []
    candidates: list[str] = []
    pending_prim: str | None = None
    pending_path: str | None = None
    pending_articulation = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        prim_match = _PRIM_LINE.match(raw_line)
        if prim_match is not None:
            pending_prim = prim_match.group(1)
            pending_path = "/" + "/".join([*stack, pending_prim])
            pending_articulation = "ArticulationRootAPI" in line or "PhysxArticulationAPI" in line
        elif pending_prim is not None and ("ArticulationRootAPI" in line or "PhysxArticulationAPI" in line):
            pending_articulation = True
        if "{" in line:
            if pending_prim is not None:
                stack.append(pending_prim)
                if pending_articulation and pending_path is not None:
                    candidates.append(pending_path)
                pending_prim = None
                pending_path = None
                pending_articulation = False
        if "}" in line and stack:
            for _ in range(line.count("}")):
                if stack:
                    stack.pop()
    return sorted(dict.fromkeys(candidates))


def strip_mimic_joint_lines(text: str) -> str:
    """Remove PhysX/Newton mimic-joint metadata while preserving non-mimic USD data."""

    sanitized_lines: list[str] = []
    for raw_line in text.splitlines():
        if any(token in raw_line for token in _MIMIC_TOKENS):
            continue
        if "PhysxMimicJointAPI" in raw_line and "apiSchemas" in raw_line:
            raw_line = _strip_physx_mimic_api_from_schema_line(raw_line)
            if raw_line.strip() == "":
                continue
        sanitized_lines.append(raw_line)
    trailing_newline = "\n" if text.endswith("\n") else ""
    return "\n".join(sanitized_lines) + trailing_newline


def align_articulation_root_to_robot_prim(text: str, *, robot_prim: str = "svh") -> str:
    """Promote articulation root APIs to the robot prim and remove nested root APIs.

    The Isaac URDF importer can place `PhysicsArticulationRootAPI` on the first
    link while joints remain rooted through `/svh`. For control probing, align
    the articulation root to the robot prim so Isaac views can find the DOFs.
    """

    demoted = _remove_nested_articulation_root_apis(text, robot_prim=robot_prim)
    return _promote_robot_prim_apis(demoted, robot_prim=robot_prim)


def rewrite_invalid_svh_body0_targets(text: str) -> str:
    """Rewrite imported joint `body0=/svh` targets to real parent rigid-body paths.

    The SCHUNK USD importer output can compose movable joints with `body0`
    pointing at `/svh`, which is a robot/articulation root but not a rigid body.
    PhysX then creates no articulation DOFs. For tree-shaped imported links, the
    rigid parent path is the parent directory of the `body1` link target.
    """

    lines = text.splitlines()
    output: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if "Physics" in line and "Joint" in line and '"' in line:
            block, next_index = _collect_brace_block(lines, index)
            output.extend(_rewrite_joint_body0_block(block))
            index = next_index
            continue
        output.append(line)
        index += 1
    trailing_newline = "\n" if text.endswith("\n") else ""
    return "\n".join(output) + trailing_newline


def add_default_svh_drive_gains(text: str, *, stiffness: float = 50000.0, damping: float = 1000.0) -> str:
    """Author default angular drive stiffness/damping for imported SCHUNK joints.

    The Isaac URDF importer can emit SCHUNK joints with `PhysicsDriveAPI:angular`
    and `maxForce` only. In that state, position targets are accepted by the
    articulation tensor API but the rendered PhysX step does not hold the target
    pose. The sanitized control USD needs explicit gains for static RPS proof
    renders and later actuator-control experiments.
    """

    lines = text.splitlines()
    output: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if "Physics" in line and "Joint" in line and '"' in line:
            block, next_index = _collect_brace_block(lines, index)
            output.extend(_add_drive_gains_to_joint_block(block, stiffness=stiffness, damping=damping))
            index = next_index
            continue
        output.append(line)
        index += 1
    trailing_newline = "\n" if text.endswith("\n") else ""
    return "\n".join(output) + trailing_newline


def sanitize_usd_tree_for_control(source_dir: Path, destination_dir: Path) -> dict[str, Any]:
    """Copy a USD package and normalize control-oriented USDA payload metadata."""

    if not source_dir.exists() or not source_dir.is_dir():
        raise FileNotFoundError(f"USD source directory does not exist: {source_dir}")
    if destination_dir.exists():
        shutil.rmtree(destination_dir)
    shutil.copytree(source_dir, destination_dir)
    modified = 0
    for path in destination_dir.rglob("*.usda"):
        original = path.read_text(encoding="utf-8")
        sanitized = align_articulation_root_to_robot_prim(
            add_default_svh_drive_gains(rewrite_invalid_svh_body0_targets(strip_mimic_joint_lines(original))),
            robot_prim="svh",
        )
        if sanitized != original:
            path.write_text(sanitized, encoding="utf-8")
            modified += 1
    return {
        "status": "created",
        "source_dir": source_dir.as_posix(),
        "destination_dir": destination_dir.as_posix(),
        "modified_usda_files": modified,
    }


def usd_package_root_for_main_layer(usd_path: Path) -> Path:
    """Return the imported USD package directory for a nested main `.usda` layer."""

    for ancestor in [usd_path, *usd_path.parents]:
        if ancestor.name.endswith(".usd") and ancestor.is_dir():
            return ancestor
    raise ValueError(f"Could not find imported USD package root for {usd_path}")


def _strip_physx_mimic_api_from_schema_line(raw_line: str) -> str:
    match = _SCHEMA_LINE.search(raw_line)
    if match is None:
        return ""
    schemas = [item.strip() for item in match.group("schemas").split(",")]
    kept = [schema for schema in schemas if "PhysxMimicJointAPI" not in schema]
    if not kept:
        return ""
    return raw_line[: match.start("schemas")] + ", ".join(kept) + raw_line[match.end("schemas") :]


def _remove_nested_articulation_root_apis(text: str, *, robot_prim: str) -> str:
    stack: list[str] = []
    pending_prim: str | None = None
    pending_path: str | None = None
    current_schema_path: str | None = None
    output: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        prim_match = _PRIM_LINE.match(raw_line)
        if prim_match is not None:
            pending_prim = prim_match.group(1)
            pending_path = "/" + "/".join([*stack, pending_prim])
            current_schema_path = pending_path

        if "apiSchemas" in raw_line and any(schema in raw_line for schema in _ROOT_ARTICULATION_SCHEMAS):
            schema_path = current_schema_path or ("/" + "/".join(stack) if stack else None)
            if schema_path != f"/{robot_prim}":
                raw_line = _strip_schema_tokens(raw_line, _ROOT_ARTICULATION_SCHEMAS)
                if raw_line.strip() == "":
                    if "{" in line and pending_prim is not None:
                        stack.append(pending_prim)
                        pending_prim = None
                        pending_path = None
                        current_schema_path = "/" + "/".join(stack)
                    continue

        output.append(raw_line)
        if "{" in line:
            if pending_prim is not None:
                stack.append(pending_prim)
                pending_prim = None
                pending_path = None
                current_schema_path = "/" + "/".join(stack)
        if "}" in line and stack:
            for _ in range(line.count("}")):
                if stack:
                    stack.pop()
            current_schema_path = "/" + "/".join(stack) if stack else None
    trailing_newline = "\n" if text.endswith("\n") else ""
    return "\n".join(output) + trailing_newline


def _promote_robot_prim_apis(text: str, *, robot_prim: str) -> str:
    lines = text.splitlines()
    output: list[str] = []
    stack: list[str] = []
    pending_prim: str | None = None
    pending_path: str | None = None
    in_robot_header = False
    robot_header_has_schema = False
    promoted = False

    for raw_line in lines:
        line = raw_line.strip()
        prim_match = _PRIM_LINE.match(raw_line)
        if prim_match is not None:
            pending_prim = prim_match.group(1)
            pending_path = "/" + "/".join([*stack, pending_prim])
            in_robot_header = pending_path == f"/{robot_prim}" and "(" in raw_line
            robot_header_has_schema = False

        if in_robot_header and "apiSchemas" in raw_line:
            raw_line = _append_schema_tokens(raw_line, _PROMOTED_ROOT_SCHEMAS)
            robot_header_has_schema = True
            promoted = True

        if in_robot_header and line == ")" and not robot_header_has_schema:
            indent = raw_line[: len(raw_line) - len(raw_line.lstrip())]
            output.append(f'{indent}    prepend apiSchemas = ["{_PROMOTED_ROOT_SCHEMAS[0]}", "{_PROMOTED_ROOT_SCHEMAS[1]}"]')
            promoted = True

        output.append(raw_line)
        if "{" in line:
            if pending_prim is not None:
                stack.append(pending_prim)
                pending_prim = None
                pending_path = None
            in_robot_header = False
            robot_header_has_schema = False
        if "}" in line and stack:
            for _ in range(line.count("}")):
                if stack:
                    stack.pop()

    if not promoted:
        return text
    trailing_newline = "\n" if text.endswith("\n") else ""
    return "\n".join(output) + trailing_newline


def _strip_schema_tokens(raw_line: str, tokens: tuple[str, ...]) -> str:
    match = _SCHEMA_LINE.search(raw_line)
    if match is None:
        return ""
    schemas = [item.strip() for item in match.group("schemas").split(",") if item.strip()]
    kept = [schema for schema in schemas if not any(token in schema for token in tokens)]
    if not kept:
        return ""
    return raw_line[: match.start("schemas")] + ", ".join(kept) + raw_line[match.end("schemas") :]


def _append_schema_tokens(raw_line: str, tokens: tuple[str, ...]) -> str:
    match = _SCHEMA_LINE.search(raw_line)
    if match is None:
        return raw_line
    schemas = [item.strip() for item in match.group("schemas").split(",") if item.strip()]
    for token in tokens:
        quoted = f'"{token}"'
        if all(token not in schema for schema in schemas):
            schemas.append(quoted)
    return raw_line[: match.start("schemas")] + ", ".join(schemas) + raw_line[match.end("schemas") :]


def _collect_brace_block(lines: list[str], start_index: int) -> tuple[list[str], int]:
    block: list[str] = []
    depth = 0
    started = False
    index = start_index
    while index < len(lines):
        line = lines[index]
        block.append(line)
        depth += line.count("{")
        if "{" in line:
            started = True
        depth -= line.count("}")
        index += 1
        if started and depth <= 0:
            break
    return block, index


def _rewrite_joint_body0_block(block: list[str]) -> list[str]:
    body0_line_indices = [index for index, line in enumerate(block) if "physics:body0" in line]
    body0_target_indices = [index for index, line in enumerate(block) if "prepend rel physics:body0 = </svh>" in line]
    if not body0_target_indices:
        return block
    body1_target = _first_relationship_target(block, "physics:body1")
    if body1_target is None:
        return block
    replacement = _parent_rigid_body_path(body1_target)
    rewritten: list[str] = []
    for index, line in enumerate(block):
        if index in body0_target_indices:
            if replacement is None:
                continue
            rewritten.append(line.replace("</svh>", f"<{replacement}>"))
            continue
        if replacement is None and index in body0_line_indices:
            continue
        rewritten.append(line)
    return rewritten


def _add_drive_gains_to_joint_block(block: list[str], *, stiffness: float, damping: float) -> list[str]:
    if not any("PhysicsDriveAPI:angular" in line for line in block):
        return block
    has_stiffness = any("drive:angular:physics:stiffness" in line for line in block)
    has_damping = any("drive:angular:physics:damping" in line for line in block)
    if has_stiffness and has_damping:
        return block

    rewritten: list[str] = []
    inserted = False
    for line in block:
        rewritten.append(line)
        if not inserted and "drive:angular:physics:maxForce" in line:
            indent = line[: len(line) - len(line.lstrip())]
            if not has_stiffness:
                rewritten.append(f"{indent}float drive:angular:physics:stiffness = {_format_usda_float(stiffness)}")
            if not has_damping:
                rewritten.append(f"{indent}float drive:angular:physics:damping = {_format_usda_float(damping)}")
            inserted = True
    return rewritten


def _format_usda_float(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return f"{value:.6g}"


def _first_relationship_target(block: list[str], rel_name: str) -> str | None:
    pattern = re.compile(rf"prepend\s+rel\s+{re.escape(rel_name)}\s*=\s*<([^>]+)>")
    for line in block:
        match = pattern.search(line)
        if match is not None:
            return "/" + match.group(1).lstrip("/")
    return None


def _parent_rigid_body_path(body1_target: str) -> str | None:
    parent = body1_target.rsplit("/", 1)[0]
    if parent in {"", "/svh", "/svh/Geometry/base_link"}:
        return None
    return parent
