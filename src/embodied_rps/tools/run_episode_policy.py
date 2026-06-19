"""Run the actuator-feasible episode policy loop."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from embodied_rps.episode import load_episode_policy_config, run_episode_policy


def main(argv: Sequence[str] | None = None) -> int:
    """Execute synthetic policy episodes from a YAML config."""

    parser = argparse.ArgumentParser(description="Run classifier-to-actuator RPS episodes.")
    parser.add_argument("--config", required=True, type=Path, help="Path to configs/episode_policy.yaml")
    args = parser.parse_args(argv)

    summary = run_episode_policy(load_episode_policy_config(args.config))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
