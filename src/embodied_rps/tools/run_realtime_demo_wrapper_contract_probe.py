"""Run the PowerShell contract probe for the realtime demo strict wrapper."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from embodied_rps.realtime_demo_wrapper_contract_probe import (
    RealtimeDemoWrapperContractProbeConfig,
    run_realtime_demo_wrapper_contract_probe,
)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for strict-wrapper contract probing."""

    parser = argparse.ArgumentParser(description="Run the realtime demo strict-wrapper contract probe.")
    parser.add_argument("--output-root", type=Path, default=RealtimeDemoWrapperContractProbeConfig.output_root)
    parser.add_argument(
        "--strict-exit-code",
        action="store_true",
        help="Return 70 when the wrapper contract probe fails.",
    )
    args = parser.parse_args(argv)

    summary = run_realtime_demo_wrapper_contract_probe(
        RealtimeDemoWrapperContractProbeConfig(output_root=args.output_root)
    )
    print(json.dumps(summary, indent=2))
    if args.strict_exit_code and summary.get("contract_status") != "passed":
        return 70
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
