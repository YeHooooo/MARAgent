from __future__ import annotations

import argparse
from pathlib import Path

from maragent.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the MARAgent pipeline.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--input", help="Single input image, NPY, H5, or HDF5 file.")
    parser.add_argument("--input-dir", help="Directory for batch inference.")
    parser.add_argument(
        "--glob",
        default="*.png",
        help='Glob used with --input-dir, for example "*.png", "*.npy", or "**/*.h5".',
    )
    parser.add_argument("--output", help="Override output directory.")
    parser.add_argument("--tools-root", help="Override tools directory.")
    parser.add_argument("--offline", action="store_true", help="Use heuristic VLM substitute for smoke tests.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.output:
        config["paths"]["output_dir"] = args.output
    if args.tools_root:
        config["paths"]["tools_root"] = args.tools_root
    if args.offline:
        config["runtime"]["offline"] = True

    from maragent.pipeline import MARAgentPipeline

    pipeline = MARAgentPipeline(config=config)
    if args.input:
        result = pipeline.run_path(args.input)
        print(f"[Done] {result.case_id}: best={result.best_model}, summary={result.summary_path}")
        return

    if not args.input_dir:
        raise SystemExit("Provide --input or --input-dir.")

    paths = sorted(path for path in Path(args.input_dir).glob(args.glob) if path.is_file())
    if not paths:
        raise SystemExit(f"No files matched {args.glob} in {args.input_dir}")
    results = pipeline.run_many(paths)
    print(f"[Done] processed={len(results)}")


if __name__ == "__main__":
    main()
