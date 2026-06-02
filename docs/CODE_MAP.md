# Code Map

This file maps `TMI_MARAgent.pdf` concepts to the cleaned codebase.

| Paper component | Code |
| --- | --- |
| VLM-based Perception Agent | `maragent/agents/perception.py` |
| Smart Route: Low/Medium/High routing | `maragent/core/router.py` |
| Fast Restoration | `SmartRouter.route(...).route == "fast_restoration"` |
| Memory Search / RAG | `maragent/memory/bank.py` |
| All Model Race | `SmartRouter.route(...).route == "all_model_race"` |
| MAR expert model pool | `maragent/models/registry.py` and `tools/` |
| VLM no-reference selection | `maragent/agents/restoration.py` |
| Difference-map safety check | `maragent/core/diff.py` and `maragent/agents/report.py` |
| End-to-end workflow | `maragent/pipeline.py` |
| CLI entry point | `scripts/run_maragent.py` |

## Source Cleanup

The original folder mixed agent scripts, model implementations, checkpoints,
memory records, generated images, and temporary outputs. The new repository
separates them:

- reusable agent logic lives under `maragent/`;
- third-party MAR model source code lives under `tools/`;
- generated outputs go to `outputs/`;
- retrieved and reviewed cases go to `memory_bank/`;
- checkpoints are configured but not committed.

The old hard-coded API key is intentionally not copied. Use environment
variables documented in `README.md`.
