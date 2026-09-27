from __future__ import annotations

import argparse
import json
from pathlib import Path

from .golden import record_golden, validate_golden
from .spike import run_spike
from .replay import export_golden, evaluate
from .train import brain_train, brain_imitation_train


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m flydrone.tellosim")
    sub = parser.add_subparsers(dest="command", required=True)
    spike = sub.add_parser("spike", help="run SDK, operation, and headless physics spikes")
    spike.add_argument("--world-config", type=Path)
    spike.add_argument("--out", type=Path, default=Path("reports/tellosim/spike.json"))
    record = sub.add_parser("golden-record", help="record one deterministic policy episode")
    record.add_argument("--checkpoint", type=Path, required=True)
    record.add_argument("--scenario", type=Path, default=Path("configs/tellosim/golden_episode.json"))
    record.add_argument("--brain-graph", type=Path)
    record.add_argument("--brain-readout-neurons", type=int, default=64)
    record.add_argument("--seed", type=int, default=11)
    record.add_argument("--out", type=Path, default=Path("artifacts/golden_episode"))
    validate = sub.add_parser("golden-validate", help="validate a recorded Golden Episode")
    validate.add_argument("--episode", type=Path, default=Path("artifacts/golden_episode"))
    train = sub.add_parser("brain-train", help="train a reservoir-backed TelloSim policy")
    train.add_argument("--graph", type=Path, required=True)
    train.add_argument("--out", type=Path, required=True)
    train.add_argument("--total-steps", type=int, default=256)
    train.add_argument("--readout-neurons", type=int, default=64)
    train.add_argument("--device", default="cpu")
    train.add_argument("--seed", type=int, default=11)
    imitate = sub.add_parser("brain-imitation-train", help="train reservoir policy from TelloSim reference trajectories")
    imitate.add_argument("--graph", type=Path, required=True)
    imitate.add_argument("--out", type=Path, required=True)
    imitate.add_argument("--episodes", type=int, default=16)
    imitate.add_argument("--readout-neurons", type=int, default=64)
    imitate.add_argument("--device", default="cpu")
    imitate.add_argument("--seed", type=int, default=11)
    record.add_argument("--device", default="cpu")
    validate.add_argument("--brain-graph", type=Path)
    export = sub.add_parser("golden-export")
    export.add_argument("--episode", type=Path, required=True)
    export.add_argument("--out", type=Path, default=Path("reports/vis/views/golden-episode"))
    export.add_argument("--registry", type=Path, default=Path("reports/vis/views/index.json"))
    evaluation = sub.add_parser("golden-evaluate")
    evaluation.add_argument("--checkpoint", required=True)
    evaluation.add_argument("--brain-graph", required=True)
    evaluation.add_argument("--cases", required=True)
    evaluation.add_argument("--out", required=True)
    evaluation.add_argument("--device", default="cpu")
    evaluation.add_argument("--readout-neurons", type=int, default=64)
    args = parser.parse_args()
    if args.command == "spike":
        print(json.dumps(run_spike(args.world_config, args.out), ensure_ascii=False, indent=2))
    elif args.command == "golden-record":
        print(json.dumps(record_golden(args.out, args.seed, args.checkpoint, args.scenario, args.brain_graph, args.brain_readout_neurons, args.device), ensure_ascii=False, indent=2))
    elif args.command == "golden-validate":
        result = validate_golden(args.episode, args.brain_graph)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        raise SystemExit(0 if result["GOLDEN_EPISODE_READY"] else 1)
    elif args.command == "golden-export":
        m = export_golden(args.episode, args.out, registry=args.registry)
        print(json.dumps({"view_id": m["view_id"], "events": m["event_count"], "replay_sha256": m["replay_sha256"]}))
    elif args.command == "golden-evaluate":
        print(json.dumps(evaluate(args.checkpoint, args.brain_graph, args.cases, args.out, args.device, args.readout_neurons), indent=2))
    elif args.command == "brain-train":
        print(json.dumps(brain_train(args.out, args.graph, args.total_steps, args.seed, args.readout_neurons, args.device), ensure_ascii=False, indent=2))
    elif args.command == "brain-imitation-train":
        print(json.dumps(brain_imitation_train(args.out, args.graph, args.episodes, args.seed, args.readout_neurons, args.device), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
