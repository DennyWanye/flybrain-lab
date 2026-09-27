"""Build reproducible visual evidence, never train or replace a checkpoint.

Run: python -m flydrone.tellosim.visual_fixtures --project-root .
"""
from pathlib import Path
import argparse
import json
import torch

from ..policy import ActorCritic
from .golden import load_runtime, record_golden, sha256
from .visual import import_golden, atomic_json


def build(root):
    root = Path(root).resolve()
    output = root / "artifacts/tellosim_visual"
    output.mkdir(parents=True, exist_ok=True)
    original = root / "artifacts/golden_episode_verified"
    golden = json.loads((original / "manifest.json").read_text())
    checkpoint = original / "checkpoint.pt"
    graph = root / "data/male-v1.npz"
    torch.set_num_threads(4)
    trained, brain, digest = load_runtime(checkpoint, graph)
    baseline_path = output / "untrained_checkpoint.pt"
    if not baseline_path.exists():
        # The historical pre-training weights were not saved: identify this as
        # a fresh seed-11 initialization, not a recovered historical checkpoint.
        torch.manual_seed(11)
        baseline = ActorCritic(trained.feature_dim, action_dim=9)
        torch.save({"policy_state_dict":baseline.state_dict(), "feature_dim":trained.feature_dim,
            "action_dim":9, "graph_sha256":brain.graph_sha256,"mapping_sha256":brain.mapping_sha256,
            "seed":11,"mode":"fresh_untrained_baseline_not_historical_initial"}, baseline_path)
    payload = torch.load(baseline_path,map_location="cpu",weights_only=False)
    baseline = ActorCritic(trained.feature_dim,action_dim=9)
    baseline.load_state_dict(payload["policy_state_dict"])
    baseline.eval()
    scenario = golden["scenario_config"]
    baseline_dir = output / "untrained_same_case"
    if not (baseline_dir / "manifest.json").exists():
        record_golden(baseline_dir, golden["seed"], baseline_path, scenario,
                      runtime=(baseline,brain,sha256(baseline_path)))
    import_golden(root, str(baseline_dir.relative_to(root)), "untrained-same-case", "fresh_untrained_reservoir_baseline")
    cases = [json.loads(line) for line in (root / "reports/golden_evaluation_final/episodes.jsonl").read_text().splitlines()]
    failure = next(case for case in cases if case["scenario_config"]["group"] == "reverse_direction" and not case["success"])
    failure_dir = output / "trained_failure"
    if not (failure_dir / "manifest.json").exists():
        record_golden(failure_dir,failure["seed"],checkpoint,failure["scenario_config"],runtime=(trained,brain,digest))
    import_golden(root,str(failure_dir.relative_to(root)),"trained-reverse-failure")
    initial = json.loads((baseline_dir / "manifest.json").read_text())
    result = {"schema_version":"tellosim.visual_comparison/1.0", "scenario":scenario,
              "same_scenario":initial["scenario_config"] == golden["scenario_config"],
              "same_graph":initial["graph_sha256"] == golden["graph_sha256"],
              "same_mapping":initial["brain_mapping_sha256"] == golden["brain_mapping_sha256"],
              "note":"Fresh untrained seed-11 baseline versus existing trained checkpoint; historical initial checkpoint was not saved. Single illustrative case, not a learning/generalization gate.",
              "runs":[]}
    for label, run_id, m in [("新初始化基线","untrained-same-case",initial),("已有训练模型","golden-episode",golden)]:
        result["runs"].append({"label":label,"run_id":run_id,**{k:m[k] for k in ("success","episode_return","final_distance_m","episode_steps","policy_checkpoint_sha256","replay_sha256")}})
    atomic_json(root / "reports/vis/tellosim/comparison.json",result)
    return result


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--project-root",default=".")
    args=parser.parse_args()
    print(json.dumps(build(args.project_root),ensure_ascii=False,indent=2))
