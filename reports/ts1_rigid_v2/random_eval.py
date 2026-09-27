from pathlib import Path
import json
from reports.ts1_rigid_v2.preflight import RuleClock
from flydrone.tellosim.training.c0_campaign import evaluate
ROOT=Path('.').resolve();OUT=ROOT/'reports/ts1_rigid_v2'
assert (OUT/'frozen-checkpoints.json').exists()
class RandomClock(RuleClock):
    feature_source='uniform_random_no_neural_model'
    training_method='Uniform random baseline; no training or neural features'
    def snapshot(self):return {'status':'not_recorded','neurons':[]}
cases=json.loads((OUT/'cases.json').read_text())['sealed_test']
evaluate(ROOT,RandomClock(),cases,OUT/'random-sealed.json','rigid-random-sealed','random',record_indices=(0,))
