"""Exercise extended methods on the tiny baseline produced by main.py --smoke."""
from pathlib import Path
import shutil
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from common import configure, set_data, tf, save_json, environment_metadata
from baseline_model import create_student_model
from part1_pruning import PruningComparator
from part2_quantization import QuantizationPipeline
from part3_distillation import DistillationFramework
from part4_interaction_analysis import CompressionInteractionAnalyzer
from compression_evaluator import CompressionEvaluator

cfg = configure(output_dir=Path("outputs/extended_smoke"), seed=42, batch_size=6,
    baseline_epochs=1, finetune_epochs=1, distill_epochs=1, quick=True,
    sensitivity_samples=6, calibration_samples=6, verbose=0, benchmark_repeats=2)
rng = np.random.default_rng(42)
x = rng.integers(0, 256, (24, 32, 32, 3), dtype=np.uint8)
y = np.arange(24, dtype=np.int32) % 100
set_data((x[:12], y[:12], x[12:18], y[12:18], x[18:], y[18:], x[:6]))
source = Path("outputs/smoke")
shutil.copyfile(source / "checkpoints/baseline_initial.weights.h5",
                cfg.output_dir / "checkpoints/baseline_initial.weights.h5")
model = tf.keras.models.load_model(source / "baseline_model.keras", compile=False)
pruner = PruningComparator(str(source / "baseline_model.keras"))
quantizer = QuantizationPipeline(model)
distiller = DistillationFramework(model, create_student_model(width=.125))
analyzer = CompressionInteractionAnalyzer(model)
evaluator = CompressionEvaluator(model)
tasks = [("sensitivity", pruner.analyze_pruning_sensitivity),
         ("lottery_tickets", pruner.lottery_ticket_hypothesis_test),
         ("dynamic_quantization", quantizer.dynamic_quantization_analysis),
         ("quantization_errors", quantizer.quantization_error_analysis),
         ("feature_matching", distiller.feature_matching_distillation),
         ("self_ensemble", distiller.self_distillation_experiments),
         ("six_orderings", analyzer.compression_pipeline_optimization),
         ("robustness", lambda: evaluator.analyze_compression_robustness(model))]
for name, function in tasks:
    destination = cfg.output_dir / "results" / f"{name}.json"
    if destination.exists():
        print("Already checked:", name, flush=True)
        continue
    print("Checking:", name, flush=True)
    save_json(function(), destination)
save_json({**environment_metadata(), "data_source": "synthetic extended execution checks",
           "completed_methods": [name for name, _ in tasks]}, cfg.output_dir / "results/environment.json")
print("Extended synthetic checks completed.", flush=True)
