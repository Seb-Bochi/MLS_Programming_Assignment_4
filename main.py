"""Run the assignment's six-step compression pipeline without producing a report."""
from __future__ import annotations

import argparse
import gc
import weakref
from pathlib import Path
import numpy as np
from common import (tf, ExperimentConfig, configure, get_config, get_data, set_data,
                    fit_model, save_json, json_ready, environment_metadata, safe_name)
from baseline_model import (prepare_compression_datasets, train_baseline_model,
                            create_student_model)
from part1_pruning import PruningComparator
from part2_quantization import QuantizationPipeline
from part3_distillation import DistillationFramework
from part4_interaction_analysis import CompressionInteractionAnalyzer
from compression_evaluator import CompressionEvaluator
from visualization_tools import create_compression_visualizations, export_benchmarks
from application_scenarios import ApplicationScenarioAnalysis, generate_compression_guidelines
from neural_architecture_search import search_student_architecture


def _smoke_baseline():
    """Tiny residual fixture and random data for execution checks only."""
    cfg = get_config()
    rng = np.random.default_rng(cfg.seed)
    images = rng.integers(0, 256, (24, 32, 32, 3), dtype=np.uint8)
    labels = np.arange(24, dtype=np.int32) % 100
    set_data((images[:12], labels[:12], images[12:18], labels[12:18],
              images[18:], labels[18:], images[:6]))
    inputs = tf.keras.Input((32, 32, 3), name="images")
    x = tf.keras.layers.Rescaling(1 / 255, name="normalize")(inputs)
    x = tf.keras.layers.Conv2D(16, 1, name="stem")(x)
    for stage in range(2, 5):
        shortcut = x
        prefix = f"conv{stage}_block1"
        for index, width in ((1, 8), (2, 8), (3, 16)):
            x = tf.keras.layers.Conv2D(width, 3 if index == 2 else 1, padding="same",
                                      name=f"{prefix}_{index}_conv")(x)
            x = tf.keras.layers.BatchNormalization(name=f"{prefix}_{index}_bn")(x)
            if index < 3:
                x = tf.keras.layers.ReLU(name=f"{prefix}_{index}_relu")(x)
        x = tf.keras.layers.Add(name=f"{prefix}_add")([shortcut, x])
        x = tf.keras.layers.ReLU(name=f"{prefix}_out")(x)
        x = tf.keras.layers.AveragePooling2D(name=f"pool{stage}")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    model = tf.keras.Model(inputs, tf.keras.layers.Dense(100, name="logits")(x), name="synthetic_smoke_resnet")
    model.save_weights(str(cfg.output_dir / "checkpoints" / "baseline_initial.weights.h5"))
    fit_model(model, epochs=1, name="smoke_baseline")
    model.save(cfg.output_dir / "baseline_model.keras")
    return model


def run_pipeline(args):
    cfg = ExperimentConfig(seed=args.seed, output_dir=Path(args.output_dir),
        baseline_epochs=args.baseline_epochs, finetune_epochs=args.finetune_epochs,
        distill_epochs=args.distill_epochs, batch_size=args.batch_size,
        imagenet_weights=not args.no_pretrained, verbose=0 if args.quiet else 2)
    if args.quick or args.smoke:
        cfg.quick = True
        cfg.baseline_epochs = cfg.finetune_epochs = cfg.distill_epochs = 1
        cfg.warmup_epochs = 1
        cfg.train_limit, cfg.val_limit, cfg.test_limit = 1000, 200, 200
        cfg.calibration_samples, cfg.sensitivity_samples = 100, 32
        cfg.benchmark_repeats = 3
        cfg.batch_size = min(cfg.batch_size, 16)
    if args.smoke:
        cfg.batch_size, cfg.sensitivity_samples, cfg.calibration_samples = 6, 6, 6
    configure(cfg)
    metadata = environment_metadata()
    metadata["data_source"] = "synthetic execution test, not CIFAR-100 evidence" if args.smoke else "CIFAR-100"
    metadata["experiments"] = args.parts
    save_json(metadata, cfg.output_dir / "results" / "environment.json")

    # Step 1: Prepare data and train/load baseline.
    print("Preparing datasets and baseline model...")
    baseline_path = cfg.output_dir / "baseline_model.keras"
    if args.smoke:
        baseline_model = _smoke_baseline()
    else:
        prepare_compression_datasets()
        if baseline_path.exists() and not args.retrain:
            baseline_model = tf.keras.models.load_model(baseline_path, compile=False)
        else:
            baseline_model = train_baseline_model()
    if args.baseline_only:
        return {"baseline_model": str(baseline_path)}

    # Step 2: Initialize frameworks as each family is used, limiting retained models.
    evaluator = CompressionEvaluator(baseline_model=baseline_model, power_watts=args.power_watts)
    data = get_data()
    test_data = (data[4], data[5])
    all_results = {"baseline": evaluator.benchmark_compressed_model(baseline_model, test_data, "baseline")}
    experiment_results = {}
    evaluated = weakref.WeakSet([baseline_model])
    evaluated_files = set()

    # Step 4 is streamed after each Step 3 experiment to release large graphs.
    def evaluate_models(value, name):
        if isinstance(value, tf.keras.Model):
            if value not in evaluated:
                all_results[name] = evaluator.benchmark_compressed_model(value, test_data, name)
                evaluated.add(value)
            return
        if isinstance(value, dict):
            for key, child in value.items():
                if isinstance(child, (dict, tf.keras.Model)):
                    evaluate_models(child, f"{name}_{key}")
                elif isinstance(child, (str, Path)) and str(child).endswith(".tflite") and str(child) not in evaluated_files:
                    evaluated_files.add(str(child))
                    all_results[f"{name}_{key}"] = evaluator.benchmark_compressed_model(child, test_data, f"{name}_{key}")

    def collect(family, experiments):
        for name, function in experiments:
            print(f"Running {family}/{name}...")
            result = function()
            evaluate_models(result, f"{family}_{name}")
            summary = json_ready(result)
            experiment_results.setdefault(family, {})[name] = summary
            save_json(summary, cfg.output_dir / "results" / f"{family}_{name}.json")
            save_json(all_results, cfg.output_dir / "performance_benchmarks.json")
            del result
            gc.collect()

    # Step 3: Compression experiments with original method names.
    if "pruning" in args.parts:
        comparator = PruningComparator(str(baseline_path))
        tasks = [("magnitude_pruning", comparator.magnitude_based_pruning),
                 ("structured_pruning", comparator.structured_pruning),
                 ("gradual_vs_oneshot", comparator.gradual_vs_oneshot_pruning)]
        if args.extended:
            tasks += [("sensitivity", comparator.analyze_pruning_sensitivity),
                      ("lottery_ticket", comparator.lottery_ticket_hypothesis_test)]
        collect("pruning", tasks)
        del tasks, comparator
        gc.collect()
    if "quantization" in args.parts:
        pipeline = QuantizationPipeline(baseline_model)
        tasks = [("mixed_bit", pipeline.mixed_bit_quantization),
                 ("ptq_vs_qat", pipeline.post_training_vs_qat_comparison),
                 ("extreme_quantization", pipeline.extreme_quantization)]
        if args.extended:
            tasks += [("dynamic_vs_static", pipeline.dynamic_quantization_analysis),
                      ("error_analysis", pipeline.quantization_error_analysis)]
        collect("quantization", tasks)
        del tasks, pipeline
        gc.collect()
    if "distillation" in args.parts:
        student = create_student_model(width=0.125) if args.smoke else "mobile"
        framework = DistillationFramework(baseline_model, student_architecture=student)
        tasks = [("temperature_optimization", lambda: framework.temperature_optimization(num_trials=2 if cfg.quick else 10)),
                 ("progressive_distillation", lambda: framework.progressive_distillation(
                     intermediate_sizes=[0.5] if args.smoke else ([0.02] if cfg.quick else [0.75, 0.5, 0.25]))),
                 ("attention_transfer", framework.attention_transfer)]
        if args.extended:
            tasks += [("feature_matching", framework.feature_matching_distillation),
                      ("self_distillation", framework.self_distillation_experiments)]
        collect("distillation", tasks)
        del tasks, framework
        gc.collect()
    if "interaction" in args.parts:
        analyzer = CompressionInteractionAnalyzer(baseline_model)
        result = analyzer.comprehensive_compression_analysis()
        evaluate_models(result, "interaction")
        experiment_results["interaction"] = json_ready(result)
        save_json(result, cfg.output_dir / "results" / "interactions.json")
        save_json(analyzer.pareto_frontier_analysis(), cfg.output_dir / "results" / "pareto_analysis.json")
        save_json(analyzer.failure_mode_analysis(), cfg.output_dir / "results" / "failure_modes.json")
        if args.extended:
            ordering = analyzer.compression_pipeline_optimization()
            evaluate_models(ordering, "ordering")
            save_json(ordering, cfg.output_dir / "results" / "orderings.json")
        del analyzer, result
        gc.collect()
    if "nas" in args.parts:
        collect("nas", [("architecture_search", lambda: search_student_architecture(
            baseline_model, widths=(0.125, 0.25) if args.smoke else ((0.5, 1.0) if cfg.quick else (0.5, 1.0, 1.5)),
            depths=(1,) if cfg.quick else (1, 2)))])
    if args.extended:
        # Choose using validation, then report robustness once on the untouched test set.
        candidates = [(name, metrics) for name, metrics in all_results.items() if name != "baseline"]
        if candidates:
            from common import accuracy
            from part2_quantization import evaluate_tflite
            validation = {}
            for name, metrics in candidates:
                path = metrics["artifact_path"]
                if path and path.endswith(".tflite"):
                    validation[name] = evaluate_tflite(path)
                elif path:
                    model = tf.keras.models.load_model(path, compile=False)
                    validation[name] = accuracy(model)
            chosen = max(validation, key=validation.get)
            path = all_results[chosen]["artifact_path"]
            model = path if path.endswith(".tflite") else tf.keras.models.load_model(path, compile=False)
            save_json({"selected_model": chosen, "selection_split": "validation",
                       "results": evaluator.analyze_compression_robustness(model)},
                      cfg.output_dir / "results" / "robustness.json")

    # Step 5: Analysis and visualizations. No LaTeX or PDF report is produced.
    print("Generating benchmark files and visualizations...")
    export_benchmarks(all_results, include_workbook=not args.no_workbook)
    create_compression_visualizations(all_results, experiment_results)
    guidelines = generate_compression_guidelines(all_results, cfg.output_dir / "compression_guidelines.md")

    # Step 6: Application scenario analysis.
    scenarios = ApplicationScenarioAnalysis(all_results)
    save_json({"mobile": scenarios.mobile_deployment_optimization(),
               "edge": scenarios.edge_device_optimization(),
               "cloud": scenarios.cloud_inference_optimization()}, cfg.output_dir / "results" / "application_scenarios.json")
    save_json(experiment_results, cfg.output_dir / "results" / "all_experiments.json")
    save_json(guidelines, cfg.output_dir / "results" / "guidelines.json")
    print(f"Compression pipeline complete. Artifacts: {cfg.output_dir.resolve()}")
    if args.smoke:
        print("Synthetic smoke run only; these numbers do not establish CIFAR-100 accuracy.")
    return all_results


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--parts", nargs="+", choices=["pruning", "quantization", "distillation", "interaction", "nas"],
                        default=["pruning", "quantization", "distillation", "interaction", "nas"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--baseline-epochs", type=int, default=100)
    parser.add_argument("--finetune-epochs", type=int, default=10)
    parser.add_argument("--distill-epochs", type=int, default=30)
    parser.add_argument("--quick", action="store_true", help="One-epoch, small real CIFAR-100 run")
    parser.add_argument("--smoke", action="store_true", help="Tiny synthetic run with no downloads; not accuracy evidence")
    parser.add_argument("--extended", action="store_true", help="Sensitivity, lottery tickets, feature/ensemble KD, all six orders and robustness")
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--retrain", action="store_true", help="Replace baseline in selected output directory")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-workbook", action="store_true", help="Export JSON/CSV only")
    parser.add_argument("--power-watts", type=float, help="Optional measured device power for rough energy estimate")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if min(args.batch_size, args.baseline_epochs, args.finetune_epochs, args.distill_epochs) < 1:
        parser.error("batch size and epoch counts must be positive")
    if args.quick and args.smoke:
        parser.error("choose either --quick or --smoke")
    args.output_dir = args.output_dir or ("outputs/smoke" if args.smoke else "outputs/quick" if args.quick else ".")
    if (args.quick or args.smoke) and Path(args.output_dir).resolve() == Path(".").resolve():
        parser.error("quick/smoke runs require a separate output directory")
    return args


if __name__ == "__main__":
    run_pipeline(parse_args())
