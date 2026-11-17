"""Repeat INT8 QAT with three training seeds and a fixed baseline/data split."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (tf, configure, set_seed, accuracy, fit_model, save_json,
                    save_model, environment_metadata)
from baseline_model import prepare_compression_datasets
from part2_quantization import quantize_model
from compression_evaluator import CompressionEvaluator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--num-seeds', type=int, default=3)
    args = parser.parse_args()
    if args.num_seeds < 3 or args.epochs < 1:
        parser.error('Use at least three seeds and one epoch.')
    destination = Path(args.output_dir)
    if (destination / 'seed_results.json').exists():
        parser.error('Results already exist; choose a new output directory.')
    configure(output_dir=destination, seed=42, finetune_epochs=args.epochs)
    prepare_compression_datasets()  # Split fixed before changing training RNG seeds.
    baseline = tf.keras.models.load_model(args.baseline, compile=False)
    baseline_score = accuracy(baseline, 'test')
    save_json(environment_metadata(), destination / 'results/environment.json')

    def experiment(seed):
        set_seed(seed)
        model = quantize_model(baseline, bits=8)
        training = fit_model(model, learning_rate=1e-5, name=f'qat8_seed_{seed}')
        result = {'seed': seed, 'test_accuracy': accuracy(model, 'test'),
                  'validation_accuracy': accuracy(model, 'val'),
                  'baseline_accuracy': baseline_score,
                  'artifact_path': save_model(model, f'qat8_seed_{seed}'),
                  'training_time_seconds': training['training_time_seconds']}
        save_json(result, destination / 'results' / f'seed_{seed}.json')
        return result

    result = CompressionEvaluator().cross_validate_compression(experiment, args.num_seeds)
    result['scope'] = 'INT8 reference QAT only; fixed trained baseline and CIFAR-100 split.'
    result['limitation'] = ('Measures compression-training variability conditional on one baseline; '
                           'does not measure baseline-training variability or other techniques.')
    save_json(result, destination / 'seed_results.json')
    print(f'Seed evaluation complete: {destination / "seed_results.json"}')


if __name__ == '__main__':
    main()
