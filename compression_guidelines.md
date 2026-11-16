# Compression technique selection guidelines

Use validation data to choose hyperparameters and report held-out test results only after selection. The assignment targets are >75% baseline test accuracy and at least 10x actual artifact compression with less than 5 percentage points of accuracy loss; these are measured goals, not guaranteed outcomes.

All size metrics use MiB (2**20 bytes). `accuracy_loss_percent` denotes percentage points. Host measurements screen candidates; final mobile, edge and cloud recommendations require the intended hardware/runtime.

## Mobile Deployment

Priorities: model_size, inference_latency.

Candidate techniques: compact student distillation; structured channel pruning; native INT8 PTQ or QAT.

Suggested sequence to test: choose student architecture → distill/fine-tune → structured prune if needed → fine-tune → calibrate and quantize → validate on device.

- **Accuracy:** Select temperature, width and pruning strength on validation data; use QAT if INT8 PTQ loses too much accuracy.
- **Size:** Report actual exported bytes. Packed INT4/binary weights are storage experiments unless a compatible execution backend is supplied.
- **Latency:** Native operator/delegate support and activation traffic determine gains; sparse weights alone do not reduce dense-kernel work.
- **Battery:** Measure device energy and duty cycle before estimating battery impact.

## Edge Computing

Priorities: energy_consumption, memory_usage.

Candidate techniques: small student architecture search; structured pruning; full-integer quantization.

Suggested sequence to test: set flash/RAM/deadline budgets → search compact students → distill → prune and fine-tune → calibrate INT8 → profile target runtime.

- **Memory:** Measure activation/tensor-arena peak separately from model file size and Python process RSS.
- **Latency:** Validate deadline tails under sustained load; a mean host latency is not a real-time guarantee.
- **Portability:** Package preprocessing and supported operators for offline operation; unsupported operators may force a slower fallback.

## Cloud Inference

Priorities: throughput, accuracy.

Candidate techniques: distillation; hardware-aligned structured pruning; backend-supported low precision.

Suggested sequence to test: choose serving batch/concurrency → distill → compare pruning and quantization orders → benchmark actual backend → select a Pareto-efficient candidate.

- **Throughput:** Choose from measured batch throughput subject to an accuracy budget; smaller files do not imply higher throughput.
- **Cost:** Amortize extra compression training and evaluate serving utilization before claiming cost savings.
- **Accuracy:** Use the same dataset split and multiple seeds; do not choose hyperparameters using held-out test accuracy.

## Interaction and failure checks

- Compare compression orderings at a common starting checkpoint and record each stage's training cost.
- Recalibrate after pruning or distillation; previously estimated activation ranges can become stale.
- Keep pruning masks enforced during fine-tuning; otherwise zeroed weights can regrow.
- Inspect layer sensitivity before applying severe pruning or low-bit quantization to every layer.
- Treat INT4 and XNOR-style floating-point reference kernels as algorithm studies; report their measured runtime and packed storage separately.
- Report accuracy, confidence calibration and corruption robustness together; a compact network can preserve accuracy while becoming overconfident.
- Count preprocessing, calibration provenance, initialization and random seeds as part of reproducibility.
