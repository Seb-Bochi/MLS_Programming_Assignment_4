"""Part 2: calibrated PTQ, QAT, mixed-bit and binary quantization.

INT8 deployment uses TensorFlow Lite integer kernels. INT4/INT2/binary layers
are explicit floating-point reference kernels with straight-through gradients;
their measured latency is not a claim of native low-bit acceleration. The .npz
archives really pack low-bit weights, but are storage artifacts, not Lite models.
"""

from pathlib import Path
import json
import time

import numpy as np

from common import (tf, get_config, get_data, clone_model, fit_model, accuracy,
                    save_model, model_weight_bytes, set_seed, safe_name, make_lite_interpreter)


def _validate_bits(bits):
    if isinstance(bits, bool) or int(bits) != bits or int(bits) not in (1, 2, 4, 8):
        raise ValueError("Supported bit widths are 1, 2, 4 and 8.")
    return int(bits)


def _layers(model):
    return [layer for layer in model.layers
            if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D,
                                  tf.keras.layers.Dense))]


def _kernel(layer):
    return (layer.depthwise_kernel if isinstance(layer, tf.keras.layers.DepthwiseConv2D)
            else layer.kernel)


def _ste_sign(value):
    # The clipped straight-through estimator has zero gradient outside the
    # representable latent range.  An identity STE can let binary latent
    # weights grow without bound during long fine-tuning runs.
    clipped = tf.clip_by_value(value, -1.0, 1.0)
    signed = tf.where(value >= 0, tf.ones_like(value), -tf.ones_like(value))
    return clipped + tf.stop_gradient(signed - clipped)


def _weight_quantize(value, bits, axes=None):
    """Symmetric per-output-channel quantization, with STE for training."""
    axes = list(range(len(value.shape) - 1)) if axes is None else axes
    if bits == 1:
        # Alpha is a quantization statistic, not another trainable path.
        scale = tf.stop_gradient(
            tf.reduce_mean(tf.abs(value), axis=axes, keepdims=True))
        return scale * _ste_sign(value)
    qmax = float(2 ** (bits - 1) - 1)
    scale = tf.maximum(tf.reduce_max(tf.abs(value), axis=axes, keepdims=True)
                       / qmax, tf.keras.backend.epsilon())
    rounded = tf.clip_by_value(tf.round(value / scale), -qmax, qmax) * scale
    return value + tf.stop_gradient(rounded - value)


def _activation_quantize(value, bits, minimum, maximum):
    if bits is None:
        return value
    if bits == 1:
        return _ste_sign(value)
    return tf.quantization.fake_quant_with_min_max_vars(
        value, min=float(minimum), max=float(maximum), num_bits=bits,
        narrow_range=False)


@tf.keras.utils.register_keras_serializable(package="Compression")
class QuantizedConv2D(tf.keras.layers.Conv2D):
    """Reference low-bit convolution; binary mode uses XNOR-style alpha/beta."""

    def __init__(self, *args, bits=8, activation_bits=8,
                 activation_min=-1.0, activation_max=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.bits = _validate_bits(bits)
        self.activation_bits = (None if activation_bits is None
                                else _validate_bits(activation_bits))
        self.activation_min = float(activation_min)
        self.activation_max = float(activation_max)
        if self.data_format != "channels_last" or self.groups != 1:
            raise ValueError("Reference kernels require channels_last and groups=1.")
        if self.bits == 1 and self.activation_bits == 1 and self.dilation_rate != (1, 1):
            raise ValueError("XNOR-style patch scaling requires dilation_rate=(1, 1).")

    def call(self, inputs):
        strides = [1, *self.strides, 1]
        dilation = [1, *self.dilation_rate, 1]
        if self.bits == 1 and self.activation_bits == 1:
            outputs = tf.nn.conv2d(_ste_sign(inputs), _ste_sign(self.kernel),
                                   strides, self.padding.upper(), dilations=dilation)
            # One magnitude per input patch and per output filter, as in XNOR-Net.
            beta = tf.nn.avg_pool2d(tf.reduce_mean(tf.abs(inputs), -1, keepdims=True),
                                    self.kernel_size, self.strides, self.padding.upper())
            alpha = tf.reduce_mean(tf.abs(self.kernel), (0, 1, 2))
            outputs = outputs * beta * alpha
        else:
            inputs = _activation_quantize(inputs, self.activation_bits,
                                          self.activation_min, self.activation_max)
            outputs = tf.nn.conv2d(inputs, _weight_quantize(self.kernel, self.bits),
                                   strides, self.padding.upper(), dilations=dilation)
        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias)
        return self.activation(outputs) if self.activation is not None else outputs

    def get_config(self):
        return dict(super().get_config(), bits=self.bits,
                    activation_bits=self.activation_bits,
                    activation_min=self.activation_min,
                    activation_max=self.activation_max)


@tf.keras.utils.register_keras_serializable(package="Compression")
class QuantizedDense(tf.keras.layers.Dense):
    """Reference quantized matrix multiplication with differentiable rounding."""

    def __init__(self, *args, bits=8, activation_bits=8,
                 activation_min=-1.0, activation_max=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.bits = _validate_bits(bits)
        self.activation_bits = (None if activation_bits is None
                                else _validate_bits(activation_bits))
        self.activation_min = float(activation_min)
        self.activation_max = float(activation_max)

    def call(self, inputs):
        if self.bits == 1 and self.activation_bits == 1:
            outputs = tf.matmul(_ste_sign(inputs), _ste_sign(self.kernel))
            outputs *= (tf.reduce_mean(tf.abs(inputs), -1, keepdims=True)
                        * tf.reduce_mean(tf.abs(self.kernel), 0))
        else:
            inputs = _activation_quantize(inputs, self.activation_bits,
                                          self.activation_min, self.activation_max)
            outputs = tf.matmul(inputs, _weight_quantize(self.kernel, self.bits))
        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias)
        return self.activation(outputs) if self.activation is not None else outputs

    def get_config(self):
        return dict(super().get_config(), bits=self.bits,
                    activation_bits=self.activation_bits,
                    activation_min=self.activation_min,
                    activation_max=self.activation_max)


@tf.keras.utils.register_keras_serializable(package="Compression")
class QuantizedDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
    """Depthwise reference kernel, with a separate scale for each output channel."""

    def __init__(self, *args, bits=8, activation_bits=8,
                 activation_min=-1.0, activation_max=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.bits = _validate_bits(bits)
        self.activation_bits = (None if activation_bits is None
                                else _validate_bits(activation_bits))
        self.activation_min = float(activation_min)
        self.activation_max = float(activation_max)
        if self.data_format != "channels_last":
            raise ValueError("Reference kernels require channels_last.")
        if self.bits == 1 and self.activation_bits == 1 and self.dilation_rate != (1, 1):
            raise ValueError("XNOR-style patch scaling requires dilation_rate=(1, 1).")

    def call(self, inputs):
        strides = [1, *self.strides, 1]
        if self.bits == 1 and self.activation_bits == 1:
            outputs = tf.nn.depthwise_conv2d(
                _ste_sign(inputs), _ste_sign(self.depthwise_kernel), strides,
                self.padding.upper(), dilations=self.dilation_rate)
            beta = tf.nn.avg_pool2d(tf.abs(inputs), self.kernel_size, self.strides,
                                    self.padding.upper())
            beta = tf.repeat(beta, self.depth_multiplier, axis=-1)
            alpha = tf.reshape(tf.reduce_mean(tf.abs(self.depthwise_kernel), (0, 1)), [-1])
            outputs = outputs * beta * alpha
        else:
            inputs = _activation_quantize(inputs, self.activation_bits,
                                          self.activation_min, self.activation_max)
            outputs = tf.nn.depthwise_conv2d(
                inputs, _weight_quantize(self.depthwise_kernel, self.bits, axes=(0, 1)),
                strides, self.padding.upper(), dilations=self.dilation_rate)
        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias)
        return self.activation(outputs) if self.activation is not None else outputs

    def get_config(self):
        return dict(super().get_config(), bits=self.bits,
                    activation_bits=self.activation_bits,
                    activation_min=self.activation_min,
                    activation_max=self.activation_max)


def calibrate_activation_ranges(model, calibration_data=None):
    """Collect quantizable-layer input ranges using training-only examples."""
    data = get_data()[-1] if calibration_data is None else calibration_data
    if not len(data):
        raise ValueError("Calibration requires at least one sample.")
    layers = _layers(model)
    probe = tf.keras.Model(model.inputs, [layer.input for layer in layers])
    ranges = {layer.name: [0.0, 0.0] for layer in layers}
    # Small batches bound the memory required by multi-output activation probes.
    for start in range(0, len(data), 8):
        outputs = probe(np.asarray(data[start:start + 8], dtype=np.float32), training=False)
        if not isinstance(outputs, (list, tuple)):
            outputs = [outputs]
        for layer, output in zip(layers, outputs):
            ranges[layer.name][0] = min(ranges[layer.name][0], float(tf.reduce_min(output)))
            ranges[layer.name][1] = max(ranges[layer.name][1], float(tf.reduce_max(output)))
    return {name: (lo, max(hi, lo + 1e-6)) for name, (lo, hi) in ranges.items()}


def quantize_model(model, bits=8, activation_bits=None, calibration_data=None,
                   activation_ranges=None):
    """Clone a model into trainable reference kernels without changing its graph.

    ``bits`` can be an integer or layer-name mapping. By default, activations
    receive the same precision as weights. All other layers retain FP32 state.
    """
    assignment = bits if isinstance(bits, dict) else {l.name: bits for l in _layers(model)}
    for value in assignment.values():
        _validate_bits(value)
    ranges = activation_ranges or calibrate_activation_ranges(model, calibration_data)

    def clone_layer(layer):
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D,
                              tf.keras.layers.Dense)):
            config = layer.get_config()
            for key in ("bits", "activation_bits", "activation_min", "activation_max"):
                config.pop(key, None)
            width = int(assignment.get(layer.name, 8))
            act_width = (activation_bits.get(layer.name, width)
                         if isinstance(activation_bits, dict)
                         else width if activation_bits is None else activation_bits)
            lo, hi = ranges[layer.name]
            if isinstance(layer, tf.keras.layers.DepthwiseConv2D):
                layer_type = QuantizedDepthwiseConv2D
            elif isinstance(layer, tf.keras.layers.Conv2D):
                layer_type = QuantizedConv2D
            else:
                layer_type = QuantizedDense
            return layer_type.from_config(dict(config, bits=width, activation_bits=act_width,
                                                activation_min=lo, activation_max=hi))
        return layer.__class__.from_config(layer.get_config())

    quantized = tf.keras.models.clone_model(model, clone_function=clone_layer)
    quantized.set_weights(model.get_weights())
    return quantized


def _quantized_numpy(kernel, bits, axes=None):
    if not np.isfinite(kernel).all():
        count = int(kernel.size - np.count_nonzero(np.isfinite(kernel)))
        raise ValueError(
            f"Cannot pack a quantized kernel containing {count} non-finite values.")
    axes = tuple(range(kernel.ndim - 1)) if axes is None else axes
    if bits == 1:
        scale = np.mean(np.abs(kernel), axis=axes, keepdims=True).astype(np.float32)
        codes = (kernel >= 0).astype(np.uint8)
        offset = 0
    else:
        offset = 2 ** (bits - 1) - 1
        scale = np.maximum(np.max(np.abs(kernel), axis=axes, keepdims=True) / offset,
                           tf.keras.backend.epsilon()).astype(np.float32)
        codes = (np.clip(np.rint(kernel / scale), -offset, offset) + offset).astype(np.uint8)
    return codes, scale, offset


class _ClipBinaryLatentWeights(tf.keras.callbacks.Callback):
    """Keep full-precision binary-QAT kernels in the STE's latent range."""

    def on_train_batch_end(self, batch, logs=None):
        for layer in _layers(self.model):
            if getattr(layer, "bits", None) == 1:
                kernel = _kernel(layer)
                kernel.assign(tf.clip_by_value(kernel, -1.0, 1.0))
        for weight in self.model.weights:
            tf.debugging.assert_all_finite(
                weight, f"Non-finite value produced in {weight.name}")


def packed_weights(model, name):
    """Save a reloadable archive with actual 1/2/4/8-bit kernel bit packing."""
    destination = Path(get_config().output_dir) / "compressed_models" / f"{safe_name(name)}.packed.npz"
    destination.parent.mkdir(parents=True, exist_ok=True)
    tensors, descriptions = {}, []
    for index, layer in enumerate(model.layers):
        for weight_index, value in enumerate(layer.get_weights()):
            key = f"l{index}_w{weight_index}"
            item = {"key": key, "shape": list(value.shape), "layer": layer.name}
            if isinstance(layer, (QuantizedConv2D, QuantizedDepthwiseConv2D, QuantizedDense)) and weight_index == 0:
                axes = (0, 1) if isinstance(layer, QuantizedDepthwiseConv2D) else None
                codes, scale, offset = _quantized_numpy(value, layer.bits, axes=axes)
                flat = codes.reshape(-1)
                bit_plane = ((flat[:, None] >> np.arange(layer.bits, dtype=np.uint8)) & 1).reshape(-1)
                tensors[key] = np.packbits(bit_plane, bitorder="little")
                tensors[key + "_scale"] = scale
                item.update(bits=layer.bits, offset=offset)
            else:
                tensors[key] = value
            descriptions.append(item)
    tensors["metadata"] = np.asarray(json.dumps({"format_version": 1,
                                                  "model_json": model.to_json(),
                                                  "weights": descriptions}))
    np.savez_compressed(destination, **tensors)
    return str(destination)


def load_packed_model(path):
    """Unpack storage artifacts into reference kernels (FP32 execution memory)."""
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
        model = tf.keras.models.model_from_json(metadata["model_json"])
        layer_weights = {}
        for item in metadata["weights"]:
            key, shape = item["key"], item["shape"]
            if "bits" in item:
                count, bits = int(np.prod(shape)), item["bits"]
                unpacked = np.unpackbits(archive[key], bitorder="little")[:count * bits]
                codes = (unpacked.reshape(count, bits) * (1 << np.arange(bits))).sum(1).reshape(shape)
                signed = 2.0 * codes - 1.0 if bits == 1 else codes - item["offset"]
                value = (signed * archive[key + "_scale"]).astype(np.float32)
            else:
                value = archive[key]
            layer_weights.setdefault(item["layer"], []).append(value)
        for layer in model.layers:
            if layer.name in layer_weights:
                layer.set_weights(layer_weights[layer.name])
    return model


def materialize_quantized_weights(model):
    """Bake reference-kernel weights into ordinary layers for INT8 conversion.

    Lite recalibrates activations and applies its own INT8 per-channel scales.
    This does not turn INT4 or binary reference kernels into native low-bit ops.
    """
    def clone_layer(layer):
        config = layer.get_config()
        for key in ("bits", "activation_bits", "activation_min", "activation_max"):
            config.pop(key, None)
        if isinstance(layer, QuantizedDepthwiseConv2D):
            return tf.keras.layers.DepthwiseConv2D.from_config(config)
        if isinstance(layer, QuantizedConv2D):
            return tf.keras.layers.Conv2D.from_config(config)
        if isinstance(layer, QuantizedDense):
            return tf.keras.layers.Dense.from_config(config)
        return layer.__class__.from_config(config)

    materialized = tf.keras.models.clone_model(model, clone_function=clone_layer)
    for layer in model.layers:
        weights = layer.get_weights()
        if isinstance(layer, (QuantizedConv2D, QuantizedDepthwiseConv2D, QuantizedDense)):
            axes = (0, 1) if isinstance(layer, QuantizedDepthwiseConv2D) else None
            weights[0] = _weight_quantize(tf.convert_to_tensor(weights[0]), layer.bits, axes).numpy()
        if weights:
            materialized.get_layer(layer.name).set_weights(weights)
    return materialized


def export_tflite(model, name, mode="int8", calibration_data=None):
    """Convert to a real executable float, dynamic-range or full-INT8 artifact."""
    if mode not in ("float32", "dynamic", "int8"):
        raise ValueError("mode must be float32, dynamic or int8")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if mode != "float32":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    if mode == "int8":
        data = get_data()[-1] if calibration_data is None else calibration_data
        if not len(data):
            raise ValueError("INT8 conversion requires representative samples.")

        def representative_dataset():
            for sample in data:
                yield [np.asarray(sample[None], dtype=np.float32)]

        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
    result = converter.convert()
    destination = Path(get_config().output_dir) / "compressed_models" / f"{safe_name(name)}.tflite"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(result)
    return str(destination)


def tflite_predict(model, samples):
    """Run Lite inference, honoring input and output quantization parameters."""
    content = bytes(model) if isinstance(model, (bytes, bytearray)) else Path(model).read_bytes()
    interpreter = make_lite_interpreter(model_content=content)
    interpreter.allocate_tensors()
    input_detail, output_detail = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
    predictions = []
    for sample in samples:
        value = np.asarray(sample[None], dtype=np.float32)
        dtype = input_detail["dtype"]
        if np.issubdtype(dtype, np.integer):
            scale, zero = input_detail["quantization"]
            if scale <= 0:
                raise ValueError("Integer Lite input has no valid quantization scale.")
            limits = np.iinfo(dtype)
            value = np.clip(np.rint(value / scale + zero), limits.min, limits.max).astype(dtype)
        else:
            value = value.astype(dtype)
        interpreter.set_tensor(input_detail["index"], value)
        interpreter.invoke()
        output = interpreter.get_tensor(output_detail["index"])
        if np.issubdtype(output_detail["dtype"], np.integer):
            scale, zero = output_detail["quantization"]
            output = (output.astype(np.float32) - zero) * scale
        predictions.append(output[0])
    return np.asarray(predictions)


def evaluate_tflite(model, split="val"):
    data = get_data()
    offset = {"train": 0, "val": 2, "test": 4}[split]
    return float(np.mean(np.argmax(tflite_predict(model, data[offset]), axis=1)
                         == np.asarray(data[offset + 1]).reshape(-1)))


def _latency(model, samples, repeats=10):
    """Median measured reference latency after warmup, synchronizing via numpy."""
    value = tf.convert_to_tensor(np.asarray(samples[:1], dtype=np.float32))
    predict = tf.function(lambda x: model(x, training=False))
    for _ in range(3):
        predict(value).numpy()
    durations = []
    for _ in range(repeats):
        start = time.perf_counter()
        predict(value).numpy()
        durations.append(1000 * (time.perf_counter() - start))
    return float(np.median(durations))


class QuantizationPipeline:
    def __init__(self, model):
        self.model = model
        self.quantization_results = {}
        self._activation_ranges = None

    def _ranges(self):
        if self._activation_ranges is None:
            self._activation_ranges = calibrate_activation_ranges(self.model)
        return self._activation_ranges

    def _gradient_sensitivity(self):
        """Empirical diagonal-Fisher proxy from validation gradients."""
        data = get_data()
        samples = min(get_config().sensitivity_samples, len(data[2]))
        layers = _layers(self.model)
        sums = {layer.name: 0.0 for layer in layers}
        count = 0
        for start in range(0, samples, 8):
            end = min(start + 8, samples)
            x = np.asarray(data[2][start:end], dtype=np.float32)
            y = np.asarray(data[3][start:end]).reshape(-1)
            with tf.GradientTape() as tape:
                tape.watch([_kernel(layer) for layer in layers])
                logits = self.model(x, training=False)
                loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y, logits, from_logits=True))
            grads = tape.gradient(loss, [_kernel(layer) for layer in layers])
            for layer, grad in zip(layers, grads):
                if grad is not None:
                    sums[layer.name] += len(x) * float(tf.reduce_sum(tf.square(grad * _kernel(layer))))
            count += len(x)
        return {name: value / max(1, count) for name, value in sums.items()}

    def _record_model(self, model, name):
        path = save_model(model, name)
        packed = packed_weights(model, name)
        quantized_kernel_count = sum(int(np.prod(_kernel(layer).shape)) for layer in _layers(model))
        return {"model": model, "validation_accuracy": accuracy(model),
                "model_path": path, "packed_weights_path": packed,
                "keras_artifact_bytes": Path(path).stat().st_size,
                "packed_storage_bytes": Path(packed).stat().st_size,
                "quantized_kernel_parameters": quantized_kernel_count,
                "remaining_fp32_parameters": model.count_params() - quantized_kernel_count,
                "remaining_fp32_state": "biases, batch normalization and non-kernel layer weights",
                "execution": "FP32 reference kernels with simulated low-bit values",
                "training_history": {}}

    def mixed_bit_quantization(self, bit_configurations=(8, 4, 2)):
        """Assign more bits to sensitive layers and measure the resulting tradeoffs."""
        choices = sorted({_validate_bits(bit) for bit in bit_configurations})
        if not choices:
            raise ValueError("At least one bit width is required.")
        sensitivity = self._gradient_sensitivity()
        ordered = sorted(sensitivity, key=sensitivity.get)
        assignment = {name: choices[min(len(choices) - 1, index * len(choices) // len(ordered))]
                      for index, name in enumerate(ordered)}
        # Input stem and classifier often need more precision irrespective of score.
        quantizable = _layers(self.model)
        assignment[quantizable[0].name] = max(choices)
        assignment[quantizable[-1].name] = max(choices)
        models = {}
        for bits in choices:
            model = quantize_model(self.model, bits, activation_ranges=self._ranges())
            models[f"uniform_{bits}"] = self._record_model(model, f"quant_uniform_{bits}")
        mixed = quantize_model(self.model, assignment, activation_ranges=self._ranges())
        models["mixed"] = self._record_model(mixed, "quant_mixed")
        best = max(models, key=lambda name: (models[name]["validation_accuracy"],
                                             -models[name]["packed_storage_bytes"]))
        results = {
            "sensitivity_analysis": sensitivity,
            "sensitivity_method": "mini-batch gradient-squared weighted by kernel magnitude; empirical Fisher proxy",
            "optimal_bit_assignment": assignment,
            "assignment_method": "sensitivity-ranked precision tiers; heuristic, not a global optimum",
            "mixed_bit_models": models,
            "compression_analysis": {
                name: {"packed_storage_ratio": model_weight_bytes(self.model) / result["packed_storage_bytes"],
                       "validation_accuracy": result["validation_accuracy"],
                       "packed_storage_bytes": result["packed_storage_bytes"],
                       "keras_artifact_bytes": result["keras_artifact_bytes"]}
                for name, result in models.items()},
            "best_validation_model": best,
        }
        self.quantization_results["mixed_bit"] = results
        return results

    def post_training_vs_qat_comparison(self, bit_widths=(8, 4)):
        """Compare calibrated reference PTQ and equally initialized STE-QAT."""
        comparison = {"ptq_results": {}, "qat_results": {}, "accuracy_comparison": {},
                      "training_efficiency": {}, "calibration_analysis": {}}
        for bits in bit_widths:
            bits = _validate_bits(bits)
            start = time.perf_counter()
            ptq = quantize_model(self.model, bits, activation_ranges=self._ranges())
            ptq_result = self._record_model(ptq, f"ptq_{bits}")
            ptq_seconds = time.perf_counter() - start
            set_seed(get_config().seed)
            qat = quantize_model(self.model, bits, activation_ranges=self._ranges())
            trained = fit_model(qat, learning_rate=1e-5, name=f"qat_{bits}")
            qat_result = self._record_model(qat, f"qat_{bits}")
            qat_result["training_history"] = trained["history"]
            qat_result["training_time_seconds"] = trained["training_time_seconds"]
            if bits == 8:
                # The portable integer model is a separate, measured deployment artifact.
                lite_path = export_tflite(self.model, "ptq_int8", mode="int8")
                ptq_result["tflite_path"] = lite_path
                ptq_result["tflite_validation_accuracy"] = evaluate_tflite(lite_path)
                ptq_result["tflite_bytes"] = Path(lite_path).stat().st_size
                qat_path = export_tflite(materialize_quantized_weights(qat), "qat_int8", mode="int8")
                qat_result["tflite_path"] = qat_path
                qat_result["tflite_validation_accuracy"] = evaluate_tflite(qat_path)
                qat_result["tflite_bytes"] = Path(qat_path).stat().st_size
                qat_result["tflite_export_method"] = (
                    "Bake QAT kernel values, then recalibrate activation scales for "
                    "Lite's native INT8 kernels; deployment accuracy measured separately.")
            comparison["ptq_results"][str(bits)] = ptq_result
            comparison["qat_results"][str(bits)] = qat_result
            comparison["accuracy_comparison"][str(bits)] = {
                "ptq": ptq_result["validation_accuracy"], "qat": qat_result["validation_accuracy"],
                "qat_gain_percentage_points": 100 * (qat_result["validation_accuracy"]
                                                        - ptq_result["validation_accuracy"])}
            comparison["training_efficiency"][str(bits)] = {
                "ptq_setup_and_evaluation_seconds": ptq_seconds,
                "qat_training_seconds": trained["training_time_seconds"]}
        calibration = get_data()[-1]
        sizes = sorted({min(len(calibration), n) for n in (32, 128, 1000)})
        for size in sizes:
            subset_model = quantize_model(self.model, 8, calibration_data=calibration[:size])
            comparison["calibration_analysis"][str(size)] = {
                "calibration_samples": size, "validation_accuracy": accuracy(subset_model)}
            del subset_model
        comparison["calibration_requirements"] = (
            "PTQ and this QAT initialization use training-only, unlabeled calibration data; "
            "QAT additionally uses labeled training data. Selection uses validation data.")
        self.quantization_results["ptq_vs_qat"] = comparison
        return comparison

    def extreme_quantization(self):
        """Run INT4 and XNOR-style binary reference networks, including timings."""
        baseline_accuracy = accuracy(self.model)
        samples = get_data()[2]
        baseline_ms = _latency(self.model, samples)
        int4 = quantize_model(self.model, 4, activation_ranges=self._ranges())
        assignment = {layer.name: 1 for layer in _layers(self.model)}
        assignment[_layers(self.model)[0].name] = 8
        assignment[_layers(self.model)[-1].name] = 8
        binary = quantize_model(self.model, assignment, activation_ranges=self._ranges())
        results = {"int4_quantization": {}, "binary_quantization": {},
                   "accuracy_degradation_analysis": {}, "performance_improvements": {}}
        for label, model in (("int4", int4), ("binary", binary)):
            callbacks = [_ClipBinaryLatentWeights()] if label == "binary" else None
            trained = fit_model(
                model, learning_rate=1e-5, name=f"extreme_{label}",
                callbacks=callbacks, clipnorm=1.0 if label == "binary" else None)
            result = self._record_model(model, f"extreme_{label}")
            result["training_history"] = trained["history"]
            result["training_time_seconds"] = trained["training_time_seconds"]
            latency = _latency(model, samples)
            result["reference_latency_ms"] = latency
            results[f"{label}_quantization"] = result
            results["accuracy_degradation_analysis"][label] = {
                "accuracy_loss_percentage_points": 100 * (baseline_accuracy - result["validation_accuracy"]),
                "layer_output_error": self._error_analysis(model)}
            results["performance_improvements"][label] = {
                "baseline_latency_ms": baseline_ms, "reference_latency_ms": latency,
                "measured_speedup": baseline_ms / max(latency, 1e-12),
                "native_low_bit_acceleration": False}
        results["binary_quantization"]["method"] = (
            "XNOR-style sign convolutions with per-filter alpha and per-patch beta; "
            "8-bit stem/classifier; floating-point reference execution, not bitwise XNOR/popcount")
        self.quantization_results["extreme_quantization"] = results
        return results

    def dynamic_quantization_analysis(self):
        """Compare actual dynamic-range and calibrated INT8 Lite artifacts."""
        samples = get_data()[2]
        results = {}
        for mode in ("float32", "dynamic", "int8"):
            start = time.perf_counter()
            path = export_tflite(self.model, f"quant_{mode}", mode=mode)
            conversion_seconds = time.perf_counter() - start
            # Warm the interpreter/converter code paths before a measured full pass.
            tflite_predict(path, samples[:1])
            start = time.perf_counter()
            predictions = tflite_predict(path, samples)
            elapsed = time.perf_counter() - start
            results[mode] = {
                "tflite_path": path, "artifact_bytes": Path(path).stat().st_size,
                "validation_accuracy": float(np.mean(predictions.argmax(1) == np.asarray(get_data()[3]).reshape(-1))),
                "conversion_seconds": conversion_seconds,
                "end_to_end_ms_per_sample": 1000 * elapsed / len(samples),
                "calibration_samples": len(get_data()[-1]) if mode == "int8" else 0,
                "timing_includes_interpreter_setup": True}
        results["tradeoffs"] = (
            "Dynamic-range quantization calibrates no representative activation ranges; "
            "supported kernels quantize activations at inference. Full INT8 fixes activation "
            "ranges using calibration. Inspect measured accuracy and target-device latency.")
        self.quantization_results["dynamic_vs_static"] = results
        return results

    def _error_analysis(self, quantized):
        """Measure cumulative activation error relative to corresponding FP32 layers."""
        names = [layer.name for layer in _layers(self.model)]
        original_probe = tf.keras.Model(self.model.inputs, [self.model.get_layer(n).output for n in names])
        quantized_probe = tf.keras.Model(quantized.inputs, [quantized.get_layer(n).output for n in names])
        samples = get_data()[2][:get_config().sensitivity_samples]
        sums = {name: {"squared_error": 0.0, "signal_power": 0.0, "elements": 0} for name in names}
        for start in range(0, len(samples), 8):
            inputs = np.asarray(samples[start:start + 8], dtype=np.float32)
            originals, outputs = original_probe(inputs, training=False), quantized_probe(inputs, training=False)
            if not isinstance(originals, (list, tuple)):
                originals, outputs = [originals], [outputs]
            for name, original, output in zip(names, originals, outputs):
                sums[name]["squared_error"] += float(tf.reduce_sum(tf.square(original - output)))
                sums[name]["signal_power"] += float(tf.reduce_sum(tf.square(original)))
                sums[name]["elements"] += int(tf.size(original))
        return {name: {"mse": value["squared_error"] / max(1, value["elements"]),
                       "relative_mse": value["squared_error"] / max(1e-12, value["signal_power"]),
                       "signal_to_noise_db": float(10 * np.log10(max(value["signal_power"], 1e-12)
                                                                 / max(value["squared_error"], 1e-12)))}
                for name, value in sums.items()}

    def quantization_error_analysis(self):
        """Compare error accumulation and identify the most affected layers."""
        results = {}
        for bits in (8, 4, 2):
            quantized = quantize_model(self.model, bits, activation_ranges=self._ranges())
            errors = self._error_analysis(quantized)
            results[str(bits)] = {
                "layer_errors": errors,
                "bottleneck_layers": sorted(errors, key=lambda n: errors[n]["relative_mse"], reverse=True)[:5],
                "interpretation": "Cumulative output error; includes all upstream quantization."}
            del quantized
        self.quantization_results["quantization_error"] = results
        return results
