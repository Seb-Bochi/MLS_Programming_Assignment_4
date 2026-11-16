import numpy as np
from common import configure, get_data, tf
from baseline_model import prepare_compression_datasets, create_baseline_model


def test_stratified_split_calibration_has_no_leakage(tmp_path, monkeypatch):
    cfg = configure(output_dir=tmp_path, calibration_samples=100, imagenet_weights=False)
    labels = np.repeat(np.arange(100), 20).astype(np.int32)
    images = np.zeros((2000, 32, 32, 3), dtype=np.uint8)
    # Encode record identity in two bytes to verify membership after splitting.
    images[:, 0, 0, 0] = np.arange(2000) % 256
    images[:, 0, 0, 1] = np.arange(2000) // 256
    test_x = np.zeros((100, 32, 32, 3), dtype=np.uint8)
    monkeypatch.setattr(tf.keras.datasets.cifar100, "load_data", lambda **kwargs:
                        ((images, labels[:, None]), (test_x, np.arange(100)[:, None])))
    train_x, train_y, val_x, val_y, test_x, test_y, calibration = prepare_compression_datasets()
    identity = lambda values: set((values[:, 0, 0, 0].astype(int) +
                                  256 * values[:, 0, 0, 1].astype(int)).tolist())
    assert not identity(train_x) & identity(val_x)
    assert identity(calibration) <= identity(train_x)
    assert len(calibration) == 100
    assert len(np.unique(train_y)) == len(np.unique(val_y)) == 100
    assert train_x.dtype == np.uint8
    saved = np.load(tmp_path / "checkpoints" / "data_split.npz")
    assert not set(saved["train_indices"]) & set(saved["validation_indices"])


def test_resnet50_builds_flat_prunable_graph(tmp_path):
    configure(output_dir=tmp_path, imagenet_weights=False, image_size=32)
    model = create_baseline_model()
    assert model.get_layer("conv2_block1_1_conv").filters == 64
    assert model.output_shape == (None, 100)
    logits = model(np.zeros((1, 32, 32, 3), np.float32), training=False)
    assert logits.shape == (1, 100)
    assert np.all(np.isfinite(logits.numpy()))
