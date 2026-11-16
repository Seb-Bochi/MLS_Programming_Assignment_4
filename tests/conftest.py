"""Keep the numerical test suite small and independent of CIFAR downloads."""
import os
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
import gc
import pytest


@pytest.fixture(autouse=True)
def release_graphs():
    yield
    from common import tf
    tf.keras.backend.clear_session()
    gc.collect()
