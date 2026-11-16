"""Small, reproducible width/depth search complementing the supplied skeleton."""
import time
import numpy as np
from baseline_model import create_student_model
from common import get_config, set_seed, model_weight_bytes, save_json
from part3_distillation import distill_student


def search_student_architecture(teacher, widths=(0.5, 1.0, 1.5), depths=(1, 2), max_size_mb=10):
    """Equal-epoch KD trials, selected by validation accuracy under a weight budget.

    This is a budgeted grid NAS baseline, not a claim of globally optimal design.
    The memory budget here is parameter bytes; final artifacts are benchmarked later.
    """
    if not widths or not depths or max_size_mb <= 0:
        raise ValueError("Search space and size budget must be nonempty and positive")
    cfg = get_config()
    trials, best = {}, None
    for width in widths:
        for depth in depths:
            set_seed(cfg.seed)
            name = f"nas_w{width:g}_d{depth}"
            student = create_student_model(width=width, depth=depth,
                num_classes=int(teacher.output_shape[-1]), input_shape=teacher.input_shape[1:])
            if model_weight_bytes(student) / 2**20 > max_size_mb:
                trials[name] = {"status": "over_parameter_budget"}
                continue
            result = distill_student(teacher, student, name=name)
            trials[name] = {"model": student, "width": width, "depth": depth, "parameters": student.count_params(),
                "weight_size_mb": model_weight_bytes(student) / 2**20,
                "final_accuracy": result["final_accuracy"],
                "training_time_seconds": result["training_time_seconds"],
                "training_history": result["training_history"],
                "artifact_path": result.get("artifact_path"), "evaluation_split": "validation"}
            if best is None or result["final_accuracy"] > best["final_accuracy"]:
                best = {**result, "width": width, "depth": depth, "trial": name}
    if best is None:
        raise ValueError("No candidate fits the requested parameter budget")
    output = {"trials": trials, "best": best,
              "method": "equal-epoch grid search, validation selection, fixed teacher"}
    save_json(output, cfg.output_dir / "results" / "architecture_search.json")
    return output
