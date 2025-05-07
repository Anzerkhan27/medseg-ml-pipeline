# orchestration/prefect_flow.py  – Prefect 2.x version
import sys, os
from pathlib import Path

# ── make scripts/ importable ───────────────────────────────────────
scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
sys.path.append(str(scripts_dir))

from generate_manifest import build_manifest
from standardize_data import run_standardization
from train_classifier import main as train_classifier
from infer_batch import run_batch_inference

from prefect import task, flow, get_run_logger

# ─────────────────── TASKS ─────────────────────────────────────────
@task(log_prints=True)
def check_and_process_dataset(dataset_path: str) -> str:
    log = get_run_logger()
    if os.path.exists(dataset_path):
        log.info(f"🗂️  Dataset already exists at {dataset_path}")
    else:
        log.info("📥 Dataset not found – downloading …")
        import kagglehub
        dataset_path = kagglehub.dataset_download(
            "mateuszbuda/lgg-mri-segmentation"
        )
        log.info(f"✅ Downloaded to {dataset_path}")
    return dataset_path


@task(log_prints=True)
def create_manifest(dataset_root: str, manifest_path: str) -> str:
    log = get_run_logger()
    Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)

    df = build_manifest(dataset_root)
    log.info(f"📝 Manifest rows: {len(df)} (labels: {df['label'].notna().sum()})")
    if df.empty:
        raise ValueError("Manifest is empty – aborting.")

    df.to_csv(manifest_path, index=False)
    return manifest_path


@task(log_prints=True)
def standardize_data(manifest_path: str, output_dir: str) -> str:
    log = get_run_logger()
    run_standardization(manifest_path, output_dir)
    n = len(list(Path(output_dir).glob("*.npz")))
    log.info(f"📦 .npz saved: {n}")
    if n == 0:
        raise ValueError("No .npz written – aborting.")
    return output_dir


@task(log_prints=True)
def train_model() -> str:
    train_classifier()
    return "outputs/models/classifier_model.keras"


@task(log_prints=True)
def generate_predictions(
    model_path: str, data_dir: str, output_dir: str
) -> str:
    log = get_run_logger()
    if not list(Path(data_dir).glob("*.npz")):
        raise ValueError("No .npz files for inference.")

    name = Path(data_dir).resolve().name
    run_batch_inference(model_path, data_dir, name, output_dir)

    latest = max(Path(output_dir).glob(f"{name}_*.csv"), key=os.path.getctime)
    log.info(f"📰 Predictions CSV → {latest}")
    return str(latest)


@task(log_prints=True)
def evaluate_model(predictions_file: str, model_path: str):
    from metrics_report import generate_metrics_report
    log = get_run_logger()
    log.info("🔍 Evaluating …")
    generate_metrics_report(predictions_file, model_path)
    log.info("🏁 Evaluation done.")


# ─────────────────── FLOW (Prefect 2) ──────────────────────────────
@flow(name="🧠 MedSeg ML Pipeline")
def medseg_flow(
    dataset_path: str = "C:/Users/lucif/.cache/kagglehub/datasets/mateuszbuda/",
    manifest_path: str = "data/manifest.csv",
    standardized_dir: str = "data/processed",
    predictions_dir: str = "outputs/predictions",
):
    ds_root   = check_and_process_dataset(dataset_path)
    manifest  = create_manifest(ds_root, manifest_path)
    std_dir   = standardize_data(manifest, standardized_dir)
    model     = train_model()
    csv_file  = generate_predictions(model, std_dir, predictions_dir)
    evaluate_model(csv_file, model)


# ─────────────────── local run ─────────────────────────────────────
if __name__ == "__main__":
    # override defaults by passing arguments:
    #   medseg_flow(dataset_path="...", standardized_dir="...")
    medseg_flow()
