"""Submit a LoRA fine-tuning job to Nebius Token Factory (OpenAI-compatible).

Uploads the training file, creates a LoRA job on a base model, and polls until
it reaches a terminal state, printing the fine-tuned model id to use for
inference. Requires NEBIUS_API_KEY. Use --dry-run to validate the dataset and
print the job request WITHOUT spending credit.

    # Validate only (no spend):
    python -m finetune.submit --dry-run

    # Launch the real job (spends credit):
    NEBIUS_API_KEY=... python -m finetune.submit \
        --base-model meta-llama/Llama-3.1-8B-Instruct --epochs 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from finetune.schema import EXTRACTION_FIELDS  # noqa: E402

NEBIUS_BASE_URL = "https://api.tokenfactory.nebius.com/v1/"
DEFAULT_BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
TERMINAL_STATES = {"succeeded", "failed", "cancelled"}


def validate_dataset(path: Path) -> int:
    """Sanity-check the JSONL before spending: chat format + required keys."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run `python -m finetune.generate_dataset` first")
    count = 0
    with path.open() as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            messages = row.get("messages")
            if not messages or messages[0]["role"] != "system" or messages[-1]["role"] != "assistant":
                raise ValueError(f"{path}:{lineno} not in chat fine-tune format")
            target = json.loads(messages[-1]["content"])
            missing = [k for k in EXTRACTION_FIELDS if k not in target]
            if missing:
                raise ValueError(f"{path}:{lineno} target missing keys: {missing}")
            count += 1
    if count == 0:
        raise ValueError(f"{path} is empty")
    return count


def build_job_request(base_model: str, training_file_id: str, args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "model": base_model,
        "training_file": training_file_id,
        "hyperparameters": {
            "n_epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
        },
        "lora": True,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit a LoRA fine-tune to Nebius Token Factory.")
    parser.add_argument("--train-file", default=str(Path(__file__).resolve().parent / "data" / "train.jsonl"))
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--dry-run", action="store_true", help="Validate + print the request; do not submit.")
    args = parser.parse_args()

    train_path = Path(args.train_file)
    n = validate_dataset(train_path)
    print(f"Validated {n} training examples in {train_path}")

    if args.dry_run:
        request = build_job_request(args.base_model, "<uploaded-file-id>", args)
        print("DRY RUN — job request that would be submitted:")
        print(json.dumps(request, indent=2))
        print("\nNo credit spent. Re-run without --dry-run (and NEBIUS_API_KEY set) to launch.")
        return

    api_key = os.getenv("NEBIUS_API_KEY")
    if not api_key:
        print("NEBIUS_API_KEY is not set. Export it or use --dry-run.")
        sys.exit(1)

    from openai import OpenAI

    client = OpenAI(base_url=NEBIUS_BASE_URL, api_key=api_key)

    print("Uploading training file...")
    uploaded = client.files.create(file=train_path.open("rb"), purpose="fine-tune")
    print(f"  file id: {uploaded.id}")

    request = build_job_request(args.base_model, uploaded.id, args)
    print(f"Creating LoRA job on {args.base_model}...")
    job = client.fine_tuning.jobs.create(**request)
    print(f"  job id: {job.id}  status: {job.status}")

    while job.status not in TERMINAL_STATES:
        time.sleep(args.poll_seconds)
        job = client.fine_tuning.jobs.retrieve(job.id)
        print(f"  status: {job.status}")

    if job.status != "succeeded":
        print(f"Job ended in state '{job.status}'.")
        sys.exit(1)

    fine_tuned = getattr(job, "fine_tuned_model", None)
    if not fine_tuned:
        checkpoints = client.fine_tuning.jobs.checkpoints.list(job.id).data
        fine_tuned = checkpoints[0].fine_tuned_model if checkpoints else None
    print("\nFine-tune succeeded.")
    print(f"Fine-tuned model id: {fine_tuned}")
    print("Evaluate it with:")
    print(f"  NEBIUS_API_KEY=... python -m finetune.evaluate --tuned-model '{fine_tuned}'")


if __name__ == "__main__":
    main()
