import argparse
import json
import os
import time
from typing import Any, Dict, Iterable

from llm_extract_openai import get_client


FINAL_JOB_STATUSES = {"succeeded", "failed", "cancelled"}


def serialize_api_object(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, (list, tuple)):
        return [serialize_api_object(item) for item in obj]
    return obj


def write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def upload_file(client: Any, path: str, purpose: str) -> Any:
    with open(path, "rb") as f:
        return client.files.create(file=f, purpose=purpose)


def wait_for_job(client: Any, job_id: str, poll_interval: float) -> Any:
    last_status = ""
    while True:
        job = client.fine_tuning.jobs.retrieve(job_id)
        status = str(getattr(job, "status", ""))
        if status != last_status:
            print(json.dumps({"job_id": job_id, "status": status}, ensure_ascii=False))
            last_status = status
        if status in FINAL_JOB_STATUSES:
            return job
        time.sleep(max(1.0, poll_interval))


def collect_job_events(client: Any, job_id: str, limit: int = 100) -> Iterable[Dict[str, Any]]:
    events = client.fine_tuning.jobs.list_events(job_id, limit=limit)
    return [serialize_api_object(event) for event in events]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit and monitor OpenAI fine-tuning job for event extraction SFT.")
    parser.add_argument("--training-file", required=True, help="OpenAI fine-tuning training JSONL path")
    parser.add_argument("--val-file", default=None, help="Optional validation JSONL path")
    parser.add_argument("--model", default="gpt-4.1-mini-2025-04-14", help="Base model name")
    parser.add_argument("--suffix", default="news-event-v1", help="Fine-tuned model suffix")
    parser.add_argument("--n-epochs", type=int, default=3, help="Number of supervised fine-tuning epochs")
    parser.add_argument("--poll-interval", type=float, default=30.0, help="Polling interval in seconds")
    parser.add_argument("--output", default="output/sft/model_info.json", help="Model info output JSON path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = get_client()

    training_file = upload_file(client, args.training_file, purpose="fine-tune")
    validation_file = upload_file(client, args.val_file, purpose="fine-tune") if args.val_file else None

    create_kwargs: Dict[str, Any] = {
        "model": args.model,
        "training_file": training_file.id,
        "hyperparameters": {"n_epochs": args.n_epochs},
        "method": {"type": "supervised"},
        "suffix": args.suffix,
    }
    if validation_file is not None:
        create_kwargs["validation_file"] = validation_file.id

    job = client.fine_tuning.jobs.create(**create_kwargs)
    final_job = wait_for_job(client, job.id, poll_interval=args.poll_interval)
    payload = {
        "training_file": serialize_api_object(training_file),
        "validation_file": serialize_api_object(validation_file) if validation_file is not None else None,
        "job": serialize_api_object(final_job),
        "events": list(collect_job_events(client, final_job.id)),
    }
    write_json(args.output, payload)
    print(
        json.dumps(
            {
                "output": args.output,
                "job_id": getattr(final_job, "id", ""),
                "status": getattr(final_job, "status", ""),
                "fine_tuned_model": getattr(final_job, "fine_tuned_model", None),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
