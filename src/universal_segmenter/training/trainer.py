"""Production trainer with independent best and exact-resume checkpoints."""

from __future__ import annotations

import json
import math
import os
import random
import signal
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

from universal_segmenter.losses import create_paper_loss
from universal_segmenter.metrics import binary_metrics, boundary_metrics
from universal_segmenter.models import create_model

from .checkpoint import (
    TrainingCursor,
    atomic_torch_save,
    build_checkpoint,
    capture_rng_state,
    restore_rng_state,
    validate_initial_checkpoint,
    validate_resume_checkpoint,
)
from .config import TrainingConfig
from .dataset import (
    DrawSampler,
    PaperDataset,
    build_epoch_plan,
    deterministic_seed,
    load_rows,
)
from .freeze import verify_dataset_freeze


class _StopController:
    def __init__(self, stop_file: Path) -> None:
        self.stop_file = stop_file
        self.requested = False
        self.reason = ""

    def request(self, reason: str) -> None:
        if not self.requested:
            print(
                f"\n[train] Stop requested ({reason}). Finishing the current "
                "optimizer update and saving last.pt...",
                flush=True,
            )
        self.requested = True
        self.reason = reason

    def poll_file(self) -> None:
        if self.stop_file.is_file():
            self.request("STOP_REQUESTED file")


@contextmanager
def _graceful_stop(controller: _StopController):
    previous: dict[int, Any] = {}

    def handler(signum, _frame):
        controller.request(signal.Signals(signum).name)

    for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        value = getattr(signal, name, None)
        if value is None:
            continue
        try:
            previous[value] = signal.getsignal(value)
            signal.signal(value, handler)
        except (OSError, ValueError):
            pass
    try:
        yield
    finally:
        for value, old_handler in previous.items():
            signal.signal(value, old_handler)


def _seed_everything(torch, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def _optimizer(torch, model, config: TrainingConfig):
    encoder_parameters = list(model.encoder.parameters())
    encoder_ids = {id(parameter) for parameter in encoder_parameters}
    decoder_parameters = [
        parameter for parameter in model.parameters()
        if id(parameter) not in encoder_ids
    ]
    return torch.optim.AdamW([
        {"params": encoder_parameters, "lr": config.encoder_learning_rate,
         "name": "encoder"},
        {"params": decoder_parameters, "lr": config.decoder_learning_rate,
         "name": "decoder"},
    ], weight_decay=config.weight_decay)


def _scheduler(torch, optimizer, total_steps: int,
               config: TrainingConfig):
    warmup_steps = max(1, round(total_steps * config.warmup_fraction))

    def factor(step: int) -> float:
        if step < warmup_steps:
            return max(config.minimum_learning_rate_factor,
                       (step + 1) / warmup_steps)
        progress = min(1.0, (step - warmup_steps) /
                       max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        minimum = config.minimum_learning_rate_factor
        return minimum + (1.0 - minimum) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def _epoch_sample_count(train_count: int, config: TrainingConfig) -> int:
    requested = config.epoch_samples or train_count
    unit = config.effective_batch_size
    return math.ceil(requested / unit) * unit


def _loader_generator(torch, *seed_parts: object):
    """Keep worker seeding from consuming the model's resumable RNG stream."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(deterministic_seed(*seed_parts) % (2**63 - 1))
    return generator


def _empty_accumulator() -> dict[str, float | int]:
    return {
        "batch_count": 0,
        "sample_count": 0,
        "loss": 0.0,
        "bce": 0.0,
        "dice": 0.0,
        "distance": 0.0,
        "boundary_bce": 0.0,
    }


def _accumulate(accumulator: dict[str, float | int], losses: dict[str, Any],
                batch_size: int) -> None:
    accumulator["batch_count"] = int(accumulator["batch_count"]) + 1
    accumulator["sample_count"] = int(accumulator["sample_count"]) + batch_size
    for name in ("loss", "bce", "dice", "distance", "boundary_bce"):
        accumulator[name] = float(accumulator[name]) + float(losses[name].item())


def _averages(accumulator: dict[str, float | int]) -> dict[str, float]:
    count = max(1, int(accumulator["batch_count"]))
    return {
        name: float(accumulator[name]) / count
        for name in ("loss", "bce", "dice", "distance", "boundary_bce")
    }


def _validation(torch, model, loss_function, loader, device,
                threshold: float) -> dict[str, Any]:
    model.eval()
    losses: list[float] = []
    rows: list[dict[str, Any]] = []
    profile_boundaries: dict[str, list[float]] = defaultdict(list)
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["mask"].to(device, non_blocking=True)
            distances = batch["distance"].to(device, non_blocking=True)
            outputs = model(images)
            current_loss = loss_function(outputs, targets, distances)["loss"]
            losses.append(float(current_loss.item()))
            probabilities = torch.sigmoid(outputs["mask_logits"])
            predictions = probabilities >= threshold
            for item in range(images.shape[0]):
                prediction = predictions[item, 0].cpu().numpy()
                target = targets[item, 0].cpu().numpy() >= 0.5
                region = binary_metrics(prediction, target)
                edge = boundary_metrics(prediction, target, 2.0)
                profile = str(batch["capture_profile"][item])
                profile_boundaries[profile].append(edge["boundary_f1"])
                rows.append({
                    "sample_id": str(batch["sample_id"][item]),
                    "capture_profile": profile,
                    **region,
                    **{f"{name}_at_2px": value for name, value in edge.items()},
                    "catastrophic": bool(
                        region["iou"] < 0.97 or edge["boundary_f1"] < 0.90),
                })
    if not rows:
        raise ValueError("validation split is empty")
    mean_boundary = float(np.mean([row["boundary_f1_at_2px"] for row in rows]))
    worst_profile = min(
        float(np.mean(values)) for values in profile_boundaries.values())
    median_iou = float(median(row["iou"] for row in rows))
    catastrophic_rate = float(np.mean([row["catastrophic"] for row in rows]))
    score = (
        0.50 * mean_boundary +
        0.25 * worst_profile +
        0.25 * median_iou -
        catastrophic_rate
    )
    return {
        "loss": float(np.mean(losses)),
        "selection_score": score,
        "median_iou": median_iou,
        "mean_boundary_f1_at_2px": mean_boundary,
        "worst_profile_boundary_f1_at_2px": worst_profile,
        "catastrophic_rate": catastrophic_rate,
        "profiles": {
            profile: {
                "count": len(values),
                "mean_boundary_f1_at_2px": float(np.mean(values)),
            }
            for profile, values in sorted(profile_boundaries.items())
        },
        "per_image": rows,
    }


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _save_checkpoint(torch, path: Path, kind: str, config: TrainingConfig,
                     fingerprint: str, cursor: TrainingCursor, model, optimizer,
                     scheduler, history: list[dict[str, Any]],
                     accumulator: dict[str, float | int], rng_state) -> None:
    value = build_checkpoint(
        kind=kind,
        config=config.serialized(),
        config_hash=config.immutable_hash(),
        dataset_fingerprint=fingerprint,
        cursor=cursor,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        history=history,
        epoch_accumulator=accumulator,
        rng_state=rng_state,
    )
    atomic_torch_save(torch, value, path)


def run_training(project_root: Path, config: TrainingConfig) -> dict[str, Any]:
    """Train or resume one candidate. Resume always uses last.pt, not best.pt."""
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError as error:
        raise RuntimeError("training requires a CUDA-enabled PyTorch installation") from error

    config.validate()
    project = Path(project_root).resolve()
    dataset_root = (project / config.dataset_root).resolve()
    freeze = verify_dataset_freeze(dataset_root)
    fingerprint = str(freeze["fingerprint"])
    rows = load_rows(dataset_root)
    train_count = sum(row["split"] == "train" for row in rows)
    validation_rows = [row for row in rows if row["split"] == "validation"]
    if not train_count or not validation_rows:
        raise ValueError("training and validation splits must both be nonempty")
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot see the GPU")
    device = torch.device(config.device)
    run_root = (project / config.experiments_root / config.run_name).resolve()
    checkpoint_root = run_root / "checkpoints"
    last_path = checkpoint_root / "last.pt"
    best_path = checkpoint_root / "best.pt"
    stop_file = run_root / "STOP_REQUESTED"
    if stop_file.exists():
        stop_file.unlink()

    _seed_everything(torch, config.random_seed)
    resuming_existing = last_path.is_file() and config.resume
    model = create_model(
        config.model_type,
        pretrained=(
            config.pretrained_encoder and not resuming_existing
            and config.initial_checkpoint is None),
    ).to(device)
    if not resuming_existing and config.initial_checkpoint is not None:
        initial_path = Path(config.initial_checkpoint)
        if not initial_path.is_absolute():
            initial_path = project / initial_path
        if not initial_path.is_file():
            raise FileNotFoundError(
                f"initial fine-tuning checkpoint is missing: {initial_path}")
        initial = torch.load(initial_path, map_location=device, weights_only=False)
        validate_initial_checkpoint(
            initial, config.model_type, config.resolution)
        model.load_state_dict(initial["model_state"], strict=True)
        print(
            f"[train] Initialized new run from {initial_path} "
            f"({initial.get('checkpoint_kind')} checkpoint).", flush=True)
    optimizer = _optimizer(torch, model, config)
    samples_per_epoch = _epoch_sample_count(train_count, config)
    batches_per_epoch = samples_per_epoch // config.physical_batch_size
    if batches_per_epoch % config.gradient_accumulation_steps:
        raise AssertionError("epoch plan does not align with gradient accumulation")
    optimizer_steps_per_epoch = (
        batches_per_epoch // config.gradient_accumulation_steps)
    scheduler = _scheduler(
        torch, optimizer, optimizer_steps_per_epoch * config.epochs, config)
    loss_function = create_paper_loss()
    cursor = TrainingCursor()
    history: list[dict[str, Any]] = []
    accumulator = _empty_accumulator()
    rng_state = capture_rng_state(torch)

    if last_path.is_file() and config.resume:
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        validate_resume_checkpoint(
            checkpoint, config.immutable_hash(), fingerprint)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        cursor = TrainingCursor.from_dict(checkpoint["cursor"])
        history = list(checkpoint.get("history", []))
        accumulator = dict(checkpoint.get(
            "epoch_accumulator", _empty_accumulator()))
        rng_state = checkpoint["rng_state"]
        restore_rng_state(torch, rng_state)
        print(
            f"[train] Resumed {config.run_name}: epoch {cursor.epoch + 1}/"
            f"{config.epochs}, next batch {cursor.next_batch + 1}/"
            f"{batches_per_epoch}, optimizer step "
            f"{cursor.global_optimizer_step}.", flush=True)
    elif last_path.is_file():
        raise FileExistsError(
            f"{last_path} already exists and resume is disabled. Use a new RUN_NAME.")

    if cursor.complete:
        print(
            f"[train] Run already complete. Best epoch: {cursor.best_epoch + 1}; "
            f"best checkpoint: {best_path}", flush=True)
        return {
            "status": "already_complete",
            "run_root": str(run_root),
            "best_checkpoint": str(best_path),
            "last_checkpoint": str(last_path),
            "best_epoch": cursor.best_epoch,
            "best_score": cursor.best_score,
        }

    train_dataset = PaperDataset(dataset_root, rows, config.resolution, training=True)
    validation_dataset = PaperDataset(
        dataset_root, validation_rows, config.resolution, training=False)
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.validation_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=config.num_workers > 0,
        generator=_loader_generator(
            torch, config.random_seed, "validation_loader"),
    )
    controller = _StopController(stop_file)
    run_root.mkdir(parents=True, exist_ok=True)
    _atomic_json(run_root / "training_config.json", config.serialized())
    print(
        f"[train] {config.model_type} | {config.resolution}px | physical batch "
        f"{config.physical_batch_size} | accumulation "
        f"{config.gradient_accumulation_steps} | effective batch "
        f"{config.effective_batch_size} | FP32", flush=True)
    print(
        f"[train] {samples_per_epoch:,} sampled images/epoch | "
        f"{optimizer_steps_per_epoch:,} optimizer updates/epoch | "
        f"{config.epochs} epochs", flush=True)
    print(
        "[train] To stop safely, press Ctrl+C once. last.pt will resume; "
        "best.pt remains the best completed validation epoch.", flush=True)

    started = time.perf_counter()
    with _graceful_stop(controller):
        for epoch in range(cursor.epoch, config.epochs):
            encoder_trainable = epoch >= config.freeze_encoder_epochs
            model.set_encoder_trainable(encoder_trainable)
            model.train()
            if not encoder_trainable:
                # Freezing includes BatchNorm/dropout state, not only gradients.
                model.encoder.eval()
            plan = build_epoch_plan(
                rows, epoch, config.random_seed, samples_per_epoch,
                config.category_weights)
            start_batch = cursor.next_batch if epoch == cursor.epoch else 0
            if start_batch > batches_per_epoch:
                raise ValueError("checkpoint batch cursor exceeds epoch length")
            if start_batch % config.gradient_accumulation_steps:
                raise ValueError("checkpoint is not at an optimizer-update boundary")
            if epoch != cursor.epoch:
                cursor.epoch = epoch
                cursor.next_batch = 0
                accumulator = _empty_accumulator()
            start_draw = start_batch * config.physical_batch_size
            sampler = DrawSampler(plan[start_draw:])
            train_loader = DataLoader(
                train_dataset,
                batch_size=config.physical_batch_size,
                sampler=sampler,
                num_workers=config.num_workers,
                pin_memory=device.type == "cuda",
                persistent_workers=config.num_workers > 0,
                generator=_loader_generator(
                    torch, config.random_seed, epoch, "train_loader"),
            )
            optimizer.zero_grad(set_to_none=True)
            epoch_started = time.perf_counter()
            for local_batch, batch in enumerate(train_loader):
                batch_index = start_batch + local_batch
                images = batch["image"].to(device, non_blocking=True)
                targets = batch["mask"].to(device, non_blocking=True)
                distances = batch["distance"].to(device, non_blocking=True)
                outputs = model(images)
                losses = loss_function(outputs, targets, distances)
                (losses["loss"] /
                 config.gradient_accumulation_steps).backward()
                _accumulate(accumulator, losses, int(images.shape[0]))
                accumulation_boundary = (
                    (batch_index + 1) % config.gradient_accumulation_steps == 0)
                if accumulation_boundary:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.gradient_clip_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    cursor.next_batch = batch_index + 1
                    cursor.global_optimizer_step += 1
                    rng_state = capture_rng_state(torch)
                    controller.poll_file()
                    periodic = (
                        cursor.global_optimizer_step %
                        config.checkpoint_every_optimizer_steps == 0)
                    if periodic or controller.requested:
                        _save_checkpoint(
                            torch, last_path, "last", config, fingerprint,
                            cursor, model, optimizer, scheduler, history,
                            accumulator, rng_state)
                    if controller.requested:
                        if stop_file.exists():
                            stop_file.unlink()
                        print(
                            f"[train] Safely stopped. Resume checkpoint: "
                            f"{last_path}", flush=True)
                        return {
                            "status": "interrupted",
                            "reason": controller.reason,
                            "run_root": str(run_root),
                            "best_checkpoint": str(best_path),
                            "last_checkpoint": str(last_path),
                            "cursor": cursor.serialized(),
                        }
                if ((batch_index + 1) % config.progress_every_batches == 0
                        or batch_index + 1 == batches_per_epoch):
                    elapsed = time.perf_counter() - epoch_started
                    completed = batch_index + 1 - start_batch
                    seconds_per_batch = elapsed / max(1, completed)
                    remaining = (batches_per_epoch - batch_index - 1) * seconds_per_batch
                    averages = _averages(accumulator)
                    print(
                        f"[train] epoch {epoch + 1}/{config.epochs} | batch "
                        f"{batch_index + 1}/{batches_per_epoch} | loss "
                        f"{averages['loss']:.5f} | ETA {remaining / 60:.1f} min",
                        flush=True,
                    )

            # If the final training update was checkpointed before validation,
            # a restart arrives here with an empty loader and evaluates once.
            validation = _validation(
                torch, model, loss_function, validation_loader, device,
                config.threshold)
            training = _averages(accumulator)
            epoch_record = {
                "epoch": epoch,
                "encoder_trainable": encoder_trainable,
                "global_optimizer_step": cursor.global_optimizer_step,
                "training": training,
                "validation": validation,
                "learning_rates": {
                    str(group.get("name", index)): float(group["lr"])
                    for index, group in enumerate(optimizer.param_groups)
                },
                "elapsed_seconds": time.perf_counter() - epoch_started,
            }
            history.append(epoch_record)
            score = float(validation["selection_score"])
            improved = score > cursor.best_score
            if improved:
                cursor.best_score = score
                cursor.best_epoch = epoch
            cursor.epoch = epoch + 1
            cursor.next_batch = 0
            accumulator = _empty_accumulator()
            rng_state = capture_rng_state(torch)
            if improved:
                _save_checkpoint(
                    torch, best_path, "best", config, fingerprint, cursor,
                    model, optimizer, scheduler, history, accumulator, rng_state)
            _save_checkpoint(
                torch, last_path, "last", config, fingerprint, cursor,
                model, optimizer, scheduler, history, accumulator, rng_state)
            _atomic_json(run_root / "history.json", history)
            print(
                f"[train] epoch {epoch + 1} complete | val score {score:.6f} | "
                f"median IoU {validation['median_iou']:.5f} | boundary F1@2 "
                f"{validation['mean_boundary_f1_at_2px']:.5f} | "
                f"{'NEW BEST' if improved else 'best epoch ' + str(cursor.best_epoch + 1)}",
                flush=True,
            )

    cursor.complete = True
    rng_state = capture_rng_state(torch)
    _save_checkpoint(
        torch, last_path, "last", config, fingerprint, cursor, model,
        optimizer, scheduler, history, accumulator, rng_state)
    report = {
        "schema_version": "universal-paper-training-report-1.0",
        "status": "complete",
        "model_type": config.model_type,
        "run_name": config.run_name,
        "dataset_fingerprint": fingerprint,
        "best_epoch": cursor.best_epoch,
        "best_score": cursor.best_score,
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "elapsed_seconds": time.perf_counter() - started,
        "history": history,
    }
    _atomic_json(run_root / "training_report.json", report)
    print(
        f"[train] COMPLETE | best epoch {cursor.best_epoch + 1} | "
        f"best.pt: {best_path} | last.pt: {last_path}", flush=True)
    return report
