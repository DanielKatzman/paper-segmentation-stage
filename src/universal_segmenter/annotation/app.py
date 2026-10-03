"""Teammate-friendly GUI launcher for the portable ML annotation workflow."""

from __future__ import annotations

import json
import os
import traceback
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from universal_segmenter.annotation.ml_proposals import (
    MLProposalProvider,
    PortableModel,
    PredictionCache,
    load_portable_models,
)
from universal_segmenter.annotation.session import annotate_manifest
from universal_segmenter.annotation.split_policy import (
    SPLIT_NAMES,
    assign_workspace_splits,
    normalize_folder_policies,
    split_manifest_rows,
)
from universal_segmenter.data.manifest import (
    discover_capture_page_folders,
    read_jsonl,
    sync_capture_manifest,
    write_jsonl,
)


TOOL_STATE_SCHEMA = "paper-segmentation-annotation-workspace-1.0"


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _workspace_state(workspace: Path, input_folder: Path,
                     primary_model: str) -> dict[str, Any]:
    path = workspace / "annotation_workspace.json"
    if path.is_file():
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("schema_version") != TOOL_STATE_SCHEMA:
            raise ValueError(f"unsupported annotation workspace: {path}")
        previous_input = str(value.get("input_folder", "")).strip()
        if previous_input and Path(previous_input).resolve() != input_folder:
            raise ValueError(
                "This output workspace belongs to a different input folder. "
                "Choose its original input folder or create a new workspace.")
    else:
        value = {
            "schema_version": TOOL_STATE_SCHEMA,
            "workspace_id": uuid.uuid4().hex,
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }
    value["input_folder"] = str(input_folder)
    value["primary_model"] = primary_model
    value["updated_utc"] = datetime.now(timezone.utc).isoformat()
    _atomic_json(path, value)
    return value


def run_annotation_workspace(
        input_folder: Path, workspace: Path, models: Sequence[PortableModel],
        primary_model: str = "segformer_b1_detail", review_saved: bool = False,
        cpu_threads: int | None = None, pdf_dpi: int = 300,
        folder_split_ratios: dict[str, dict[str, float]] | None = None,
        split_seed: int = 73191,
        reassign_automatic_splits: bool = False,
        ) -> dict[str, Any]:
    """Run the complete resumable ML-first annotation workflow."""
    input_folder = Path(input_folder).resolve()
    workspace = Path(workspace).resolve()
    if not input_folder.is_dir():
        raise FileNotFoundError(f"input folder does not exist: {input_folder}")
    if input_folder == workspace or input_folder in workspace.parents:
        raise ValueError(
            "The output workspace cannot be inside the input folder, because "
            "generated images would be rediscovered as new inputs.")
    model_by_key = {model.key: model for model in models}
    if primary_model not in model_by_key:
        raise ValueError(f"primary model is unavailable: {primary_model}")
    workspace.mkdir(parents=True, exist_ok=True)
    state = _workspace_state(workspace, input_folder, primary_model)
    workspace_id = str(state["workspace_id"])
    manifest_relative = Path(
        f"dataset/manifests/captures_{workspace_id}.jsonl")
    manifest_path = workspace / manifest_relative
    scan = sync_capture_manifest(input_folder, manifest_path, workspace)
    records = read_jsonl(manifest_path)
    retained: list[dict[str, Any]] = []
    removed_missing = 0
    for record in records:
        source = Path(str(record["image_path"]))
        source = source if source.is_absolute() else workspace / source
        if (not source.is_file()
                and record.get("annotation_status") not in {"reviewed", "rejected"}):
            removed_missing += 1
            continue
        # Flat, content-hashed filenames let multiple contributors merge their
        # dataset folders without depending on identical source folder names.
        record["output_group"] = "captures"
        record["annotation_workspace_id"] = workspace_id
        retained.append(record)
    records = retained
    scan["removed_missing_unreviewed"] = removed_missing
    scan["total"] = len(records)
    split_report = assign_workspace_splits(
        records,
        folder_policies=folder_split_ratios,
        seed=split_seed,
        reassign_automatic=reassign_automatic_splits,
    )
    state["folder_split_ratios"] = split_report["folders"]
    state["split_seed"] = int(split_seed)
    _atomic_json(workspace / "annotation_workspace.json", state)
    write_jsonl(manifest_path, records)
    split_manifest_relative = Path(
        f"dataset/manifests/splits_{workspace_id}.jsonl")
    split_manifest_path = workspace / split_manifest_relative
    write_jsonl(split_manifest_path, split_manifest_rows(records))

    pending = [
        record for record in records
        if record.get("annotation_status") != "rejected"
        and (review_saved or record.get("annotation_status") != "reviewed")
    ]
    threads = max(1, min(8, int(
        cpu_threads if cpu_threads is not None else (os.cpu_count() or 2) - 1)))
    cache = PredictionCache(
        project=workspace,
        cache_root=workspace / "cache",
        model=model_by_key[primary_model],
        cpu_threads=threads,
        pdf_dpi=pdf_dpi,
    )
    cache.start(pending)
    provider = MLProposalProvider(cache)
    try:
        report = annotate_manifest(
            workspace,
            manifest_relative,
            output_folder=Path("dataset"),
            review_saved=review_saved,
            pdf_dpi=pdf_dpi,
            materialize_annotation_images=True,
            proposal_provider=provider,
        )
    finally:
        cache.close()
    final_records = read_jsonl(manifest_path)
    write_jsonl(split_manifest_path, split_manifest_rows(final_records))
    split_report["captures_by_split"] = dict(Counter(
        str(record["dataset_split"]) for record in final_records))
    complete_report = {
        "schema_version": "paper-segmentation-annotation-run-1.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "workspace_id": workspace_id,
        "input_folder": str(input_folder),
        "workspace": str(workspace),
        "manifest": manifest_relative.as_posix(),
        "split_manifest": split_manifest_relative.as_posix(),
        "primary_model": primary_model,
        "models": {
            model.key: {
                "display_name": model.display_name,
                "onnx_sha256": model.model_sha256,
                "resolution": model.resolution,
                "threshold": model.threshold,
            }
            for model in models
        },
        "cpu_threads": threads,
        "folder_scan": scan,
        "split_assignment": split_report,
        **report,
    }
    _atomic_json(workspace / "last_annotation_run.json", complete_report)
    return complete_report


def _show_message(title: str, message: str, error: bool = False) -> None:
    import tkinter as tk
    from tkinter import messagebox
    root = tk.Tk()
    root.withdraw()
    try:
        if error:
            messagebox.showerror(title, message, parent=root)
        else:
            messagebox.showinfo(title, message, parent=root)
    finally:
        root.destroy()


def launch_annotation_tool(project: Path | None = None) -> None:
    """Choose folders/model in one window, then open the correction editor."""
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk
    except ImportError as error:
        raise RuntimeError(
            "Tkinter is required. Use the official Python installer with Tcl/Tk.") from error

    try:
        models = load_portable_models(project, require_all=True)
    except Exception as error:
        _show_message("Models are missing", str(error), error=True)
        return
    display_to_key = {model.display_name: model.key for model in models}
    key_to_display = {model.key: model.display_name for model in models}
    configuration: dict[str, Any] = {}
    root = tk.Tk()
    root.title("Paper Segmentation Annotator")
    root.resizable(True, False)
    root.columnconfigure(1, weight=1)

    input_value = tk.StringVar()
    output_value = tk.StringVar()
    model_value = tk.StringVar(
        value=key_to_display.get("segformer_b1_detail", models[0].display_name))
    review_value = tk.BooleanVar(value=False)
    reassign_value = tk.BooleanVar(value=False)
    split_summary = tk.StringVar(value="Not configured yet")
    folder_policies: dict[str, dict[str, float]] = {}

    ttk.Label(
        root,
        text=(
            "Choose the folder containing scans/photos and a separate output "
            "workspace. All subfolders are scanned automatically."),
        wraplength=650,
        justify="left",
        padding=(12, 12, 12, 8),
    ).grid(row=0, column=0, columnspan=3, sticky="ew")

    def choose_input() -> None:
        selected = filedialog.askdirectory(title="Choose scans/photos folder")
        if not selected:
            return
        input_value.set(selected)
        folder_policies.clear()
        split_summary.set("Not configured yet")
        if not output_value.get().strip():
            source = Path(selected)
            output_value.set(str(
                source.parent / f"{source.name}_paper_annotations"))

    def choose_output() -> None:
        selected = filedialog.askdirectory(
            title="Choose or create annotation workspace")
        if selected:
            output_value.set(selected)

    def configure_splits() -> bool:
        source = Path(input_value.get().strip()).expanduser()
        if not source.is_dir():
            messagebox.showerror(
                "Choose input first",
                "Choose the scans/photos folder before configuring splits.",
                parent=root,
            )
            return False
        folders = discover_capture_page_folders(source)
        if not folders:
            messagebox.showerror(
                "No captures found",
                "No supported scans or photographs were found in that folder.",
                parent=root,
            )
            return False

        saved: dict[str, dict[str, float]] = {}
        destination_text = output_value.get().strip()
        if destination_text:
            state_path = Path(destination_text).expanduser() / "annotation_workspace.json"
            if state_path.is_file():
                try:
                    state = json.loads(state_path.read_text(encoding="utf-8"))
                    same_input = Path(str(state.get("input_folder", ""))).resolve() == source.resolve()
                    if same_input and isinstance(state.get("folder_split_ratios"), dict):
                        saved = state["folder_split_ratios"]
                except (OSError, ValueError, TypeError, json.JSONDecodeError):
                    saved = {}
        initial = normalize_folder_policies(
            folders, folder_policies or saved or None)

        dialog = tk.Toplevel(root)
        dialog.title("Dataset split by folder")
        dialog.geometry("820x620")
        dialog.minsize(720, 420)
        dialog.transient(root)
        dialog.grab_set()
        accepted = {"value": False}
        variables: dict[str, dict[str, tk.StringVar]] = {}

        ttk.Label(
            dialog,
            text=(
                "Percentages are applied to complete physical pieces. Repeated "
                "lighting captures of one piece always remain in one split. "
                "Final test is never used during training. For a dedicated "
                "challenge folder, enter 0/0/0/100 in that row."),
            wraplength=770,
            justify="left",
            padding=(12, 10),
        ).pack(fill="x")

        presets = ttk.Frame(dialog, padding=(12, 0, 12, 8))
        presets.pack(fill="x")
        ttk.Label(presets, text="Apply to every folder:").pack(side="left")

        def apply_preset(values: tuple[int, int, int, int]) -> None:
            for per_folder in variables.values():
                for split, value in zip(SPLIT_NAMES, values):
                    per_folder[split].set(str(value))

        for label, values in (
            ("Standard 70/10/10/10", (70, 10, 10, 10)),
            ("Hard 60/10/10/20", (60, 10, 10, 20)),
        ):
            ttk.Button(
                presets, text=label,
                command=lambda selected=values: apply_preset(selected),
            ).pack(side="left", padx=(7, 0))

        table_container = ttk.Frame(dialog, padding=(12, 0, 12, 0))
        table_container.pack(fill="both", expand=True)
        canvas = tk.Canvas(table_container, highlightthickness=0)
        scrollbar = ttk.Scrollbar(
            table_container, orient="vertical", command=canvas.yview)
        rows = ttk.Frame(canvas)
        rows.columnconfigure(0, weight=1)
        window = canvas.create_window((0, 0), window=rows, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        rows.bind("<Configure>", lambda _event: canvas.configure(
            scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda event: canvas.itemconfigure(
            window, width=event.width))

        headers = ("Folder", "Train %", "Validation %", "Calibration %", "Final test %")
        for column, label in enumerate(headers):
            ttk.Label(rows, text=label, padding=(5, 4)).grid(
                row=0, column=column, sticky="w")
        for row_index, folder in enumerate(folders, start=1):
            ttk.Label(rows, text=folder, padding=(5, 3)).grid(
                row=row_index, column=0, sticky="w")
            variables[folder] = {}
            for column, split in enumerate(SPLIT_NAMES, start=1):
                variable = tk.StringVar(
                    master=dialog,
                    value=f"{initial[folder][split] * 100.0:g}",
                )
                variables[folder][split] = variable
                ttk.Entry(rows, textvariable=variable, width=10).grid(
                    row=row_index, column=column, padx=4, pady=2)

        def accept() -> None:
            configured: dict[str, dict[str, float]] = {}
            try:
                for folder, per_folder in variables.items():
                    percentages = {
                        split: float(per_folder[split].get().strip())
                        for split in SPLIT_NAMES
                    }
                    if any(value < 0.0 for value in percentages.values()):
                        raise ValueError(f"{folder}: percentages cannot be negative")
                    if abs(sum(percentages.values()) - 100.0) > 0.01:
                        raise ValueError(
                            f"{folder}: percentages must add up to 100")
                    configured[folder] = {
                        split: percentages[split] / 100.0
                        for split in SPLIT_NAMES
                    }
            except ValueError as error:
                messagebox.showerror(
                    "Invalid percentages", str(error), parent=dialog)
                return
            folder_policies.clear()
            folder_policies.update(configured)
            split_summary.set(
                f"Configured {len(configured)} folder(s); physical pieces stay grouped")
            accepted["value"] = True
            dialog.destroy()

        buttons = ttk.Frame(dialog, padding=12)
        buttons.pack(fill="x")
        ttk.Button(buttons, text="Cancel", command=dialog.destroy).pack(
            side="right", padx=(7, 0))
        ttk.Button(buttons, text="Save percentages", command=accept).pack(
            side="right")
        dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)
        root.wait_window(dialog)
        return bool(accepted["value"])

    ttk.Label(root, text="Input folder:", padding=(12, 6)).grid(
        row=1, column=0, sticky="w")
    ttk.Entry(root, textvariable=input_value, width=72).grid(
        row=1, column=1, sticky="ew", padx=5)
    ttk.Button(root, text="Browse…", command=choose_input).grid(
        row=1, column=2, padx=(5, 12))

    ttk.Label(root, text="Output workspace:", padding=(12, 6)).grid(
        row=2, column=0, sticky="w")
    ttk.Entry(root, textvariable=output_value, width=72).grid(
        row=2, column=1, sticky="ew", padx=5)
    ttk.Button(root, text="Browse…", command=choose_output).grid(
        row=2, column=2, padx=(5, 12))

    ttk.Label(root, text="Starting guess:", padding=(12, 6)).grid(
        row=3, column=0, sticky="w")
    ttk.Combobox(
        root,
        textvariable=model_value,
        values=list(display_to_key),
        state="readonly",
        width=40,
    ).grid(row=3, column=1, sticky="w", padx=5)
    ttk.Label(root, text="Dataset split:", padding=(12, 6)).grid(
        row=4, column=0, sticky="w")
    split_frame = ttk.Frame(root)
    split_frame.grid(row=4, column=1, columnspan=2, sticky="w", padx=5)
    ttk.Button(
        split_frame,
        text="Configure folder percentages…",
        command=configure_splits,
    ).pack(side="left")
    ttk.Label(split_frame, textvariable=split_summary, padding=(8, 0)).pack(
        side="left")
    ttk.Checkbutton(
        root,
        text="Review already completed masks again",
        variable=review_value,
    ).grid(row=5, column=1, sticky="w", padx=5, pady=(2, 2))
    ttk.Checkbutton(
        root,
        text=(
            "Reassign existing automatic splits using these percentages "
            "(manual choices remain locked)"),
        variable=reassign_value,
    ).grid(row=6, column=1, sticky="w", padx=5, pady=(2, 6))

    ttk.Label(
        root,
        text=(
            "The calibrated SegFormer v3 guess is generated on CPU in the "
            "background and cached permanently. Split buttons can override the "
            "complete physical piece."),
        wraplength=650,
        justify="left",
        padding=(12, 5, 12, 10),
    ).grid(row=7, column=0, columnspan=3, sticky="ew")

    def start() -> None:
        source_text = input_value.get().strip()
        destination_text = output_value.get().strip()
        source = Path(source_text).expanduser()
        destination = Path(destination_text).expanduser()
        if not source.is_dir():
            messagebox.showerror(
                "Invalid input", "Choose an existing input folder.", parent=root)
            return
        if not destination_text:
            messagebox.showerror(
                "Invalid output", "Choose an output workspace.", parent=root)
            return
        try:
            source_resolved = source.resolve()
            destination_resolved = destination.resolve()
            if (source_resolved == destination_resolved
                    or source_resolved in destination_resolved.parents):
                raise ValueError("Output must not be inside the input folder.")
        except (OSError, ValueError) as error:
            messagebox.showerror("Invalid folders", str(error), parent=root)
            return
        if not folder_policies and not configure_splits():
            return
        configuration.update({
            "input": source,
            "output": destination,
            "primary": display_to_key[model_value.get()],
            "review": bool(review_value.get()),
            "folder_splits": dict(folder_policies),
            "reassign_splits": bool(reassign_value.get()),
        })
        root.destroy()

    buttons = ttk.Frame(root, padding=(12, 5, 12, 12))
    buttons.grid(row=8, column=0, columnspan=3, sticky="e")
    ttk.Button(buttons, text="Cancel", command=root.destroy).pack(
        side="right", padx=(7, 0))
    ttk.Button(buttons, text="Start annotation", command=start).pack(side="right")
    root.protocol("WM_DELETE_WINDOW", root.destroy)
    root.mainloop()
    if not configuration:
        return

    try:
        report = run_annotation_workspace(
            configuration["input"],
            configuration["output"],
            models=models,
            primary_model=configuration["primary"],
            review_saved=configuration["review"],
            folder_split_ratios=configuration["folder_splits"],
            reassign_automatic_splits=configuration["reassign_splits"],
        )
        split_counts = report.get("captures_by_split", {})
        split_line = (
            f"Train {split_counts.get('train', 0)} | "
            f"Validation {split_counts.get('validation', 0)} | "
            f"Calibration {split_counts.get('calibration', 0)} | "
            f"Final test {split_counts.get('holdout', 0)}"
        )
        if report["complete"]:
            message = (
                f"Annotation complete: {report['reviewed']}/{report['total']} "
                f"captures.\n{split_line}\n\nOutput: {report['workspace']}")
        else:
            message = (
                f"Paused safely: {report['reviewed']}/{report['total']} "
                f"captures reviewed.\n{split_line}\n"
                "Run the tool again to resume.")
        _show_message("Paper Segmentation Annotator", message)
    except Exception as error:
        destination = Path(configuration["output"]).resolve()
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "annotation_error.log").write_text(
            traceback.format_exc(), encoding="utf-8")
        _show_message(
            "Annotation stopped",
            f"{error}\n\nDetails were saved to annotation_error.log.",
            error=True,
        )
