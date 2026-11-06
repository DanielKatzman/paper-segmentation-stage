"""Collection, generator and immutable dataset commands; no training GPU required."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core import SceneStore, read_json


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("annotate", help="Open the multi-piece collection application")
    p.add_argument("--workspace", type=Path)
    p = sub.add_parser("discover")
    p.add_argument("input", type=Path); p.add_argument("workspace", type=Path)
    p.add_argument("--metadata", type=Path, required=True); p.add_argument("--dpi", type=int, default=300)
    p = sub.add_parser("validate")
    p.add_argument("workspace", type=Path); p.add_argument("--complete-only", action="store_true")
    p = sub.add_parser("export")
    p.add_argument("workspace", type=Path); p.add_argument("zip", type=Path)
    p.add_argument("--include-unfinished", action="store_true")
    p = sub.add_parser("import")
    p.add_argument("zip", type=Path); p.add_argument("central", type=Path)
    p = sub.add_parser("inventory")
    p.add_argument("input_root", type=Path); p.add_argument("output", type=Path)
    p.add_argument("--declare-training-assets", action="store_true")
    p = sub.add_parser("generate")
    p.add_argument("--inventory", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True); p.add_argument("--resume", action="store_true")
    p = sub.add_parser("build-dataset")
    p.add_argument("--source", type=Path, action="append", default=[])
    p.add_argument("--central", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--ledger", type=Path, required=True)
    p.add_argument("--seed", type=int, default=73191)
    p.add_argument("--ratios", type=Path)
    p.add_argument("--legacy", type=Path, action="append", default=[])
    p.add_argument("--legacy-exhaustive-ids", type=Path,
                   help="JSON list of audited exhaustive legacy TRAIN sample IDs")
    p = sub.add_parser("resolve-annotation")
    p.add_argument("central", type=Path); p.add_argument("capture_id")
    p.add_argument("--collection", required=True)
    p = sub.add_parser("verify-dataset")
    p.add_argument("dataset", type=Path)
    p = sub.add_parser("assign-splits")
    p.add_argument("--source", type=Path, action="append", required=True)
    p.add_argument("--ledger", type=Path, required=True)
    p.add_argument("--ratios", type=Path)
    p.add_argument("--legacy", type=Path, action="append", default=[])
    p = sub.add_parser("publish-pieces")
    p.add_argument("workspace", type=Path); p.add_argument("output", type=Path)
    p.add_argument("--include-generated-fixtures",action="store_true",
                   help="Diagnostic only: retain synthetic-truth origin instead of claiming manual review")
    p = sub.add_parser("clone-workspace")
    p.add_argument("source",type=Path); p.add_argument("output",type=Path)
    p = sub.add_parser("collection-assets")
    p.add_argument("workspace", type=Path); p.add_argument("inventory", type=Path)
    p.add_argument("--ledger", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "annotate":
            from .app import launch
            launch(workspace=args.workspace)
            return 0
        if args.command == "discover":
            from .workspace import discover_into
            result = discover_into(SceneStore(args.workspace), args.input, read_json(args.metadata), args.dpi)
        elif args.command == "validate":
            result = SceneStore(args.workspace).verify(complete_only=args.complete_only)
        elif args.command == "export":
            from .exchange import export_collection
            result = export_collection(SceneStore(args.workspace), args.zip, include_unfinished=args.include_unfinished)
        elif args.command == "import":
            from .exchange import import_collection
            result = import_collection(args.zip, args.central)
        elif args.command == "inventory":
            from .generator import create_inventory
            result = create_inventory(args.input_root, args.output, declare_training_assets=args.declare_training_assets)
        elif args.command == "generate":
            from .generator import GeneratorConfig, generate
            config = GeneratorConfig(**read_json(args.config))
            result = generate(args.inventory,args.output,config,resume=args.resume)
        elif args.command == "build-dataset":
            from .dataset import build_dataset
            result = build_dataset(args.source,args.output,args.ledger,seed=args.seed,
                ratios=read_json(args.ratios) if args.ratios else None, legacy_roots=args.legacy,
                legacy_exhaustive_ids=set(read_json(args.legacy_exhaustive_ids)) if args.legacy_exhaustive_ids else None,
                central=args.central)
        elif args.command == "resolve-annotation":
            from .exchange import resolve_annotation
            result = resolve_annotation(args.central,args.capture_id,args.collection)
        elif args.command == "verify-dataset":
            from .dataset import verify_dataset
            result = verify_dataset(args.dataset)
        elif args.command == "assign-splits":
            from .dataset import assign_splits, _protected_legacy
            scenes = [SceneStore(root).load(i) for root in args.source for i in SceneStore(root).ids()
                      if SceneStore(root).load(i).eligible]
            result = assign_splits(scenes,args.ledger,
                ratios=read_json(args.ratios) if args.ratios else None,
                protected_tokens=_protected_legacy(args.legacy))
        elif args.command == "publish-pieces":
            from .workspace import publish_pieces
            result = publish_pieces(SceneStore(args.workspace),args.output,
                                    include_generated_fixtures=args.include_generated_fixtures)
        elif args.command == "clone-workspace":
            from .workspace import clone_workspace
            result = clone_workspace(SceneStore(args.source),args.output)
        elif args.command == "collection-assets":
            from .generator import collection_assets
            result = collection_assets(SceneStore(args.workspace),args.inventory,args.ledger)
        print(json.dumps(result,indent=2,sort_keys=True))
        return 0
    except (ValueError,OSError,KeyError) as error:
        print(json.dumps({"status":"failed","error":str(error)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
