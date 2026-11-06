"""Run real Stage 1 inference and the orchestrator's native boundary adapter."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/"src"))

from universal_segmenter import __version__
from universal_segmenter.release_cli import verify_output
from universal_segmenter.scenes.core import write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orchestrator",type=Path,required=True)
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    target=args.output.resolve()
    if target.exists():
        raise ValueError("Choose a new diagnostic folder")
    target.mkdir(parents=True)
    adapter=args.orchestrator.resolve()/"composition/adapter/prepare_stage1_for_stage2.py"
    environment={**os.environ,"PYTHONDONTWRITEBYTECODE":"1","PYTHONPATH":str(PROJECT/"src")}
    commands=[
        [sys.executable,"-m","universal_segmenter.release_cli","run","--input",str(args.input.resolve()),
         "--output",str(target/"raw"),"--models",str(PROJECT/"models"),"--review","none","--piece-mode","multi"],
        [sys.executable,str(adapter),"--input",str(target/"raw"),"--output",str(target/"adapted")]]
    for number,command in enumerate(commands,1):
        print(f"Checking handoff step {number}/2",flush=True)
        with (target/f"step-{number}.log").open("w",encoding="utf-8") as log:
            result=subprocess.run(command,env=environment,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"Handoff step {number} exited {result.returncode}; inspect {target/f'step-{number}.log'}")
    raw,adapted=verify_output(target/"raw"),verify_output(target/"adapted")
    adjustments=json.loads((target/"adapted/mask_boundary_adapter.json").read_text())
    if raw["pieces"]!=adapted["pieces"] or not adjustments["raw_input_unchanged"]:
        raise ValueError("Adapter changed the piece inventory or raw publication")
    report={"status":"passed","stage1_version":__version__,"raw":raw,"adapted":adapted,
        "adapter_sha256":hashlib.sha256(adapter.read_bytes()).hexdigest(),
        "adapter_policy":adjustments["policy"],
        "removed_pixels":sum(a["removed_pixels"] for a in adjustments["adjustments"]),
        "scope":"Real CPU inference, source linkage and native boundary handoff; future GUI and four-stage assembly not qualified"}
    write_json(target/"control.json",report)
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
