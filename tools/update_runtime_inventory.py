"""Explicitly refresh the development source inventory after model verification."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write",action="store_true",required=True)
    parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    metadata=json.loads((root/"compatibility.json").read_text())
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    for relative,expected in [(metadata["model"]["path"],metadata["model"]["sha256"]),
                              ("models/release.json",metadata["model"]["release_manifest_sha256"])]:
        if sha(root/relative)!=expected:
            raise ValueError("Model identity differs; inventory refresh cannot approve a changed model")
    files={p.relative_to(root).as_posix():sha(p)
        for folder in ("src","scripts","schemas","configs","models","tools")
        for p in (root/folder).rglob("*") if p.is_file() and "__pycache__" not in p.parts
        and p.suffix in (".py",".json",".onnx") and p.name!="compatibility.json"}
    files.update({p:sha(root/p) for p in ("pyproject.toml","constraints.txt","MANIFEST.in")})
    (root/"provenance/runtime_sources.json").write_bytes((json.dumps({"schema_version":
        "stage1.runtime-inventory.v1","files":files},indent=2,sort_keys=True)+"\n").encode())
    print(f"Recorded {len(files)} development runtime files; frozen model unchanged")


if __name__=="__main__":
    main()
