"""Template: verify source inventory and frozen model/configuration identities."""
from pathlib import Path
import hashlib
import json
import tomllib


def main():
    root = Path(__file__).resolve().parents[1]
    read = lambda p: json.loads(p.read_text(encoding='utf-8'))
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    compatibility = read(root/'compatibility.json')
    package = tomllib.loads((root/'pyproject.toml').read_text(encoding='utf-8'))
    if package['project']['version'] != compatibility['release_version']:
        raise ValueError('Distribution versions differ')
    if read(root/'src'/compatibility['module']/'compatibility.json') != compatibility:
        raise ValueError('Installed compatibility metadata differs')
    expected = read(root/'provenance/runtime_sources.json')['files']
    actual = {p.relative_to(root).as_posix(): sha(p)
              for folder in ('src', 'scripts', 'schemas', 'configs', 'models', 'tools')
              for p in (root/folder).rglob('*') if p.is_file()
              and '__pycache__' not in p.parts and p.suffix in ('.py', '.json', '.onnx')
              and p.name != 'compatibility.json'}
    actual.update({p: sha(root/p) for p in ('pyproject.toml', 'constraints.txt')})
    if actual != expected:
        changed = sorted(p for p in actual.keys() | expected.keys() if actual.get(p) != expected.get(p))
        raise ValueError('Runtime inventory differs: '+', '.join(changed))
    for field in ('sha256', 'release_manifest_sha256'):
        path = compatibility['model']['path'] if field == 'sha256' else 'models/release.json'
        if sha(root/path) != compatibility['model'][field]:
            raise ValueError('Frozen asset differs: '+path)
    print(json.dumps({'status': 'verified', 'release_version': compatibility['release_version'],
                      'component_software_version': compatibility['component_software_version'],
                      'runtime_files': len(actual), 'device': 'cpu'}, indent=2))


if __name__ == '__main__':
    main()
