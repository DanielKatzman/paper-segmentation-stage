"""Template copied into each component to create an isolated CPU environment."""
import argparse
from pathlib import Path
import subprocess
import sys
import venv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--venv', type=Path)
    parser.add_argument('--with-tests', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if sys.version_info[:2] != (3, 14):
        parser.error('Use Python 3.14 for this release qualification')
    destination = (args.venv or root/'.venv').resolve()
    if destination.exists():
        parser.error('Choose a new environment; setup does not replace an existing one')
    venv.EnvBuilder(with_pip=True).create(destination)
    python = destination/('Scripts/python.exe' if sys.platform == 'win32' else 'bin/python')
    extra = 'runtime' if (root/'src/universal_segmenter').exists() else 'inference'
    command = [str(python), '-m', 'pip', 'install', '-c', str(root/'constraints.txt'), str(root)+'['+extra+']']
    if args.with_tests:
        command += ['pytest', 'jsonschema']
    subprocess.run(command, check=True)
    subprocess.run([str(python), '-B', str(root/'tools/verify_release.py')], check=True)


if __name__ == '__main__':
    main()
