import subprocess
import sys
from os import chdir
from pathlib import Path


def make_relative(path: Path) -> str:
    return (
        str(path.absolute()).removeprefix(str(here.parent.absolute())).removeprefix("/")
    )


here = Path(__file__).resolve().parent
files = (
    [
        here / ".python-version",
        here / "pyproject.toml",
        here / "uv.lock",
        here / "CMAPSSData",
        here / "src",
        here / ".env",
        here / "rgrc.py",
        here / "README.md",
    ]
    + [x for x in here.iterdir() if x.suffix == ".log"]
    + [
        x
        for x in (here.parent / "rungrid").iterdir()
        if x.name != ".venv" and x.name != ".git" and "cache" not in x.name
    ]
)


for file in files:
    if not file.exists():
        print(file, "not found")
        sys.exit(1)


chdir(here.parent)

payload = here / "payload.zip"
payload.unlink(missing_ok=True)
cmd = [
    "zip",
    "-r9v",
    make_relative(payload),
] + [make_relative(x) for x in files]
print("zip returned:", subprocess.call(cmd))
