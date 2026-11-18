from pathlib import Path
import subprocess
import sys

result = subprocess.run([sys.executable, *sys.argv[1:]], capture_output=True, text=True)
sys.stdout.write(result.stdout)
sys.stderr.write(result.stderr)
if result.returncode not in (0, 77):
    print(f"::error::CTest python: {sys.executable}")
    lines = [line for line in (result.stderr + result.stdout).splitlines() if line.strip()]
    for line in lines[-8:]:
        print(f"::error::{line}")
raise SystemExit(result.returncode)
