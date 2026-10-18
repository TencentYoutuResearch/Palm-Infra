"""Compare streaming hash against hashlib at padding/read-buffer boundaries."""
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile

with tempfile.TemporaryDirectory(prefix="mollm_sha256_") as directory:
    path = Path(directory) / "input"
    for size in (0, 3, 55, 56, 63, 64, 65, 65535, 65536, 65537, 131072, 131135):
        data = bytes((i * 29 + 7) % 256 for i in range(size))
        path.write_bytes(data)
        actual = subprocess.check_output([sys.argv[1], str(path)], text=True).strip()
        assert actual == hashlib.sha256(data).hexdigest(), (size, actual)
    assert subprocess.run([sys.argv[1], str(path.parent / "missing")], capture_output=True).returncode != 0
print("SHA-256 padding and streaming boundaries pass")
