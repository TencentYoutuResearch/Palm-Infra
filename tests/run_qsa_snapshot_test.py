"""Native snapshot lifecycle/format test; no torch/real checkpoint required."""
import hashlib
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile

spec = importlib.util.spec_from_file_location("snapshot", Path(__file__).resolve().parents[1]/"tools/check_qsa_indexer_snapshot.py")
snapshot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(snapshot)

with tempfile.TemporaryDirectory(prefix="mollm_qsa_snapshot_") as directory:
    path = Path(directory)/"indexer.bin"
    command = [sys.argv[1], "--snapshot", str(path)]
    completed = subprocess.run(command)
    if completed.returncode == 77:
        sys.exit(77)
    if completed.returncode:
        sys.exit(completed.returncode)
    with path.open("rb") as source:
        metadata = snapshot.read_header(source)
        assert metadata["diagnostic_bf16"] == (os.environ.get("MOLLM_TEST_CUDA_QSA_INDEXER_BF16") == "1"), "precision flag mismatch"
        assert len(source.read()) == metadata["elements"]*4, "payload size differs from header"
    original = hashlib.sha256(path.read_bytes()).digest()
    assert subprocess.run(command).returncode not in (0, 77), "existing dump must reject"
    assert hashlib.sha256(path.read_bytes()).digest() == original, "existing artifact changed"
    print("QSA snapshot: native continuation, bounded payload and no-overwrite pass")
