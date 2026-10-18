"""Bounded BF16 snapshot replay/no-overwrite test without torch/model weights."""
import hashlib
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

with tempfile.TemporaryDirectory(prefix="mollm_qsa_replay_") as directory:
    root = Path(directory)
    source, output = root/"source.bin", root/"replay.bin"
    environment = dict(os.environ, MOLLM_TEST_CUDA_QSA_INDEXER_BF16="1")
    completed = subprocess.run([sys.argv[1], "--snapshot", str(source)], env=environment)
    if completed.returncode:
        sys.exit(completed.returncode)
    command = [sys.argv[2], "--snapshot-replay", str(source), str(output)]
    subprocess.run(command, check=True)
    assert source.read_bytes() == output.read_bytes(), "whole replay must reproduce small native snapshot exactly"
    original = hashlib.sha256(output.read_bytes()).digest()
    assert subprocess.run(command).returncode not in (0, 77), "existing replay output must reject"
    assert hashlib.sha256(output.read_bytes()).digest() == original, "existing replay output changed"
    payload = source.read_bytes()
    bad = root/"invalid.bin"
    for index, content in enumerate((b"bad", payload[:-1], payload[:8]+struct.pack("<I", 0xffffffff)+payload[12:])):
        bad.write_bytes(content)
        destination = root/f"invalid-{index}.bin"
        assert subprocess.run([sys.argv[2], "--snapshot-replay", str(bad), str(destination)]).returncode not in (0, 77)
        assert not destination.exists(), "invalid source must not create output"
    print("QSA replay: exact small snapshot, no-overwrite, truncation and oversized query rejection pass")
