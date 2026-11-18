from pathlib import Path
import re

root = Path(__file__).resolve().parent
failed_path = root / "Testing/Temporary/LastTestsFailed.log"
log_path = root / "Testing/Temporary/LastTest.log"
log = log_path.read_text()
if "Test Failed." not in log:
    raise SystemExit(0)
failed = failed_path.read_text().splitlines()
for entry in failed[:3]:
    test_id, name = entry.split(":", 1)
    match = re.search(rf"(?m)^{re.escape(test_id)}/\d+ Testing: {re.escape(name)}$", log)
    if not match:
        continue
    next_test = re.search(r"(?m)^\d+/\d+ Testing: ", log[match.end():])
    end = match.end() + next_test.start() if next_test else len(log)
    block = log[match.start():end][:16000]
    encoded = block.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    print(f"::error::{encoded}")
