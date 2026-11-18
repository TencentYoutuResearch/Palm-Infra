from pathlib import Path
import re
import sys

root = Path(__file__).resolve().parent
log = (root / "Testing/Temporary/LastTest.log").read_text()
failed = (root / "Testing/Temporary/LastTestsFailed.log").read_text().splitlines()
print(f"::error::reporter python: {sys.executable}")
entry = failed[0]
print(f"::error::failed entry: {entry}")
test_id, name = entry.split(":", 1)
match = re.search(rf"(?m)^{re.escape(test_id)}/\d+ Testing: {re.escape(name)}$", log)
if match:
    next_test = re.search(r"(?m)^\d+/\d+ Testing: ", log[match.end():])
    end = match.end() + next_test.start() if next_test else len(log)
    lines = [line for line in log[match.start():end].splitlines() if line.strip()]
    for line in lines:
        if line.startswith("Command:"):
            print(f"::error::{line}")
            break
    for line in lines[-7:]:
        print(f"::error::{line}")
