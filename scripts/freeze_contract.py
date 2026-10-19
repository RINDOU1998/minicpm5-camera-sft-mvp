"""One-time extraction from the byte-preserved 50-case fixture; no model calls."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from camera_sft.common import ROOT, read_json, write_json, sha, digest

rows = read_json(ROOT / "data/frozen/cases50.json")
tools = rows[0]["request"]["tools"]
system = rows[0]["request"]["messages"][0]["content"]
assert len(rows) == 50 and len(tools) == 10
assert all(r["request"]["tools"] == tools for r in rows)
assert all(r["request"]["messages"][0]["content"] == system for r in rows)
write_json(ROOT / "contracts/camera_v1.json", {
    "id": "camera-frozen50-v1", "system": system, "tools": tools,
    "source_file": "data/frozen/cases50.json", "source_sha256": sha(ROOT / "data/frozen/cases50.json"),
    "tools_sha256": digest(tools),
    "scoring_note": "Original additionalProperties=true is preserved. Historical exact-match scorer rejects extra parameters; validator enforces exact declared keys separately.",
    "result_contract": "mvp-simulated-v1; training fixtures only, not a production hardware API",
})
print("Frozen 10 tools, original system, and source SHA256.")
