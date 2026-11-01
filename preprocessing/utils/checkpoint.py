from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

STAGES = ["parsed", "qc", "annotated", "normalized", "written"]


def code_fingerprint(root: Path) -> str:
    digest = hashlib.sha256()
    root = Path(root)
    pipeline_paths = [
        path
        for path in root.rglob("*.py")
        if "__pycache__" not in path.parts
        and path.relative_to(root).parts[0] not in {"manual", "tests", "tools"}
    ]
    # ``manual/`` is a historical review archive.  The build only consumes
    # source metadata and mapping code, so changing an archived review must not
    # invalidate model-ready checkpoints.
    for path in sorted(pipeline_paths):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


class Checkpoints:
    def __init__(self, root: Path, fingerprint: str | None = None):
        self.root = root
        self.fingerprint = fingerprint
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, dataset: str) -> Path:
        return self.root / f"{dataset}.json"

    def read(self, dataset: str) -> dict:
        path = self.path(dataset)
        if not path.exists():
            return {"dataset": dataset, "stages": {}}
        return json.loads(path.read_text())

    def done(self, dataset: str, stage: str) -> bool:
        data = self.read(dataset)
        if self.fingerprint and data.get("code_fingerprint") != self.fingerprint:
            return False
        record = data.get("stages", {}).get(stage, {})
        if not record.get("ok"):
            return False

        artifact = record.get("artifact")
        return bool(artifact and Path(artifact).exists())

    def artifact(self, dataset: str, stage: str) -> str | None:
        return (
            self.read(dataset)
            .get("stages", {})
            .get(stage, {})
            .get("artifact")
        )

    def mark(self, dataset: str, stage: str, artifact=None, meta=None) -> None:
        data = self.read(dataset)
        if self.fingerprint and data.get("code_fingerprint") != self.fingerprint:
            data = {"dataset": dataset, "stages": {}}
        data["code_fingerprint"] = self.fingerprint
        data.setdefault("stages", {})[stage] = {
            "ok": True,
            "time": time.time(),
            "artifact": str(artifact) if artifact else None,
            "meta": meta or {},
        }

        temporary = self.path(dataset).with_suffix(".tmp")
        temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        temporary.replace(self.path(dataset))

    def invalidate_from(self, dataset: str, stage: str) -> None:
        if stage not in STAGES:
            raise ValueError(f"Unknown stage: {stage}; choose from {STAGES}")

        data = self.read(dataset)
        start = STAGES.index(stage)
        for item in STAGES[start:]:
            data.get("stages", {}).pop(item, None)

        # --from-stage explicitly asks to trust completed artifacts before the
        # selected stage, even when code changed since they were written.
        # Rebase those retained records onto the current fingerprint so the
        # pipeline does not unexpectedly restart parsing or downloading.
        data["code_fingerprint"] = self.fingerprint

        self.path(dataset).write_text(
            json.dumps(data, indent=2, ensure_ascii=False)
        )

    def clear(self, dataset: str) -> None:
        self.path(dataset).unlink(missing_ok=True)
