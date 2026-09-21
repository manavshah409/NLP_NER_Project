"""Small deterministic UTF-8 and checksum helpers."""
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone


def now():
    return datetime.now(timezone.utc).isoformat()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def checksums(root):
    root = Path(root)
    files = {p.relative_to(root).as_posix(): file_hash(p)
             for p in sorted(root.rglob("*")) if p.is_file()}
    if not files:
        raise ValueError("Cannot checksum an empty directory")
    return {"method": "SHA256 of UTF-8 canonical JSON mapping relative POSIX file paths to SHA256; sorted keys, no spaces, ensure_ascii=False; manifest stored outside root", "files": files, "combined_sha256": digest(files)}


def verify_checksums(root, manifest):
    actual = checksums(root)
    if actual != manifest:
        raise ValueError(f"Checksum mismatch: {root}")
    return actual
