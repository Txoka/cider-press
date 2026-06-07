import json
from pathlib import Path, PurePosixPath
from typing import List, Tuple

ARCHIVE_FORMAT = "ciderpress.text-archive.v1"
TEXT_CONTROL_CHARS = {"\t", "\n", "\r", "\f", "\b"}


def _relative_posix(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _validate_archive_path(path: str) -> PurePosixPath:
    rel = PurePosixPath(path)
    if rel.is_absolute() or not rel.parts:
        raise ValueError(f"Unsafe archive path: {path!r}")
    if any(part in {"", ".", ".."} for part in rel.parts):
        raise ValueError(f"Unsafe archive path: {path!r}")
    return rel


def _looks_like_text(text: str) -> bool:
    if "\x00" in text:
        return False
    for ch in text:
        if ord(ch) < 32 and ch not in TEXT_CONTROL_CHARS:
            return False
    return True


def build_text_archive(root_path: str) -> Tuple[str, int, int, int]:
    root = Path(root_path)
    if not root.is_dir():
        raise ValueError(f"Not a directory: {root_path}")

    dirs: List[str] = []
    files = []
    skipped = 0

    for path in sorted(root.rglob("*"), key=lambda p: _relative_posix(p, root)):
        rel = _relative_posix(path, root)
        if path.is_symlink():
            skipped += 1
            continue
        if path.is_dir():
            dirs.append(rel)
            continue
        if not path.is_file():
            skipped += 1
            continue

        raw = path.read_bytes()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            skipped += 1
            continue
        if not _looks_like_text(text):
            skipped += 1
            continue
        files.append({"path": rel, "text": text})

    archive = {
        "format": ARCHIVE_FORMAT,
        "dirs": dirs,
        "files": files,
    }
    text = json.dumps(archive, ensure_ascii=False, indent=2, sort_keys=True)
    return text, len(files), len(dirs), skipped


def extract_text_archive(text: str, out_path: str) -> Tuple[int, int]:
    archive = json.loads(text)
    if archive.get("format") != ARCHIVE_FORMAT:
        raise ValueError("Unsupported text archive format")

    out_root = Path(out_path)
    if out_root.exists() and not out_root.is_dir():
        raise ValueError(f"Archive output path exists and is not a directory: {out_path}")
    out_root.mkdir(parents=True, exist_ok=True)

    dirs = archive.get("dirs", [])
    files = archive.get("files", [])
    if not isinstance(dirs, list) or not isinstance(files, list):
        raise ValueError("Malformed text archive")

    for item in dirs:
        if not isinstance(item, str):
            raise ValueError("Malformed text archive directory entry")
        rel = _validate_archive_path(item)
        (out_root / Path(*rel.parts)).mkdir(parents=True, exist_ok=True)

    for item in files:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str) or not isinstance(item.get("text"), str):
            raise ValueError("Malformed text archive file entry")
        rel = _validate_archive_path(item["path"])
        dest = out_root / Path(*rel.parts)
        if dest.exists() and dest.is_dir():
            raise ValueError(f"Archive file path conflicts with existing directory: {item['path']}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(item["text"], encoding="utf-8")

    return len(files), len(dirs)
