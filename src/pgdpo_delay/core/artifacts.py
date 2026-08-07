"""Run manifests and immutable verification bundles.

Every run/verify writes a manifest containing the config hash, git commit,
problem/method, seeds, device, API versions, and solver role. Certified
multi-file verification outputs use :func:`begin_bundle`: files are built in
a same-filesystem staging directory, moved to a new immutable bundle, and
made current by one atomic JSON-pointer replacement. A failed build can
therefore never partially overwrite the previously published bundle.
"""
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
import uuid
from pathlib import Path


_POINTER_SCHEMA = 1
_SAFE_TIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _fsync_dir(path: Path) -> None:
    """Best-effort directory sync after an atomic rename."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        # Some platforms/filesystems do not support fsync on a directory.
        pass
    finally:
        os.close(fd)


def atomic_write_json(path, payload, *, default=None):
    """Write JSON through a sibling temporary file and ``os.replace``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    created = False
    try:
        with open(tmp, "x", encoding="utf-8") as fp:
            created = True
            json.dump(payload, fp, indent=1, default=default)
            fp.write("\n")
            fp.flush()
            os.fsync(fp.fileno())
        os.replace(tmp, path)
        _fsync_dir(path.parent)
    finally:
        # A failed serialization/replace must not leave a plausible pointer.
        if created:
            try:
                tmp.unlink()
            except FileNotFoundError:
                pass


def _relative_file(path) -> Path:
    """Validate a bundle-relative file name (no absolute/traversal paths)."""
    rel = Path(path)
    if rel.is_absolute() or not rel.parts or any(
            part in ("", ".", "..") for part in rel.parts):
        raise ValueError(f"unsafe bundle-relative path: {path!r}")
    return rel


class BundleTransaction:
    """Build one immutable bundle and atomically replace its tier pointer.

    ``stage_dir`` is the only path to hand to a solver. ``publish`` validates
    the complete stage, moves it to a never-reused bundle ID, and finally
    replaces ``current-<tier>.json``. If that final write fails, the moved
    directory is merely unreferenced and the old pointer remains authoritative.
    """

    def __init__(self, root, tier):
        if not isinstance(tier, str) or not _SAFE_TIER.fullmatch(tier):
            raise ValueError(f"unsafe bundle tier: {tier!r}")
        self.root = Path(root).resolve()
        self.tier = tier
        stamp = time.strftime("%Y%m%dT%H%M%S")
        self.bundle_id = f"{stamp}-{uuid.uuid4().hex}"
        self._stage_root = self.root / ".staging"
        self._bundle_root = self.root / "bundles" / tier
        self._stage_root.mkdir(parents=True, exist_ok=True)
        self._bundle_root.mkdir(parents=True, exist_ok=True)
        self.stage_dir = self._stage_root / f"{tier}-{self.bundle_id}"
        self.stage_dir.mkdir()
        self.pointer_path = self.root / f"current-{tier}.json"
        self.bundle_dir = None
        self.published = False
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if not self.published:
            self.abort()
        return False

    def publish(self, required_files=(), forbidden_globs=()):
        """Validate, move, and publish this bundle.

        ``manifest.json`` and ``config.json`` are mandatory for every bundle;
        callers list tier-specific artifacts in ``required_files``. Patterns
        such as ``p3r_*`` can be forbidden for a fast diagnostic bundle.
        """
        if self.closed:
            raise RuntimeError("bundle transaction is already closed")
        if not self.stage_dir.is_dir():
            raise RuntimeError(
                f"bundle staging directory vanished: {self.stage_dir}")

        required = [Path("manifest.json"), Path("config.json")]
        required.extend(_relative_file(path) for path in required_files)
        missing = []
        for rel in dict.fromkeys(required):
            candidate = self.stage_dir / rel
            if not candidate.is_file() or candidate.is_symlink():
                missing.append(rel.as_posix())
        if missing:
            raise FileNotFoundError(
                "bundle is incomplete; missing regular files: "
                + ", ".join(missing))

        # Parse completion metadata before the stage becomes immutable.
        for rel in (Path("manifest.json"), Path("config.json")):
            try:
                with open(self.stage_dir / rel, encoding="utf-8") as fp:
                    json.load(fp)
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"invalid bundle metadata {rel}: {exc}") from exc

        for pattern in forbidden_globs:
            if not isinstance(pattern, str) or not pattern:
                raise ValueError(f"invalid forbidden glob: {pattern!r}")
            hits = sorted(
                path.relative_to(self.stage_dir).as_posix()
                for path in self.stage_dir.glob(pattern))
            if hits:
                raise ValueError(
                    f"bundle contains forbidden artifacts for tier "
                    f"{self.tier}: {hits}")

        final_dir = self._bundle_root / self.bundle_id
        if final_dir.exists():
            raise FileExistsError(f"bundle ID collision: {final_dir}")
        os.replace(self.stage_dir, final_dir)  # destination is new; same FS
        self.bundle_dir = final_dir
        _fsync_dir(self._bundle_root)

        rel = final_dir.relative_to(self.root).as_posix()
        pointer = dict(schema=_POINTER_SCHEMA, tier=self.tier, bundle=rel)
        # This is the sole publication point. On failure final_dir remains an
        # harmless orphan and the previous pointer is untouched.
        atomic_write_json(self.pointer_path, pointer)
        self.published = True
        self.closed = True
        return final_dir

    def abort(self):
        """Discard only this transaction's private staging directory."""
        if self.closed:
            return
        if self.stage_dir.exists():
            # Never broaden this recursive deletion beyond our UUID target.
            if (self.stage_dir.parent != self._stage_root
                    or not self.stage_dir.name.startswith(f"{self.tier}-")):
                raise RuntimeError(
                    f"refusing unsafe staging cleanup: {self.stage_dir}")
            shutil.rmtree(self.stage_dir)
        self.closed = True


def begin_bundle(root, tier):
    """Create a transaction under ``root`` for the named verification tier."""
    return BundleTransaction(root, tier)


def resolve_current_bundle(root, tier):
    """Resolve and validate the immutable bundle named by the tier pointer.

    Returns ``None`` when no bundle has yet been published. A malformed,
    escaping, or dangling pointer is an integrity error rather than a silent
    fallback to an arbitrary historical bundle.
    """
    if not isinstance(tier, str) or not _SAFE_TIER.fullmatch(tier):
        raise ValueError(f"unsafe bundle tier: {tier!r}")
    root = Path(root).resolve()
    pointer_path = root / f"current-{tier}.json"
    if not pointer_path.exists():
        return None
    try:
        with open(pointer_path, encoding="utf-8") as fp:
            pointer = json.load(fp)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"invalid current-bundle pointer {pointer_path}: {exc}") from exc
    if (pointer.get("schema") != _POINTER_SCHEMA
            or pointer.get("tier") != tier
            or not isinstance(pointer.get("bundle"), str)):
        raise RuntimeError(f"invalid current-bundle pointer schema: {pointer}")
    rel = _relative_file(pointer["bundle"])
    if len(rel.parts) != 3 or rel.parts[:2] != ("bundles", tier):
        raise RuntimeError(f"pointer escapes tier bundle root: {rel}")
    bundle = (root / rel).resolve()
    expected_root = (root / "bundles" / tier).resolve()
    if bundle.parent != expected_root or not bundle.is_dir():
        raise RuntimeError(
            f"current bundle is missing or outside its tier: {bundle}")
    for name in ("manifest.json", "config.json"):
        path = bundle / name
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(
                f"current bundle lacks regular {name}: {bundle}")
    return bundle


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(
        json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:16]


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def write_manifest(outdir, *, problem, method, config, seeds=None,
                   device="cpu", api_versions=None, solver="exact", extra=None):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    man = dict(problem=problem, method=method, config_hash=config_hash(config),
               git_commit=git_commit(), seeds=seeds or {}, device=device,
               api_versions=api_versions or {}, solver=solver,
               timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"), extra=extra or {})
    atomic_write_json(outdir/"manifest.json", man)
    atomic_write_json(outdir/"config.json", config, default=str)
    return man
