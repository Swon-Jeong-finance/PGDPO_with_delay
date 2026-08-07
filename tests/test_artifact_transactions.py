"""Lifecycle guards for immutable verification artifact bundles."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pgdpo_delay.core import artifacts


P3_FULL_FILES = (
    "p3r_hjb_value.npz",
    "p3r_hjb_policy.npz",
    "p3r_hjb_residual.csv",
    "p3r_certification.csv",
    "p3d_ce_nmpc_certification.csv",
)


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ArtifactTransactionTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name) / "outputs" / "verify" / "p3"

    def tearDown(self):
        self._tmp.cleanup()

    @staticmethod
    def _metadata(stage, token, tier="full"):
        artifacts.write_manifest(
            stage, problem="p3", method="verify",
            config={"token": token, "tier": tier},
            api_versions={"p3": "test-api"},
            solver="p3r-hjb-numerical-reference",
            extra={"verify_tier": tier})

    def _publish_full(self, token):
        with artifacts.begin_bundle(self.root, "full") as tx:
            self._metadata(tx.stage_dir, token)
            for name in P3_FULL_FILES:
                (tx.stage_dir / name).write_text(
                    f"{token}:{name}", encoding="utf-8")
            return tx.publish(required_files=P3_FULL_FILES)

    def _publish_fast(self, token):
        with artifacts.begin_bundle(self.root, "fast") as tx:
            self._metadata(tx.stage_dir, token, tier="fast")
            return tx.publish(forbidden_globs=("p3r_*", "p3d_*"))

    def test_successful_full_publish_is_complete_and_immutable(self):
        old = self._publish_full("old")
        old_hashes = {name: _digest(old/name) for name in P3_FULL_FILES}
        new = self._publish_full("new")

        self.assertNotEqual(old, new)
        self.assertEqual(artifacts.resolve_current_bundle(self.root, "full"),
                         new)
        self.assertTrue((new/"manifest.json").is_file())
        self.assertTrue((new/"config.json").is_file())
        self.assertTrue(all((new/name).is_file() for name in P3_FULL_FILES))
        self.assertEqual(old_hashes,
                         {name: _digest(old/name) for name in P3_FULL_FILES})

    def test_failed_build_preserves_current_full_byte_for_byte(self):
        old = self._publish_full("certified")
        pointer = self.root/"current-full.json"
        pointer_before = pointer.read_bytes()
        old_hashes = {name: _digest(old/name) for name in P3_FULL_FILES}

        with self.assertRaisesRegex(RuntimeError, "forced failure"):
            with artifacts.begin_bundle(self.root, "full") as tx:
                self._metadata(tx.stage_dir, "failed")
                (tx.stage_dir/P3_FULL_FILES[0]).write_text("partial")
                stage = tx.stage_dir
                raise RuntimeError("forced failure")

        self.assertFalse(stage.exists())
        self.assertEqual(pointer_before, pointer.read_bytes())
        self.assertEqual(artifacts.resolve_current_bundle(self.root, "full"),
                         old)
        self.assertEqual(old_hashes,
                         {name: _digest(old/name) for name in P3_FULL_FILES})

    def test_missing_required_file_cannot_replace_current_full(self):
        old = self._publish_full("certified")
        pointer = self.root/"current-full.json"
        pointer_before = pointer.read_bytes()

        with self.assertRaises(FileNotFoundError):
            with artifacts.begin_bundle(self.root, "full") as tx:
                self._metadata(tx.stage_dir, "incomplete")
                for name in P3_FULL_FILES[:-1]:
                    (tx.stage_dir/name).write_text("partial")
                tx.publish(required_files=P3_FULL_FILES)

        self.assertEqual(pointer_before, pointer.read_bytes())
        self.assertEqual(artifacts.resolve_current_bundle(self.root, "full"),
                         old)

    def test_pointer_failure_after_bundle_move_keeps_old_current(self):
        old = self._publish_full("certified")
        pointer = self.root/"current-full.json"
        pointer_before = pointer.read_bytes()
        before = set((self.root/"bundles"/"full").iterdir())

        with artifacts.begin_bundle(self.root, "full") as tx:
            self._metadata(tx.stage_dir, "orphan")
            for name in P3_FULL_FILES:
                (tx.stage_dir/name).write_text("complete but unpublished")
            with mock.patch.object(
                    artifacts, "atomic_write_json",
                    side_effect=OSError("forced pointer failure")):
                with self.assertRaisesRegex(OSError,
                                            "forced pointer failure"):
                    tx.publish(required_files=P3_FULL_FILES)

        after = set((self.root/"bundles"/"full").iterdir())
        self.assertEqual(len(after - before), 1)  # harmless orphan retained
        self.assertEqual(pointer_before, pointer.read_bytes())
        self.assertEqual(artifacts.resolve_current_bundle(self.root, "full"),
                         old)

    def test_fast_publish_never_touches_current_full(self):
        full = self._publish_full("certified")
        full_pointer = self.root/"current-full.json"
        pointer_before = full_pointer.read_bytes()
        hashes_before = {name: _digest(full/name) for name in P3_FULL_FILES}

        fast = self._publish_fast("diagnostic")

        self.assertEqual(artifacts.resolve_current_bundle(self.root, "fast"),
                         fast)
        self.assertEqual(pointer_before, full_pointer.read_bytes())
        self.assertEqual(artifacts.resolve_current_bundle(self.root, "full"),
                         full)
        self.assertEqual(hashes_before,
                         {name: _digest(full/name) for name in P3_FULL_FILES})
        self.assertFalse(any(fast.glob("p3r_*")))

    def test_fast_bundle_rejects_p3r_artifacts(self):
        with self.assertRaisesRegex(ValueError, "forbidden artifacts"):
            with artifacts.begin_bundle(self.root, "fast") as tx:
                self._metadata(tx.stage_dir, "bad-fast", tier="fast")
                (tx.stage_dir/"p3r_stale.csv").write_text("stale")
                tx.publish(forbidden_globs=("p3r_*",))
        self.assertIsNone(
            artifacts.resolve_current_bundle(self.root, "fast"))

    def test_invalid_p3_config_preserves_current_full(self):
        from pgdpo_delay.cli import _verify_one

        old = self._publish_full("certified")
        pointer = self.root/"current-full.json"
        pointer_before = pointer.read_bytes()
        old_hashes = {name: _digest(old/name) for name in P3_FULL_FILES}

        with self.assertRaises(SystemExit):
            _verify_one("p3", full=True, config="unsupported",
                        output_root=self.root.parent)

        self.assertEqual(pointer_before, pointer.read_bytes())
        self.assertEqual(artifacts.resolve_current_bundle(self.root, "full"),
                         old)
        self.assertEqual(old_hashes,
                         {name: _digest(old/name) for name in P3_FULL_FILES})

    def test_atomic_json_replace_failure_preserves_old_pointer(self):
        pointer = self.root/"current-full.json"
        artifacts.atomic_write_json(pointer, {"old": True})
        before = pointer.read_bytes()
        with mock.patch.object(
                artifacts.os, "replace",
                side_effect=OSError("forced replace failure")):
            with self.assertRaisesRegex(OSError, "forced replace failure"):
                artifacts.atomic_write_json(pointer, {"new": True})
        self.assertEqual(before, pointer.read_bytes())
        self.assertFalse(list(pointer.parent.glob(
            f".{pointer.name}.tmp-*")))

    def test_resolver_rejects_pointer_path_traversal(self):
        self.root.mkdir(parents=True)
        (self.root/"current-full.json").write_text(json.dumps({
            "schema": 1,
            "tier": "full",
            "bundle": "../outside",
        }))
        with self.assertRaises(ValueError):
            artifacts.resolve_current_bundle(self.root, "full")


if __name__ == "__main__":
    unittest.main()
