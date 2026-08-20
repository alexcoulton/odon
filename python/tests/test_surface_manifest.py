from __future__ import annotations

import importlib
import json
import unittest
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "api" / "application-surface.json"


def _resolve(reference: str) -> Any:
    module_name, attributes = reference.split(":", 1)
    value: Any = importlib.import_module(module_name)
    for attribute in attributes.split("."):
        value = getattr(value, attribute)
    return value


class ApplicationSurfaceManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    def test_manifest_ids_and_canonical_methods_are_unique(self) -> None:
        entries = self.manifest["entries"]
        ids = [entry["id"] for entry in entries]
        self.assertEqual(len(ids), len(set(ids)))
        methods = [method for entry in entries for method in entry.get("methods", [])]
        self.assertEqual(len(methods), len(set(methods)))
        self.assertTrue(all("." in method for method in methods))

    def test_implemented_python_references_exist(self) -> None:
        for entry in self.manifest["entries"]:
            if not entry.get("semantic", True):
                continue
            if entry["status"] not in {"covered", "partial"}:
                continue
            for key in ("python_sync", "python_async"):
                references = entry.get(key, [])
                self.assertTrue(references, f"{entry['id']} has no {key} references")
                for reference in references:
                    with self.subTest(entry=entry["id"], reference=reference):
                        self.assertTrue(callable(_resolve(reference)))


if __name__ == "__main__":
    unittest.main()
