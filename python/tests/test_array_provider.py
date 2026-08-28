from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import numpy as np

from odon.providers import ArrayProvider


class CountingArray:
    def __init__(self, values: np.ndarray) -> None:
        self.values = values
        self.shape = values.shape
        self.dtype = values.dtype
        self.requests: list[tuple[slice, ...]] = []

    def __getitem__(self, slices):
        self.requests.append(slices)
        return self.values[slices]


class FailingArray(CountingArray):
    def __getitem__(self, slices):
        if self.requests:
            raise RuntimeError("synthetic chunk failure")
        return super().__getitem__(slices)


class ArrayProviderTests(unittest.TestCase):
    def test_invalid_scheduling_limits_are_rejected(self) -> None:
        values = np.zeros((2, 2), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "max_concurrency"):
            ArrayProvider(values, axes=("y", "x"), max_concurrency=0)
        with self.assertRaisesRegex(ValueError, "cache_chunks"):
            ArrayProvider(values, axes=("y", "x"), cache_chunks=-1)

    def test_failed_overwrite_preserves_existing_destination(self) -> None:
        source = FailingArray(np.arange(16, dtype=np.uint16).reshape(4, 4))
        provider = ArrayProvider(source, axes=("y", "x"), chunks=(2, 2))
        try:
            with tempfile.TemporaryDirectory() as directory:
                destination = Path(directory) / "snapshot.ome.zarr"
                destination.mkdir()
                marker = destination / "existing.txt"
                marker.write_text("keep", encoding="utf-8")
                with self.assertRaises(RuntimeError):
                    provider.materialize(destination, overwrite=True)
                self.assertEqual(marker.read_text(encoding="utf-8"), "keep")
        finally:
            provider.close()

    def test_loopback_zarr_serves_only_requested_chunks_and_caches_them(self) -> None:
        source = CountingArray(np.arange(30, dtype=np.uint16).reshape(5, 6))
        provider = ArrayProvider(
            source, axes=("y", "x"), chunks=(3, 4), cache_chunks=2
        )
        try:
            with urlopen(f"{provider.uri}/.zattrs", timeout=2) as response:
                attrs = json.load(response)
            self.assertEqual(attrs["multiscales"][0]["axes"][0]["name"], "y")

            with urlopen(f"{provider.uri}/0/1.1", timeout=2) as response:
                first = response.read()
            with urlopen(f"{provider.uri}/0/1.1", timeout=2) as response:
                second = response.read()
            self.assertEqual(first, second)
            self.assertEqual(len(source.requests), 1)
            decoded = np.frombuffer(first, dtype=np.uint16).reshape(3, 4)
            np.testing.assert_array_equal(decoded[:2, :2], source.values[3:5, 4:6])
            np.testing.assert_array_equal(decoded[2:, :], np.zeros((1, 4), dtype=np.uint16))

            secretless = provider.uri.rsplit("/", 2)[0] + "/wrong/1/.zattrs"
            with self.assertRaises(HTTPError) as denied:
                urlopen(secretless, timeout=2)
            self.assertEqual(denied.exception.code, 404)
            with tempfile.TemporaryDirectory() as directory:
                snapshot = provider.materialize(f"{directory}/snapshot.ome.zarr")
                import zarr

                persisted = zarr.open_group(str(snapshot), mode="r")["0"][:]
                np.testing.assert_array_equal(persisted, source.values)
        finally:
            provider.close()
        with self.assertRaises(URLError):
            urlopen(f"{provider.uri}/.zattrs", timeout=1)


if __name__ == "__main__":
    unittest.main()
