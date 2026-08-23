from __future__ import annotations

import unittest

from odon.async_resources import AsyncAnnotations
from odon.resources import Annotations


class _Tasks:
    def __init__(self, calls: list[tuple[str, dict]]) -> None:
        self.calls = calls

    def start(self, method: str, params: dict, **_: object) -> dict:
        self.calls.append((method, params))
        return {"task": method}


class _Client:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.tasks = _Tasks(self.calls)

    def call(self, method: str, params: dict | None = None) -> dict:
        self.calls.append((method, params or {}))
        return {"method": method}


class AnnotationResourceTests(unittest.TestCase):
    def test_crud_and_source_tasks(self) -> None:
        client = _Client()
        annotations = Annotations(client)  # type: ignore[arg-type]

        annotations.list_layers()
        annotations.get_layer(2)
        annotations.create_layer("Cells", opacity=0.5, if_revision=3)
        annotations.update_layer(2, visible=False)
        annotations.inspect(2, "cells.parquet")
        annotations.load(2, "cells.parquet", value_column="phenotype")
        annotations.reload(2)
        annotations.clear_source(2)
        annotations.delete_layer(2)

        self.assertEqual(client.calls[0], ("viewer.annotations.layers.list", {}))
        self.assertEqual(client.calls[2][1]["if_revision"], 3)
        self.assertEqual(client.calls[3][1]["state"], {"visible": False})
        self.assertEqual(client.calls[4][0], "viewer.annotations.source.inspect")
        self.assertEqual(client.calls[5][1]["value_column"], "phenotype")
        self.assertEqual(client.calls[6][0], "viewer.annotations.source.reload")
        self.assertEqual(client.calls[8][0], "viewer.annotations.layers.delete")


class _AsyncTasks:
    def __init__(self, calls: list[tuple[str, dict]]) -> None:
        self.calls = calls

    async def start(self, method: str, params: dict, **_: object) -> dict:
        self.calls.append((method, params))
        return {"task": method}


class _AsyncClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.tasks = _AsyncTasks(self.calls)

    async def call(self, method: str, params: dict | None = None) -> dict:
        self.calls.append((method, params or {}))
        return {"method": method}


class AsyncAnnotationResourceTests(unittest.IsolatedAsyncioTestCase):
    async def test_async_crud_and_load(self) -> None:
        client = _AsyncClient()
        annotations = AsyncAnnotations(client)  # type: ignore[arg-type]

        await annotations.create_layer("Cells")
        await annotations.load(1, "cells.parquet")
        await annotations.update_layer(1, radius_screen_px=8.0)
        await annotations.delete_layer(1)

        self.assertEqual(client.calls[0][0], "viewer.annotations.layers.create")
        self.assertEqual(client.calls[1][0], "viewer.annotations.source.load")
        self.assertEqual(client.calls[2][1]["state"], {"radius_screen_px": 8.0})
        self.assertEqual(client.calls[3][0], "viewer.annotations.layers.delete")


if __name__ == "__main__":
    unittest.main()
