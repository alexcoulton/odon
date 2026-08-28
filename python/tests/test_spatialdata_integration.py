from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from odon.integrations import spatialdata as adapter


class Payload:
    def __init__(self, token: str) -> None:
        self.token = token
        self.shape = (1, 16, 16)
        self.dtype = "uint16"

    def compute(self):  # pragma: no cover - called only if the adapter regresses
        raise AssertionError("the adapter must not compute pixels")

    def __getitem__(self, slices):
        import numpy as np

        return np.zeros(self.shape, dtype=self.dtype)[slices]


class Transform:
    def __init__(self, matrix: list[list[float]]) -> None:
        self.matrix = matrix

    def to_affine_matrix(self, *, input_axes, output_axes):
        if input_axes != ("x", "y") or output_axes != ("x", "y"):
            raise AssertionError("unexpected axes")
        return self.matrix


class Element:
    def __init__(
        self,
        token: str,
        backing_file: Path,
        *,
        transforms: dict[str, Transform] | None = None,
    ) -> None:
        self.data = Payload(token)
        self.shape = (1, 16, 16)
        self.dtype = "uint16"
        self.dims = ("c", "y", "x")
        self.backing_file = backing_file
        self.transforms = transforms or {
            "global": Transform([[2.0, 0.0, 7.0], [0.0, 3.0, 11.0], [0.0, 0.0, 1.0]])
        }


class VectorElement:
    def __init__(self, token: str = "vector") -> None:
        self.token = token
        self.dims = ("x", "y")
        self.transforms = {
            "global": Transform([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        }

    def equals(self, other: object) -> bool:
        return isinstance(other, VectorElement) and other.token == self.token


class FakeSpatialData:
    def __init__(
        self,
        root: Path,
        images: dict[str, Element],
        labels: dict[str, Element] | None = None,
        shapes: dict[str, object] | None = None,
        points: dict[str, object] | None = None,
        tables: dict[str, object] | None = None,
    ) -> None:
        self.path = root
        self.images = images
        self.labels = labels or {}
        self.shapes = shapes or {}
        self.points = points or {}
        self.tables = tables or {}
        self.fresh: FakeSpatialData = self
        self.self_contained = True

    def is_backed(self) -> bool:
        return self.path is not None

    def is_self_contained(self, name: str) -> bool:
        return self.self_contained

    def locate_element(self, element: Element) -> list[str]:
        for kind in ("images", "labels", "shapes", "points", "tables"):
            for name, candidate in getattr(self, kind).items():
                if candidate is element:
                    return [f"{kind}/{name}"]
        return []

    def elements_paths_on_disk(self) -> list[str]:
        return [
            f"{kind}/{name}"
            for kind in ("images", "labels", "shapes", "points", "tables")
            for name in getattr(self, kind)
        ]


class Resource:
    def __init__(self, resource_id: str, calls: list[tuple[str, dict]]) -> None:
        self.resource_id = resource_id
        self.calls = calls

    def remove(self) -> None:
        self.calls.append(("data.resources.remove", {"resource_id": self.resource_id}))


class Layer:
    def __init__(self, layer_id: str, calls: list[tuple[str, dict]]) -> None:
        self.layer_id = layer_id
        self.calls = calls

    def remove(self) -> None:
        self.calls.append(("viewer.layers.remove", {"layer_id": self.layer_id}))


class Task:
    def __init__(self, layers=()) -> None:
        self.layers = list(layers)

    def wait(self, timeout=None):
        return {
            "resource_id": "resource:0",
            "model_ready": True,
            "resources_ready": True,
            "presentation_ready": False,
            "layers": [
                {
                    **layer,
                    "layer_id": layer.get("layer_id", f"layer:{index}"),
                    "revision": 1,
                    "order": index,
                }
                for index, layer in enumerate(self.layers)
            ],
        }


class FailingTask(Task):
    def wait(self, timeout=None):
        raise RuntimeError("synthetic open failure")


class DataApi:
    def __init__(self, calls: list[tuple[str, dict]]) -> None:
        self.calls = calls

    def register(self, root, **params):
        uri = root.resolve().as_uri() if isinstance(root, Path) else str(root)
        params = {"uri": uri, **params}
        self.calls.append(("data.resources.register", params))
        return Resource(f"resource:{len(self.calls)}", self.calls)


class LayerApi:
    def __init__(self, calls: list[tuple[str, dict]]) -> None:
        self.calls = calls

    def add(self, resource, **params):
        params = {"resource_id": resource.resource_id, **params}
        self.calls.append(("viewer.layers.add", params))
        return Layer(f"layer:{len(self.calls)}", self.calls)

    def remove(self, layer_id: str) -> None:
        self.calls.append(("viewer.layers.remove", {"layer_id": layer_id}))


class DatasetApi:
    def __init__(self, calls: list[tuple[str, dict]]) -> None:
        self.calls = calls

    def open_resource(self, resource, *, layers=()):
        specs = [layer.to_dict() for layer in layers]
        self.calls.append(
            (
                "datasets.open_resource",
                {"resource_id": resource.resource_id, "layers": specs},
            )
        )
        return Task(specs)


class ObjectApi:
    def __init__(self, calls: list[tuple[str, dict]]) -> None:
        self.calls = calls

    def get_selection(self, **selector):
        self.calls.append(("viewer.objects.get_selection", dict(selector)))
        return {
            "target": "spatial_shape",
            "layer_id": selector["layer_id"],
            "selection": {
                "selection_count": 2,
                "selected": [{"id": "cell-a"}, {"id": "cell-b"}],
                "primary": {"id": "cell-b"},
                "truncated": False,
                "generation": 4,
            },
        }


class Client:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.data = DataApi(self.calls)
        self.layers = LayerApi(self.calls)
        self.datasets = DatasetApi(self.calls)
        self.objects = ObjectApi(self.calls)


class AsyncResource(Resource):
    async def remove(self) -> None:
        super().remove()


class AsyncLayer(Layer):
    async def remove(self) -> None:
        super().remove()


class AsyncTask:
    def __init__(self, layers=()) -> None:
        self.layers = list(layers)

    async def wait(self, timeout=None):
        return Task(self.layers).wait(timeout)


class AsyncDataApi(DataApi):
    async def register(self, root, **params):
        uri = root.resolve().as_uri() if isinstance(root, Path) else str(root)
        params = {"uri": uri, **params}
        self.calls.append(("data.resources.register", params))
        return AsyncResource(f"resource:{len(self.calls)}", self.calls)


class AsyncLayerApi(LayerApi):
    async def add(self, resource, **params):
        params = {"resource_id": resource.resource_id, **params}
        self.calls.append(("viewer.layers.add", params))
        return AsyncLayer(f"layer:{len(self.calls)}", self.calls)

    async def remove(self, layer_id: str) -> None:
        self.calls.append(("viewer.layers.remove", {"layer_id": layer_id}))


class AsyncDatasetApi(DatasetApi):
    async def open_resource(self, resource, *, layers=()):
        specs = [layer.to_dict() for layer in layers]
        self.calls.append(
            (
                "datasets.open_resource",
                {"resource_id": resource.resource_id, "layers": specs},
            )
        )
        return AsyncTask(specs)


class AsyncObjectApi(ObjectApi):
    async def get_selection(self, **selector):
        return super().get_selection(**selector)


class AsyncClient(Client):
    def __init__(self) -> None:
        super().__init__()
        self.data = AsyncDataApi(self.calls)
        self.layers = AsyncLayerApi(self.calls)
        self.datasets = AsyncDatasetApi(self.calls)
        self.objects = AsyncObjectApi(self.calls)


def fake_api(sdata: FakeSpatialData) -> adapter._SpatialDataApi:
    def get_transformation(element, *, get_all=False, to_coordinate_system=None):
        if get_all:
            return element.transforms
        return element.transforms[to_coordinate_system]

    return adapter._SpatialDataApi(
        spatialdata_type=FakeSpatialData,
        version="0.8.0",
        get_axes_names=lambda element: element.dims,
        get_transformation=get_transformation,
        get_table_keys=lambda table: table.table_keys,
        get_dask_backing_files=lambda element: [str(element.backing_file)],
        read_zarr=lambda root, selection: sdata.fresh,
        tokenize=lambda payload: payload.token,
    )


def backed_fixture(root: Path) -> FakeSpatialData:
    image_path = root / "images" / "morphology" / "0"
    labels_path = root / "labels" / "nuclei" / "0"
    image_path.parent.mkdir(parents=True)
    labels_path.parent.mkdir(parents=True)
    image_path.touch()
    labels_path.touch()
    current = FakeSpatialData(
        root,
        {"morphology": Element("image", image_path)},
        {"nuclei": Element("labels", labels_path)},
    )
    current.fresh = FakeSpatialData(
        root,
        {"morphology": Element("image", image_path)},
        {"nuclei": Element("labels", labels_path)},
    )
    return current


class SpatialDataIntegrationTests(unittest.TestCase):
    def test_selected_names_must_be_unique_across_element_families(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sdata = backed_fixture(root)
            duplicate_path = root / "labels" / "morphology" / "0"
            duplicate_path.parent.mkdir(parents=True)
            duplicate_path.touch()
            sdata.labels["morphology"] = Element("duplicate", duplicate_path)
            client = Client()
            with patch.object(adapter, "_load_spatialdata_api", return_value=fake_api(sdata)):
                with self.assertRaises(adapter.SpatialDataAdapterError) as raised:
                    adapter.display(client, sdata, labels=["morphology"])
            self.assertEqual(raised.exception.code, "ambiguous_element_name")
            self.assertEqual(client.calls, [])

    def test_failed_promoted_open_removes_only_new_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            new_output = root / "new.ome.zarr"
            new_output.mkdir()
            marker = new_output / "0"
            marker.write_bytes(b"new")
            calls: list[tuple[str, dict]] = []
            element = adapter.ElementDisplay(
                name="morphology",
                kind="image",
                group_path=None,
                transfer=adapter.TransferDecision("snapshot", "snapshot", "test"),
                resource=Resource("resource:new", calls),
                source=new_output,
                format="ome-zarr",
                source_lifetime="durable",
                coordinate_space=adapter.CoordinateSpace(axes=("y", "x")),
                ownership="project",
            )
            shown = adapter.SpatialDataDisplay(
                task=FailingTask(),
                source_uri=new_output.as_uri(),
                coordinate_system="global",
                elements={"morphology": element},
                _app=Client(),
                _pending_created_sources=(new_output,),
            )
            with self.assertRaisesRegex(RuntimeError, "synthetic open failure"):
                shown.wait_ready()
            self.assertFalse(new_output.exists())

            source = root / "source.parquet"
            source.write_bytes(b"replacement")
            destination_root = root / "durable"
            destination_root.mkdir()
            existing = destination_root / "cells.parquet"
            existing.write_bytes(b"original")
            vector = adapter.ElementDisplay(
                name="cells",
                kind="shapes",
                group_path=None,
                transfer=adapter.TransferDecision("snapshot", "snapshot", "test"),
                resource=Resource("resource:cells", calls),
                source=source,
                format="geoparquet",
                source_lifetime="session",
                coordinate_space=adapter.CoordinateSpace(axes=("y", "x")),
                ownership="session",
            )
            durable, created = adapter._materialize_durable_sources(
                {"cells": vector}, (), destination_root, overwrite=True
            )
            self.assertEqual(Path(durable["cells"]).resolve(), existing.resolve())
            self.assertEqual(created, ())
            self.assertEqual(existing.read_bytes(), b"replacement")

    def test_real_spatialdata_backed_composite_and_table_projection(self) -> None:
        try:
            import anndata as ad
            import geopandas as gpd
            import numpy as np
            import pandas as pd
            import spatialdata
            from shapely.geometry import box
            from spatialdata import SpatialData
            from spatialdata.models import (
                Image2DModel,
                Labels2DModel,
                PointsModel,
                ShapesModel,
                TableModel,
            )
            from spatialdata.transformations import Identity
        except ImportError as error:  # pragma: no cover - optional dependency lane
            self.skipTest(f"SpatialData test dependencies are unavailable: {error}")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "scene.zarr"
            transform = {"global": Identity()}
            image = Image2DModel.parse(
                np.arange(64, dtype=np.uint16).reshape(1, 8, 8),
                dims=("c", "y", "x"),
                transformations=transform,
            )
            labels = Labels2DModel.parse(
                np.arange(64, dtype=np.uint16).reshape(8, 8),
                dims=("y", "x"),
                transformations=transform,
            )
            shapes = ShapesModel.parse(
                gpd.GeoDataFrame(
                    {"instance_id": [1, 2]},
                    geometry=[box(0, 0, 2, 2), box(4, 4, 7, 7)],
                ),
                transformations=transform,
            )
            points = PointsModel.parse(
                pd.DataFrame(
                    {"x": [1.0, 6.0], "y": [1.0, 6.0], "feature": ["a", "b"]}
                ),
                transformations=transform,
            )
            table = TableModel.parse(
                ad.AnnData(
                    X=np.zeros((2, 1), dtype=np.float32),
                    obs=pd.DataFrame(
                        {
                            "region": pd.Categorical(["cells", "cells"]),
                            "instance_id": [1, 2],
                            "score": [0.25, 0.75],
                        },
                        index=["row-a", "row-b"],
                    ),
                ),
                region="cells",
                region_key="region",
                instance_key="instance_id",
            )
            SpatialData(
                images={"morphology": image},
                labels={"nuclei": labels},
                shapes={"cells": shapes},
                points={"transcripts": points},
                tables={"table": table},
            ).write(root)
            backed = spatialdata.read_zarr(root)
            direct_shapes = adapter.display(
                Client(),
                backed,
                image="morphology",
                labels=["nuclei"],
                shapes=["cells"],
            )
            direct_shapes.wait_ready()
            self.assertEqual(
                direct_shapes.elements["cells"].transfer.chosen, "reference"
            )
            self.assertEqual(
                direct_shapes.elements["nuclei"].transfer.chosen, "reference"
            )
            direct_shapes.close()

            client = Client()
            result = adapter.display(
                client,
                backed,
                image="morphology",
                shapes=["cells"],
                points=["transcripts"],
                tables=["table"],
                table_columns={"table": ["score"]},
            )
            result.wait_ready()

            self.assertEqual(
                result.elements["morphology"].transfer.chosen,
                "reference",
                result.elements["morphology"].transfer.reason,
            )
            self.assertEqual(result.elements["cells"].transfer.chosen, "snapshot")
            self.assertIn("table", result.table_projections)
            projected = gpd.read_parquet(result.elements["cells"].source)
            self.assertEqual(projected["score"].tolist(), [0.25, 0.75])
            self.assertTrue(
                result.elements["cells"].layer.layer_id.startswith("spatial_shape:")
            )
            self.assertEqual(result.selected_ids("cells"), ("cell-a", "cell-b"))
            result.close()

    def test_backed_image_and_labels_use_reference_without_pixel_compute(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sdata = backed_fixture(Path(directory))
            self.assertTrue(
                adapter._source_contains(
                    Path(directory).resolve(),
                    sdata.images["morphology"].backing_file.as_uri(),
                )
            )
            client = Client()
            with patch.object(adapter, "_load_spatialdata_api", return_value=fake_api(sdata)):
                result = adapter.display(
                    client, sdata, image="morphology", labels=["nuclei"]
                )
                document = result.wait_ready()

            self.assertEqual(result.elements["morphology"].group_path, "images/morphology")
            self.assertEqual(result.elements["nuclei"].group_path, "labels/nuclei")
            self.assertIsNotNone(result.elements["nuclei"].layer)
            self.assertEqual(result.elements["morphology"].transfer.chosen, "reference")
            self.assertEqual(document["resource_id"], "resource:0")
            self.assertEqual(result.readiness["presentation"], False)
            methods = [method for method, _ in client.calls]
            self.assertEqual(methods.count("data.resources.register"), 2)
            self.assertIn("datasets.open_resource", methods)
            self.assertNotIn("datasets.open_spatialdata", methods)
            image_registration = client.calls[0][1]
            self.assertEqual(image_registration["group_path"], "images/morphology")
            self.assertEqual(
                image_registration["coordinate_space"].scale, (1.0, 1.0, 1.0)
            )

    def test_diverged_in_memory_raster_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sdata = backed_fixture(Path(directory))
            sdata.images["morphology"].data.token = "computed-image"
            client = Client()
            with patch.object(adapter, "_load_spatialdata_api", return_value=fake_api(sdata)):
                with self.assertRaises(adapter.SpatialDataAdapterError) as raised:
                    adapter.display(client, sdata, image="morphology", transfer="reference")
            self.assertEqual(raised.exception.code, "in_memory_raster_diverged")
            self.assertEqual(client.calls, [])

    def test_raster_snapshot_is_never_an_implicit_provider_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sdata = backed_fixture(Path(directory))
            client = Client()
            with patch.object(adapter, "_load_spatialdata_api", return_value=fake_api(sdata)):
                with self.assertRaises(adapter.SpatialDataAdapterError) as snapshot:
                    adapter.display(client, sdata, transfer="snapshot")
            self.assertEqual(snapshot.exception.code, "raster_snapshot_unavailable")

    def test_unbacked_raster_uses_session_provider_without_disk_materialization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sdata = backed_fixture(Path(directory))
            sdata.path = None
            client = Client()
            with patch.object(adapter, "_load_spatialdata_api", return_value=fake_api(sdata)):
                result = adapter.display(client, sdata, transfer="auto")
            provider = result._providers[0]
            self.assertEqual(result.elements["morphology"].transfer.chosen, "provider")
            self.assertTrue(provider.uri.startswith("http://127.0.0.1:"))
            self.assertEqual(client.calls[0][1]["uri"], provider.uri)
            with self.assertRaises(adapter.SpatialDataAdapterError) as raised:
                with patch.object(adapter, "_load_spatialdata_api", return_value=fake_api(sdata)):
                    adapter.display(client, sdata, transfer="provider", ownership="project")
            self.assertEqual(raised.exception.code, "ephemeral_ownership_invalid")
            persisted = result.persist(Path(directory) / "durable")
            persisted.wait_ready()
            snapshot = Path(directory) / "durable" / "morphology.ome.zarr"
            self.assertTrue(snapshot.is_dir())
            project_registrations = [
                params
                for method, params in client.calls
                if method == "data.resources.register" and params["ownership"] == "project"
            ]
            self.assertEqual(project_registrations[-1]["source_lifetime"], "durable")
            removals_before = len(
                [method for method, _ in client.calls if method == "data.resources.remove"]
            )
            persisted.close()
            removals_after = len(
                [method for method, _ in client.calls if method == "data.resources.remove"]
            )
            self.assertEqual(removals_before, removals_after)
            result.close()

    def test_compatible_shapes_reference_existing_geoparquet(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sdata = backed_fixture(root)
            shapes_path = root / "shapes" / "cells" / "shapes.parquet"
            shapes_path.parent.mkdir(parents=True)
            shapes_path.touch()
            sdata.shapes["cells"] = VectorElement()
            sdata.fresh.shapes["cells"] = VectorElement()
            client = Client()
            with patch.object(adapter, "_load_spatialdata_api", return_value=fake_api(sdata)):
                result = adapter.display(client, sdata, shapes=["cells"])
                result.wait_ready()

            shape = result.elements["cells"]
            self.assertEqual(shape.transfer.chosen, "reference")
            self.assertIsNone(shape.group_path)
            self.assertIsNotNone(shape.layer)
            self.assertTrue(shape.layer.layer_id.startswith("spatial_shape:"))
            selection = result.selection("cells")
            self.assertEqual(selection.ids, ("cell-a", "cell-b"))
            self.assertEqual(selection.primary_id, "cell-b")
            self.assertEqual(selection.selection_count, 2)
            registration = next(
                params
                for method, params in client.calls
                if method == "data.resources.register" and params["format"] == "geoparquet"
            )
            self.assertEqual(registration["uri"], shapes_path.resolve().as_uri())

    def test_selected_incompatible_vector_is_an_explicit_session_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sdata = backed_fixture(root)
            sdata.points["transcripts"] = VectorElement()
            sdata.fresh.points["transcripts"] = VectorElement()
            client = Client()

            def write_snapshot(kind, name, element, destination, joins):
                self.assertEqual((kind, name), ("points", "transcripts"))
                self.assertEqual(tuple(joins), ())
                destination.touch()

            with (
                patch.object(adapter, "_load_spatialdata_api", return_value=fake_api(sdata)),
                patch.object(adapter, "_write_vector_snapshot", side_effect=write_snapshot),
            ):
                result = adapter.display(
                    client,
                    sdata,
                    points=["transcripts"],
                    transfer={"transcripts": "snapshot", "*": "auto"},
                )
                result.wait_ready()
                temporary = result._temporary_directories[0]
                self.assertTrue(temporary.is_dir())
                self.assertEqual(result.elements["transcripts"].transfer.chosen, "snapshot")
                result.close()
                self.assertFalse(temporary.exists())

    def test_selected_table_requires_explicit_columns(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sdata = backed_fixture(Path(directory))
            table = SimpleNamespace(table_keys=("cells", "region", "instance_id"))
            sdata.tables["table"] = table
            client = Client()
            with patch.object(adapter, "_load_spatialdata_api", return_value=fake_api(sdata)):
                with self.assertRaises(adapter.SpatialDataAdapterError) as raised:
                    adapter.display(client, sdata, tables=["table"])
            self.assertEqual(raised.exception.code, "table_columns_required")

    def test_remote_backed_raster_preserves_store_uri_and_group(self) -> None:
        root = "https://example.test/sample.zarr"
        image = Element("image", Path("unused"))
        image.backing_file = f"{root}/images/morphology/0"
        sdata = FakeSpatialData(root, {"morphology": image})  # type: ignore[arg-type]
        fresh_image = Element("image", Path("unused"))
        fresh_image.backing_file = image.backing_file
        sdata.fresh = FakeSpatialData(root, {"morphology": fresh_image})  # type: ignore[arg-type]
        client = Client()
        with patch.object(adapter, "_load_spatialdata_api", return_value=fake_api(sdata)):
            result = adapter.display(client, sdata)
        self.assertEqual(result.source_uri, root)
        self.assertEqual(client.calls[0][1]["uri"], root)
        self.assertEqual(client.calls[0][1]["group_path"], "images/morphology")

    def test_s3_secondary_raster_fails_before_misleading_presentation(self) -> None:
        root = "s3://example-bucket/sample.zarr"
        image = Element("image", Path("unused"))
        labels = Element("labels", Path("unused"))
        image.backing_file = f"{root}/images/morphology/0"
        labels.backing_file = f"{root}/labels/nuclei/0"
        sdata = FakeSpatialData(  # type: ignore[arg-type]
            root, {"morphology": image}, {"nuclei": labels}
        )
        sdata.fresh = FakeSpatialData(  # type: ignore[arg-type]
            root, {"morphology": image}, {"nuclei": labels}
        )
        with patch.object(adapter, "_load_spatialdata_api", return_value=fake_api(sdata)):
            with self.assertRaises(adapter.SpatialDataAdapterError) as raised:
                adapter.display(Client(), sdata, labels=["nuclei"])
        self.assertEqual(raised.exception.code, "s3_secondary_raster_unavailable")

    def test_s3_primary_remains_session_lifetime_with_session_credentials(self) -> None:
        root = "s3://example-bucket/sample.zarr"
        image = Element("image", Path("unused"))
        image.backing_file = f"{root}/images/morphology/0"
        sdata = FakeSpatialData(root, {"morphology": image})  # type: ignore[arg-type]
        fresh = Element("image", Path("unused"))
        fresh.backing_file = image.backing_file
        sdata.fresh = FakeSpatialData(root, {"morphology": fresh})  # type: ignore[arg-type]
        client = Client()
        with patch.object(adapter, "_load_spatialdata_api", return_value=fake_api(sdata)):
            shown = adapter.display(client, sdata)
            with self.assertRaises(adapter.SpatialDataAdapterError) as raised:
                adapter.display(Client(), sdata, ownership="project")
        self.assertEqual(shown.elements["morphology"].source_lifetime, "session")
        self.assertEqual(client.calls[0][1]["source_lifetime"], "session")
        self.assertEqual(raised.exception.code, "ephemeral_ownership_invalid")
        shown.close()

    def test_secondary_rotation_and_shear_are_normalized_relative_to_primary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sdata = backed_fixture(Path(directory))
            sdata.images["morphology"].transforms["global"] = Transform(
                [[2.0, 0.0, 10.0], [0.0, 2.0, 20.0], [0.0, 0.0, 1.0]]
            )
            sdata.labels["nuclei"].transforms["global"] = Transform(
                [[0.0, -4.0, 14.0], [2.0, 1.0, 24.0], [0.0, 0.0, 1.0]]
            )
            sdata.fresh.images["morphology"].transforms = sdata.images[
                "morphology"
            ].transforms
            sdata.fresh.labels["nuclei"].transforms = sdata.labels["nuclei"].transforms
            client = Client()
            with patch.object(adapter, "_load_spatialdata_api", return_value=fake_api(sdata)):
                adapter.display(client, sdata, labels=["nuclei"])
            label_registration = client.calls[1][1]
            self.assertEqual(
                label_registration["coordinate_space"].affine.matrix,
                ((0.0, -2.0, 2.0), (1.0, 0.5, 2.0), (0.0, 0.0, 1.0)),
            )

    def test_refresh_commits_a_new_generation_before_releasing_the_old_one(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sdata = backed_fixture(Path(directory))
            client = Client()
            with patch.object(adapter, "_load_spatialdata_api", return_value=fake_api(sdata)):
                current = adapter.display(client, sdata, labels=["nuclei"])
                current.wait_ready()
                refreshed = current.refresh(sdata)
            methods = [method for method, _ in client.calls]
            second_open = [
                index
                for index, method in enumerate(methods)
                if method == "datasets.open_resource"
            ][1]
            first_remove = methods.index("viewer.layers.remove")
            self.assertLess(second_open, first_remove)
            self.assertTrue(current._closed)
            self.assertFalse(refreshed._closed)
            refreshed.close()


class AsyncSpatialDataIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_async_adapter_has_the_same_generic_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sdata = backed_fixture(Path(directory))
            client = AsyncClient()
            with patch.object(adapter, "_load_spatialdata_api", return_value=fake_api(sdata)):
                result = await adapter.display_async(client, sdata, labels="nuclei")
                await result.wait_ready()
            methods = [method for method, _ in client.calls]
            self.assertIn("datasets.open_resource", methods)
            self.assertNotIn("datasets.open_spatialdata", methods)


if __name__ == "__main__":
    unittest.main()
