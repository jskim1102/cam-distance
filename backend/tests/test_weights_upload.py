"""시스템 전역 YOLO 가중치 업로드 계약."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import config, inference_api
from app.inference import models_dir
from app.inference.worker import Detection, InferenceResult, InferenceWorker
from app.streaming.manager import StreamManager, detections_to_json


@pytest.fixture()
def weights_dir(monkeypatch, tmp_path: Path) -> Path:
    monkeypatch.setattr(config, "WEIGHTS_DIR", tmp_path)
    return tmp_path


@pytest.fixture()
def client(
    monkeypatch,
    weights_dir: Path,
) -> tuple[TestClient, MagicMock, MagicMock, list[tuple[object, ...]]]:
    classes = [{"id": 0, "name": "forklift"}, {"id": 4, "name": "pallet"}]
    monkeypatch.setattr(models_dir, "extract_model_classes", lambda _path: classes)

    activation_calls: list[tuple[object, ...]] = []
    current_model = [config.YOLO_DEFAULT_MODEL]

    def record_model(name: str, *, force_reload: bool = False) -> None:
        activation_calls.append(("model", name, force_reload))
        current_model[0] = name

    set_model = MagicMock(side_effect=record_model)
    set_all_sources = MagicMock(
        side_effect=lambda names: activation_calls.append(("sources", list(names)))
    )
    get_config = MagicMock(side_effect=lambda: {
        "enabled": True,
        "model": current_model[0],
        "conf_threshold": 0.5,
        "device": "auto",
        "gpu_util_target": 1.0,
        "gpu_util_duty": 0.0,
    })
    monkeypatch.setattr(inference_api.stream_manager, "set_inference_model", set_model)
    monkeypatch.setattr(inference_api.stream_manager, "get_inference_config", get_config)
    monkeypatch.setattr(
        inference_api.stream_manager,
        "set_all_source_models",
        set_all_sources,
        raising=False,
    )

    app = FastAPI()
    app.include_router(inference_api.inference_router)
    return TestClient(app), set_model, set_all_sources, activation_calls


def test_default_weights_status_uses_yolo26x_and_exposes_coco_classes(
    weights_dir: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(config, "YOLO_DEFAULT_MODEL", "yolo26x.pt")
    status = models_dir.get_active_weights_status()

    assert status == {
        "active_name": "yolo26x.pt",
        "default_name": "yolo26x.pt",
        "type": "preset",
        "uploaded_at": None,
        "size_mb": None,
        "class_count": 80,
    }
    assert models_dir.get_active_model_name() == "yolo26x.pt"
    assert models_dir.get_active_model_names() == ["yolo26x.pt"]
    assert len(models_dir.list_active_classes()) == 80
    assert models_dir.list_active_classes()[0] == {"id": 0, "name": "person"}


@pytest.mark.parametrize(
    ("old_models", "active_name", "force_reload", "allowed_models"),
    [
        (["yolo26x.pt"], "best.pt", True, {"yolo26x.pt", "best.pt"}),
        (["best.pt"], "yolo26x.pt", False, {"yolo26x.pt"}),
    ],
)
def test_refresh_source_models_never_syncs_a_stale_or_empty_lane(
    monkeypatch,
    old_models: list[str],
    active_name: str,
    force_reload: bool,
    allowed_models: set[str],
) -> None:
    class SyncingManager:
        def __init__(self) -> None:
            self.source_models = list(old_models)
            self.current_model = old_models[0]
            self.transitions: list[list[str]] = []

        def get_inference_config(self) -> dict[str, str]:
            return {"model": self.current_model}

        def sync(self) -> None:
            self.transitions.append(
                [name for name in self.source_models if name in allowed_models]
            )

        def set_all_source_models(self, names: list[str]) -> None:
            self.source_models = list(names)
            self.sync()

        def set_inference_model(
            self,
            _name: str,
            *,
            force_reload: bool = False,
        ) -> None:
            del force_reload
            self.current_model = _name
            self.sync()

    manager = SyncingManager()
    monkeypatch.setattr(inference_api, "stream_manager", manager)
    monkeypatch.setattr(models_dir, "get_active_model_name", lambda: active_name)

    inference_api._refresh_source_models(force_reload=force_reload)

    assert manager.transitions == [[active_name], [active_name]]


@pytest.mark.parametrize(
    ("current_name", "active_name", "requested_force_reload", "expected_force_reload"),
    [
        ("yolo26x.pt", "best.pt", True, False),
        ("best.pt", "best.pt", True, True),
        ("best.pt", "yolo26x.pt", False, False),
    ],
)
def test_refresh_source_models_loads_the_active_lane_once(
    monkeypatch,
    current_name: str,
    active_name: str,
    requested_force_reload: bool,
    expected_force_reload: bool,
) -> None:
    class CountingManager:
        def __init__(self) -> None:
            self.current_model = current_name
            self.lanes = {current_name}
            self.loads: list[str] = []
            self.force_reload_values: list[bool] = []

        def get_inference_config(self) -> dict[str, str]:
            return {"model": self.current_model}

        def set_all_source_models(self, names: list[str]) -> None:
            desired = set(names)
            for name in desired - self.lanes:
                self.loads.append(name)
            self.lanes = desired

        def set_inference_model(
            self,
            name: str,
            *,
            force_reload: bool = False,
        ) -> None:
            self.force_reload_values.append(force_reload)
            if force_reload and name in self.lanes:
                self.loads.append(name)
            self.current_model = name

    manager = CountingManager()
    monkeypatch.setattr(inference_api, "stream_manager", manager)
    monkeypatch.setattr(models_dir, "get_active_model_name", lambda: active_name)

    inference_api._refresh_source_models(force_reload=requested_force_reload)

    assert manager.loads == [active_name]
    assert manager.force_reload_values == [expected_force_reload]


def test_upload_persists_custom_weights_and_replaces_the_single_active_model(
    client: tuple[TestClient, MagicMock, MagicMock, list[tuple[object, ...]]],
    weights_dir: Path,
) -> None:
    http, set_model, set_all_sources, activation_calls = client

    response = http.post(
        "/api/inference/weights",
        files={"file": ("warehouse-v3.pt", b"model-bytes", "application/octet-stream")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["active_name"] == "warehouse-v3.pt"
    assert body["type"] == "custom"
    assert body["class_count"] == 2
    assert body["size_mb"] == pytest.approx(
        len(b"model-bytes") / 1024 / 1024
    )
    assert body["uploaded_at"].endswith("Z")
    assert (weights_dir / "custom.pt").read_bytes() == b"model-bytes"
    metadata = json.loads((weights_dir / "custom.json").read_text(encoding="utf-8"))
    assert metadata == {
        "original_name": "warehouse-v3.pt",
        "uploaded_at": body["uploaded_at"],
        "size_bytes": len(b"model-bytes"),
        "classes": [{"id": 0, "name": "forklift"}, {"id": 4, "name": "pallet"}],
    }
    set_model.assert_called_once_with("warehouse-v3.pt", force_reload=False)
    set_all_sources.assert_called_once_with(["warehouse-v3.pt"])
    assert activation_calls == [
        ("sources", ["warehouse-v3.pt"]),
        ("model", "warehouse-v3.pt", False),
    ]

    assert http.get("/api/inference/weights").json() == body
    assert http.get("/api/inference/classes").json() == metadata["classes"]
    assert models_dir.get_active_model_name() == "warehouse-v3.pt"
    assert models_dir.get_active_model_names() == ["warehouse-v3.pt"]
    assert models_dir.resolve_model_path("warehouse-v3.pt") == str(weights_dir / "custom.pt")
    assert http.put(
        "/api/inference/config",
        json={"model": "warehouse-v3.pt"},
    ).status_code == 200
    assert http.put(
        "/api/inference/config",
        json={"model": "yolo26x.pt"},
    ).status_code == 400


def test_reuploading_the_same_named_weights_force_reloads_once(
    client: tuple[TestClient, MagicMock, MagicMock, list[tuple[object, ...]]],
    weights_dir: Path,
) -> None:
    http, set_model, set_all_sources, activation_calls = client
    assert http.post(
        "/api/inference/weights",
        files={"file": ("warehouse.pt", b"first", "application/octet-stream")},
    ).status_code == 200
    set_model.reset_mock()
    set_all_sources.reset_mock()
    activation_calls.clear()

    response = http.post(
        "/api/inference/weights",
        files={"file": ("warehouse.pt", b"second", "application/octet-stream")},
    )

    assert response.status_code == 200
    assert (weights_dir / "custom.pt").read_bytes() == b"second"
    set_all_sources.assert_called_once_with(["warehouse.pt"])
    set_model.assert_called_once_with("warehouse.pt", force_reload=True)
    assert activation_calls == [
        ("sources", ["warehouse.pt"]),
        ("model", "warehouse.pt", True),
    ]


@pytest.mark.parametrize("default_name", ["yolo26x.pt", "yolo26n.pt"])
def test_delete_custom_weights_restores_configured_preset(
    client: tuple[TestClient, MagicMock, MagicMock, list[tuple[object, ...]]],
    weights_dir: Path,
    monkeypatch,
    default_name: str,
) -> None:
    monkeypatch.setattr(config, "YOLO_DEFAULT_MODEL", default_name)
    http, set_model, set_all_sources, activation_calls = client
    assert models_dir.get_active_model_name() == default_name
    assert http.get("/api/inference/weights").json()["active_name"] == default_name
    upload = http.post(
        "/api/inference/weights",
        files={"file": ("custom-model.pt", b"weights", "application/octet-stream")},
    )
    assert upload.status_code == 200
    assert upload.json()["active_name"] == "custom-model.pt"
    assert upload.json()["default_name"] == default_name
    set_model.reset_mock()
    set_all_sources.reset_mock()
    activation_calls.clear()

    response = http.delete("/api/inference/weights")

    assert response.status_code == 200
    assert response.json() == {
        "active_name": default_name,
        "default_name": default_name,
        "type": "preset",
        "uploaded_at": None,
        "size_mb": None,
        "class_count": 80,
    }
    assert not (weights_dir / "custom.pt").exists()
    assert not (weights_dir / "custom.json").exists()
    set_model.assert_called_once_with(default_name, force_reload=False)
    set_all_sources.assert_called_once_with([default_name])
    assert activation_calls == [
        ("sources", [default_name]),
        ("model", default_name, False),
    ]
    classes_response = http.get("/api/inference/classes")
    assert classes_response.status_code == 200
    assert len(classes_response.json()) == 80


def test_upload_reports_applied_weights_when_lane_refresh_fails(
    client: tuple[TestClient, MagicMock, MagicMock, list[tuple[object, ...]]],
    weights_dir: Path,
) -> None:
    http, set_model, set_all_sources, _activation_calls = client
    set_all_sources.side_effect = ValueError("lane refresh failed")

    response = http.post(
        "/api/inference/weights",
        files={"file": ("warehouse.pt", b"new-weights", "application/octet-stream")},
    )

    assert response.status_code == 202
    body = response.json()
    assert body["active_name"] == "warehouse.pt"
    assert "다시 시도" in body["refresh_error"]
    assert (weights_dir / "custom.pt").read_bytes() == b"new-weights"
    assert http.get("/api/inference/weights").json()["active_name"] == body["active_name"]
    set_model.assert_not_called()


def test_delete_reports_preset_as_active_when_lane_refresh_fails(
    client: tuple[TestClient, MagicMock, MagicMock, list[tuple[object, ...]]],
    weights_dir: Path,
) -> None:
    http, set_model, set_all_sources, _activation_calls = client
    (weights_dir / "custom.pt").write_bytes(b"weights")
    (weights_dir / "custom.json").write_text(
        json.dumps(
            {
                "original_name": "warehouse.pt",
                "uploaded_at": "2026-08-27T00:00:00Z",
                "size_bytes": 7,
                "classes": [{"id": 0, "name": "forklift"}],
            }
        ),
        encoding="utf-8",
    )
    set_model.side_effect = ValueError("lane refresh failed")

    response = http.delete("/api/inference/weights")

    assert response.status_code == 202
    body = response.json()
    assert body["active_name"] == body["default_name"] == config.YOLO_DEFAULT_MODEL
    assert "다시 시도" in body["refresh_error"]
    assert not (weights_dir / "custom.pt").exists()
    assert not (weights_dir / "custom.json").exists()
    assert http.get("/api/inference/weights").json()["active_name"] == body["active_name"]
    set_all_sources.assert_called_once_with([config.YOLO_DEFAULT_MODEL])


def test_invalid_extension_and_oversized_upload_leave_no_files(
    client: tuple[TestClient, MagicMock, MagicMock, list[tuple[object, ...]]],
    weights_dir: Path,
    monkeypatch,
) -> None:
    http, set_model, set_all_sources, activation_calls = client

    response = http.post(
        "/api/inference/weights",
        files={"file": ("weights.onnx", b"not-pt", "application/octet-stream")},
    )
    assert response.status_code == 400

    response = http.post(
        "/api/inference/weights",
        files={"file": ("yolo26x.pt", b"alias-collision", "application/octet-stream")},
    )
    assert response.status_code == 400
    assert "preset" in response.json()["detail"]

    monkeypatch.setattr(models_dir, "MAX_UPLOAD_BYTES", 3)
    response = http.post(
        "/api/inference/weights",
        files={"file": ("too-large.pt", b"1234", "application/octet-stream")},
    )
    assert response.status_code == 400
    assert list(weights_dir.iterdir()) == []
    set_model.assert_not_called()
    set_all_sources.assert_not_called()
    assert activation_calls == []


def test_class_extraction_failure_removes_temporary_upload(
    client: tuple[TestClient, MagicMock, MagicMock, list[tuple[object, ...]]],
    weights_dir: Path,
    monkeypatch,
) -> None:
    http, set_model, set_all_sources, activation_calls = client

    def fail_extract(_path: Path) -> list[dict]:
        raise ValueError("not a detection model")

    monkeypatch.setattr(models_dir, "extract_model_classes", fail_extract)
    response = http.post(
        "/api/inference/weights",
        files={"file": ("broken.pt", b"bad", "application/octet-stream")},
    )

    assert response.status_code == 400
    assert list(weights_dir.iterdir()) == []
    set_model.assert_not_called()
    set_all_sources.assert_not_called()
    assert activation_calls == []


def test_extract_model_classes_runs_ultralytics_in_short_subprocess(monkeypatch, tmp_path: Path) -> None:
    weight_path = tmp_path / "candidate.pt"
    weight_path.write_bytes(b"weights")
    completed = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout='ultralytics log\n[{"id": 2, "name": "car"}]\n',
        stderr="",
    )
    run = MagicMock(return_value=completed)
    monkeypatch.setattr(models_dir.subprocess, "run", run)

    assert models_dir.extract_model_classes(weight_path) == [{"id": 2, "name": "car"}]
    command = run.call_args.args[0]
    assert command[-1] == str(weight_path)
    assert "ultralytics" in command[2]
    assert run.call_args.kwargs["timeout"] <= 60


def test_only_presets_and_the_active_custom_alias_can_resolve(weights_dir: Path) -> None:
    (weights_dir / "custom.pt").write_bytes(b"weights")
    (weights_dir / "custom.json").write_text(
        json.dumps(
            {
                "original_name": "site-model.pt",
                "uploaded_at": "2026-08-18T00:00:00Z",
                "size_bytes": 7,
                "classes": [{"id": 0, "name": "person"}],
            }
        ),
        encoding="utf-8",
    )

    assert models_dir.is_preset("yolo26n.pt")
    assert not models_dir.is_preset("site-model.pt")
    assert models_dir.is_allowed_model("site-model.pt")
    assert models_dir.resolve_model_path("site-model.pt") == str(weights_dir / "custom.pt")
    assert models_dir.resolve_model_path("yolo26n.pt") == str(config.YOLO_WEIGHTS_DIR / "yolo26n.pt")
    for rejected in ("custom.pt", "/tmp/site-model.pt", "../site-model.pt", "other.pt"):
        assert not models_dir.is_allowed_model(rejected)
        with pytest.raises(ValueError):
            models_dir.resolve_model_path(rejected)


def test_existing_custom_metadata_restores_active_model_after_restart(weights_dir: Path) -> None:
    (weights_dir / "custom.pt").write_bytes(b"weights")
    (weights_dir / "custom.json").write_text(
        json.dumps(
            {
                "original_name": "persisted.pt",
                "uploaded_at": "2026-08-18T01:02:03Z",
                "size_bytes": 7,
                "classes": [{"id": 0, "name": "worker"}],
            }
        ),
        encoding="utf-8",
    )

    assert models_dir.get_active_model_name() == "persisted.pt"
    assert models_dir.get_active_model_names() == ["persisted.pt"]
    assert models_dir.get_active_weights_status()["active_name"] == "persisted.pt"
    assert models_dir.list_active_classes() == [{"id": 0, "name": "worker"}]

    worker = InferenceWorker()
    assert worker.get_status()["model"] == "persisted.pt"
    worker.configure_models({"persisted.pt", "../escape.pt"})
    assert worker.get_pool_status()["models"] == ["persisted.pt"]

    manager = StreamManager()
    assert manager.get_source_models("ipcam-after-restart") == ["persisted.pt"]


def test_force_reload_replaces_same_named_active_worker_lane(weights_dir: Path) -> None:
    (weights_dir / "custom.pt").write_bytes(b"weights")
    (weights_dir / "custom.json").write_text(
        json.dumps(
            {
                "original_name": "site-model.pt",
                "uploaded_at": "2026-08-18T01:02:03Z",
                "size_bytes": 7,
                "classes": [{"id": 0, "name": "forklift"}],
            }
        ),
        encoding="utf-8",
    )
    worker = InferenceWorker(model_name="site-model.pt")

    class ExistingLane:
        stopped = False

        def stop(self) -> None:
            self.stopped = True

    existing = ExistingLane()
    worker._desired_models = {"site-model.pt"}
    worker._lanes["site-model.pt"] = existing  # type: ignore[assignment]
    worker.set_model("site-model.pt", force_reload=True)

    assert existing.stopped
    assert worker._lanes.get("site-model.pt") is not existing
    assert worker.get_pool_status()["models"] == ["site-model.pt"]


def test_all_source_models_recalculate_to_the_single_active_model(weights_dir: Path) -> None:
    manager = StreamManager()
    manager._per_source_models = {
        "ipcam-a": ["yolo26n.pt"],
        "ipcam-b": [],
    }
    manager._recompute_cadence = MagicMock()  # type: ignore[method-assign]

    manager.set_all_source_models(["active.pt"])

    assert manager._per_source_models == {
        "ipcam-a": ["active.pt"],
        "ipcam-b": ["active.pt"],
    }
    assert manager._default_source_models == ["active.pt"]
    manager._recompute_cadence.assert_called_once()

    with pytest.raises(ValueError, match="하나"):
        manager.set_all_source_models(["yolo26x.pt", "active.pt"])
    with pytest.raises(ValueError, match="하나"):
        manager.set_source_models("ipcam-c", ["yolo26x.pt", "active.pt"])


def test_detection_payload_preserves_each_lane_model_name() -> None:
    payload = json.loads(
        detections_to_json(
            InferenceResult(
                source_id="ipcam-a",
                timestamp=1.0,
                frame_w=640,
                frame_h=360,
                detections=[
                    Detection(0, "person", 0.9, (0, 0, 10, 20), "yolo26x.pt"),
                    Detection(0, "forklift", 0.8, (20, 0, 40, 20), "warehouse.pt"),
                ],
            )
        )
    )

    assert [item["model"] for item in payload["items"]] == [
        "yolo26x.pt",
        "warehouse.pt",
    ]
