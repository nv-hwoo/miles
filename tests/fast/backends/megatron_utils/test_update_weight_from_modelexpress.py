import threading
from argparse import Namespace

import pytest

from miles.backends.megatron_utils.update_weight import update_weight_from_modelexpress as mx_module
from miles.backends.megatron_utils.update_weight.update_weight_from_modelexpress import (
    ModelExpressUpdateError,
    UpdateWeightFromModelExpress,
)


class RemoteMethod:
    def __init__(self, fn):
        self._fn = fn

    def remote(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


class FakeCatalog:
    def __init__(self):
        self.commits = []

    def commit_revision(self, model_id, version):
        self.commits.append((model_id, version))


class FakePublisher:
    def __init__(self):
        self.catalog = FakeCatalog()
        self.target_digest = "sha256:launch"
        self.pending_digest = None
        self.publishes = []
        self.baselines = []
        self.waits = []
        self.metrics = {
            "perf/update_weights_density": 0.25,
            "perf/update_weights_wire_bytes": 123.0,
            "perf/mx_encode_delta": 4.0,
            "perf/mx_publish_time": 5.0,
        }

    def publish_version(self, version, **kwargs):
        self.publishes.append((version, kwargs, threading.current_thread().name))
        if version != "0":
            self.pending_digest = f"sha256:{version}"

    def capture_baseline(self, gather, read):
        self.baselines.append((gather, read, threading.current_thread().name))

    def wait_for_commit(self, version, completion=None):
        if completion is not None:
            completion.result()
        self.waits.append(version)
        if version != "0":
            self.target_digest = self.pending_digest
            self.pending_digest = None

    def pop_metrics(self):
        metrics, self.metrics = self.metrics, {}
        return metrics


class FakeEngine:
    def __init__(self, events, install_success=True):
        self.version = "0"
        self.digest = "sha256:launch"
        self.install_success = install_success
        self.events = events
        self.prepare_weights_from_modelexpress = RemoteMethod(self._prepare)
        self.pause_generation = RemoteMethod(self._pause)
        self.flush_cache = RemoteMethod(self._flush)
        self.update_weights_from_modelexpress = RemoteMethod(self._install)
        self.get_modelexpress_status = RemoteMethod(self._status)
        self.continue_generation = RemoteMethod(self._continue)

    def _event(self, name):
        self.events.append((name, threading.current_thread().name))

    def _prepare(self, target):
        self._event(f"prepare:{target}")
        return {
            "success": True,
            "metrics": {"perf/mx_receive_prepare_time": 2.0},
        }

    def _pause(self, mode):
        self._event(f"pause:{mode}")

    def _flush(self):
        self._event("flush")

    def _install(self, target):
        self._event(f"install:{target}")
        if self.install_success:
            self.version = target
            self.digest = f"sha256:{target}"
        return {
            "success": self.install_success,
            "installed_version": self.version,
            "detail": "install failed" if not self.install_success else "",
            "metrics": {"perf/mx_receive_install_time": 3.0},
        }

    def _status(self):
        return {
            "success": True,
            "installed_version": self.version,
            "target_digest": self.digest,
            "state": "VERIFIED",
        }

    def _continue(self):
        self._event("continue")


def args():
    return Namespace(
        hf_checkpoint="/models/model",
        modelexpress_config={
            "initial_version": "0",
            "model_id": "policy",
        },
        pause_generation_mode="retract",
    )


@pytest.fixture(autouse=True)
def patch_runtime(monkeypatch):
    monkeypatch.setattr(mx_module.ray, "get", lambda refs: refs)
    monkeypatch.setattr(mx_module.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(mx_module.dist, "barrier", lambda group=None: None)
    monkeypatch.setattr(
        mx_module.dist,
        "broadcast_object_list",
        lambda values, src, group=None: None,
    )
    monkeypatch.setattr(mx_module, "get_gloo_group", lambda: object())
    monkeypatch.setattr(UpdateWeightFromModelExpress, "_init_lora", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        UpdateWeightFromModelExpress,
        "_for_each_hf_bucket",
        lambda self, consume: None,
    )
    monkeypatch.setattr(mx_module, "make_tensor_reader", lambda _path: object())


def updater(publisher):
    return UpdateWeightFromModelExpress(
        args(),
        model=[],
        weights_getter=lambda: {},
        model_name="qwen3",
        quantization_config=None,
        publisher=publisher,
    )


def test_receive_metrics_merge_by_max_latency():
    assert mx_module._receiver_metrics(
        [
            {"metrics": {"perf/mx_receive_prepare_time": 2.0}},
            {"metrics": {"perf/mx_receive_prepare_time": 3.0}},
        ]
    ) == {"perf/mx_receive_prepare_time": 3.0}


def test_lora_is_rejected_before_publisher_use():
    with pytest.raises(ValueError, match="does not support LoRA"):
        UpdateWeightFromModelExpress(
            args(),
            model=[],
            weights_getter=lambda: {},
            model_name="qwen3",
            quantization_config=None,
            is_lora=True,
            publisher=FakePublisher(),
        )


def test_miles_supplies_main_thread_buckets_and_control_thread_rollout():
    publisher = FakePublisher()
    instance = updater(publisher)
    events = []
    instance.connect_rollout_engines([FakeEngine(events)], object())

    instance.update_weights()
    instance.update_weights()
    instance._control.shutdown()

    main = threading.current_thread().name
    assert publisher.publishes[0][0] == "0"
    assert publisher.baselines[0][2] == main
    assert publisher.publishes[1] == (
        "1",
        {
            "base_version": "0",
            "gather_hf_buckets": instance._for_each_hf_bucket,
        },
        main,
    )
    assert publisher.waits == ["0", "1"]
    assert all(thread.startswith("modelexpress-control") for _event, thread in events)
    assert publisher.catalog.commits == [("policy", "0"), ("policy", "1")]
    assert instance.pop_metrics() == {
        "perf/update_weights_density": 0.25,
        "perf/update_weights_wire_bytes": 123.0,
        "perf/mx_encode_delta": 4.0,
        "perf/mx_publish_time": 5.0,
        "perf/mx_receive_prepare_time": 2.0,
        "perf/mx_receive_install_time": 3.0,
    }
    assert instance.weight_version == 1


def test_failed_install_is_not_committed_or_resumed():
    publisher = FakePublisher()
    instance = updater(publisher)
    events = []
    instance.connect_rollout_engines([FakeEngine(events, install_success=False)], object())
    instance.update_weights()

    with pytest.raises(ModelExpressUpdateError, match="install failed"):
        instance.update_weights()
    instance._control.shutdown()

    assert publisher.catalog.commits == [("policy", "0")]
    assert not any(event == "continue" for event, _thread in events)
    assert instance.weight_version == 0


def test_receive_metrics_are_broadcast_to_the_logging_rank(monkeypatch):
    instance = updater(FakePublisher())
    instance.connect_rollout_engines([FakeEngine([])], object())
    instance.update_weights()
    monkeypatch.setattr(mx_module.dist, "get_rank", lambda: 1)

    def broadcast(values, src, group=None):
        values[0] = {"perf/mx_receive_prepare_time": 7.0}

    monkeypatch.setattr(mx_module.dist, "broadcast_object_list", broadcast)

    instance.update_weights()
    assert instance._control is not None
    instance._control.shutdown()

    assert instance.pop_metrics()["perf/mx_receive_prepare_time"] == 7.0
