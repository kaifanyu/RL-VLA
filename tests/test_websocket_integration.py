"""Verify the installed upstream wire protocol without a VLA or model weights."""

import json
import queue
import threading

import numpy as np
import pytest

from rl_vla.base_policy import make_base_policy


def test_upstream_client_numpy_websocket_round_trip():
    msgpack_numpy = pytest.importorskip("openpi_client.msgpack_numpy")
    pytest.importorskip("openpi_client.websocket_client_policy")
    websocket_server = pytest.importorskip("websockets.sync.server")
    expected_chunk = np.linspace(-0.2, 0.2, 70, dtype=np.float32).reshape(10, 7)
    image = np.arange(224 * 224 * 3, dtype=np.uint8).reshape(224, 224, 3)
    payload = {
        "observation/image": image,
        "observation/wrist_image": image[:, ::-1].copy(),
        "observation/state": np.linspace(0.0, 0.7, 8, dtype=np.float32),
        "prompt": "pick up the red block",
    }
    metadata = {"checkpoint": "mock-wire-test", "reset_pose": np.zeros(7, dtype=np.float32)}
    requests = []
    server_errors = queue.Queue()
    result = queue.Queue()
    connections = []
    handler_finished = threading.Event()

    def handle(connection):
        connections.append(connection)
        try:
            packer = msgpack_numpy.Packer()
            # OpenPI sends metadata immediately after the websocket handshake.
            connection.send(packer.pack(metadata))
            for _ in range(2):
                requests.append(msgpack_numpy.unpackb(connection.recv(timeout=5)))
                connection.send(packer.pack({"actions": expected_chunk, "server_timing": {"infer_ms": 0.1}}))
        except Exception as error:  # noqa: BLE001 - Relay thread failures to the test thread.
            server_errors.put(error)
        finally:
            handler_finished.set()

    def query(port):
        policy = None
        try:
            policy = make_base_policy({"kind": "openpi", "host": "127.0.0.1", "port": port}, 7, 10)
            first = policy.infer({"openpi": payload, "privileged": "must not be sent"})
            policy.reset()  # Upstream reset is local and must not add a protocol frame.
            second = policy.infer({"openpi": payload})
            result.put((policy.metadata, first, second))
        except Exception as error:  # noqa: BLE001 - Relay thread failures to the test thread.
            result.put(error)
        finally:
            if policy is not None:
                # Upstream exposes no public close method; close the test connection explicitly.
                policy._client._ws.close()

    # serve() binds/listens before the client starts, avoiding upstream's retry loop.
    with websocket_server.serve(
        handle, "127.0.0.1", 0, compression=None, max_size=None,
        open_timeout=3, close_timeout=1,
    ) as server:
        serving = threading.Thread(target=server.serve_forever, daemon=True)
        serving.start()
        client = threading.Thread(target=query, args=(server.socket.getsockname()[1],), daemon=True)
        client.start()
        try:
            client.join(timeout=10)
            assert not client.is_alive(), "Upstream websocket client did not complete within 10 seconds"
            response = result.get(timeout=1)
            if isinstance(response, Exception):
                raise response
            server_metadata, first, second = response
            assert server_errors.empty(), list(server_errors.queue)
            assert handler_finished.wait(timeout=1)
            assert server_metadata["server"] == {"checkpoint": "mock-wire-test", "reset_pose": [0.0] * 7}
            json.dumps(server_metadata, allow_nan=False)
            for chunk in (first, second):
                assert chunk.shape == (10, 7)
                assert chunk.dtype == np.float32
                np.testing.assert_array_equal(chunk, expected_chunk)
            assert len(requests) == 2
            for request in requests:
                assert set(request) == set(payload)
                assert request["prompt"] == payload["prompt"]
                for key in ("observation/image", "observation/wrist_image", "observation/state"):
                    assert request[key].dtype == payload[key].dtype
                    np.testing.assert_array_equal(request[key], payload[key])
        finally:
            for connection in connections:
                connection.close()
            server.shutdown()
            serving.join(timeout=5)
            client.join(timeout=5)
            assert not serving.is_alive(), "Mock websocket server did not shut down"
            assert not client.is_alive(), "Mock websocket client did not shut down"
