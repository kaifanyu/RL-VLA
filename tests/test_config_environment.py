"""Deployment endpoint overrides must preserve configuration and fail early."""

import pytest

from rl_vla.config import load_config


@pytest.fixture
def config_path(tmp_path, monkeypatch):
    monkeypatch.delenv("RL_VLA_OPENPI_HOST", raising=False)
    monkeypatch.delenv("RL_VLA_OPENPI_PORT", raising=False)
    path = tmp_path / "config.toml"
    path.write_text(
        'algorithm = "sac"\n'
        '[env]\nkind = "toy"\n'
        '[base]\nkind = "openpi"\nhost = "original-server"\nport = 8765\n'
        '[chunk]\ngamma = 0.99\n'
        '[train]\ndevice = "cpu"\n',
        encoding="utf-8",
    )
    return path


def test_unset_overrides_preserve_config(config_path):
    config = load_config(config_path)
    assert config["base"] == {"kind": "openpi", "host": "original-server", "port": 8765}


@pytest.mark.parametrize(
    ("overrides", "host", "port"),
    [
        ({"RL_VLA_OPENPI_HOST": "  openpi  "}, "openpi", 8765),
        ({"RL_VLA_OPENPI_PORT": " 8000 "}, "original-server", 8000),
        (
            {"RL_VLA_OPENPI_HOST": "host.docker.internal", "RL_VLA_OPENPI_PORT": "9000"},
            "host.docker.internal",
            9000,
        ),
        ({"RL_VLA_OPENPI_PORT": "1"}, "original-server", 1),
        ({"RL_VLA_OPENPI_PORT": "65535"}, "original-server", 65535),
    ],
)
def test_endpoint_overrides(config_path, monkeypatch, overrides, host, port):
    original = config_path.read_bytes()
    for name, value in overrides.items():
        monkeypatch.setenv(name, value)
    config = load_config(config_path)
    assert config["base"] == {"kind": "openpi", "host": host, "port": port}
    assert config["train"]["device"] == "cpu"
    assert config_path.read_bytes() == original


@pytest.mark.parametrize("host", ["", "  \t\n"])
def test_empty_host_override_rejected(config_path, monkeypatch, host):
    monkeypatch.setenv("RL_VLA_OPENPI_HOST", host)
    with pytest.raises(ValueError, match="RL_VLA_OPENPI_HOST.*nonempty"):
        load_config(config_path)


@pytest.mark.parametrize("port", ["", " ", "abc", "8000.0", "0", "-1", "65536"])
def test_invalid_port_override_rejected(config_path, monkeypatch, port):
    monkeypatch.setenv("RL_VLA_OPENPI_PORT", port)
    with pytest.raises(ValueError, match="RL_VLA_OPENPI_PORT.*integer from 1 through 65535"):
        load_config(config_path)


def test_mock_base_ignores_endpoint_environment(config_path, monkeypatch):
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace('kind = "openpi"', 'kind = "mock"'),
        encoding="utf-8",
    )
    monkeypatch.setenv("RL_VLA_OPENPI_HOST", " ")
    monkeypatch.setenv("RL_VLA_OPENPI_PORT", "not-a-port")
    config = load_config(config_path)
    assert config["base"] == {"kind": "mock", "host": "original-server", "port": 8765}
