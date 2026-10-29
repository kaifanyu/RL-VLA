"""Include the local OpenPI client in the isolated LIBERO-Plus dependency overlay."""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def complete(output):
    overlay = ROOT / ".venv-plus/lib/python3.11/site-packages/benchmark_dependencies.pth"
    client = str(ROOT / "third_party/openpi/packages/openpi-client/src")
    paths = overlay.read_text().splitlines()
    if client not in paths:
        paths.insert(2, client)
        overlay.write_text("\n".join(paths) + "\n")
    environment = os.environ | {"LIBERO_CONFIG_PATH": str(ROOT / ".cache/libero-plus-screen")}
    result = subprocess.check_output([
        str(ROOT / ".venv-plus/bin/python"), "-c",
        ("import json,libero.libero,openpi_client,rl_vla.robustness; "
         "print(json.dumps({'libero':libero.libero.__file__,'openpi_client':openpi_client.__file__}))")],
        env=environment, text=True)
    imported = json.loads(result)
    if not Path(imported["libero"]).is_relative_to(ROOT / "third_party/libero_plus"):
        raise ValueError("Wrong LIBERO fork in Plus overlay")
    path = output / "plus_runtime.json"
    report = json.loads(path.read_text())
    report.update(client_import=imported["openpi_client"],
                  dependency_overlay_paths=paths,
                  overlay_sha256=hashlib.sha256(overlay.read_bytes()).hexdigest(),
                  completion_script=str(Path(__file__)),
                  runtime_import_check="libero.libero + openpi_client + rl_vla.robustness passed")
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    complete(parser.parse_args().output.resolve())
