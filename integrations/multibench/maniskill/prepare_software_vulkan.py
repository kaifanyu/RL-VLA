"""Prepare a persistent, process-local Ubuntu lavapipe Vulkan driver."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from download_verified_file import download, use_cached_ipv4

PACKAGE = "mesa-vulkan-drivers_25.2.8-0ubuntu0.24.04.4_amd64.deb"
SHA256 = "a4ec58cb952b4e2a8b7c7a9b4ea80fa3d2914e347ad13b11dfb67518b409fb57"
URL = "https://archive.ubuntu.com/ubuntu/pool/main/m/mesa/" + PACKAGE


def main():
    folder = ROOT / ".cache/native-maniskill"
    use_cached_ipv4()
    package = download(
        URL, folder / PACKAGE, SHA256, size=17496794, workers=8, request_bytes=262144
    )
    subprocess.run(["dpkg-deb", "-x", str(package), str(folder / "mesa")], check=True)
    library = folder / "mesa/usr/lib/x86_64-linux-gnu/libvulkan_lvp.so"
    dependencies = subprocess.check_output(["ldd", str(library)], text=True)
    if "not found" in dependencies:
        raise RuntimeError("Missing system libraries for lavapipe:\n" + dependencies)
    source = folder / "mesa/usr/share/vulkan/icd.d/lvp_icd.json"
    definition = json.loads(source.read_text())
    definition["ICD"]["library_path"] = str(library)
    icd = folder / "lvp_icd.json"
    icd.write_text(json.dumps(definition, indent=2) + "\n")
    manifest = {
        "package": PACKAGE,
        "url": URL,
        "sha256": SHA256,
        "driver": str(library),
        "icd": str(icd),
        "source": "https://packages.ubuntu.com/noble-updates/amd64/mesa-vulkan-drivers/download",
        "library_dependencies": dependencies,
        "system_driver_modified": False,
    }
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Set VK_DRIVER_FILES and VK_ICD_FILENAMES in the simulator process to:", icd)


if __name__ == "__main__":
    main()
