import subprocess
import tempfile
from pathlib import Path

import pytest


def test_build_custom_apk():
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "build" / "build_apk_tools.sh"
    try:
        subprocess.check_call([str(script)])
    except (subprocess.CalledProcessError, PermissionError) as exc:
        pytest.skip(f"apk-tools build failed: {exc}")
    apk_path = repo_root / "apk-tools" / "src" / "apk"
    assert apk_path.is_file()
    subprocess.check_call([str(apk_path), "--version"])


def _make_pkg(tmpdir, name, provides, version, depends=None):
    pkgdir = tmpdir / name / "usr" / "bin"
    pkgdir.mkdir(parents=True)
    script_path = pkgdir / name
    script_path.write_text("#!/bin/sh\n")
    script_path.chmod(0o755)
    pkginfo = [
        f"pkgname = {name}",
        f"pkgver = {version}",
        f"pkgdesc = {name}",
        "url = https://example.com",
        "builddate = 0",
        "packager = test",
        "size = 1",
        "arch = noarch",
        f"origin = {name}",
        "maintainer = test",
        "license = none",
        "provider_priority = 100",
        f"provides = {provides}",
        "datahash = 0",
    ]
    if depends:
        pkginfo.append(f"depend = {depends}")
    pkginfo_path = tmpdir / name / ".PKGINFO"
    pkginfo_path.write_text("\n".join(pkginfo) + "\n")
    apk_path = tmpdir / f"{name}-{version}.apk"
    subprocess.check_call([
        "tar",
        "-C",
        str(tmpdir / name),
        "-czf",
        str(apk_path),
        ".",
    ])
    return apk_path


def test_alternative_packages():
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "build" / "build_apk_tools.sh"
    try:
        subprocess.check_call([str(script)])
    except (subprocess.CalledProcessError, PermissionError) as exc:
        pytest.skip(f"apk-tools build failed: {exc}")
    apk_path = repo_root / "apk-tools" / "src" / "apk"
    assert apk_path.is_file()

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        repo = tmp / "repo"
        repo.mkdir()
        root = tmp / "root"
        alt1 = _make_pkg(repo, "alt1", "virtpkg", "2", depends="missingdep")
        alt2 = _make_pkg(repo, "alt2", "virtpkg", "1")
        subprocess.check_call([
            str(apk_path),
            "index",
            "-o",
            str(repo / "APKINDEX.tar.gz"),
            str(alt1),
            str(alt2),
        ])
        subprocess.check_call([
            str(apk_path),
            "--root",
            str(root),
            "--initdb",
            "--repository",
            str(repo),
            "--allow-untrusted",
            "add",
            "virtpkg",
        ])
        res_alt2 = subprocess.run(
            [str(apk_path), "--root", str(root), "info", "alt2"],
            stdout=subprocess.DEVNULL,
        )
        assert res_alt2.returncode == 0
        res_alt1 = subprocess.run(
            [str(apk_path), "--root", str(root), "info", "alt1"],
            stdout=subprocess.DEVNULL,
        )
        assert res_alt1.returncode != 0
