from __future__ import annotations

import os
import shutil
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "install.sh"


class InstallScriptStaticTests(unittest.TestCase):
    def test_posix_script_exists(self):
        self.assertTrue(SCRIPT.is_file())
        text = SCRIPT.read_text(encoding="utf-8")
        self.assertTrue(text.startswith("#!/usr/bin/env bash\n"))
        self.assertIn("Host assumptions: curl (or wget) and tar", text)
        self.assertIn("cortex.shizuha.com/deployer/install.sh", text)
        self.assertNotIn("--break-system-packages", text)
        self.assertNotIn("pip3 install", text)
        self.assertIn('pip install --python', text)
        self.assertIn("UV_PYTHON_INSTALL_DIR", text)
        self.assertIn("UV_PYTHON_PREFERENCE=only-managed", text)
        self.assertIn('info() { echo "cortex-deployer-install: $*" >&2; }', text)
        self.assertNotIn('uv="$(ensure_uv)"', text)
        self.assertNotIn("if have uv; then", text)

    def test_help_and_logs_stay_off_stdout(self):
        if not shutil.which("bash"):
            self.skipTest("bash not available")
        proc = subprocess.run(
            ["bash", str(SCRIPT), "--help"],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.stdout, "")
        self.assertIn("Usage:", proc.stderr)

    def test_bash_syntax(self):
        if not shutil.which("bash"):
            self.skipTest("bash not available")
        subprocess.run(["bash", "-n", str(SCRIPT)], check=True)

    def test_no_system_pip_invocation(self):
        text = SCRIPT.read_text(encoding="utf-8")
        for needle in (
            "python3 -m pip",
            "python -m pip",
            "pip3 install",
            "apt install",
            "sudo ",
        ):
            self.assertNotIn(needle, text)

    def test_windows_script_exists(self):
        ps1 = ROOT / "scripts" / "install.ps1"
        self.assertTrue(ps1.is_file())
        text = ps1.read_text(encoding="utf-8")
        self.assertIn("irm https://cortex.shizuha.com/deployer/install.ps1", text)
        self.assertNotIn("pip3 install", text)
        self.assertIn("pip install --python", text)


@unittest.skipUnless(
    os.environ.get("CORTEX_DEPLOYER_RUN_INSTALL_E2E") == "1",
    "set CORTEX_DEPLOYER_RUN_INSTALL_E2E=1 to run the isolated installer",
)
class InstallScriptE2ETests(unittest.TestCase):
    def test_installs_into_prefix_without_system_pip(self):
        if not shutil.which("bash"):
            self.skipTest("bash not available")
        home = Path(tempfile.mkdtemp(prefix="cd-install-e2e-"))
        self.addCleanup(lambda: shutil.rmtree(home, ignore_errors=True))
        prefix = home / "prefix"
        bindir = home / "bin"
        env = os.environ.copy()
        env["CORTEX_DEPLOYER_HOME"] = str(prefix)
        env["CORTEX_DEPLOYER_BIN_DIR"] = str(bindir)
        env["HOME"] = str(home)
        env.pop("VIRTUAL_ENV", None)
        env.pop("PYTHONHOME", None)
        # Prove we do not need a host uv/python on PATH.
        env["PATH"] = "/usr/bin:/bin"
        proc = subprocess.run(
            ["bash", str(SCRIPT)],
            check=True,
            env=env,
            cwd=str(home),
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.stdout, "", proc.stderr)
        wrapper = bindir / "cortex-deployer"
        self.assertTrue(wrapper.is_file())
        self.assertTrue(wrapper.stat().st_mode & stat.S_IXUSR)
        out = subprocess.run(
            [str(wrapper), "--version"],
            check=True,
            capture_output=True,
            text=True,
            env={**env, "PATH": f"{bindir}:{env.get('PATH', '')}"},
        )
        self.assertTrue(out.stdout.strip())
        self.assertTrue((prefix / "venv" / "bin" / "cortex-deployer").is_file())


if __name__ == "__main__":
    unittest.main()
