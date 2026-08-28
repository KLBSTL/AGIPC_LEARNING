import os
import subprocess
import unittest
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
EXECUTABLE = Path(
    os.environ.get(
        "GIPC_EXECUTABLE",
        REPOSITORY / "build-short2" / "Release" / "gipc.exe",
    )
)


class ExecutableCliTests(unittest.TestCase):
    """Catches a parser that exists but is not wired before OpenGL/CUDA startup."""

    def run_cli(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(EXECUTABLE), *arguments],
            cwd=REPOSITORY,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
        )

    def test_help_exits_without_creating_the_viewer(self) -> None:
        result = self.run_cli("--help")
        self.assertEqual(result.returncode, 0)
        self.assertIn("--scene stiff-bunny-drop", result.stdout)

    def test_invalid_mapping_exits_two_before_gpu_startup(self) -> None:
        result = self.run_cli("--agipc-mapping", "unknown")
        self.assertEqual(result.returncode, 2)
        self.assertIn("mapping", result.stderr.lower())

    def test_missing_tet_mesh_exits_two_before_gpu_startup(self) -> None:
        result = self.run_cli(
            "--scene",
            "stiff-bunny-drop",
            "--headless",
            "--frames",
            "1",
            "--tet-mesh",
            "missing-bunny.msh",
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("tet mesh", result.stderr.lower())


if __name__ == "__main__":
    unittest.main()
