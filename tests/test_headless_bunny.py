import json
import subprocess
import unittest
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
EXECUTABLE = REPOSITORY / "build-short2" / "Release" / "gipc.exe"
METRICS = REPOSITORY / "tests" / "headless_bunny_metrics.json"
SHORT_ROOT = Path("S:/")


class HeadlessBunnyTests(unittest.TestCase):
    """Catches viewer startup, wrong scene selection, or missing batch metrics."""

    @classmethod
    def setUpClass(cls) -> None:
        result = subprocess.run(
            ["subst.exe", "S:", str(REPOSITORY)], capture_output=True, text=True
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr)

    @classmethod
    def tearDownClass(cls) -> None:
        subprocess.run(["subst.exe", "S:", "/D"], capture_output=True, text=True)

    def test_one_frame_stiff_bunny_writes_traceable_metrics(self) -> None:
        METRICS.unlink(missing_ok=True)
        result = subprocess.run(
            [
                str(SHORT_ROOT / "build-short2" / "Release" / "gipc.exe"),
                "--scene",
                "stiff-bunny-drop",
                "--solver",
                "stiffgipc",
                "--tet-mesh",
                str(SHORT_ROOT / "Assets" / "tetMesh" / "bunny2.msh"),
                "--frames",
                "1",
                "--young-modulus",
                "1e7",
                "--dt",
                "0.01",
                "--headless",
                "--metrics-path",
                str(SHORT_ROOT / "tests" / "headless_bunny_metrics.json"),
            ],
            cwd=SHORT_ROOT,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=600,
        )
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        report = json.loads(METRICS.read_text(encoding="utf-8"))
        self.assertEqual(report["scene"], "stiff-bunny-drop")
        self.assertEqual(report["solver"], "stiffgipc")
        self.assertEqual(report["frames_completed"], 1)
        self.assertEqual(report["fem_vertices"], 19193)
        self.assertEqual(report["young_modulus"], 1e7)
        self.assertEqual(report["dt"], 0.01)
        self.assertGreater(report["wall_time_ms"], 0.0)
        self.assertGreaterEqual(report["simulation_time_ms"], 0.0)
        self.assertTrue(report["finite_vertices"])


if __name__ == "__main__":
    unittest.main()
