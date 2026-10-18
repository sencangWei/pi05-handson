import contextlib
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from live_preview import LivePreview
from viewer_controls import floor_basis, paint_world_axes

try:
    import mujoco
except ImportError:
    mujoco = None


class FloorBasisTests(unittest.TestCase):
    def test_floor_directions_do_not_mutate_camera_arrays(self):
        forward, up = np.array([0., 1, -1]), np.array([0., 1, 1])
        before = forward.copy(), up.copy()
        right, top = floor_basis(forward, up)
        np.testing.assert_allclose(right, [1, 0])
        np.testing.assert_allclose(top, [0, 1])
        np.testing.assert_array_equal(forward, before[0])
        np.testing.assert_array_equal(up, before[1])

    def test_horizontal_view_uses_forward_on_table(self):
        right, top = floor_basis([1, 0, 0], [0, 0, 1])
        np.testing.assert_allclose(right, [0, -1])
        np.testing.assert_allclose(top, [1, 0])
        for forward, up in [([0, 0, 0], [0, 0, 0]), ([float("nan"), 0, 1], [0, 1, 0])]:
            with self.assertRaises(ValueError):
                floor_basis(forward, up)


@unittest.skipIf(mujoco is None, "MuJoCo simulator environment required")
class NativeViewerControlsTests(unittest.TestCase):
    def setUp(self):
        self.model = mujoco.MjModel.from_xml_string(
            '<mujoco><worldbody><body pos="0 0 1"><freejoint/>'
            '<geom type="sphere" size=".03"/><site name="grip" pos="0 0 .1"/>'
            '</body></worldbody></mujoco>')
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.view = LivePreview.__new__(LivePreview)
        self.view.env = SimpleNamespace(sim=SimpleNamespace(
            model=SimpleNamespace(_model=self.model),
            data=SimpleNamespace(_data=self.data, site_xpos=self.data.site_xpos)))
        self.view._eef_site_id = 0
        self.view._camera_scene = mujoco.MjvScene(self.model, maxgeom=1)
        self.view.handle = SimpleNamespace(lock=contextlib.nullcontext,
                                           user_scn=mujoco.MjvScene(self.model, maxgeom=7),
                                           cam=mujoco.MjvCamera(), sync=Mock())
        self.view.handle.cam.distance = 1

    def test_current_camera_directions_for_orbit_and_top_view(self):
        for azimuth, expected in [(0, ([0, -1], [1, 0])),
                                  (90, ([1, 0], [0, 1])), (180, ([0, 1], [-1, 0]))]:
            for elevation in (-30, -90, 0):
                self.view.handle.cam.azimuth, self.view.handle.cam.elevation = azimuth, elevation
                actual = self.view.translation_basis()
                np.testing.assert_allclose(actual, expected, atol=1e-6)
        self.view.top_view()
        np.testing.assert_allclose(self.view.translation_basis(), ([1, 0], [0, 1]), atol=1e-6)

    def test_axes_are_world_aligned_and_only_touch_user_scene(self):
        qpos, qvel, geoms = self.data.qpos.copy(), self.data.qvel.copy(), self.model.ngeom
        self.view.sync()
        scene = self.view.handle.user_scn
        self.assertEqual(scene.ngeom, 7)
        for i in range(3):
            np.testing.assert_allclose(scene.geoms[i].mat.reshape(3, 3)[:, 2], np.eye(3)[i], atol=1e-6)
            label = scene.geoms[i + 3]
            self.assertEqual(label.label, ("X+", "Y+", "Z+")[i])
            np.testing.assert_allclose(label.pos, self.data.site_xpos[0] + [.16, .12, .02]
                                       + np.eye(3)[i] * .14)
        np.testing.assert_array_equal(self.data.qpos, qpos)
        np.testing.assert_array_equal(self.data.qvel, qvel)
        self.assertEqual(self.model.ngeom, geoms)
        self.view.handle.sync.assert_called_once()
        with self.assertRaises(ValueError):
            paint_world_axes(mujoco.MjvScene(self.model, maxgeom=6), [0, 0, 0])


if __name__ == "__main__":
    unittest.main()
