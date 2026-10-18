"""Viewer-only world axes and table-plane controls from the current camera."""
import numpy as np


def floor_basis(forward, up):
    forward, up = np.asarray(forward, dtype=float).copy(), np.asarray(up, dtype=float).copy()
    if (forward.shape != (3,) or up.shape != (3,)
            or not np.isfinite(forward).all() or not np.isfinite(up).all()):
        raise ValueError("Invalid viewer camera orientation")
    right_xy = np.cross(forward, up)[:2].copy()
    up_xy = up[:2].copy()
    if np.linalg.norm(up_xy) < 1e-6:
        up_xy = forward[:2].copy()
    if min(np.linalg.norm(right_xy), np.linalg.norm(up_xy)) < 1e-6:
        raise ValueError("Camera does not define a table-plane control direction")
    return right_xy / np.linalg.norm(right_xy), up_xy / np.linalg.norm(up_xy)


def paint_world_axes(scene, origin):
    """Decorate only a passive viewer user_scn, never the robot model or renderer."""
    import mujoco
    origin = np.asarray(origin, dtype=float)
    if scene.maxgeom < 7 or origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError("Invalid viewer axis scene or origin")
    # Keep the annotation beside the fingers so the wrist does not hide +Z.
    anchor = origin + [.16, .12, .02]
    for i, (name, color) in enumerate((("X+", [1, .1, .1, 1]),
                                       ("Y+", [.1, 1, .1, 1]),
                                       ("Z+", [.1, .35, 1, 1]))):
        geom = scene.geoms[i]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3),
                           anchor, np.eye(3).ravel(), np.asarray(color, dtype=np.float32))
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, .004,
                            anchor, anchor + np.eye(3)[i] * .12)
        geom.label = ""
        label = scene.geoms[i + 3]
        mujoco.mjv_initGeom(label, mujoco.mjtGeom.mjGEOM_LABEL, np.zeros(3),
                           anchor + np.eye(3)[i] * .14, np.eye(3).ravel(),
                           np.asarray(color, dtype=np.float32))
        label.label = name
    leader = scene.geoms[6]
    mujoco.mjv_initGeom(leader, mujoco.mjtGeom.mjGEOM_LINE, np.zeros(3),
                       origin, np.eye(3).ravel(), np.ones(4, dtype=np.float32))
    mujoco.mjv_connector(leader, mujoco.mjtGeom.mjGEOM_LINE, 1, origin, anchor)
    scene.ngeom = 7
