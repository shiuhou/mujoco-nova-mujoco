from __future__ import annotations

from pathlib import Path

import numpy as np

from collision_review import review_model


ROOT = Path(__file__).resolve().parent


def test_every_official_mesh_has_an_enclosing_collision_proxy() -> None:
    rows = review_model()
    assert [row.link for row in rows] == [
        "base_link",
        "Link1",
        "Link2",
        "Link3",
        "Link4",
        "Link5",
        "Link6",
    ]
    assert all(row.encloses_mesh for row in rows)
    assert all(
        np.isclose(row.proxy_volume_ratio, 1.0, rtol=0, atol=1e-5) for row in rows
    )
    for row in rows:
        errors = np.abs(np.asarray(row.boundary_error_min_m + row.boundary_error_max_m))
        np.testing.assert_array_less(errors, 2e-6)


def test_collision_audit_is_independent_of_mujoco() -> None:
    rows = review_model(ROOT / "nova2.xml", ROOT / "assets")
    assert len(rows) == 7
    assert sum(row.triangle_count for row in rows) > 0
