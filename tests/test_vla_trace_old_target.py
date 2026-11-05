"""Geometry convention checks independent of native rendered evidence."""

import numpy as np
import pytest

from rl_vla.diagnostics.old_target import convex_hull, project_pixels


def test_projection_uses_pixel_centers_and_convention_rotation():
    # Identity pinhole places homogeneous point (64, 32, 1) at pixel (64, 32).
    points=np.array([[64.,32.,1.]])
    actual=project_pixels(points,np.eye(4),1)
    assert np.allclose(actual,[[(255-64+.5)*224/256-.5,(32+.5)*224/256-.5]])
    rotated=project_pixels(points,np.eye(4),-1)
    assert np.allclose(rotated[0,1],(255-32+.5)*224/256-.5)
    with pytest.raises(ValueError,match="behind"):
        project_pixels(np.array([[0.,0.,-1.]]),np.eye(4),1)


def test_prism_hull_excludes_interior_and_repeated_vertices():
    points=np.array([[0,0],[2,0],[2,2],[0,2],[1,1],[2,2]])
    assert np.array_equal(convex_hull(points),[[0,0],[2,0],[2,2],[0,2]])
