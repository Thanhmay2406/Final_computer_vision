"""Geometry contracts detect pixel-axis/scale mistakes and incompatible artifacts."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
from handgesture.preprocessing import PreprocessingConfig, HandPreprocessor, normalize_landmarks, square_roi


def points():
    p=np.zeros((21,3));p[:,0]=np.linspace(.1,.9,21);p[:,1]=.5;p[:,2]=np.linspace(0,.2,21)
    return p


def test_geometry_uses_pixel_axes_and_wrist_relative_3d_scale():
    p=points(); got=normalize_landmarks(p,200,100).reshape(21,3)
    pixel=p*np.array([200,100,200]);relative=pixel-pixel[0]
    np.testing.assert_allclose(got,relative/np.linalg.norm(relative,axis=1).max(),atol=1e-7)
    np.testing.assert_allclose(got[0],0)
    with pytest.raises(ValueError):normalize_landmarks(np.zeros((21,3)),100,100)
    p[2,1]=np.nan
    with pytest.raises(ValueError):normalize_landmarks(p,100,100)


def test_square_roi_preserves_outside_region_with_black_padding():
    image=np.full((20,30,3),255,dtype=np.uint8);p=points();p[:,0]=np.linspace(-.2,.2,21)
    roi,bbox=square_roi(image,p,1.4)
    assert roi.shape[0]==roi.shape[1] and bbox[0]<0
    assert (roi[:,0]==0).all() and roi.max()==255


def test_preprocessor_handles_no_hand_invalid_and_checks_fingerprint():
    config=PreprocessingConfig(detector_sha256='test')
    processor=HandPreprocessor(config,detector=lambda image,timestamp: None)
    assert processor.process(np.zeros((30,30,3),dtype=np.uint8)).status=='no_hand'
    with pytest.raises(ValueError,match='fingerprint'):config.assert_matches('wrong')
    processor=HandPreprocessor(config,detector=lambda image,timestamp: points())
    result=processor.process(np.zeros((30,30,3),dtype=np.uint8))
    assert result.status=='ok' and result.features.shape==(63,)
    assert config.fingerprint==PreprocessingConfig(detector_sha256='test').fingerprint


def test_detector_padding_maps_points_back_to_original_pixel_geometry():
    config=PreprocessingConfig('test',detector_input_padding=.5)
    # Original width40,height20 -> padded80x40. Center (.5,.5) stays at (20,10).
    # A normalized padded x=.625 maps to original x=.75, not .625.
    processor=HandPreprocessor(config,detector=lambda image,timestamp: None)
    padded=points();padded[:,0]=.625;padded[:,1]=.5
    mapped=processor.map_detector_landmarks(padded,40,20)
    np.testing.assert_allclose(mapped[:,0],.75)
    np.testing.assert_allclose(mapped[:,1],.5)
    np.testing.assert_allclose(mapped[:,2],padded[:,2]*2)
