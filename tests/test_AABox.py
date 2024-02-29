import pytest
import torch
import numpy as np

from photonlib.meta import AABox
from tests.fixtures import fake_photon_library


@pytest.fixture
def ranges():
    return [[0, 1], 
            [0, 2],
            [0, 3]]

@pytest.fixture
def aabox(ranges):
    return AABox(ranges)

def test_AABox_init(aabox, ranges, fake_photon_library):
    # test already done by fixture
    assert torch.allclose(aabox.lengths, torch.tensor([1,2,3], dtype=torch.float32))
    assert torch.allclose(aabox.ranges, torch.as_tensor(ranges, dtype=torch.float32))
    
    # test with wrong shape
    with pytest.raises(ValueError):
        AABox(torch.zeros(2,3))
    with pytest.raises(ValueError):
        AABox(torch.zeros(4))
        
    # test init with numpy array
    a = AABox(np.array(ranges))
    assert len(a.ranges.shape) == 2 and a.ranges.shape[1] == 2
    
    # test init with torch tensor
    a =AABox(torch.as_tensor(ranges))
    assert len(a.ranges.shape) == 2 and a.ranges.shape[1] == 2
    
    # test init from photon library h5 file
    a = AABox.load(fake_photon_library)
    assert len(a.ranges.shape) == 2 and a.ranges.shape[1] == 2
    
    # test init from cfg
    cfg = dict(photonlib=dict(filepath=fake_photon_library))
    a = AABox.load(cfg)
    assert len(a.ranges.shape) == 2 and a.ranges.shape[1] == 2
    
    # test init fail
    with pytest.raises(TypeError):
        AABox.load(1)
    
    
def test_AABox_properties(aabox, ranges):
    assert np.allclose(aabox.x.cpu().numpy(),ranges[0])
    assert np.allclose(aabox.y.cpu().numpy(),ranges[1])
    assert np.allclose(aabox.z.cpu().numpy(),ranges[2])
    
def test_AABox_norm_coord(aabox, ranges):
    trange = torch.as_tensor(ranges)
    
    pos = torch.diff(trange).ravel()/2 # center of the box
    norm_pos = aabox.norm_coord(pos)
    assert torch.allclose(norm_pos, torch.zeros(3))
    
    pos = trange[:,0]                  # close edge
    norm_pos = aabox.norm_coord(pos)
    assert torch.allclose(norm_pos, -torch.ones(3))
    
    pos = trange[:,1]                  # far edge
    norm_pos = aabox.norm_coord(pos)
    assert torch.allclose(norm_pos, +torch.ones(3))