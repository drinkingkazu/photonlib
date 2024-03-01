import pytest
import torch
import numpy as np

from photonlib.meta import VoxelMeta
from tests.fixtures import fake_photon_library, fake_photon_library_ranges, fake_photon_library_shape


@pytest.fixture
def ranges(fake_photon_library_ranges):
    return fake_photon_library_ranges
    
@pytest.fixture
def shape(fake_photon_library_shape):
    return fake_photon_library_shape

@pytest.fixture
def voxelmeta(shape, ranges):
    return VoxelMeta(shape, ranges)

def test_VoxelMeta_init(voxelmeta, ranges, shape, fake_photon_library):
    # test already done by fixture
    tranges = torch.as_tensor(ranges).float()
    tshape = torch.as_tensor(shape).long()
    
    # test init from photon library h5 file
    a = VoxelMeta.load(fake_photon_library)
    assert torch.allclose(a.ranges, tranges), 'ranges should be the same as the h5 file'
    assert torch.allclose(a.shape, tshape), 'shape should be the same as the h5 file'
    
    # test init from cfg
    cfg = dict(photonlib=dict(filepath=fake_photon_library))
    a = VoxelMeta.load(cfg)
    assert torch.allclose(a.ranges, tranges), 'ranges should be the same as the h5 file'
    assert torch.allclose(a.shape, tshape), 'shape should be the same as the h5 file'
    
    # fail init with bad input
    with pytest.raises(TypeError):
        VoxelMeta.load(1)

    # test fail init with bad shape array
    with pytest.raises(ValueError):
        VoxelMeta((1,2,3,4), ranges)
    

def test_VoxelMeta_bins(voxelmeta, ranges, shape):
    bins = voxelmeta.bins
    xshape, yshape, zshape = shape
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = ranges
    
    assert all(np.allclose([bins[i][0], bins[i][-1]], ranges[i]) for i in range(3)), \
        "bins should have the same range as the voxelmeta"
    assert tuple(len(b)-1 for b in bins) == shape, "bins should have the same shape as the voxelmeta"
    
    # test bin centers
    bin_centers = voxelmeta.bin_centers
    cen = lambda x: (x[1:] + x[:-1])/2   # noqa: E731
    expected_bin_centers = (cen(b) for b in bins)
    assert all(torch.allclose(exp, act) for exp, act in zip(expected_bin_centers, bin_centers)), \
                    "bin_centers should be the center of the bins"
    
    
def test_VoxelMeta_len(voxelmeta):
    assert len(voxelmeta) == voxelmeta.shape[0]*voxelmeta.shape[1]*voxelmeta.shape[2], \
        "len(VoxelMeta) should be the product of the shape"
    
def test_VoxelMeta_norm_step_size(voxelmeta, shape):
    assert torch.allclose(voxelmeta.norm_step_size,2/torch.as_tensor(shape)), \
        "norm_step_size should be 2/shape"
    
    
    """ ---------------------- axis (idx, idy, idz) inputs --------------------- """
def test_VoxelMeta_idx_to_voxel(voxelmeta):
    # [idx, idy, idz] -> [voxid]
    voxaxes = [0, 0, 0]
    tensor_vox = torch.as_tensor(voxaxes)
    array_vox = np.array(voxaxes)

    for vox in [voxaxes, tensor_vox, array_vox]:
        voxid = voxelmeta.idx_to_voxel(vox)
        assert isinstance(voxid, torch.Tensor), "voxid should be a torch.Tensor"
        assert torch.allclose(voxid, torch.tensor([0])), "voxid should be [0]"
        
    voxaxes = [[0, 0, 0], [1, 1, 1]]
    tensor_vox = torch.tensor(voxaxes)
    array_vox = np.array(voxaxes)
    
    for vox in [voxaxes, tensor_vox, array_vox]:
        second_answer = (1 + voxelmeta.shape[0] + voxelmeta.shape[0]*voxelmeta.shape[1]).long()
        voxid = voxelmeta.idx_to_voxel(vox)
        assert isinstance(voxid, torch.Tensor), "voxid should be a torch.Tensor"
        assert torch.allclose(voxid, torch.tensor([0, second_answer])), f"voxid should be [0, {second_answer}]"

def test_VoxelMeta_idx_to_coord(voxelmeta):
    """not tested for accuracy of position, just that it returns a tensor of the right shape"""
    
    # [idx, idy, idz] -> [x, y, z]
    voxaxes = [0, 0, 0]
    tensor_vox = torch.as_tensor(voxaxes)
    array_vox = tensor_vox.cpu().numpy()
    
    for vox in [voxaxes, tensor_vox, array_vox]:
        pos = voxelmeta.idx_to_coord(vox)
        assert isinstance(pos, torch.Tensor), "pos should be a torch.Tensor"
        assert pos.shape == (3,), "pos should have 3 dimensions" 
        assert (pos >= voxelmeta.ranges[:,0]).all(), "pos out of min range"
        assert (pos <= voxelmeta.ranges[:,1]).all(), "pos out of max range"

    voxaxes = [[0, 0, 0], [1, 1, 1]]
    tensor_vox = torch.tensor(voxaxes)
    array_vox = np.array(voxaxes)
    for vox in [voxaxes, tensor_vox, array_vox]:
        pos = voxelmeta.idx_to_coord(vox)
        assert isinstance(pos, torch.Tensor), "pos should be a torch.Tensor"
        assert pos.shape == (len(voxaxes), 3), "pos should have 3 dimensions" 
        assert (pos >= voxelmeta.ranges[:,0]).all(), "pos out of min range"
        assert (pos <= voxelmeta.ranges[:,1]).all(), "pos out of max range"

""" ------------------------ position (x,y,z) inputs ----------------------- """
def test_VoxelMeta_coord_to_idx(voxelmeta, torch_rng):
    # [x, y, z] -> [voxid]
    mins = voxelmeta.ranges[:,0]
    maxs = voxelmeta.ranges[:,1]
    
    # 1D tensor
    rand_pos = torch.rand(size=(3,), generator=torch_rng)*(maxs-mins)+mins
    axisid = voxelmeta.coord_to_idx(rand_pos)
    assert isinstance(axisid, torch.Tensor), "rand_pos should be a torch.Tensor"
    assert axisid.shape == (3,), "axisid should have 3 dimensions"

    # 1D list
    axisid = voxelmeta.coord_to_idx(list(rand_pos))
    assert isinstance(axisid, torch.Tensor), "rand_pos should be a torch.Tensor"
    assert axisid.shape == (3,), "axisid should have 3 dimensions"

    # 2D tensor
    rand_pos = torch.rand(size=(100,3), generator=torch_rng)*(maxs-mins)+mins
    axisid = voxelmeta.coord_to_idx(rand_pos)
    assert isinstance(axisid, torch.Tensor), "rand_pos should be a torch.Tensor"
    assert axisid.shape == (100, 3), "axisid should have 3 dimensions"
    
    # check that axisid is in the right range
    axisid = axisid.squeeze()
    assert torch.all(axisid >= 0)
    assert all(torch.all(axisid[:, i] < voxelmeta.shape[i]) for i in range(3)), \
            f"axisid should be in the right range ({voxelmeta.shape}), {axisid}"

def test_VoxelMeta_coord_to_voxel(voxelmeta, torch_rng):
    """not tested for accuracy of voxid just that the returned voxids are in range"""
    # [x, y, z] -> [voxid]
    mins = voxelmeta.ranges[:,0]
    maxs = voxelmeta.ranges[:,1]
    
    rand_pos = torch.rand(size=(100, 3), generator=torch_rng)*(maxs-mins)+mins
    voxids = voxelmeta.coord_to_voxel(rand_pos)    
    
    maxvox = voxelmeta.shape[0]*voxelmeta.shape[1]*voxelmeta.shape[2] - 1
    assert torch.all(voxids>=0), "voxids should be positive"
    assert torch.all(voxids<=maxvox), f"voxids should be less than {maxvox}"

""" ---------------------------- voxel id inputs --------------------------- """
def test_voxel_to_idx(voxelmeta, torch_rng):
    # [voxid] -> [idx, idy, idz]
    maxvox = voxelmeta.shape[0]*voxelmeta.shape[1]*voxelmeta.shape[2] - 1
    voxids = torch.randint(0, int(maxvox), size=(100,), generator=torch_rng)
    
    axisid = voxelmeta.voxel_to_idx(voxids)
    assert all(torch.all(axisid[:, i] < voxelmeta.shape[i]) for i in range(3)), \
            f"axisid should be in the right range ({voxelmeta.shape}), {axisid}"

def test_voxel_to_coord(voxelmeta, torch_rng):
    # [voxid] -> [x, y, z], ...]
    maxvox = voxelmeta.shape[0]*voxelmeta.shape[1]*voxelmeta.shape[2] - 1
    voxids = torch.randint(0, int(maxvox), size=(100,), generator=torch_rng)

    pos = voxelmeta.voxel_to_coord(voxids)

    assert isinstance(pos, torch.Tensor), "pos should be a torch.Tensor"
    assert pos.shape == (100, 3), "pos should have 3 dimensions" 
    assert (pos >= voxelmeta.ranges[:,0]).all(), "pos out of min range"
    assert (pos <= voxelmeta.ranges[:,1]).all(), "pos out of max range"
    

""" ------------------------- ring around the rosie ------------------------ """
def test_axis2pos2axis(voxelmeta, torch_rng):
    input = (torch.rand(size=(100, 3), generator=torch_rng)*voxelmeta.shape).long()
    output = voxelmeta.coord_to_idx(voxelmeta.idx_to_coord(input))
    assert torch.allclose(input, output), '(ix,iy,iz) -> (x,y,z) -> (ix,iy,iz) failed'

def test_vox2axis2vox(voxelmeta, torch_rng):
    maxvox = int(voxelmeta.shape[0]*voxelmeta.shape[1]*voxelmeta.shape[2]) - 1
    input = torch.randint(0, maxvox, size=(100,), generator=torch_rng)
    output = voxelmeta.idx_to_voxel(voxelmeta.voxel_to_idx(input))
    assert torch.allclose(input, output), '(voxid) -> (ix, iy, iz) -> (voxid) failed'
    
    
""" --------------------------------- other -------------------------------- """

def test_VoxelMeta_select_axis(rng, voxelmeta):
    axis = rng.integers(0, 3)
    
    ax, other = voxelmeta.select_axis(axis)
    assert ax == axis, "selected axis should be the same as the input axis"
    assert {*other} == {0,1,2} - {axis}, "other axes should be the other two axes"
    
    
    ax_str = 'xyz'[axis]
    ax, other = voxelmeta.select_axis(ax_str)
    assert ax == axis, "selected axis should be the same as the input axis"
    assert {*other} == {0,1,2} - {axis}, "other axes should be the other two axes"
    
    with pytest.raises(IndexError):
        voxelmeta.select_axis(5)
        
def test_VoxelMeta_slice_shape(voxelmeta):
    axes = [0,1,2]
    for axis in axes:
        other_axes = list({*axes} - {axis})
        
        shape = voxelmeta.slice_shape(axis)
        assert {*shape} == {*voxelmeta.shape[other_axes].cpu().numpy()}, "shape should be the same as the voxelmeta shape without the selected axis"


    with pytest.raises(IndexError):
        voxelmeta.slice_shape(5)

def test_VoxelMeta_check_valid_idx(voxelmeta, torch_rng, shape):
    idx_length = torch.randint(2,100, size=(1,), generator=torch_rng).item()
    
    bad_idx = torch.randint(max(shape)+1, 5*max(shape), size=(idx_length,len(shape)), generator=torch_rng)    
    assert ~voxelmeta.check_valid_idx(bad_idx).all(), "all idx should be invalid"
    assert voxelmeta.check_valid_idx(bad_idx, return_components=True).shape == (idx_length, len(shape))
    
    
    good_idx = torch.randint(0, min(shape), size=(idx_length,len(shape)), generator=torch_rng)
    assert voxelmeta.check_valid_idx(good_idx).all(), "all idx should be valid"
    assert voxelmeta.check_valid_idx(good_idx, return_components=True).shape == (idx_length, len(shape))
    
    
def test_VoxelMeta_idx_at(voxelmeta):
    axis = 0
    for i in range(voxelmeta.shape[axis]):
        idx = torch.as_tensor(voxelmeta.idx_at(axis, i))
        assert idx.shape == (voxelmeta.shape[(axis+1)%3] * voxelmeta.shape[(axis+2)%3], 3), f"shape of idx is incorrect for axis {axis} and i {i}"
        assert torch.all(idx[:,axis] == i), f"axis {axis} of idx should be {i}"
        assert torch.all(idx[:,(axis+1)%3] == torch.arange(voxelmeta.shape[(axis+1)%3]).repeat_interleave(voxelmeta.shape[(axis+2)%3])), f"axis {(axis+1)%3} of idx is incorrect for axis {axis} and i {i}"
        assert torch.all(idx[:,(axis+2)%3] == torch.arange(voxelmeta.shape[(axis+2)%3]).repeat(voxelmeta.shape[(axis+1)%3])), f"axis {(axis+2)%3} of idx is incorrect for axis {axis} and i {i}"
        

    axis = 1
    for i in range(voxelmeta.shape[axis]):
        idx = torch.as_tensor(voxelmeta.idx_at(axis, i))
        assert idx.shape == (voxelmeta.shape[(axis+1)%3] * voxelmeta.shape[(axis+2)%3], 3), f"shape of idx is incorrect for axis {axis} and i {i}"
        assert torch.all(idx[:,axis] == i), f"axis {axis} of idx should be {i}"
        assert torch.all(idx[:,(axis+1)%3] == torch.arange(voxelmeta.shape[(axis+1)%3]).repeat(voxelmeta.shape[(axis+2)%3])), f"axis {(axis+1)%3} of idx is incorrect for axis {axis} and i {i}"
        assert torch.all(idx[:,(axis+2)%3] == torch.arange(voxelmeta.shape[(axis+2)%3]).repeat_interleave(voxelmeta.shape[(axis+1)%3])), f"axis {(axis+2)%3} of idx is incorrect for axis {axis} and i {i}"
        

    axis = 2
    for i in range(voxelmeta.shape[axis]):
        idx = torch.as_tensor(voxelmeta.idx_at(axis, i))
        assert idx.shape == (voxelmeta.shape[(axis+1)%3] * voxelmeta.shape[(axis+2)%3], 3), f"shape of idx is incorrect for axis {axis} and i {i}"
        assert torch.all(idx[:,axis] == i), f"axis {axis} of idx should be {i}"
        assert torch.all(idx[:,(axis+1)%3] == torch.arange(voxelmeta.shape[(axis+1)%3]).repeat(voxelmeta.shape[(axis+2)%3])), f"axis {(axis+1)%3} of idx is incorrect for axis {axis} and i {i}"
        assert torch.all(idx[:,(axis+2)%3] == torch.arange(voxelmeta.shape[(axis+2)%3]).repeat_interleave(voxelmeta.shape[(axis+1)%3])), f"axis {(axis+2)%3} of idx is incorrect for axis {axis} and i {i}"
        
def test_VoxelMeta_coord_at(voxelmeta):
    # this is just a wrapper for idx_to_coord(idx_at(axis, i))
    for axes in range(3):
        for i in range(voxelmeta.shape[axes]):
            coord = voxelmeta.coord_at(axes, i)
            assert coord.shape == (voxelmeta.shape[(axes+1)%3] * voxelmeta.shape[(axes+2)%3], 3), f"shape of coord is incorrect for axis {axes} and i {i}"

def test_VoxelMeta_digitize(voxelmeta, torch_rng):
    mins = voxelmeta.ranges[0,0]
    maxs = voxelmeta.ranges[0,1]
    rand_pos_len = torch.randint(5, 100, size=(1,), generator=torch_rng).item()
    
    for axis in range(3):
        # 1D tensor
        rand_pos = torch.rand(size=(1,), generator=torch_rng)*(maxs-mins)+mins
        axisid = voxelmeta.digitize(rand_pos, axis)
        assert np.allclose(axisid, voxelmeta.coord_to_idx(rand_pos)[axis]), "axisid should be the same as the coord_to_idx for that axis"

        # 1D list
        axisid = voxelmeta.digitize(list(rand_pos), axis)
        assert np.allclose(axisid, voxelmeta.coord_to_idx(rand_pos)[axis]), "axisid should be the same as the coord_to_idx for that axis"

        # 2D tensor
        rand_pos = torch.rand(size=(rand_pos_len,), generator=torch_rng)*(maxs-mins)+mins
        axisid = voxelmeta.digitize(rand_pos, axis)
        assert torch.allclose(voxelmeta.digitize(rand_pos, axis),voxelmeta.coord_to_idx(rand_pos.reshape(-1,1).repeat(1,3))[:,axis])