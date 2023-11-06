from photonlib import PhotonLib
from photonlib import VoxelMeta
import pytest
import h5py
import numpy as np
import torch


from tests.fixtures import fake_photon_library, writable_temp_file, plib, fake_photon_library_shape, fake_photon_library_ranges, num_pmt

@pytest.fixture()
def shapes(fake_photon_library_shape):
    return fake_photon_library_shape

@pytest.fixture()
def ranges(fake_photon_library_ranges):
    return fake_photon_library_ranges

@pytest.fixture()
def plib_with_eff(rng, fake_photon_library, shapes, ranges):
    with h5py.File(fake_photon_library, 'r') as f:
        vis = f['vis'][:]
    meta = VoxelMeta.load(fake_photon_library)
    rand_eff = rng.random()
    return PhotonLib(meta, vis, eff=rand_eff)

   
def test_PhotonLib_init(fake_photon_library, rng):
    # init with constructor
    with h5py.File(fake_photon_library, 'r') as f:
        vis = f['vis'][:]
    meta = VoxelMeta.load(fake_photon_library)
    rand_eff = rng.random()
    plib = PhotonLib(meta, vis, eff=rand_eff)
    assert np.allclose(plib.vis, vis), 'PhotonLib.__init__ does not load visibility correctly'
    assert plib.eff == rand_eff, 'PhotonLib.__init__ does not load err correctly'

    # init with PhotonLib.load(str)
    plib = PhotonLib.load(fake_photon_library)
    assert np.allclose(plib.vis, vis), 'PhotonLib.load does not load visibility correctly'
    
    # init with PhotonLib.load(dict)
    plib = PhotonLib.load(dict(photonlib=dict(filepath=fake_photon_library)))
    assert np.allclose(plib.vis, vis), 'PhotonLib.load does not load visibility correctly'
    
    # fail init with bad input
    with pytest.raises(ValueError):
        PhotonLib.load(123)
        
def test_PhotonLib_save(fake_photon_library, shapes, num_pmt, rng):
    # save to a new file
    with h5py.File(fake_photon_library, 'r') as f:
        vis = f['vis'][:]
    meta = VoxelMeta.load(fake_photon_library)
    
    # test for all keys except eff (numvox, min, max, vis)
    vis_reshaped = np.swapaxes(vis.reshape(list(shapes)[::-1] + [-1]), 0, 2)
    for vis,test in zip([vis, torch.as_tensor(vis), vis_reshaped], ["np", "torch", "np_reshaped"]):
        new_file = writable_temp_file(suffix='.h5')
        PhotonLib.save(new_file, vis, meta)
        
        with h5py.File(fake_photon_library, 'r') as f_old, h5py.File(new_file, 'r') as f_new:
            for key in f_old.keys():
                assert np.allclose(f_old[key][:], f_new[key][:]), f'PhotonLib.save does not save {key} correctly for {test} input'
                
                
    # test for eff
    rand_eff = rng.random()
    new_file = writable_temp_file(suffix='.h5')
    PhotonLib.save(new_file, vis, meta, eff=rand_eff)
    with h5py.File(new_file, 'r') as f_new:
        assert np.allclose(f_new['eff'], rand_eff), 'PhotonLib.save does not save eff correctly'

def test_PhotonLib_view(plib, shapes, num_pmt):
    # test vis_view, which is plib.view(plib.vis)
    assert plib.vis_view.shape == (*shapes, num_pmt), 'PhotonLib.vis_view does not return the correct shape'


def test_PhotonLib_visibility(plib, torch_rng, num_pmt, ranges):
    tranges = torch.as_tensor(ranges)
    num_pos = torch.randint(low=1, high=100, size=(1,), generator=torch_rng).item()
    rand_pos = torch.rand(size=(num_pos, 3), generator=torch_rng)*(tranges[:,1]-tranges[:,0])+tranges[:,0]
    
    # test for single point and multiple points
    for x in [rand_pos[0], rand_pos]:
        assert plib.visibility(x).shape == (*x.shape[:-1], num_pmt)
        
def test_PhotonLib_len(plib, fake_photon_library):
    with h5py.File(fake_photon_library, 'r') as f:
        vis = f['vis'][:]
    assert len(plib) == len(vis), 'PhotonLib.__len__ does not return the correct length'
    
def test_PhotonLib_n_pmts(plib, num_pmt):
    assert plib.n_pmts == num_pmt, 'PhotonLib.n_pmts does not return the correct number of PMTs'
    
def test_PhotonLib_get_item(plib, fake_photon_library):
    with h5py.File(fake_photon_library, 'r') as f:
        vis = f['vis'][:]
    
    for i in range(len(vis)):
        assert np.allclose(plib[i], vis[i]), 'PhotonLib.__getitem__ does not return the correct visibility'

    with pytest.raises(IndexError):
        plib[len(vis)]

def test_PhotonLib_call(plib, plib_with_eff, torch_rng, num_pmt, ranges):
    for plib in [plib, plib_with_eff]:    
        tranges = torch.as_tensor(ranges)
        num_pos = torch.randint(low=1, high=100, size=(1,), generator=torch_rng).item()
        rand_pos = torch.rand(size=(num_pos, 3), generator=torch_rng)*(tranges[:,1]-tranges[:,0])+tranges[:,0]
        
        xs = [rand_pos[0], rand_pos]
        for x in xs:
            assert plib(x).shape == (*x.shape[:-1], num_pmt)
            assert np.allclose(plib(x), plib.visibility(x)*plib.eff), 'PhotonLib.__call__ does not return the correct visibility'
            
def test_PhotonLib_gradient_on_fly(plib, torch_rng, num_pmt, shapes):
    num_vox = np.prod(shapes)
    rand_idx = torch.randint(low=0, high=num_vox, size=(1,), generator=torch_rng).item()
    
    # internal function
    grad = plib._gradient_on_fly(rand_idx)
    assert grad.shape == (len(shapes), num_pmt)
    
    # internal, multiple pts should fail
    with pytest.raises(ValueError):
        plib._gradient_on_fly([rand_idx]*2)
    
    # external function
    grad_ext = plib.gradient_on_fly(rand_idx)
    assert np.allclose(grad, grad_ext), 'PhotonLib.gradient_on_fly does not return the correct gradient'
    assert grad_ext.shape == (len(shapes), num_pmt)
    
    # internal, multiple pts shouldn't fail
    grad_ext = plib.gradient_on_fly(range(10))
    assert grad_ext.shape == (10, len(shapes), num_pmt)
    
    # since there's no cache, gradient should be equal to gradient_on_fly
    assert np.allclose(plib.gradient(range(10)), grad_ext), 'PhotonLib.gradient does not return the correct gradient'
    
    
def test_PhotonLib_gradient_from_cache(plib, torch_rng, num_pmt, shapes, rng):
    num_vox = np.prod(shapes)
    
    # without cache set, gradient_from_cache should fail
    with pytest.raises(RuntimeError):
        plib.gradient_from_cache(...)
    
    # set cache
    random_cache = 10**torch.randint(low=-7, high=-3, size=(num_vox, len(shapes), num_pmt), generator=torch_rng, dtype=torch.float32)
    plib.grad_cache = random_cache
    
    # now that it's set, plib.gradient should return the cached value
    rand_int = torch.randint(low=0, high=num_vox, size=(100,), generator=torch_rng,)
    
    assert torch.allclose(plib.gradient(rand_int[0]), random_cache[rand_int[0]]), 'PhotonLib.gradient does not return the correct gradient from cache'

    # multiple pts shouldn't fail
    assert torch.allclose(plib.gradient_from_cache(rand_int), random_cache[rand_int]), 'PhotonLib.gradient_from_cache does not return the correct gradient from cache'
    
def test_PhotonLib_grad_view(plib, torch_rng, shapes, num_pmt):
    
    for axis in range(3):
        with pytest.raises(RuntimeError):
            plib.grad_view(axis)
            
    # set cache
    num_vox = np.prod(shapes)    
    random_cache = 10**torch.randint(low=-7, high=-3, size=(num_vox, num_pmt, len(shapes)), generator=torch_rng)
    plib.grad_cache = random_cache

    for axis in range(3):
        plib.grad_view(axis)