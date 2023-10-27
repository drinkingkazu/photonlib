import h5py
import torch
import numpy as np
from scipy.ndimage import sobel
from .meta import VoxelMeta

class PhotonLib:
    def __init__(self, meta: VoxelMeta, vis:torch.Tensor, eff:float = 1.):
        '''
        Constructor

        Parameters
        ----------
        meta : VoxelMeta
            Defines the volume and its voxelization scheme for a photon library
        vis  : torch.Tensor
            Visibility map as 1D array indexed by the voxel IDs
        eff  : float
            Overall scaling factor for the visibility. Does not do anything if 1.0
        '''
        self._meta = meta
        self._eff = eff
        self.grad_cache = None
        self._vis = vis
    
    @classmethod
    def load(cls, cfg_or_fname:str):
        '''
        Constructor method that can take either a config dictionary or the data file path

        Parameters
        ----------
        cfg_or_fname : str
            If string type, it is interpreted as a path to a photon library data file.
            If dictionary type, it is interpreted as a configuration.
        '''

        if isinstance(cfg_or_fname,dict):
            filepath=cfg_or_fname['photonlib']['filepath']
        elif isinstance(cfg_or_fname,str):
            filepath=cfg_or_fname
        else:
            raise ValueError(f'The argument of load function must be str or dict (received{cfg_or_fname} {type(filepath)})')

        meta = VoxelMeta.load(filepath)
        
        print(f'[PhotonLib] loading {filepath}')
        with h5py.File(filepath, 'r') as f:
            vis = torch.as_tensor(f['vis'][:])
            eff = torch.as_tensor(f.get('eff', default=1.))
        print('[PhotonLib] file loaded')

        #pmt_pos = None
        #if pmt_loc is not None:
        #    pmt_pos = PhotonLib.load_pmt_loc(pmt_loc)

        plib = cls(meta, vis, eff)

        return plib


    @property
    def meta(self):
        return self._meta


    def visibility(self, x):
        '''
        A function meant for analysis/inference (not for training) that returns
        the visibilities for all PMTs given the position(s) in x. Note x is not 
        a normalized coordinate.

        Parameters
        ----------
        x : torch.Tensor
            A (or an array of) 3D point in the absolute coordinate
        
        Returns
        -------
        torch.Tensor
            An instance holding the visibilities in linear scale for the position(s) x.
        '''

        vox_ids = self.meta.coord_to_voxel(x)

        return self._vis[vox_ids]


    #@staticmethod
    #def load_pmt_loc(fpath):
    #    df = pd.read_csv(fpath)
    #    pmt_pos = df[['x', 'y', 'z']].to_numpy()
    #    return pmt_pos

    @property
    def eff(self):
        return self._eff

    @property
    def vis(self):
        return self._vis

    def view(self, arr):
        shape = list(self.meta.shape.numpy()[::-1]) + [-1]
        return torch.swapaxes(arr.reshape(shape), 0, 2)

    @property
    def vis_view(self):
        return self.view(self.vis)

    def __repr__(self):
        return f'{self.__class__} [:memory:]'
    
    def __len__(self):
        return len(self.vis)
    
    @property
    def n_pmts(self):
        return self.vis.shape[1]
     
    def __getitem__(self, vox_id):    
        return self.vis[vox_id]

    def __call__(self, coords):
        vox = self.meta.coord_to_voxel(coords)
        vis = self[vox]
        return vis * self.eff

    def _gradient_on_fly(self, voxel_id):

        idx = self.meta.voxel_to_idx(voxel_id)

        center = torch.ones_like(idx)
        center[idx == 0] = 0
        center = tuple(center)

        high = idx + 2
        low = idx - 1
        low[low<0] = 0
        selected = tuple(slice(l,h) for l,h in zip(low, high))

        data = self.vis_view[selected]
        grad = torch.column_stack([
            [sobel(data[...,pmt], i)[center] for i in range(3)]
            for pmt in range(self.n_pmts)
        ])

        return grad

    def gradient_on_fly(self, voxels):
        voxels = torch.as_tensor(voxels)
        if voxels.dim() == 0:
            return torch.as_tensor([self._gradient_on_fly(voxel)])
        elif voxels.dim() == 1:
            return torch.as_tensor([self._gradient_on_fly(v) for v in voxels])

    def gradient_from_cache(self, voxel_id):
        if self.grad_cache is None:
            raise RunTimeError('grad_cache not loaded')

        return self.grad_cache[voxel_id]

    def gradient(self, voxel_id):
        if self.grad_cache is not None:
            grad = self.gradient_from_cache(voxel_id)
        else:
            grad = self.gradient_on_fly(voxel_id)

        # convert to dV/dx for comparison with torch.autograd.grad
        # sobel = gaus [1,2,1] (x) gaus [1,2,1] (x) diff [1,0,-1]
        # resacle with a factor of  4x4 (gauss) and 2 (finite diff.)
        # grad /= self.meta.norm_step_size * 32
        return grad

    def grad_view(self, d_axis):
        if self.grad_cache is None:
            raise NotImplementedError('gradient_view requires caching')

        d_axis = self.meta.select_axis(d_axis)[0]
        return self.view(self.grad_cache[:,d_axis])

    @staticmethod
    def save(outpath, vis, meta, eff=None):

        if isinstance(vis, torch.Tensor):
            vis = vis.cpu().detach().numpy()
        else:
            vis = np.asarray(vis)

        if vis.ndim == 4:
            vis = np.swapaxes(vis, 0, 2).reshape(len(meta), -1)

        # TODO check dim(vis) and dim(meta)

        print('[PhotonLib] saving to', outpath)
        with h5py.File(outpath, 'w') as f:
            f.create_dataset('numvox', data=meta.shape.cpu().detach().numpy())
            f.create_dataset('min', data=meta.ranges[:,0].cpu().detach().numpy())
            f.create_dataset('max', data=meta.ranges[:,1].cpu().detach().numpy())
            f.create_dataset('vis', data=vis, compression='gzip')

            if eff is not None:
                f.create_dataset('eff', data=eff)

        print('[PhotonLib] file saved')
