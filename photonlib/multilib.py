import h5py
import torch
import numpy as np
from scipy.ndimage import sobel
from .meta import VoxelMeta, AABox

class MultiLib:
    def __init__(self):
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
        self._vis_v = []
        self._model_id_v = []
        self._meta_v = []
        self._meta = AABox([[0.,0.],[0.,0.],[0.,0.]])

    def add_data(self, vis):

        self._vis_v.append(torch.as_tensor(vis,dtype=torch.float32))
        assert len(self._vis_v[-1].shape) == 2, f'The tensor shape must be 2D array!'
        assert self._vis_v[-1].shape[1] == self._vis_v[0].shape[1], f'The detector count must be {self._vis_v[0].shape[1]}'

        return len(self._vis_v)-1

    def add_meta(self, model_id:int, meta:VoxelMeta):

        # Check the model_id
        assert model_id < len(self.vis_array), f'Invalid model_id ({model_id}) must be < {len(self.vis_array)}'

        # Check the voxel count
        assert len(meta) == len(self.vis_array[model_id]), f'The number of voxels in the meta {len(meta)} and data {len(self.vis_array[meta_id])} are different'

        # Check to make sure the meta is not overlapping
        for abox in self.meta_array:
            assert not abox.overlaps(meta), f"The given meta overlaps with one of registered meta {meta}\n{abox}"

        self._model_id_v.append(model_id)
        self._meta_v.append(meta)
        self._meta.merge(meta)

    @property
    def meta(self):
        return self._meta

    @property
    def meta_array(self):
        return self._meta_v
    

    @property
    def vis_array(self):
        return self._vis_v

    @property
    def model_id_array(self):
        return self._model_id_v    
    
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
            filepath=cfg_or_fname['multilib']['filepath']
        elif isinstance(cfg_or_fname,str):
            filepath=cfg_or_fname
        else:
            raise ValueError(f'The argument of load function must be str or dict (received {cfg_or_fname} {type(cfg_or_fname)})')

        print(f'[MultiLib] loading {filepath}')
        mlib = cls()
        with h5py.File(filepath, 'r') as f:

            data_index = 0
            while True:
                name = f'vis{data_index}'
                if name in f.keys():
                    mlib.add_data(torch.as_tensor(f[name][:],dtype=torch.float32))
                else:
                    break
            model_id_v = [int(v) for v in f['model_id']]
            for i,model_id in enumerate(model_id_v):
                meta = VoxelMeta(f['meta_nvox'][i], f['meta_range'][i])
                mlib.add_meta(model_id,meta)
        print('[MultiLib] file loaded')

        return mlib  

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
        if len(x.shape) == 1:
            x = x[None,:]

        vis = torch.zeros(x.shape[0])
        for i,m in enumerate(self.meta_array):
            mask = m.contain(x)
            vis[mask] += self.vis_array[self.model_id_array[i]][m.coord_to_voxel(x[mask])]
        return vis

    def __repr__(self):
        return f'{self.__class__} [:memory:]'
    
    def __len__(self):
        return sum([len(v) for v in self.vis_array])

    @property
    def n_pmts(self):
        return self.vis_array[0].shape[1]

    def __call__(self, coords):
        return self.visibility(coords)

    @staticmethod
    def save(outpath):

        if isinstance(vis, torch.Tensor):
            vis = vis.cpu().detach().numpy()
        else:
            vis = np.asarray(vis)

        # TODO check dim(vis) and dim(meta)
        print('[MultiLib] saving to', outpath)
        with h5py.File(outpath, 'w') as f:

            f.create_dataset('model_id', data=np.array(self.model_id_array))

            data_range = np.zeros(shape=(len(self.meta_array),3,2),dtype=float)
            data_nvox  = np.zeros(shape=(len(self.meta_array),3),dtype=int)
            for i,m in enumerate(self.meta_array):
                data_range[i] = m.ranges.cpu().detach().numpy()
                data_nvox[i]  = m.shape.cpu().detach().numpy()

            f.create_dataset('meta_range', data=data_range  )
            f.create_dataset('meta_nvox',  data=data_nvox   )
            f.create_dataset('model_id',    data=self.model_id_array)

            for i,vis in enumerate(self.vis_array):
                f.create_dataset(f'vis{i}', data=vis.cpu().detach().numpy(), compression='gzip')

        print('[MultiLib] file saved')
