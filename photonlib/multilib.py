from __future__ import annotations

import h5py
import torch
import numpy as np
from photonlib import VoxelMeta, AABox, PhotonLib

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
        self._data_id_v = []
        self._plib_v = []
        self._n_pmts_v = []
        self._meta = None

    def to(self, device=None):
        '''
        Perform device conversion on the PhotonLib.

        Arguments
        ---------
        device: :class:`torch.memory_format`, optinal
            The desired memory format of the returned PhotonLib. Default: `None`.

        Returns
        -------
        plib: MultiLib`
            If `device=None` or ``self`` already on same device, returns
            ``self``. Otherwise, return a new instance of MultiLib on the
            desired device.
        '''

        if device is None or self.device == torch.device(device):
            return self

        for i in range(len(self._vis_v)):
            self._vis_v[i] = self._vis_v[i].to(device)

        for i in range(len(self._data_id_v)):
            data_id = self._data_id_v[i]
            plib    = self._plib_v[i]
            self._plib_v[i] = PhotonLib(plib.meta.clone(), self._vis_v[data_id])

        return self

    def add_data(self, vis):

        self._vis_v.append(torch.as_tensor(vis,dtype=torch.float32))
        assert len(self._vis_v[-1].shape) == 2, f'The tensor shape must be 2D array!'
        assert self._vis_v[-1].shape[1] == self._vis_v[0].shape[1], f'The detector count must be {self._vis_v[0].shape[1]}'

        return len(self._vis_v)-1

    def add_meta(self, data_id:int, meta:VoxelMeta):

        # Check the data_id
        assert data_id < len(self._vis_v), f'Invalid data_id ({data_id}) must be < {len(self._vis_v)}'

        # Check the voxel count
        assert len(meta) == len(self._vis_v[data_id]), f'The number of voxels in the meta {len(meta)} and data {len(self._vis_v[meta_id])} are different'

        # Check to make sure the meta is not overlapping
        for i,plib in enumerate(self._plib_v):
            abox=plib.meta
            assert not abox.overlaps(meta), f"The given meta overlaps with one of registered meta {meta}\n{abox}"
        self._data_id_v.append(data_id)
        self._plib_v.append(PhotonLib(meta.clone(),self._vis_v[data_id]))
        self._n_pmts_v.append(self._vis_v[data_id].shape[-1])
        # Update own meta
        if self._meta is None:
            self._meta = AABox(meta.ranges.clone())
        else:
            self._meta.merge(meta)

    @property
    def device(self):
        if len(self._vis_v)<1:
            return 'cpu'
        return self._vis_v[0].device    

    @property
    def meta(self):
        return self._meta

    @property
    def plib(self):
        return self._plib_v

    @property
    def data_id(self):
        return self._data_id_v    
    
    @classmethod
    def load(cls, cfg_or_fname : str | dict):
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
                    print('    Loading',name)
                    mlib.add_data(torch.as_tensor(f[name][:],dtype=torch.float32))
                else:
                    break
                data_index += 1
            print('    Finished loading',data_index,'tensors')
            data_id_v = [int(v) for v in f['data_id']]
            for i,data_id in enumerate(data_id_v):
                meta = VoxelMeta(f['meta_nvox'][i], f['meta_range'][i])
                mlib.add_meta(data_id,meta)
            print('    Found',len(mlib.plib),'meta info')
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
        #x = torch.as_tensor(x,dtype=torch.float32)
        if len(x.shape) == 1:
            x = x[None,:]

        vis = torch.zeros(size=(len(x),self.n_pmts),dtype=torch.float32)
        for i,plib in enumerate(self.plib):
            mask = plib.meta.contain(x)
            if mask.sum() < 1: continue
            vis[mask,sum(self._n_pmts_v[:i]):sum(self._n_pmts_v[:i+1])] += plib.visibility(x[mask])
        return vis


    def gradx(self, x):
        if len(x.shape) == 1:
            x = x[None,:]

        grad = torch.zeros(size=(len(x),self.n_pmts),dtype=torch.float32)
        for i in range(len(self._plib_v)):
            grad[:,sum(self._n_pmts_v[:i]):sum(self._n_pmts_v[:i+1])] += self.plib[self.data_id[i]].gradx(x)
        return grad

    def __repr__(self):
        return f'{self.__class__} [:memory:]'
    
    def __len__(self):
        return sum([len(v) for v in self.plib])

    @property
    def n_pmts(self):
        return sum(self._n_pmts_v)

    def __call__(self, coords):
        return self.visibility(coords)

    def save(self, outpath):

        # TODO check dim(vis) and dim(meta)
        print('[MultiLib] saving to', outpath)
        with h5py.File(outpath, 'w') as f:

            data_range = np.zeros(shape=(len(self._plib_v),3,2),dtype=float)
            data_nvox  = np.zeros(shape=(len(self._plib_v),3),dtype=int)
            for i,plib in enumerate(self._plib_v):
                data_range[i] = plib.meta.ranges.cpu().detach().numpy()
                data_nvox[i]  = plib.meta.shape.cpu().detach().numpy()

            f.create_dataset('meta_range', data=data_range  )
            f.create_dataset('meta_nvox',  data=data_nvox   )
            f.create_dataset('data_id',    data=np.array(self.data_id))

            for i,vis in enumerate(self._vis_v):
                f.create_dataset(f'vis{i}', data=vis.cpu().detach().numpy(), compression='gzip')

        print('[MultiLib] file saved')