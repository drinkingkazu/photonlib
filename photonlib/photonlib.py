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
        self._eff = torch.as_tensor(eff,dtype=torch.float32)
        self._vis = torch.as_tensor(vis,dtype=torch.float32)
        self.grad_cache = None

    def contain(self, pts):
        return self._meta.contain(pts)
    
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
            raise ValueError(f'The argument of load function must be str or dict (received {cfg_or_fname} {type(cfg_or_fname)})')

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

    @property
    def device(self):
        return self._vis.device

    def to(self, device=None):
        '''
        Perform device conversion on the PhotonLib.

        Arguments
        ---------
        device: :class:`torch.memory_format`, optinal
            The desired memory format of the returned PhotonLib. Default: `None`.

        Returns
        -------
        plib: PhotonLib`
            If `device=None` or ``self`` already on same device, returns
            ``self``. Otherwise, return a new instance of PhotonLib on the
            desired device.
        '''

        if device is None or self.device == torch.device(device):
            return self

        return PhotonLib(self.meta, self.vis.to(device), self.eff.to(device))

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
        pos = x
        squeeze=False
        if len(x.shape) == 1:
            pos = pos[None,:]
            squeeze=True
        vis = torch.zeros(size=(pos.shape[0],self.n_pmts),dtype=torch.float32).to(self.device)
        mask = self.meta.contain(pos)
        vis[mask] = self.vis[self.meta.coord_to_voxel(pos[mask])]

        return vis if not squeeze else vis.squeeze()

    def gradx(self, x):

        pos = x
        if len(x.shape) == 1:
            pos = pos[None,:]

        grad = torch.zeros(size=(pos.shape[0],self.n_pmts),dtype=torch.float32).to(self.device)

        mask0   = self.meta.contain(pos)
        vox_ids = self.meta.coord_to_voxel(pos[mask0]) + 1
        mask1   = vox_ids >= (len(self.meta)-1)
        vox_ids[mask1] = len(self.meta)-1

        grad[mask0] = (self.vis[vox_ids] - self.vis[vox_ids-1])/self.meta.voxel_size[0]

        return grad


    def visibility2(self, x):
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
        pos = x
        squeeze=False
        if len(x.shape) == 1:
            pos = pos[None,:]
            squeeze=True
        #vis = torch.zeros(size=(pos.shape[0],self.n_pmts),dtype=torch.float32).to(self.device)
        #mask = self.meta.contain(pos)
        #vis[mask] = self.vis[self.meta.coord_to_voxel(pos[mask])]

        return self._vis[self.meta.coord_to_voxel(pos)]


    def gradx2(self, x):

        pos = x
        if len(x.shape) == 1:
            pos = pos[None,:]

        if not hasattr(self,'grad'):
            self.grad = torch.zeros(size=(len(self),self.n_pmts),dtype=torch.float32,device=self.device)
            self.grad[:-1,:] = self._vis[1:] - self._vis[:-1]
            self.grad[-1,:]  = self.grad[-2,:]

        vox_ids = self.meta.coord_to_voxel(pos) + 1
        return self.grad[vox_ids]



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
        return f'{self.__class__} ({self.device})'
    
    def __len__(self):
        return len(self.vis)
    
    @property
    def n_pmts(self):
        return self.vis.shape[1]
     
    def __getitem__(self, vox_id):    
        return self.vis[vox_id]

    def __call__(self, coords):
        return self.visibility(coords) * self.eff

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

