from __future__ import annotations

import h5py
import torch
import numpy as np

from tqdm import tqdm
from contextlib import contextmanager

class AABox:
    '''
    Axis-Aligned bounding box in the N-dim cartesian coordinate
    '''
    def __init__(self, ranges):
        '''
        Constructor
        
        Parameters
        ----------
        ranges : array-like
            shape (N,2) holding two N-dimentional points.
            The first point [:,0] is the minimum point of the bounding box.
            The second point [:,1] is the maximum point of the bounding box.
        '''
        ranges = torch.as_tensor(ranges, dtype=torch.float32)
        if len(ranges.shape) != 2 or ranges.shape[1] != 2:
            raise ValueError('ranges must be a 2D array with shape (2,N)')
        self._ranges  = ranges
        self._lengths = torch.diff(self._ranges).flatten()    

    def __repr__(self):
        s = 'Meta'
        for i,var in enumerate('xyz'):
            x0, x1 = self.ranges[i]
            s += f' {var}:({x0},{x1})'
        return s

    def to(self,device):
        self._ranges  = self._ranges.to(device)
        self._lengths = self._lengths.to(device)
        return self

    @property
    def device(self):
        return self._ranges.device
    

    @property
    def x(self):
        return self._ranges[0]
    
    @property
    def y(self):
        return self._ranges[1]

    @property
    def z(self):
        return self._ranges[2]    
    

    @property
    def ranges(self):
        '''
        Access the minimum ([0,:]) and the maximum ([1,:]) points of the bounding box.
        '''
        return self._ranges

    @property
    def lengths(self):
        '''
        Access the lengths of the box along each axis
        '''
        return self._lengths

    def update(self, ranges:torch.Tensor):
        assert self.ranges.shape == ranges.shape, f"Cannot update VoxelMeta ranges by changing dimensions!"
        self._ranges[:]  = ranges[:]
        self._lengths[:] = (torch.diff(self._ranges).flatten())[:]

    def merge(self, abox:AABox):
        '''
        Expand the range to cover the subject AABox instance

        Parameters
        ----------
        abox : AABox
            The subject axis-aligned rectangular box to be contained by this box
        '''
        for dim in range(len(self._ranges)):
            self._ranges[dim][0] = min(self.ranges[dim][0],abox.ranges[dim][0])
            self._ranges[dim][1] = max(self.ranges[dim][1],abox.ranges[dim][1])
        self._lengths[:] = (torch.diff(self._ranges).flatten())[:]

    def overlaps(self, abox:AABox):
        '''
        Check if two boxes overlap (adjacent is considered not overlapping)
        
        Parameters
        ----------
        abox : AABox
            The comparison subject

        Returns
        -------
        bool
            True if this box overlaps with the subject.
        '''

        x = self.x[1] > abox.x[0] and self.x[0] < abox.x[1]
        y = self.y[1] > abox.y[0] and self.y[0] < abox.y[1]
        z = self.z[1] > abox.z[0] and self.z[0] < abox.z[1]

        return x and y and z

    def norm_coord(self, pos):
        '''
        Transform the absolute position to the normalized (-1 to 1 along each axis) within the box
        Parameters
        ----------
        pos : (array-like)
            a (or an array of) position(s) in the absolute coordinate.

        Returns
        -------
        torch.Tensor 
            instance holding the positions in the normalized coordinate using the box
            definition (the range along each axis -1 to 1).
        '''
        pos = torch.as_tensor(pos)

        ranges = torch.as_tensor(self.ranges, device=pos.device)

        norm_pos = pos - ranges[:,0]
        norm_pos /= torch.as_tensor(self.lengths, device=pos.device)
        norm_pos *= 2.
        norm_pos -= 1.

        return norm_pos


    def contain(self, pos):
        '''
        Returns a boolean (array). True = the point is contained. Otherwise False.

        Parameters
        ----------
        pos : (array-like)
            a (or an array of) position(s) in the absolute coordinate

        Returns
        -------
        bool or torch.Tensor
            bool if pos is a single position. a boolean array if the input is an array of positions.
        '''
        pos = torch.as_tensor(pos)
        squeeze = False
        if len(pos.shape) == 1:
            pos=pos[None,:]
            squeeze = True

        contained = torch.ones_like(pos[...,0]).bool()
        for i in range(self.ranges.shape[0]):
            contained = contained & (self.ranges[i,0] < pos[:,i]) & (pos[:,i] < self.ranges[i,1])

        return contained if not squeeze else contained.squeeze()


    @classmethod
    def load(cls, cfg_or_fname):
        '''
        A factory method to construct an instance using a photon library file
        '''

        if isinstance(cfg_or_fname,dict):
            fname = cfg_or_fname['photonlib']['filepath']
        elif isinstance(cfg_or_fname,str):
            fname = cfg_or_fname
        else:
            raise TypeError('The argument must be a configuration dict or a filepath string')

        with h5py.File(fname, 'r') as f:
            ranges = torch.column_stack((
                torch.as_tensor(f['min'],dtype=torch.float32),
                torch.as_tensor(f['max'],dtype=torch.float32)
                )
            )
        return cls(ranges)



class VoxelMeta(AABox):
    '''
    A voxelized version of the AABox. Voxel size is uniform along each axis.
    This class introduces two ways to define a position within the specified volume
    (i.e. within AABox) in addition to the absolute positions.
    
    Index (or idx) ... an array of (N) represents a position by the voxel index along each axis.

    Voxel id (or voxel) ... a single integer uniquely assigned to every voxel.

    The attributes in this class provide ways to convert between these and the absolute coordinates.
    '''

    def __init__(self, shape, ranges):
        '''
        Constructor

        Parameters
        ----------
        shape  : array-like
            N-dim array specifying the voxel count along each axis.
        ranges : array-like
            (N, 2) array given to the AABox constructor (see description).
        '''
        super().__init__(ranges)
        
        self._shape = torch.as_tensor(shape, dtype=torch.int64)
        if self._shape.shape != (self.ranges.shape[0],):
            raise ValueError('shape must be a 1D array with length equal to number of axes')
        self._voxel_size = torch.diff(self.ranges).flatten() / self.shape
       
    def __repr__(self):
        s = 'Meta'
        for i,var in enumerate('xyz'):
            bins = self.shape[i]
            x0, x1 = self.ranges[i]
            s += f' {var}:({x0},{x1},{bins})'
        return s

    def __len__(self):
        return torch.prod(self.shape)

    def to(self,device):
        super().to(device)
        self._shape = self._shape.to(device)
        self._voxel_size = self._voxel_size.to(device)

    def clone(self):
        return VoxelMeta(self.shape.clone(),self.ranges.clone())

    @property
    def shape(self):
        return self._shape

    @property
    def voxel_size(self):
        return self._voxel_size
    
    @property
    def bins(self):
        output = tuple(
            torch.linspace(ranges[0], ranges[1], nbins)
            for ranges, nbins in zip(self.ranges, self.shape+1)
        )

        return output

    @property
    def bin_centers(self):
        centers = tuple((b[1:] + b[:-1]) / 2. for b in self.bins)
        return centers
    
    @property
    def norm_step_size(self):
        # !TODO: (2023-11-05 sy) what is this?
        return 2. / self.shape


    def idx_to_voxel(self, idx):
        '''
        Converts from the index coordinate (N) to the voxel ID (1)

        Parameters
        ----------
        idx : array-like (2D or 3D)
            An array of positions in terms of voxel index along xyz axis

        Returns
        -------
        torch.Tensor
            A 1D array of voxel IDs corresponding to the input axis index(es)
        '''

        idx = torch.as_tensor(idx)

        if len(idx.shape) == 1:
            idx = idx[None,:]

        nx, ny = self.shape[:2]
        vox = idx[:,0] + idx[:,1]*nx + idx[:,2]*nx*ny

        return vox.squeeze()
    
    def voxel_to_idx(self, voxel):
        '''
        Converts from the voxel ID (1) to the index coordinate (N)

        Parameters
        ----------
        voxel : int or array-like (1D)
            A voxel ID or a list of voxel IDs

        Returns
        -------
        torch.Tensor
            A list of index IDs. Shape (3) if the input is a single point. Otherwise (-1,3).

        '''
        voxel = torch.as_tensor(voxel)
        nx, ny = self.shape[:2]

        idx = torch.column_stack([
            voxel % nx,
            torch.floor_divide(voxel, nx) % ny,
            torch.floor_divide(voxel, nx*ny)]
            )

        return idx.squeeze()

    
    def idx_to_coord(self, idx):
        '''
        Converts from the index coordinate (N) to the absolute coordinate (N)

        Parameters
        ----------
        idx : array-like (1D or 2D)
            An array of positions in terms of voxel index along xyz axis

        Returns
        -------
        torch.Tensor
            An array of corresponding positions in the absolute coordinate (at each voxel center)
        '''
        idx = torch.as_tensor(idx)

        voxel_size = torch.as_tensor(self.voxel_size, device=idx.device)
        ranges = torch.as_tensor(self.ranges, device=idx.device)
        coord = (idx+0.5) * voxel_size
        coord += ranges[:, 0]
        return coord


    def voxel_to_coord(self, voxel):
        '''
        Converts from the voxel ID (1) to the absolute coordinate (N)

        Parameters
        ----------
        voxel : int or array-like (1D)
            A voxel ID or a list of voxel IDs

        Returns
        -------
        torch.Tensor
            An array of corresponding positions in the absolute coordinate (at each voxel center)
        '''
        idx = self.voxel_to_idx(voxel)
        return self.idx_to_coord(idx)
    

    def coord_to_idx(self, coord):
        '''
        Converts from the absolute coordinate (N) to the index coordinate (N)

        Parameters
        ----------
        coord : array-like (1D or 2D)
            A (or an array of) position(s) in the absolute coordinate

        Returns
        torch.Tensor
            An array of corresponding voxels represented as index along xyz axis
        -------
        '''
        # TODO(2021-10-29 kvt) validate coord_to_idx
        # TODO(2021-10-29 kvt) check ranges
        coord = torch.as_tensor(coord)

        step = torch.as_tensor(self.voxel_size, device=coord.device)
        ranges = torch.as_tensor(self.ranges, device=coord.device)
        idx = (coord - ranges[:,0]) / step

        idx = self.as_int64(idx)
        idx[idx<0] = 0
        for axis in range(3):
            n = self.shape[axis]
            mask = idx[...,axis] >= n
            idx[mask,axis] = n-1

        return idx


    def coord_to_voxel(self, coord):
        '''
        Converts from the absolute coordinate (N) to the voxel ID (1)

        Parameters
        ----------
        coord : array-like (1D or 2D)
            A (or an array of) position(s) in the absolute coordinate

        Returns
        torch.Tensor
            An array of corresponding voxels represented as integer voxel ID
        '''
        idx = self.coord_to_idx(coord)
        vox = self.idx_to_voxel(idx)
        return vox


    def as_int64(self, idx):
        idx = idx.type(torch.int64)
        return idx
        

    @classmethod
    def load(cls, cfg_or_fname):
        '''
        A factory method to construct an instance using a photon library file

        Parameters
        ----------

        Returns
        -------
        '''

        if isinstance(cfg_or_fname,dict):
            fname = cfg_or_fname['photonlib']['filepath']
        elif isinstance(cfg_or_fname,str):
            fname = cfg_or_fname
        else:
            raise TypeError('The argument must be a configuration dict or a filepath string')

        with h5py.File(fname, 'r') as f:
            shape = torch.as_tensor(f['numvox'][:],dtype=torch.float32)
            ranges = torch.column_stack((
                torch.as_tensor(f['min'],dtype=torch.float32),
                torch.as_tensor(f['max'],dtype=torch.float32)
                )
            )
        return cls(shape, ranges)


    def idx_at(self, axis : int | str, i : int):
        '''
        Slice the volume at the i-th voxel along the specified axis.

        Parameters
        ----------
        axis : int or str
            The axis to slice, str (x, y, or z) or int (0, 1, or 2).
        i : int
            The index along the axis to slice.

        Returns
        -------
        torch.Tensor
            An array of 3D axis index that represents the sliced plane. Shape (N,3) where
            N is the product of voxel count along the sliced plane. For instance, if the
            input argument axis is 0 or 'x', N = ny * nz where ny and nz are the number of
            voxels along the y and z axis.
        '''
        axis, axis_others = self.select_axis(axis)
        axis_a, axis_b = axis_others

        grid = [None] * 3
        grid[axis] = i
        grid[axis_a] = torch.arange(self.shape[axis_a])
        grid[axis_b] = torch.arange(self.shape[axis_b])

        idx = np.column_stack([g.flatten() for g in np.meshgrid(*grid)])
        return idx


    def coord_at(self, axis : int | str, i : int):
        '''
        Slice the volume at the i-th voxel along the specified axis.

        Parameters
        ----------
        axis : int or str
            The axis to slice, str (x, y, or z) or int (0, 1, or 2).
        i : int
            The index along the axis to slice.

        Returns
        -------
        torch.Tensor
            An array of 3D coordinate positions that represent the sliced plane. Shape (N,3).
            See idx_at explanation. This function converts the axis index to 3D position.
        '''

        return self.idx_to_coord(self.idx_at(axis,i))



    def slice_shape(self, axis: int | str):
        '''
        Return the shape of a sliced tensor along the specified axis.
        Use this to create a 2D slice of a visibility map for positions obtained through
        idx_at and coord_at functions.

        Arguments
        ---------
        axis : str | int
            The axis along which a slice is obtained. [0,1,2] or ['x','y','z']

        Returns
        -------
            A tuple of the tensor shape
        '''

        if axis not in ['x','y','z',0,1,2]:
            raise IndexError(f'unknown axis {axis}')

        if axis in [0,'x']:
            return (self.shape[1].item(),self.shape[2].item())

        elif axis in [1,'y']:
            return (self.shape[0].item(),self.shape[2].item())

        else:
            return (self.shape[1].item(),self.shape[0].item())

    
    @staticmethod
    def select_axis(axis):
        axis_to_num = dict(x=0, y=1, z=2)
        
        if isinstance(axis, str) and axis in axis_to_num:
            axis = axis_to_num[axis]
            
        axis_others = [0, 1, 2]
        if axis not in axis_others:
            raise IndexError(f'unknown axis {axis}')
        axis_others.pop(axis)

        return axis, axis_others


    def check_valid_idx(self, idx, return_components=False):
        idx = torch.as_tensor(idx)
        shape = torch.as_tensor(self.shape,device=idx.device)
        mask = (idx >= 0) & (idx < shape)

        if return_components:
            return mask

        return torch.all(mask, axis=-1)

    def digitize(self, x, axis):
        x = torch.as_tensor(x)
        axis = self.select_axis(axis)[0]
        n = self.shape[axis]

        xmin = self.ranges[axis, 0]
        step = self.voxel_size[axis]

        idx = self.as_int64((x - xmin) / step)

        # TODO: (2021-10-29 kvt) exception?
        idx[idx<0] = 0
        idx[idx>=n] = n-1

        return idx


