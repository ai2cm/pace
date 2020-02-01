import f90nml
import os
import fv3.utils.gt4py_utils as utils
from fv3.utils.grid import Grid
#from mpi4py import MPI

#comm = MPI.COMM_WORLD
#rank = comm.Get_rank()
#size = comm.Get_size()
#ranks_per_tile = total_ranks // 6
#tile_number =  rank // ranks_per_tile + 1

def namelist_to_dict(source):
    namelist = dict(source)
    for name, value in namelist.items():
        if isinstance(value, f90nml.Namelist):
            namelist[name] = namelist_to_dict(value)
    flatter_namelist = {}
    for key, value in namelist.items():
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                if subkey in flatter_namelist:
                    raise Exception("Cannot flatten this namelist, duplicate keys: " +subkey)
                flatter_namelist[subkey] = subvalue
        else:
            flatter_namelist[key] = value
    return flatter_namelist

def merge_namelist_defaults(nml):
    defaults = {'grid_type': 0,
                'do_f3d': False,
                'inline_q': False,
                'do_skeb': False  # save dissipation estimate
    }
    defaults.update(nml)
    return defaults

# TODO: Before this can be used, we need to write a module to make the grid data from files on disk and call it
def make_grid_from_namelist(namelist):
    shape_params = {}
    split = namelist['ntiles'] / 6
    for narg in ['npx', 'npy', 'npz']:
        shape_params[narg] = namelist[narg]
    indices = {
        'isd': 0, 'ied': namelist['npx'] + 2 * utils.halo - 2,
        'is_': utils.halo, 'ie': namelist['npx'] + utils.halo - 2,
        'jsd': 0, 'jed': namelist['npy'] + 2 * utils.halo - 2,
        'js': utils.halo, 'je': namelist['npy'] + utils.halo - 2,
    }
    #fv_mp_mod.f90
    # domain_decomp
    # in: npx, npy, nregions, layout, out Atm
    # nregions=6, num_contact=12, npes_per_tile=npes_x*npes_y
    # mpp_define_layout((1, npx-1.1,npy-1), npes_per_tile, layout)
    # in python( 0, npx +2*halo - 2)
    # square domain
    # pe_start(n) = pelist(1) + (n-1)*layout(1)*layout(2)
    # pe_end(n) = pe_start(n) + layout(1)*layout(2) - 1
    # mpp_get_compute_domain, mpp_get_data_domain
    # sorc/fv3gfs.fd/FV3/fms/mpp/include/mpp_domains_util.inc
    '''
   
sorc/fv3gfs.fd/FV3/fms/mpp/include/mpp_domains_define.inc
mpp_compute_extent
    '''
    return Grid(indices, shape_params)
def define_domains(isg, ieg, jsg, jeg):
    isds, ieds = compute_extent(isg, ieg, npx, ibegin, iend)
    jsds, jeds = compute_extent(jsg, jeg, npy, jbegin, jend)

    #def domain_given_rank(num_tiles_per_face, tile_number):
        
def compute_extent(isg, ieg, ndivs):
    ibegin = Array.new
    iend = Array.new
    even_ndivs= ndivs%2 == 0
    even_length = (ieg - isd + 1)%2 == 0
    symmetrize = (even_ndivs and even_length) or\
        ((not even_ndivs) and (not even_length)) or \
        ((not even_ndivs) and even_length and
         ndivs < (ieg - isg + 1)/2)
    is_ = isg
    for ndiv in range(ndivs):
        if ndiv == 0:
            imax = ieg
            ndmax = ndivs
        if ndiv < (ndivs - 1)/2+1:
            ie = is_ + ceiling((imax - is_ + 1) / (ndmax - ndiv)) - 1
            ndmirror = (ndivs - 1) - ndiv
            if (ndmirror > ndiv and symmetrize):
                ibegin[ndmirror] = max(isg+ieg-ie, ie + 1)
                iend[ndmirror] = max(isg+ieg - is_, ie + 1)
                ndmax = ndmax - 1
        else:
            if symmetrize:
                is_ = ibegin[ndiv]
                ie = iend[ndiv]
            else:
                ie = is_ + ceiling((imax - is_ + 1) / (ndmax - ndiv)) - 1
        ibegin[ndiv] = is_
        iend[ndiv] = ie
        if ie < is_:
            raise Exception("ie="+ ie + " cannot be smaller than is=" + is_)
        if (ndiv ==ndivs - 1 and iend[ndiv] != ieg):
            raise Exception("domain extent does not span space completely")
        is_ = ie + 1
    return ibegin, iend

def set_grid(in_grid):
    global grid
    grid = in_grid


namelist = namelist_to_dict(f90nml.read(os.environ['NAMELIST_FILENAME']).items())
namelist = merge_namelist_defaults(namelist)
try:
    grid
except NameError:
    grid = make_grid_from_namelist(namelist)
