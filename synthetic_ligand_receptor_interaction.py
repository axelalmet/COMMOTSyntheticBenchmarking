import numpy as np
import numba as nb
from pde import PDEBase, CartesianGrid, ScalarField, FieldCollection, PlotTracker, FileStorage
import h5py
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import matplotlib as mpl
mpl.rcParams['font.size'] = 16


# Simulate ligand-receptor binding using simple PDE models
# package: py-pde https://py-pde.readthedocs.io/en/latest/index.html

def gaussian_2d(theta, sigma_x, sigma_y):
    a = np.cos(theta)**2/(2*sigma_x**2) + np.sin(theta)**2/(2*sigma_y**2)
    b = -np.sin(2*theta)/(4*sigma_x**2) + np.sin(2*theta)/(4*sigma_y**2)
    c = np.sin(theta)**2/(2*sigma_x**2) + np.cos(theta)**2/(2*sigma_y**2)
    return a, b, c

def rescale(x, xl, xr):
    return ( 1 - x ) * xl + x * xr

def plot_fields(
    field_list, 
    field_labels, 
    ncol = 5, 
    cmap = 'cividis', 
    xlimit = None,
    ylimit = None,
    filename = None,
    vmin = None,
    vmax = None
):
    nf = len(field_list)
    nrow = int( np.ceil(nf/ncol) )
    ncol = min(ncol, nf)
    fig, ax = plt.subplots(nrow, ncol, figsize=(ncol*5,nrow*5), squeeze=False)
    # gs = gridspec.GridSpec(nrow, ncol)
    # gs.update(wspace=0.025, hspace=0.05)
    
    cnt = 0
    for i in range(nf):
        irow = int(i / ncol)
        icol = i % ncol
        im = ax[irow, icol].imshow(
            field_list[i].T,
            origin = 'lower',
            cmap = cmap,
            extent = (xlimit[0],xlimit[1],ylimit[0],ylimit[1]),
            vmin = vmin,
            vmax = vmax
        )
        divider = make_axes_locatable(ax[irow,icol])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax)
        ax[irow, icol].set_title(field_labels[i])
    if nf < ncol * nrow:
        for i in range(nf, ncol*nrow):
            irow = int(i / ncol)
            icol = i % ncol
            ax[irow,icol].axis('off')
    fig.subplots_adjust(hspace=-0.1, wspace=0.5)
    plt.savefig(filename, bbox_inches='tight')

def plot_field_1D(
    x = None,
    fields = None,
    colors = ['#a1caf1', '#ffcccb', '#e60012', '#002fa7'],
    alphas = [0.75, 0.75, 0.8, 0.8],
    figsize = (10,3),
    filename = None
):
    """
    In default, the fields should be Ligand, Receptor, Occupied receptor, and Sent ligand.
    To not include one, simply set the corresponding alpha value to 0.
    """
    fig = plt.figure(figsize=figsize)
    for i in range(len(fields)):
        plt.fill_between(x, 0, fields[i], color=colors[i], alpha=alphas[i], lw=0)
    plt.savefig(filename, dpi=500, bbox_inches='tight')

def synthesize_data(
    ligand_diffusivity = None,
    production_rate_scale = None,
    receptor_init_scale = None,
    binding_rate = None,
    dissociation_rate = None,
    degradation_rate_ligand = None,
    degradation_rate_receptor = None,
    degradation_rate_complex = None,
    boundary_condition = 'natural',
    xlimit = None,
    ylimit = None,
    xngrid = None,
    yngrid = None,
    ligand_field = {},
    receptor_field = {},
    ligand_nblob = None,
    receptor_nblob = None,
    ligand_noise = None,
    receptor_noise = None,
    ligand_blob_bandwidth_bound = None,
    receptor_blob_bandwidth_bound = None,
    ligand_cutoff = 0.1,
    receptor_cutoff = 0.1,
    ligand_upperbound = 1.0,
    receptor_upperbound = 1.0,    
    simulation_path = None,
    plot_live_simulation = False,
    pde_t_end = 50,
    pde_dt = 0.025
):
    """
    Generate synthetic data by running simulations
    
    Considering n ligands and m receptors

    Parameters
    ----------
    ligand_diffusivity
        (n,) array of ligand diffusivity coefficients
    production_rate_scale
        (n,) array of ligand production rate
    receptor_init_scale
        (m,) array of receptor initial density
    binding_rate
        (n,m) array of binding rate between the n ligands and m receptors
    dissociation_rate
        (n,m) array of dissociation rates of the ligand-receptor complexes
    degradation_rate_ligand
        (n,) array of degradation rates of ligands
    degradation_rate_receptor
        (m,) array of degradation rates of receptors
    degradation_rate_complex
        (n,m) array of degradation rates of ligand-receptor complexes
    boundary_condition
        Boundary condition of the model. Defaults to 'natural'.
    xlimit
        (2,) array of limits on x-axis
    ylimit
        (2,) array of limits on y-axis
    xngrid
        int, number of grids on x-axis
    yngrid
        int, number of grids on y-axis
    ligand_field
        A dictionary of prescribed fields for ligands of shape (xngrid, yngrid)
    receptor_field
        A dictionary of prescribed fields for receptors of shape (xngrid, yngrid)
    ligand_nblob
        (n,) array of number of random blobs to generate for each ligand species
    receptor_nblob
        (m,) array of number of random blobs to generate for each receptor species
    ligand_noise
        (n,) array of noise level when generating ligand_field
    receptor_noise
        (m,) array of noise level when generating receptor_field
    ligand_blob_bandwidth_bound
        (2,) array of lower and upper bound of generated blob
        or
        (n,2) array of lower and upper boundas of generated blob for each ligand species
    receptor_blob_bandwidth_bound
        (2,) array of lower and upper bound of generated blob
        or
        (m,2) array of lower and upper boundas of generated blob for each receptor species
    simulation_path
        All related files will be saved to this folder
    plot_live_simulation
        Whether to plot the live simulation.
    """
    # Initialize basic information
    n = binding_rate.shape[0]
    m = binding_rate.shape[1]
    l = len(binding_rate.nonzero()[0])
    xl, xr = xlimit
    yl, yr = ylimit
    if not ligand_blob_bandwidth_bound is None:
        if len(ligand_blob_bandwidth_bound.shape) == 1:
            ligand_blob_bandwidth_bound = np.repeat(ligand_blob_bandwidth_bound.reshape(1,-1), n, axis=0)
    if not receptor_blob_bandwidth_bound is None:
        if len(receptor_blob_bandwidth_bound.shape) == 1:
            receptor_blob_bandwidth_bound = np.repeat(receptor_blob_bandwidth_bound.reshape(1,-1), m, axis=0)
    
    if ligand_noise is None:
        ligand_noise = np.zeros(n)
    if receptor_noise is None:
        receptor_noise = np.zeros(m)

    if degradation_rate_ligand is None:
        degradation_rate_ligand = np.zeros(n)
    if degradation_rate_receptor is None:
        degradation_rate_receptor = np.zeros(m)
    if degradation_rate_complex is None:
        degradation_rate_complex = np.zeros([n,m], float)
    degradation_rate = list(degradation_rate_ligand) + list(degradation_rate_receptor)

    grid = CartesianGrid(([xl, xr], [yl, yr]), (xngrid, yngrid))
    grid_coord = grid.cell_coords
    initial_data = []
    input_field_list = []
    input_field_labels = []
    # Initialize ligand fields
    ligand_production_rate = []
    for i in range(n):
        if not i in ligand_field.keys():
            bwl = ligand_blob_bandwidth_bound[i,0]
            bwr = ligand_blob_bandwidth_bound[i,1]
            prod_data = ScalarField(grid, 0)
            z = np.zeros([xngrid, yngrid], float)
            for j in range(ligand_nblob[i]):
                x, y, r = np.random.rand(3)
                x = rescale(x, xl, xr)
                y = rescale(y, yl, yr)
                r = rescale(r, bwl, bwr)
                d_sq = (grid_coord[:,:,0]-x) ** 2 + (grid_coord[:,:,1]-y) ** 2
                z = z + np.exp(-d_sq / (r ** 2))
            prod_data.data = prod_data.data + z * production_rate_scale[i]
            noise = ScalarField.random_normal(grid)
            prod_data.data += noise.data * ligand_noise[i]
            prod_data.data[np.where(prod_data.data < ligand_cutoff)] = 0
            prod_data.data[np.where(prod_data.data > ligand_upperbound)] = ligand_upperbound
        else:
            prod_data = ScalarField(grid, 0)
            prod_data.data = ligand_field[i]
        ligand_production_rate.append(prod_data)
        input_field_list.append(prod_data.data)
        input_field_labels.append('Ligand_source_%d' % (i+1))
        init_data = ScalarField(grid, 0, label = 'Ligand_%d' % (i+1))
        initial_data.append(init_data)
    # Initialize receptor fields
    for i in range(m):
        if not i in receptor_field.keys():
            bwl = receptor_blob_bandwidth_bound[i,0]
            bwr = receptor_blob_bandwidth_bound[i,1]
            init_data = ScalarField(grid, 0, label = 'Receptor_%d' % (i+1))
            z = np.zeros([xngrid, yngrid], float)
            for j in range(receptor_nblob[i]):
                x, y, r = np.random.rand(3)
                x = rescale(x, xl, xr)
                y = rescale(y, yl, yr)
                r = rescale(r, bwl, bwr)
                d_sq = (grid_coord[:,:,0]-x) ** 2 + (grid_coord[:,:,1]-y) ** 2
                z = z + np.exp(-d_sq / (r ** 2))
            init_data.data = init_data.data + z * receptor_init_scale[i]
            noise = ScalarField.random_normal(grid)
            init_data.data += receptor_noise[i] * noise.data
            init_data.data[np.where(init_data.data < receptor_cutoff)] = 0
            init_data.data[np.where(init_data.data > receptor_upperbound)] = receptor_upperbound
        else:
            init_data = ScalarField(grid, 0, label = 'Receptor_%d' % (i+1))
            init_data.data = receptor_field[i]
        input_field_list.append(init_data.data)
        input_field_labels.append('Receptor_initial_%d' % (i+1))
        initial_data.append(init_data)
    # Initialize complex fields
    dg_cpx  = []
    for i in range(n):
        for j in range(m):
            if binding_rate[i,j] > 0:
                init_data = ScalarField(grid, 0, label='Complex_%d_%d' % (i+1,j+1))
                initial_data.append(init_data)
                degradation_rate.append( degradation_rate_complex[i,j] )
    degradation_rate = np.array( degradation_rate )
            
    # Perform simulation
    initial_data = FieldCollection(initial_data)
    t_end = pde_t_end # End timestep
    t_track = 2.5 # How often we track simulations
    dt = pde_dt

    eq = LigandReceptorsPDE(
        diffs = ligand_diffusivity,
        prod_rates= ligand_production_rate,
        bind_rates = binding_rate,
        diss_rates = dissociation_rate,
        degrad_rates = degradation_rate,
        bc = boundary_condition,
        num_ligands = n,
        num_receptors = m,
        num_complexes = l
    )

    fileName = simulation_path + '/result.h5'
    storage_write = FileStorage(fileName)
    if plot_live_simulation:
        plot_tracker = PlotTracker(interval=t_track, plot_args={"cmap":'cividis', "vmin":0, "vmax":1}) # To plot the solutions at certain time intervals
        trackers = ['progress', plot_tracker, storage_write.tracker(t_track)] # To save the solutions
    else:
        trackers = ['progress', storage_write.tracker(t_track)]
    sol = eq.solve(initial_data, t_range=t_end, dt = dt, tracker=trackers, method='scipy') # Run the solution
    
    field_list = [sol.fields[i].data for i in range(n+m+l)]
    field_labels = sol.labels
    plot_fields(field_list, field_labels, filename='%s/result.pdf' % simulation_path, xlimit=xlimit, ylimit=ylimit)

    plot_fields(input_field_list, input_field_labels, filename='%s/input.pdf' % simulation_path, xlimit=xlimit, ylimit=ylimit)


    # Write out files
    np.save('%s/grid.npy' % (simulation_path), grid.cell_coords)
    for i in range(n):
        np.save("%s/ligand_prod_%d.npy" % (simulation_path, i+1), ligand_production_rate[i].data)
    for i in range(m):
        np.save("%s/receptor_init_%d.npy" % (simulation_path, i+1), initial_data[n+i].data)
    result_labels = np.array(sol.labels, str)
    np.savetxt('%s/result_labels.txt' % simulation_path, result_labels, fmt="%s")

    meta_data = {}
    meta_data['ligand_diffusivity'] = ligand_diffusivity
    meta_data['binding_rate'] = binding_rate
    meta_data['dissociation_rate'] = dissociation_rate
    meta_data['degradation_rate_ligand'] = degradation_rate_ligand
    meta_data['degradation_rate_receptor'] = degradation_rate_receptor
    meta_data['degradation_rate_complex'] = degradation_rate_complex
    meta_data['xlimit'] = xlimit
    meta_data['ylimit'] = ylimit
    f = open('%s/meta_data.pkl' % simulation_path,"wb")
    pickle.dump(meta_data,f)
    f.close()


def synthesize_data_1D(
    ligand_diffusivity = None,
    production_rate_scale = None,
    receptor_init_scale = None,
    binding_rate = None,
    dissociation_rate = None,
    degradation_rate_ligand = None,
    degradation_rate_receptor = None,
    degradation_rate_complex = None,
    boundary_condition = 'natural',
    xlimit = None,
    xngrid = None,
    ligand_field = {},
    receptor_field = {},
    ligand_nblob = None,
    receptor_nblob = None,
    ligand_noise = None,
    receptor_noise = None,
    ligand_blob_bandwidth_bound = None,
    receptor_blob_bandwidth_bound = None,
    ligand_cutoff = 0.1,
    receptor_cutoff = 0.1,
    ligand_upperbound = 1.0,
    receptor_upperbound = 1.0,    
    simulation_path = None,
    plot_live_simulation = False,
    pde_t_end = 50,
    pde_dt = 0.025
):
    """
    Generate synthetic data by running simulations
    
    Considering n ligands and m receptors

    Parameters
    ----------
    ligand_diffusivity
        (n,) array of ligand diffusivity coefficients
    production_rate_scale
        (n,) array of ligand production rate
    receptor_init_scale
        (m,) array of receptor initial density
    binding_rate
        (n,m) array of binding rate between the n ligands and m receptors
    dissociation_rate
        (n,m) array of dissociation rates of the ligand-receptor complexes
    degradation_rate_ligand
        (n,) array of degradation rates of ligands
    degradation_rate_receptor
        (m,) array of degradation rates of receptors
    degradation_rate_complex
        (n,m) array of degradation rates of ligand-receptor complexes
    boundary_condition
        Boundary condition of the model. Defaults to 'natural'.
    xlimit
        (2,) array of limits on x-axis
    xngrid
        int, number of grids on x-axis
    ligand_field
        A dictionary of prescribed fields for ligands of shape (xngrid)
    receptor_field
        A dictionary of prescribed fields for receptors of shape (xngrid)
    ligand_nblob
        (n,) array of number of random blobs to generate for each ligand species
    receptor_nblob
        (m,) array of number of random blobs to generate for each receptor species
    ligand_noise
        (n,) array of noise level when generating ligand_field
    receptor_noise
        (m,) array of noise level when generating receptor_field
    ligand_blob_bandwidth_bound
        (2,) array of lower and upper bound of generated blob
        or
        (n,2) array of lower and upper boundas of generated blob for each ligand species
    receptor_blob_bandwidth_bound
        (2,) array of lower and upper bound of generated blob
        or
        (m,2) array of lower and upper boundas of generated blob for each receptor species
    simulation_path
        All related files will be saved to this folder
    plot_live_simulation
        Whether to plot the live simulation.
    """
    # Initialize basic information
    n = binding_rate.shape[0]
    m = binding_rate.shape[1]
    l = len(binding_rate.nonzero()[0])
    xl, xr = xlimit
    if not ligand_blob_bandwidth_bound is None:
        if len(ligand_blob_bandwidth_bound.shape) == 1:
            ligand_blob_bandwidth_bound = np.repeat(ligand_blob_bandwidth_bound.reshape(1,-1), n, axis=0)
    if not receptor_blob_bandwidth_bound is None:
        if len(receptor_blob_bandwidth_bound.shape) == 1:
            receptor_blob_bandwidth_bound = np.repeat(receptor_blob_bandwidth_bound.reshape(1,-1), m, axis=0)
    
    if ligand_noise is None:
        ligand_noise = np.zeros(n)
    if receptor_noise is None:
        receptor_noise = np.zeros(m)

    if degradation_rate_ligand is None:
        degradation_rate_ligand = np.zeros(n)
    if degradation_rate_receptor is None:
        degradation_rate_receptor = np.zeros(m)
    if degradation_rate_complex is None:
        degradation_rate_complex = np.zeros([n,m], float)
    degradation_rate = list(degradation_rate_ligand) + list(degradation_rate_receptor)

    grid = CartesianGrid(([[xl, xr]]), (xngrid))
    grid_coord = grid.cell_coords
    initial_data = []
    input_field_list = []
    input_field_labels = []
    # Initialize ligand fields
    ligand_production_rate = []
    for i in range(n):
        if not i in ligand_field.keys():
            bwl = ligand_blob_bandwidth_bound[i,0]
            bwr = ligand_blob_bandwidth_bound[i,1]
            prod_data = ScalarField(grid, 0)
            z = np.zeros([xngrid], float)
            for j in range(ligand_nblob[i]):
                x, r = np.random.rand(2)
                x = rescale(x, xl, xr)
                r = rescale(r, bwl, bwr)
                d_sq = (grid_coord[:,0]-x) ** 2
                z = z + np.exp(-d_sq / (r ** 2))
            prod_data.data = prod_data.data + z * production_rate_scale[i]
            noise = ScalarField.random_normal(grid)
            prod_data.data += noise.data * ligand_noise[i]
            prod_data.data[np.where(prod_data.data < ligand_cutoff)] = 0
            prod_data.data[np.where(prod_data.data > ligand_upperbound)] = ligand_upperbound
        else:
            prod_data = ScalarField(grid, 0)
            prod_data.data = ligand_field[i]
        ligand_production_rate.append(prod_data)
        input_field_list.append(prod_data.data)
        input_field_labels.append('Ligand_source_%d' % (i+1))
        init_data = ScalarField(grid, 0, label = 'Ligand_%d' % (i+1))
        initial_data.append(init_data)
    # Initialize receptor fields
    for i in range(m):
        if not i in receptor_field.keys():
            bwl = receptor_blob_bandwidth_bound[i,0]
            bwr = receptor_blob_bandwidth_bound[i,1]
            init_data = ScalarField(grid, 0, label = 'Receptor_%d' % (i+1))
            z = np.zeros([xngrid], float)
            for j in range(receptor_nblob[i]):
                x, r = np.random.rand(2)
                x = rescale(x, xl, xr)
                r = rescale(r, bwl, bwr)
                d_sq = (grid_coord[:,0]-x) ** 2
                z = z + np.exp(-d_sq / (r ** 2))
            init_data.data = init_data.data + z * receptor_init_scale[i]
            noise = ScalarField.random_normal(grid)
            init_data.data += receptor_noise[i] * noise.data
            init_data.data[np.where(init_data.data < receptor_cutoff)] = 0
            init_data.data[np.where(init_data.data > receptor_upperbound)] = receptor_upperbound
        else:
            init_data = ScalarField(grid, 0, label = 'Receptor_%d' % (i+1))
            init_data.data = receptor_field[i]
        input_field_list.append(init_data.data)
        input_field_labels.append('Receptor_initial_%d' % (i+1))
        initial_data.append(init_data)
    # Initialize complex fields
    dg_cpx  = []
    for i in range(n):
        for j in range(m):
            if binding_rate[i,j] > 0:
                init_data = ScalarField(grid, 0, label='Complex_%d_%d' % (i+1,j+1))
                initial_data.append(init_data)
                degradation_rate.append( degradation_rate_complex[i,j] )
    degradation_rate = np.array( degradation_rate )
            
    # Perform simulation
    initial_data = FieldCollection(initial_data)
    t_end = pde_t_end # End timestep
    t_track = 2.5 # How often we track simulations
    dt = pde_dt

    eq = LigandReceptorsPDE(
        diffs = ligand_diffusivity,
        prod_rates= ligand_production_rate,
        bind_rates = binding_rate,
        diss_rates = dissociation_rate,
        degrad_rates = degradation_rate,
        bc = boundary_condition,
        num_ligands = n,
        num_receptors = m,
        num_complexes = l
    )

    fileName = simulation_path + '/result.h5'
    storage_write = FileStorage(fileName)
    if plot_live_simulation:
        plot_tracker = PlotTracker(interval=t_track, plot_args={"cmap":'cividis', "vmin":0, "vmax":1}) # To plot the solutions at certain time intervals
        trackers = ['progress', plot_tracker, storage_write.tracker(t_track)] # To save the solutions
    else:
        trackers = ['progress', storage_write.tracker(t_track)]
    sol = eq.solve(initial_data, t_range=t_end, dt = dt, tracker=trackers, method='scipy') # Run the solution
    
    field_list = [sol.fields[i].data for i in range(n+m+l)]
    field_labels = sol.labels

    # Write out files
    np.save('%s/grid.npy' % (simulation_path), grid.cell_coords)
    for i in range(n):
        np.save("%s/ligand_prod_%d.npy" % (simulation_path, i+1), ligand_production_rate[i].data)
    for i in range(m):
        np.save("%s/receptor_init_%d.npy" % (simulation_path, i+1), initial_data[n+i].data)
    result_labels = np.array(sol.labels, str)
    np.savetxt('%s/result_labels.txt' % simulation_path, result_labels, fmt="%s")

    meta_data = {}
    meta_data['ligand_diffusivity'] = ligand_diffusivity
    meta_data['binding_rate'] = binding_rate
    meta_data['dissociation_rate'] = dissociation_rate
    meta_data['degradation_rate_ligand'] = degradation_rate_ligand
    meta_data['degradation_rate_receptor'] = degradation_rate_receptor
    meta_data['degradation_rate_complex'] = degradation_rate_complex
    meta_data['xlimit'] = xlimit
    f = open('%s/meta_data.pkl' % simulation_path,"wb")
    pickle.dump(meta_data,f)
    f.close()



# Define the PDE class for a single ligand and multiple receptors 
class LigandReceptorsPDE(PDEBase):
    """ PDE class consisting of a model that considers n ligand and m receptors species.
        We assume that each ligand can bind to multiple receptors and each receptor
        can bind to multiple ligands. Assuming single ligand-receptor complexes, this gives
        n * m possible complexes and n + m + n * m total species to consider. We will however
        refine the model as we go along.
        
        @param diffs, n x 1 vector of diffusivities of the ligand molecules
        @param prod_rates, n x 1 Field collection for production rates (based on expression) for ligand molecules
        @param bind_rates, n x m matrix of binding affinities between ligand-receptor pairs
        @param diss_rates, n x m matrix of disassociation rates of ligand-receptor complexes
        @param degra_rates, (n + m + l) x 1 vector of degradation rates of ligands, receptors, then complexes,
        @param bc, specifies the boundary condition. For now, we go with Neumann BCs
    """
    def __init__(self, diffs, prod_rates, bind_rates, diss_rates, degrad_rates, bc, num_ligands, num_receptors, num_complexes):
        """ Initialise the diffusivity of the ligand, the binding affinities of the receptors, the degradation
            rates of the molecules, and the natural (no-flux boundary conditions. """
        self.diffs = diffs # Ligand diffusivities
        self.prod_rates = prod_rates # Production rate sof ligands
        self.bind_rates = bind_rates # Binding affinities 
        self.diss_rates = diss_rates # Disassociation rates of complexes
        self.degrad_rates = degrad_rates # Degradation rates
        self.num_ligands = num_ligands # Number of ligand species
        self.num_receptors = num_receptors # Number of receptor species
        self.num_complexes = num_complexes # Number of bound complexes
        self.bc = bc # Boundary condition

    def evolution_rate(self, state, t=0):
        """ Function to define the PDEs for the ligand-receptor model. We assume only the ligand molecule
            can diffuse in space, which is then consumed by the receptors. """
        diffs = self.diffs
        bind_rates = self.bind_rates
        diss_rates = self.diss_rates
        prod_rates = self.prod_rates
        degrad_rates = self.degrad_rates
        num_ligands = self.num_ligands
        num_receptors = self.num_receptors
        num_complexes = self.num_complexes

        # Initialise rhs
        rhs = state.copy()
        # First figure out where the non-zero binding (and dissociation rates) are.
        # We're kind of assuming that a complex will have  non-zero binding and
        # dissociation (may need to change this later).
        nonzero_indices = np.nonzero(bind_rates)
        bound_ligand_indices = nonzero_indices[0]
        bound_receptor_indices = nonzero_indices[1]
        nonzero_bind_rates = bind_rates[bound_ligand_indices, bound_receptor_indices]
        nonzero_diss_rates = diss_rates[bound_ligand_indices, bound_receptor_indices]

        for i in range(num_ligands + num_receptors + num_complexes):
            # Define the state 
            u = state[i]

            # Construct PDEs
            if (i < num_ligands): # For ligands, we add diffusion, a production term, and degradation
                u_t = diffs[i] * u.laplace(self.bc) + prod_rates[i] - degrad_rates[i]*u
                
                ligand_index = i # Ligands are first in the list, so match 1-1 with their index

                # Account for any binding to receptors that may occur
                if (ligand_index in bound_ligand_indices):
                    bound_receptors = np.where(bound_ligand_indices == ligand_index)[0] # Get the receptors bound to this ligand
                    for j in bound_receptors:
                        receptor_index = int(bound_receptor_indices[int(j)])
                        # Account for binding with all possible receptor partners
                        receptor_u = state[receptor_index + num_ligands] # Define the receptor state
                        complex_u = state[num_ligands + num_receptors + int(np.where((bound_ligand_indices==ligand_index)&(bound_receptor_indices==receptor_index))[0][0])]
                        u_t += diss_rates[ligand_index, receptor_index] * complex_u - bind_rates[ligand_index, receptor_index] * u * receptor_u
                    
                rhs[i] = u_t # Add equation to list

            elif ((i >= num_ligands)&(i < num_ligands + num_receptors)): # For receptors we consider binding, dissociation and degradation
                u_t = -1.0 * degrad_rates[i] * u

                receptor_index = i - num_ligands # Shift the index across to determine the receptor index (wrt the bindin gmatrix)

                if (receptor_index in bound_receptor_indices):
                    bound_ligands = np.where(bound_receptor_indices == receptor_index)[0] # Get the receptors bound to this ligand

                    for j in bound_ligands:
                        ligand_index = int(bound_ligand_indices[int(j)])
                        # Account for binding and dissociation
                        ligand_u = state[ligand_index]
                        complex_u = state[num_ligands + num_receptors + int(np.where((bound_ligand_indices==ligand_index)&(bound_receptor_indices==receptor_index))[0][0])]
                        u_t += diss_rates[ligand_index, receptor_index] * complex_u - bind_rates[ligand_index, receptor_index] * ligand_u * u

                rhs[i] = u_t # Add equation to list

            else: # Complexes are formed by ligands binding to receptors and dissociate/degrade

                # Shift index over to get the complex index
                complex_index = i - (num_ligands + num_receptors)

                # Work out where the binding rates are equal to the current binding rate considered
                bind_rate = nonzero_bind_rates[complex_index]
                diss_rate = nonzero_diss_rates[complex_index]
                degrad_rate = degrad_rates[i]

                # The actual index pair is the pair such that p*m + q = i
                ligand_index = int(bound_ligand_indices[complex_index]) # # Ligand index
                receptor_index = int(bound_receptor_indices[complex_index]) + num_ligands# Receptor index
                u_ligand = state[ligand_index] # Associated ligand
                u_receptor = state[receptor_index] # Associated receptor
                u_t = bind_rate * u_ligand * u_receptor - diss_rate * u - degrad_rate * u # Binding, dissociation, then degradation
            
                rhs[i] = u_t

        return rhs


def main():
    np.random.seed(123)
    ligand_diffusivity = np.array([200.0, 200.0])
    production_rate_scale = np.array([1.0, 1.0])
    receptor_init_scale = np.array([1.0, 1.0, 1.0])
    binding_rate = np.array([[1.0,1.0,0.0],[0.0,1.0,1.0]])
    dissociation_rate = np.array([[0.001,0.001,0.0],[0.0,0.001,0.001]])
    degradation_rate_ligand = np.array([0.2, 0.2])
    xlimit = [-200, 200]
    ylimit = [-200, 200]
    xngrid = 100
    yngrid = 100
    ligand_nblob = np.array([5,5], int)
    receptor_nblob = np.array([4,4,1], int)
    ligand_blob_bandwidth_bound = np.array([[10,50],[10,50]])
    receptor_blob_bandwidth_bound = np.array([[10,50],[10,50],[10,50]])
    simulation_path = './data/test'
    synthesize_data(
        ligand_diffusivity = ligand_diffusivity,
        production_rate_scale = production_rate_scale,
        receptor_init_scale = receptor_init_scale,
        binding_rate = binding_rate,
        dissociation_rate = dissociation_rate,
        degradation_rate_ligand = degradation_rate_ligand,
        xlimit = xlimit,
        ylimit = ylimit,
        xngrid = xngrid,
        yngrid = yngrid,
        ligand_nblob = ligand_nblob,
        receptor_nblob = receptor_nblob,
        ligand_blob_bandwidth_bound = ligand_blob_bandwidth_bound,
        receptor_blob_bandwidth_bound = receptor_blob_bandwidth_bound,
        simulation_path = simulation_path
    )

if __name__ == '__main__':
    main()

