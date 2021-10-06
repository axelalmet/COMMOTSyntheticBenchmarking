import os
import numpy as np
import numba as nb
from pde import PDEBase, CartesianGrid, ScalarField, FieldCollection, PlotTracker, FileStorage
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import matplotlib as mpl
mpl.rcParams['font.size'] = 16

from synthetic_ligand_receptor_interaction import synthesize_data

#     R1  R2
# L1  *   *
# L2  *   *
# L3  *   *
# L4  *   *
# L5  *   *



def main():
    case_name = 'case_4'
    case_dir = './data/%s' % case_name
    if not os.path.isdir(case_dir):
        os.mkdir(case_dir)
    ligand_diffusivity = 200.0 * np.ones(5)
    production_rate_scale = np.ones(5)
    receptor_init_scale = np.array([1.0, 1.0])
    binding_rate = np.ones([5,2])
    dissociation_rate = np.ones([5,2]) * 0.001
    degradation_rate_ligand = 0.2 * np.ones(5)
    xlimit = [-200, 200]
    ylimit = [-200, 200]
    xngrid = 50
    yngrid = 50
    ligand_nblob = np.array([5,5,5,5,5], int)
    receptor_nblob = np.array([5,5], int)
    ligand_blob_bandwidth_bound = np.array([[30,50],[30,50],[30,50],[30,50],[30,50]])
    receptor_blob_bandwidth_bound = np.array([[30,50],[30,50]])
    np.random.seed(4)
    for i in range(10):
        simulation_path_parent = './data/' + case_name + '/sim_' + str(i+1)
        if not os.path.isdir(simulation_path_parent):
            os.mkdir(simulation_path_parent)
        simulation_path = './data/' + case_name + '/sim_' + str(i+1) + '/pde'
        if not os.path.isdir(simulation_path):
            os.mkdir(simulation_path)
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