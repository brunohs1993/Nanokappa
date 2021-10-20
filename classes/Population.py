# calculations
import numpy as np
from scipy.interpolate import interp1d, LinearNDInterpolator
from datetime import datetime

# plotting
import matplotlib.cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import imageio
from scipy.interpolate.ndgriddata import NearestNDInterpolator
from scipy.spatial.transform import Rotation as rot

# geometry
import trimesh as tm
from trimesh.ray.ray_pyembree import RayMeshIntersector # FASTER
# from trimesh.ray.ray_triangle import RayMeshIntersector # SLOWER
from trimesh.triangles import closest_point, points_to_barycentric 

# other
import sys
import copy
from functools import partial
from itertools import repeat
import time

from classes.Constants import Constants

np.set_printoptions(precision=3, threshold=sys.maxsize, linewidth=np.nan)

# matplotlib.use('Agg')

#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=

#   Class with information about the particles contained in the domain.

#   TO DO
#   - Add option to apply heat flux as boundary condition
#   
#   QUESTIONS
#   
#   - Do the q-poins even matter? Because they are not used in any calculation, and they
#     can be recovered by using self.modes i.e. q-point and branch references.

class Population(Constants):
    '''Class comprising the particles to be simulated.'''

    def __init__(self, arguments, geometry, phonon):
        super(Population, self).__init__()

        self.args = arguments

        self.norm = self.args.energy_normal[0]
        
        self.n_of_subvols   = geometry.n_of_subvols

        self.empty_subvols      = self.args.empty_subvols
        self.n_of_empty_subvols = len(self.empty_subvols)
        
        self.particle_type = self.args.particles[0]

        if self.particle_type == 'pmps':
            self.particles_pmps   = float(self.args.particles[1])
            self.N_p              = int(self.particles_pmps*phonon.number_of_modes*(self.n_of_subvols-self.n_of_empty_subvols))
            self.particle_density = self.N_p/geometry.volume
        elif self.particle_type == 'total':
            self.N_p              = int(float(self.args.particles[1]))
            self.particles_pmps   = self.N_p/(phonon.number_of_modes*(self.n_of_subvols - self.n_of_empty_subvols))
            self.particle_density = self.N_p/geometry.volume
        elif self.particle_type == 'pv':
            self.particle_density = float(self.args.particles[1])
            self.N_p              = int(self.particle_density*geometry.volume)
            self.particles_pmps   = self.N_p/(phonon.number_of_modes*(self.n_of_subvols - self.n_of_empty_subvols))
        
        self.dt = float(self.args.timestep[0])   # ps
        self.t  = 0.0
        
        if geometry.subvol_type == 'slice':
            self.slice_axis    = geometry.slice_axis
            self.slice_length  = geometry.slice_length  # angstrom
        
        self.subvol_volume  = geometry.subvol_volume  # angstrom³

        self.facet_indexes    = self.args.bound_facets
        self.bound_cond       = np.array(self.args.bound_cond)
        self.bound_values     = self.args.bound_values
        
        
        self.connected_facets = np.array(self.args.connect_facets).reshape(-1, 2)
        
        self.T_distribution   = self.args.temp_dist[0]

        self.colormap = self.args.colormap[0]
        self.fig_plot = self.args.fig_plot
        self.rt_plot  = self.args.rt_plot

        self.current_timestep = 0

        print('Initialising population...')

        self.initialise_all_particles(geometry, phonon) # initialising particles

        print('Initialising reservoirs...')
        self.initialise_reservoirs(geometry, phonon)            # initialise reservoirs

        self.n_timesteps, self.collision_facets, self.collision_positions, self.positions = self.timesteps_to_boundary(self.positions, self.group_vel, geometry) # calculating timesteps to boundary

        self.results_folder_name = self.args.results_folder

        print('Creating convergence file...')
        self.open_convergence()
        self.write_convergence(phonon)

        if len(self.rt_plot) > 0:
            print('Starting real-time plot...')
            self.rt_graph, self.rt_fig = self.init_plot_real_time(geometry, phonon)
        
        print('Initialisation done!')
        # self.print_sim_info(geometry, phonon)
        
    def initialise_modes(self, phonon):
        '''Generate first modes.'''

        # creating unique mode matrix
        self.unique_modes = np.stack(np.meshgrid( np.arange(phonon.number_of_qpoints), np.arange(phonon.number_of_branches) ), axis = -1 ).reshape(-1, 2).astype(int)

        if self.particle_type == 'pmps':
            # if particles per mode, per subvolume are defined, use tiling
            modes = np.tile( self.unique_modes, (int( self.particles_pmps*(self.n_of_subvols-self.n_of_empty_subvols) ), 1) )
        else:
            # if not, generate randomly
            modes = np.random.randint(low = 0, high = phonon.number_of_modes , size = self.positions.shape[0])
            modes = self.unique_modes[modes, :]

        return modes.astype(int)

    def enter_probability(self, geometry, phonon):
        '''Calculates the probability of a particle with the modes of a given material (phonon)
        to enter the facets of a given geometry (geometry) with imposed boundary conditions.'''

        vel = np.transpose(phonon.group_vel, (0, 2, 1))           # shape = (Q, 3, J) - Group velocities of each mode
        normals = -geometry.mesh.facets_normal[self.res_facet, :] # shape = (R, 3)    - unit normals of each facet with boundary conditions
                                                                  # OBS: normals are reversed to point inwards (in the direction of entering particles).
        group_vel_parallel = np.dot(normals, vel)                 # shape = (R, Q, J) - dot product = projection of velocity over normal
        
        # filtering particles that don't travel in this direction (V . n <= 0)
        # group_vel_parallel = np.where(group_vel_parallel > 0, group_vel_parallel, 0) 

        # Probability of a particle entering the domain
        enter_prob = group_vel_parallel*self.dt/self.bound_thickness.reshape(-1, 1, 1)   # shape = (F, Q, J)

        return enter_prob

    def generate_positions(self, number_of_particles, mesh, key):
        '''Initialise positions of a given number of particles'''
        
        if key == 'random':
            positions = tm.sample.volume_mesh(mesh, number_of_particles)
            in_mesh = mesh.contains(positions)
            positions = positions[in_mesh, :]
            while positions.shape[0]<number_of_particles:
                new_positions = tm.sample.volume_mesh(mesh, number_of_particles-positions.shape[0])
                in_mesh = mesh.contains(new_positions)
                new_positions = new_positions[in_mesh, :]
                positions = np.concatenate((positions, new_positions), axis = 0)

        elif key == 'center':
            center = mesh.center_mass
            positions = np.ones( (number_of_particles, 3) )*center # Generate all points at the center of the bounding box, positions in angstrom

        return positions

    def initialise_all_particles(self, geometry, phonon):
        '''Uses Population atributes to generate all particles.'''
        
        # initialising positions one mode at a time (slower but uses less memory)
        self.positions = np.zeros((0, 3))

        if self.args.part_dist[0] == 'random_domain':
            number_of_particles = self.N_p
            self.positions = self.generate_positions(number_of_particles, geometry.mesh, key = 'random')
        
        elif self.args.part_dist[0] == 'center_domain':
            number_of_particles = self.N_p
            self.positions = self.generate_positions(number_of_particles, geometry.mesh, key = 'center')
        
        elif self.args.part_dist[0] == 'random_subvol':
            for i in range(self.n_of_subvols):
                if i not in self.empty_subvols:
                    
                    number_of_particles = int(np.ceil(self.particles_pmps*phonon.number_of_modes))

                    new_positions  = self.generate_positions(number_of_particles, geometry.subvol_meshes[i], key = 'random')
                    self.positions = np.vstack((self.positions, new_positions))
            
        elif self.args.part_dist[0] == 'center_subvol':
            for i in range(self.n_of_subvols):
                if i not in self.empty_subvols:

                    number_of_particles = int(np.ceil(self.particles_pmps*phonon.number_of_modes))
                    
                    new_positions = self.generate_positions(number_of_particles, geometry.subvol_meshes[i], key = 'center')
                    self.positions = np.vstack((self.positions, new_positions))

        # initialising slice id
        self.subvol_id = self.get_subvol_id(geometry)

        # assign temperatures
        self.temperatures, self.subvol_temperature = self.assign_temperatures(self.positions, geometry)

        # assigning properties from Phonon
        self.modes = self.initialise_modes(phonon) # getting modes
        self.omega, self.group_vel = self.assign_properties(self.modes, phonon)

        # occupation considering reference
        self.occupation      = phonon.calculate_occupation(self.temperatures, self.omega, reference = True)
        self.energies        = self.hbar*self.omega*self.occupation
        
        # getting scattering arrays
        self.n_timesteps, self.collision_facets, self.collision_positions, self.positions = self.timesteps_to_boundary(self.positions, self.group_vel, geometry)

        self.collision_cond = self.get_collision_condition(self.collision_facets)

        # initialising subvol arrays
        self.subvol_energy    = phonon.calculate_crystal_energy(self.subvol_temperature)
        self.subvol_heat_flux = np.zeros(self.n_of_subvols)
        self.subvol_N_p       = (np.ones(self.n_of_subvols)*self.N_p/self.n_of_subvols).astype(int)

    def initialise_reservoirs(self, geometry, phonon):
        
        # number of reservoirs
        self.n_of_reservoirs = (self.bound_cond == 'T').sum()+(self.bound_cond == 'F').sum()
        
        # which facets contain attached reservoirs
        mask           = np.logical_or(self.bound_cond == 'T', self.bound_cond == 'F')
        self.res_facet = self.facet_indexes[mask]

        # setting temperatures
        facets_temp = np.where(self.bound_cond == 'T')[0] # which FACETS have imposed temperature, indexes
        facets_flux = np.where(self.bound_cond == 'F')[0] # which FACETS have imposed heat flux  , indexes

        mask_temp = np.isin(self.res_facet, facets_temp) # which RESERVOIRS (res1, res2, res3...) have imposed temperature, boolean
        mask_flux = np.isin(self.res_facet, facets_flux) # which RESERVOIRS (res1, res2, res3...) have imposed heat flux  , boolean
        
        self.res_facet_temperature            = np.ones(self.n_of_reservoirs)         # initialise array with None, len = n_of_reservoirs
        self.res_facet_temperature[:]         = None
        self.res_facet_temperature[mask_temp] = self.bound_values[facets_temp]        # attribute imposed temperatures
        self.res_facet_temperature[mask_flux] = self.bound_values[facets_temp].mean() # initialising with the mean of the imposed temperatures

        # The virtual volume of the reservoir is considered a extrusion of the facet.
        # The extrusion thickness is calculated such that one particle per mode is created,
        # obbeying the density of particles generated (i.e. thickness = number_of_modes/(particle_density*facet_area) );
        
        self.bound_thickness = phonon.number_of_modes/(self.particle_density*geometry.mesh.facets_area[self.res_facet])

        self.enter_prob = self.enter_probability(geometry, phonon)

        self.fill_reservoirs(geometry, phonon)  # Generate first particles

    def fill_reservoirs(self, geometry, phonon):

        # generate random numbers
        dice = np.random.rand(self.n_of_reservoirs, phonon.number_of_qpoints, phonon.number_of_branches)

        # check if particles entered the domain comparing with their probability
        in_modes_mask = dice < self.enter_prob # shape = (R, Q, J)

        # calculate how many particles entered each facet
        N_p_facet = in_modes_mask.astype(int).sum(axis = (1, 2))    # shape = (R,)

        # initialise new arrays
        self.res_positions   = np.zeros((0, 3))
        self.res_modes       = np.zeros((0, 2))
        self.res_facet_id    = np.zeros(0)

        for i in range(self.n_of_reservoirs):                         # for each reservoir
            facet  = self.res_facet[i]                                # get its facet index
            mesh   = geometry.res_meshes[i]                           # select boundary
            normal = -np.array(geometry.mesh.facets_normal)[facet, :] # get boundary normal
            n      = N_p_facet[i]                                     # the number of particles on that facet

            new_modes = np.vstack(np.where(in_modes_mask[i, :, :])).T # gets which modes entered that facet

            # generate positions on boundary and add a the drift to the domain
            new_positions, _ = tm.sample.sample_surface(mesh, n)
            new_positions    = new_positions + normal*((self.enter_prob[i, new_modes[:, 0], new_modes[:, 1]] - dice[i, new_modes[:, 0], new_modes[:, 1]] )*self.bound_thickness[i]).reshape(-1, 1)

            outside = tm.proximity.signed_distance(geometry.mesh, new_positions) <= 0
                      #~geometry.mesh.contains(new_positions) # see if any new particles are generated outside

            while np.any(outside):
                # print('Res', i, outside.sum())
                new_positions[outside, :], _ = tm.sample.sample_surface(mesh, int(outside.sum()))
                new_positions[outside, :]    = new_positions[outside, :] + normal*((self.enter_prob[i, new_modes[outside, 0], new_modes[outside, 1]] - dice[i, new_modes[outside, 0], new_modes[outside, 1]])*self.bound_thickness[i]).reshape(-1, 1)
                outside[outside] = tm.proximity.signed_distance(geometry.mesh, new_positions[outside, :]) <= 0
                                   #~geometry.mesh.contains(new_positions[outside, :]) # see if any new particles are generated outside

            # adding to the new population
            self.res_positions = np.vstack((self.res_positions , new_positions ))                  # add to the positions
            self.res_modes     = np.vstack((self.res_modes     , new_modes     )).astype(int)      # add to the modes
            self.res_facet_id  = np.concatenate((self.res_facet_id, np.ones(n)*facet)).astype(int) # add to the reservoir id

        self.res_group_vel  = phonon.group_vel[self.res_modes[:, 0], self.res_modes[:, 1], :]        # retrieve velocities
        self.res_omega      = phonon.omega[self.res_modes[:, 0], self.res_modes[:, 1]]               # retrieve frequencies

        facets_flux = np.where(self.bound_cond == 'F')[0] # which FACETS have imposed heat flux  , indexes
        mask_flux = np.isin(self.res_facet, facets_flux)  # which RESERVOIRS (res1, res2, res3...) have imposed heat flux  , boolean
        
        self.res_facet_temperature[mask_flux] = np.array(list(map(self.calculate_temperature_for_flux       ,
                                                                  np.arange(self.n_of_reservoirs)[mask_flux],
                                                                  repeat(geometry)                          ,
                                                                  repeat(phonon)                          ))) # calculate temperature
        
        indexes = np.where(self.res_facet_id.reshape(-1, 1) == self.res_facet)[1]   # getting RESERVOIR indexes

        self.res_temperatures = self.res_facet_temperature[indexes] # impose temperature values to the right particles

        self.res_occupation = phonon.calculate_occupation(self.res_temperatures, self.res_omega, reference = True)
        self.res_energies   = self.hbar*self.res_omega*self.res_occupation

    def calculate_temperature_for_flux(self, reservoir, geometry, phonon):
        facet  = self.res_facet[reservoir]
        normal = -geometry.mesh.facets_normal[facet, :]
        
        indexes_out = np.logical_and(self.n_timesteps<=0, self.collision_facets == facet)                      # particles leaving through that facet

        print('Checking logic')
        print((self.n_timesteps<=0).sum(), (self.collision_facets == facet).sum())
        
        modes_out      = self.modes[indexes_out, :]
        omega_out      = self.omega[indexes_out]
        occupation_out = self.occupation[indexes_out]+phonon.reference_occupation[modes_out[:, 0], modes_out[:, 1]]
        
        energies_out  = self.hbar*omega_out*occupation_out
        flux_out     = (energies_out.reshape(-1, 1)*self.group_vel[indexes_out, :]*normal).sum() # eV angstrom/ps
        flux_out     = phonon.normalise_to_density(flux_out)                  # eV angstrom/ps

        print('e out shape', energies_out.shape)
        
        indexes_in  = self.res_facet_id == facet                # particles coming from that facet
        
        total_flux  = self.bound_values[facet]                  # imposed flux in W/m^2

        flux_in = (total_flux + flux_out)/self.eVpsa2_in_Wm2    # flux comming in, in eV/ps angs^2

        T_new = 300
        T_old = 10 # initial temperature
        
                
        # K1 = self.hbar/(flux_in*phonon.volume_unitcell)
        # K2 = (self.res_omega[indexes_in].reshape(-1, 1)*self.res_group_vel[indexes_in, :]*normal).sum(axis = 1)
        # K3 = np.exp(self.hbar*self.res_omega[indexes_in])

        K1 = self.hbar/self.kb
        K2 = self.res_omega[indexes_in]*(self.res_group_vel[indexes_in, :]*normal).sum(axis = 1) - flux_in
        K3 = (self.res_group_vel[indexes_in, :]*normal).sum(axis = 1)*np.exp(self.res_omega[indexes_in]*K1)

        print('Calculating flux reservoir temperature...')
        print('Flux out', flux_out)
        print('Flux in', flux_in)

        while np.abs((T_new-T_old)/T_old) > 1e-6:

            T_old = T_new

            occupation = phonon.calculate_occupation(T_old, self.res_omega[indexes_in], reference = False)

            dT = K1*((K2*occupation).sum()/((K3*np.log(occupation)).sum()*np.exp(-T_old)))

            T_new = T_old - dT
            
            print(T_old, T_new, dT)

            
        
        return T_new

    def add_reservoir_particles(self, geometry, phonon):
        '''Add the particles that came from the reservoir to the main population. Calculates flux balance for each reservoir.'''

        self.res_n_timesteps, self.res_collision_facets, self.res_collision_positions, self.res_positions = self.timesteps_to_boundary(self.res_positions, self.res_group_vel, geometry)

        self.res_collision_cond = self.get_collision_condition(self.res_collision_facets)
        # self.res_collision_cond                                       = np.empty(self.res_collision_facets.shape, dtype = str)
        # self.res_collision_cond[ np.isnan(self.res_collision_facets)] = 'N'
        # non_nan_facets = self.res_collision_facets[~np.isnan(self.res_collision_facets)].astype(int)
        # self.res_collision_cond[~np.isnan(self.res_collision_facets)] = self.bound_cond[non_nan_facets]

        self.positions           = np.vstack((self.positions, self.res_positions))
        self.modes               = np.vstack((self.modes    , self.res_modes    ))
        self.group_vel           = np.vstack((self.group_vel, self.res_group_vel))

        self.n_timesteps         = np.concatenate((self.n_timesteps        , self.res_n_timesteps        ))
        self.collision_facets    = np.concatenate((self.collision_facets   , self.res_collision_facets   ))
        self.collision_positions = np.concatenate((self.collision_positions, self.res_collision_positions))
        self.collision_cond      = np.concatenate((self.collision_cond     , self.res_collision_cond     ))
        self.temperatures        = np.concatenate((self.temperatures       , self.res_temperatures       ))
        self.omega               = np.concatenate((self.omega              , self.res_omega              ))
        self.occupation          = np.concatenate((self.occupation         , self.res_occupation         ))

    def assign_properties(self, modes, phonon):
        '''Get properties from the indexes.'''

        omega              = phonon.omega[ modes[:,0], modes[:,1] ]        # THz * rad
        group_vel          = phonon.group_vel[ modes[:,0], modes[:,1], : ] # THz * angstrom

        return omega, group_vel

    def assign_temperatures(self, positions, geometry):
        '''Atribute initial temperatures imposing fixed temperatures on first and last slice. Constant at T_cold unless specified otherwise.'''

        number_of_particles = positions.shape[0]
        key = self.T_distribution
        
        bound_T = self.bound_values[self.bound_cond == 'T'] # np.array([self.bound_values[i] for i in range(len(self.bound_values)) if self.bound_cond[i] == 'T'])
        
        temperatures = np.zeros(number_of_particles)    # initialise temperature array

        if key == 'linear':
            # calculates T at the center for each subvolume
            
            bound_positions = geometry.res_centroid[self.bound_cond == 'T', :] #np.array([geometry.res_centroid[i, :] for i in range(len(self.bound_values)) if self.bound_cond[i] == 'T'])

            if len(bound_T) > 2:
                subvol_temperatures = LinearNDInterpolator(bound_positions, bound_T, fill = bound_T.mean())(geometry.subvol_center)
            elif len(bound_T) == 1:
                subvol_temperatures = np.ones(self.n_of_subvols)*bound_T
            elif len(bound_T) == 2:
                direction = bound_positions[1, :]-bound_positions[0, :]
                K = ((bound_positions[0, :]-geometry.subvol_center)*direction).sum(axis = 1)
                alphas = K/(direction**2).sum()

                subvol_temperatures = bound_T[0]+alphas*(bound_T[1]-bound_T[0])
                temperatures = (self.subvol_id*subvol_temperatures[self.subvol_id]).sum(axis = 1)/self.subvol_id.sum(axis = 1)

        elif key == 'random':
            temperatures        = np.random.rand(number_of_particles)*(bound_T.ptp() ) + bound_T.min()
            
        elif key == 'constant_hot':
            temperatures       = np.ones(number_of_particles)*bound_T.max()
            subvol_temperatures = np.ones(   self.n_of_subvols)*bound_T.max()
        elif key == 'constant_cold':
            temperatures        = np.ones(number_of_particles)*bound_T.min()
            subvol_temperatures = np.ones(   self.n_of_subvols)*bound_T.min()
        elif key == 'constant_mean':
            temperatures       = np.ones(number_of_particles)*bound_T.mean()
            subvol_temperatures = np.ones(   self.n_of_subvols)*bound_T.mean()
        elif key == 'custom':
            subvol_temperatures = np.array(self.args.subvol_temp)
            temperatures = (self.subvol_id*subvol_temperatures[self.subvol_id]).sum(axis = 1)/self.subvol_id.sum(axis = 1)
        
        return temperatures, subvol_temperatures
    
    def get_collision_condition(self, collision_facets):
        
        collision_cond = np.empty(collision_facets.shape, dtype = str) # initialise as an empty string array
        
        collision_cond[ np.isnan(collision_facets)] = 'N'              # identify all nan facets with 'N'

        non_nan_facets = collision_facets[~np.isnan(collision_facets)].astype(int) # get non-nan facets
        
        collision_cond[~np.isnan(collision_facets)] = self.bound_cond[non_nan_facets] # save their condition

        return collision_cond

    def get_subvol_id(self, geometry):
        
        distance_tolerance = 1e-3  # defining tolerance

        # checking if particles inside bounding boxes - should be faster, since it saves signed_distance calls

        bounds_coordinates = np.array([mesh.bounds for mesh in geometry.subvol_meshes]) # getting bounds - SV, 2, 3

        lower_bounds = bounds_coordinates[:, 0, :].reshape(len(geometry.subvol_meshes), 1, 3)
        upper_bounds = bounds_coordinates[:, 1, :].reshape(len(geometry.subvol_meshes), 1, 3)

        in_bounds = np.all(np.logical_and(self.positions >= lower_bounds-distance_tolerance, self.positions <= upper_bounds+distance_tolerance), axis = 2).T

        # using signed_distance when used spherical subvolumes

        if geometry.subvol_type in ['box', 'slice']:
            subvol_id = in_bounds
        else:
            subvol_id = np.zeros(in_bounds.shape)
            for i in range(self.n_of_subvols):
                mesh = geometry.subvol_meshes[i]
                
                distances = tm.proximity.signed_distance(mesh, self.positions[in_bounds[:, i], :])
                subvol_id[in_bounds[:, i], i] = distances > -distance_tolerance
        
        no_subvol = subvol_id.sum(axis = 1) == 0

        while np.any(no_subvol): # while there is any particle not contained in a subvol
            
            self.bring_back_particles(self.positions[no_subvol, :], geometry) # try to bring particles back
            
            # calculate new collision
            (self.n_timesteps[no_subvol]           ,
             self.collision_facets[no_subvol]      ,
             self.collision_positions[no_subvol, :],
             self.positions[no_subvol, :]          ) = self.timesteps_to_boundary(self.positions[no_subvol, :],
                                                                                  self.group_vel[no_subvol, :],
                                                                                  geometry                    )
            
            self.collision_cond[no_subvol] = self.get_collision_condition(self.collision_facets[no_subvol])

            # re-find subvol
            in_bounds[no_subvol, :] = np.all(np.logical_and(self.positions[no_subvol, :] >= lower_bounds-distance_tolerance,
                                                            self.positions[no_subvol, :] <= upper_bounds+distance_tolerance), axis = 2).T

            if geometry.subvol_type in ['box', 'slice']:
                subvol_id[no_subvol, :] = in_bounds[no_subvol, :]
            else:
                for i in range(self.n_of_subvols):
                    mesh = geometry.subvol_meshes[i]
                    
                    distances = tm.proximity.signed_distance(mesh, self.positions[in_bounds[no_subvol, i], :])
                    subvol_id[in_bounds[no_subvol, i], i] = distances > -distance_tolerance
            
            # update those without subvols
            no_subvol = subvol_id.sum(axis = 1) == 0

        if np.any(subvol_id.sum(axis = 1) == 0):
            print('0 subvol particles', self.positions[subvol_id.sum(axis = 1) == 0, :])
            print('group vel', self.group_vel[subvol_id.sum(axis = 1) == 0, :])
            print('n ts', self.n_timesteps[subvol_id.sum(axis = 1) == 0])
            print('col cond', self.collision_cond[subvol_id.sum(axis = 1) == 0])
            print('col facet', self.collision_facets[subvol_id.sum(axis = 1) == 0])
            print('col pos', self.collision_positions[subvol_id.sum(axis = 1) == 0])

            print('2 subvol particles', self.positions[subvol_id.sum(axis = 1) == 2, :])
            
        self.subvol_N_p = subvol_id.sum(axis = 0)  # number of particles in each subvolume

        return subvol_id

    def refresh_temperatures(self, geometry, phonon):
        '''Refresh energies and temperatures while enforcing boundary conditions as given by geometry.'''
        self.energies = self.occupation*self.omega*self.hbar    # eV
        
        self.subvol_id = self.get_subvol_id(geometry)

        n_of_subvols = self.subvol_id.sum(axis = 1)

        self.subvol_energy = (self.subvol_id*self.energies.reshape(-1, 1)).sum(axis = 0)

        if self.norm == 'fixed':
            normalisation = phonon.number_of_modes/(self.particle_density*geometry.subvol_volume)
        elif self.norm == 'mean':
            normalisation = phonon.number_of_modes/self.subvol_N_p
            normalisation = np.where(np.isnan(normalisation), 0, normalisation)
        
        self.subvol_energy = self.subvol_energy*normalisation
        self.subvol_energy = phonon.normalise_to_density(self.subvol_energy)

        self.subvol_energy += phonon.reference_energy

        self.subvol_temperature = phonon.temperature_function(self.subvol_energy)

        self.temperatures = (self.subvol_temperature*self.subvol_id).sum(axis = 1)/n_of_subvols

        if np.any(n_of_subvols == 0):
            index = n_of_subvols == 0
            print(self.omega[index])
            print(self.positions[index, :])
            print(self.group_vel[index, :])
            print(self.collision_cond[index])
            print(self.collision_facets[index])
            print(self.n_timesteps[index])
            print(self.collision_positions[index, :])

            fig = plt.figure(figsize = (10, 10), dpi = 100)
            ax = fig.add_subplot(111, projection = '3d')
            ax.scatter(geometry.mesh.vertices[:, 0],
                       geometry.mesh.vertices[:, 1],
                       geometry.mesh.vertices[:, 2], c = 'k')
            ax.scatter(self.positions[index, 0],
                       self.positions[index, 1],
                       self.positions[index, 2], c = 'b')
            ax.scatter(self.collision_positions[index, 0],
                       self.collision_positions[index, 1],
                       self.collision_positions[index, 2], c = 'r')
            for i in range(index.sum()):
                init = self.positions[index, :][i, :]
                end  = init + self.group_vel[index, :][i, :] * self.dt
                ax.plot([init[0], end[0]],
                        [init[1], end[1]],
                        [init[2], end[2]], linewidth = 1, color = 'k')
            plt.tight_layout()
            plt.show()
            plt.savefig('error_particle.png')

    def calculate_heat_flux(self, geometry, phonon):
        
        # velocity * energies * subvolume (1 or 0)
        VxE = self.subvol_id.astype(int).reshape(self.n_of_subvols, -1, 1)*self.group_vel*self.energies.reshape(-1, 1) # shape = (SV, N, 3)

        VxE = VxE.sum(axis = 1)

        if self.norm == 'fixed':
            heat_flux = VxE*phonon.number_of_modes/(self.particle_density*geometry.subvol_volume.reshape(-1, 1))
        elif self.norm == 'mean':
            heat_flux = VxE*phonon.number_of_modes/self.subvol_N_p.reshape(-1, 1)
        
        heat_flux = phonon.normalise_to_density(heat_flux)

        self.subvol_heat_flux = heat_flux*self.eVpsa2_in_Wm2

    def drift(self):
        '''Drift operation.'''

        self.positions += self.group_vel*self.dt # move forward by one v*dt step

        self.n_timesteps -= 1                    # -1 timestep in the counter to boundary scattering

    def bring_back_particles(self, positions, geometry):
        '''Brings particles that are slightly outside the domain back inside,
        by mirroring them in relation to the closest point on the surface of the mesh,
        and adding a small noise.'''

        noise = 1e-3 # angstrom

        n = positions.shape[0]  # number of particles being nudged

        # finding closest points
        close_points, _, _ = tm.proximity.closest_point(geometry.mesh, positions)

        positions += 2*(close_points - positions) + (np.random.rand(n, 3)-0.5)*noise

        return positions

    def find_boundary(self, positions, velocities, geometry):
        '''Finds which mesh triangle will be hit by the particle, given an initial position and velocity
        direction. It works with individual particles as well with a group of particles.
        Returns: array of faces indexes for each particle'''

        # check if all are inside
        is_out = tm.proximity.signed_distance(geometry.mesh, positions) <= 0 # for some reason mesh.contains does not work

        while np.any(is_out):  # if any is out
            # print('Is out', is_out.sum())

            positions[is_out, :] = self.bring_back_particles(positions[is_out, :], geometry)

            is_out[is_out] = tm.proximity.signed_distance(geometry.mesh, positions[is_out, :]) <= 0

        # get where the particle will hit a wall, and which face (triangle)
        index_faces, index_ray, boundary_pos = geometry.mesh.ray.intersects_id(ray_origins      = positions ,
                                                                               ray_directions   = velocities,
                                                                               return_locations = True      ,
                                                                               multiple_hits    = False     )
        
        with_ray   = np.in1d(np.arange(positions.shape[0]), index_ray) # find which particles have rays
        stationary = (np.linalg.norm(velocities, axis = 1) == 0)       # find which particles have zero velocity

        all_boundary_pos                = np.zeros( (positions.shape[0], 3) )
        all_boundary_pos[stationary, :] = np.inf                       # those stationary have infinite collision position
        all_boundary_pos[with_ray  , :] = boundary_pos                 # those with ray have correspondent collision position

        all_faces = np.ones(with_ray.shape[0])*np.nan
        all_faces[with_ray] = index_faces

        on_edge = np.logical_and(~stationary, ~with_ray)    # those that are not stationary but don't have rays are leaking

        if np.any(on_edge):
            all_boundary_pos[on_edge, :], all_faces[on_edge] = self.find_collision_naive(positions[on_edge, :], velocities[on_edge, :], geometry)

        # get which facet those faces are part of
        nan_faces    = np.isnan(all_faces)
        index_facets = geometry.faces_to_facets(all_faces[~nan_faces])

        all_facets           = np.ones( positions.shape[0] )*np.nan  # all particles that do not collide do not have corresponding facets
        all_facets[~nan_faces] = index_facets

        return all_boundary_pos, all_facets, positions  # also return the corrected (nudged) positions

    def timesteps_to_boundary(self, positions, velocities, geometry):
        '''Calculate how many timesteps to boundary scattering.'''

        if positions.shape[0] == 0:
            ts_to_boundary      = np.zeros(0)
            index_facets        = np.zeros(0)
            boundary_pos        = np.zeros((0, 3))
            corrected_positions = np.zeros((0, 3))
            
            return ts_to_boundary, index_facets, boundary_pos, corrected_positions

        boundary_pos, index_facets, corrected_positions = self.find_boundary(positions, velocities, geometry)

        # calculate distances for given particles
        boundary_dist = np.linalg.norm(boundary_pos - corrected_positions, axis = 1 )

        ts_to_boundary = boundary_dist/( np.linalg.norm(velocities, axis = 1) * self.dt )  # such that particle hits the boundary when n_timesteps == 0 (crosses boundary)
        
        return ts_to_boundary, index_facets, boundary_pos, corrected_positions
    
    def delete_particles(self, indexes):
        '''Delete all information about particles according to the given indexes'''

        self.positions           = np.delete(self.positions          , indexes, axis = 0)
        self.group_vel           = np.delete(self.group_vel          , indexes, axis = 0)
        self.omega               = np.delete(self.omega              , indexes, axis = 0)
        self.occupation          = np.delete(self.occupation         , indexes, axis = 0)
        self.temperatures        = np.delete(self.temperatures       , indexes, axis = 0)
        self.n_timesteps         = np.delete(self.n_timesteps        , indexes, axis = 0)
        self.modes               = np.delete(self.modes              , indexes, axis = 0)
        self.collision_facets    = np.delete(self.collision_facets   , indexes, axis = 0)
        self.collision_positions = np.delete(self.collision_positions, indexes, axis = 0)
        self.collision_cond      = np.delete(self.collision_cond     , indexes, axis = 0)
        # self.subvol_id           = np.delete(self.subvol_id          , indexes, axis = 0)
        
        # self.collision_cond = [self.collision_cond[index] for index in range(len(self.collision_cond)) if index not in indexes]
    
    def calculate_specularity(self, facets, modes, geometry, phonon):
        
        normals = geometry.mesh.facets_normal[facets]        # get their normals

        wavevectors = phonon.wavevectors[modes[:, 0], :]    # get wavevectors of the particles
        wv_norms    = phonon.norm_wavevectors[modes[:, 0]]  # and their norms

        dot = (wavevectors*normals).sum(axis = 1)       # k . n --> should all be positive
        angle = np.arccos(dot/wv_norms)                 # incident angle

        eta = self.bound_values[facets]                 # get roughness values

        specularity = np.exp(-(2*np.cos(angle)*wv_norms*eta)**2)  # calculate specularity

        return specularity

    def select_reflected_modes(self, incident_modes, facet_normals, phonon):

        branches = incident_modes[:, 1]

        k = phonon.wavevectors[incident_modes[:, 0], :]  # get wavevectors
        n = -facet_normals                               # make it point inwards

        n_p     = incident_modes.shape[0] # number of particles being reflected

        # wavevector operations
        k_dot_n = (k*n).sum(axis = 1)         # dot product of k . normal
        k_size  = np.linalg.norm(k, axis = 1) # k size
        k_spec  = k - 2*n*k_dot_n.reshape(-1, 1)           # specularly reflected wavevector

        # standard deviations
        sigma_theta = 1
        sigma_phi   = 1
        sigma_r     = 1

        dtheta = np.random.normal(loc = 0, scale = sigma_theta, size = n_p).reshape(-1, 1)
        dphi   = np.random.normal(loc = 0, scale = sigma_phi  , size = n_p).reshape(-1, 1)
        dr     = np.random.normal(loc = 0, scale = sigma_r    , size = n_p)

        # First rotation - Theta

        k_vec_n = np.cross(k_spec, n, axis = 1)  # cross products
        indexes = np.all(k_vec_n == 0, axis = 1) # k parallel to n produce 0 cross products

        k_vec_n[indexes, :] = n[indexes, :] # in this case the axis is the normal

        # k_vec_n_norm = np.linalg.norm(k_vec_n, axis = 1).reshape(-1, 1) # and their norm

        a1 = k_vec_n/np.linalg.norm(k_vec_n, axis = 1).reshape(-1, 1)# k_vec_n_norm # rotation axis = k x n, normalised
        a1 = a1*np.tan(dtheta/4) # adjusting rodrigues parameters

        R1 = rot.from_mrp(a1)    # creating first rotation object

        k_try = R1.apply(k_spec) # applying rotation in theta

        # Second rotation - Phi
        
        a1_cross_k = np.cross(a1, k_try, axis = 1)                          # second axis of rotation - a1 x k
        a2 = a1_cross_k/np.linalg.norm(a1_cross_k, axis = 1).reshape(-1, 1) # normalising
        a2 = a2*np.tan(dphi/4)                                              # adjusting rodrigues parameters

        R2 = rot.from_mrp(a2)   # creating second rotation object

        k_try = R2.apply(k_try) # applying rotation in phi

        # applying scale in R

        k_try = k_try*((k_size+dr)/k_size).reshape(-1, 1) 
        
        q_try = phonon.k_to_q(k_try)

        # sign of reflected waves
        v_dot_n = phonon.group_vel[:, branches, :]*n.reshape(1, n_p, 3) # dot product V x n
        v_dot_n = v_dot_n.sum(axis = 2)

        sign = np.around(v_dot_n, decimals = 3) > 0    # where V agrees with normal

        # starting functions

        # output              Interpolator           datapoints                    values                 input            looping
        new_modes = np.array([NearestNDInterpolator(phonon.q_points[sign[:, i]], np.where(sign[:, i])[0])(q_try[i, :]) for i in range(n_p)])

        new_modes = np.hstack((new_modes.reshape(-1, 1), branches.reshape(-1, 1)))

        return new_modes

    def periodic_boundary_condition(self, positions, group_velocities, collision_facets, collision_positions, calculated_ts, geometry):
        
        # extra step to avoid the particle to be slightly out of the domain
        extra_ts = np.where(1-calculated_ts > 1e-3, 1e-3, 0)

        collision_facets = collision_facets.astype(int) # ensuring collision facets are integers

        # getting which face is connected
        rows             = np.array([np.where(i == self.connected_facets)[0][0] for i in collision_facets])  
        mask             = self.connected_facets[rows, :] == collision_facets.reshape(-1, 1)
        connected_facets = (self.connected_facets[rows, :]*~mask).sum(axis = 1).astype(int)

        previous_positions        = copy.deepcopy(positions)            # the path of some particles is calculated from their current position
        first                     = calculated_ts == 0                  # but for those at the beginning of the timestep
        previous_positions[first] -= group_velocities[first]*self.dt    # their starting position is the one before the collision

        # finding new position on the connected facet
        new_positions = np.around(collision_positions + geometry.facet_centroid[connected_facets, :] - geometry.facet_centroid[collision_facets, :], decimals = 6)

        new_positions = new_positions + group_velocities*extra_ts.reshape(-1, 1)

        buffer_new_positions = new_positions    # just to print in case of error

        # finding next scattering event
        new_ts_to_boundary, new_collision_facets, new_collision_positions, new_positions = self.timesteps_to_boundary(new_positions, group_velocities, geometry)

        calculated_ts += np.linalg.norm((collision_positions - previous_positions), axis = 1)/np.linalg.norm((group_velocities*self.dt), axis = 1) + extra_ts

        ######### DEBUG #####################
        if np.any(np.isnan(new_collision_facets)):
            i = np.isnan(new_collision_facets)
            print(first[i])
            print(calculated_ts[i])
            print(previous_positions[i])
            print(positions[i])
            print(collision_positions[i])
            print(group_velocities[i])
            print(collision_facets[i])
            print(buffer_new_positions[i])
            print(new_positions[i])
            print(new_collision_facets[i])
            print(new_collision_positions[i])
        ################################################

        new_collision_facets = new_collision_facets.astype(int)
            
        new_collision_cond = self.bound_cond[new_collision_facets]

        return new_positions, new_ts_to_boundary, new_collision_facets, new_collision_positions, new_collision_cond, calculated_ts

    def find_collision_naive(self, positions, velocities, geometry):
        '''To be used when trimesh fails to find it with ray casting.'''

        # print('Using naive!!')

        geo_bounds = geometry.bounds                                                     # grab bounding box

        positive_vel = velocities > 0                                                    # get which components are positive

        up_points = (positive_vel*geo_bounds[1, :] + ~positive_vel*geo_bounds[0, :])*(1+1e-2) # grab the proper bounds with a margin

        down_points = copy.deepcopy(positions)                            # initialise old opposite point for convergence criteria

        criteria = 1e-6                                                                  # define criteria

        not_converged = np.zeros(positions.shape[0], dtype = bool)

        while np.any(not_converged): # while criteria is not met

            steps = 4

            n_not_converged = not_converged.astype(int).sum()

            all_points = down_points[not_converged, :] + (up_points[not_converged, :] - down_points[not_converged, :])*(np.arange(0, steps+2)/5).reshape(-1, 1, 1)  # calculate 4 midpoints between down and up, including them

            outside = np.zeros(all_points.shape[[1, 0]]) # initialise inside array

            for i in range(steps+2):
                outside[:, i] = tm.proximity.signed_distance(geometry.mesh, all_points[i, :, :]) <= 0
                                #~geometry.mesh.contains(all_points[i, :, :]) # see whether they are inside the mesh

            indexes_out = np.argmax(outside, axis = 1) # get the first point that is outside
            
            indexes_in = indexes_out - 1    # by definition, the last points inside are those that come before the first points outside

            up_points[not_converged, :]   = all_points[indexes_out, np.arange(n_not_converged), :]  # update up points to the first outside
            down_points[not_converged, :] = all_points[indexes_in , np.arange(n_not_converged), :]  # update down points to the last inside

            not_converged = np.linalg.norm(down_points - up_points, axis = 1) > criteria # update convergence -> distance < criteria

        collision_points = (up_points + down_points)/2   # takes the collision as the midpoint

        # try to get the closest point and use the respective face as the collision facet

        # print(collision_points)

        _, _, collision_faces = tm.proximity.closest_point(geometry.mesh, collision_points)

        collision_facets = geometry.faces_to_facets(collision_faces)

        return collision_points, collision_facets

    def roughness_boundary_condition(self, positions, group_velocities, incident_modes, collision_facets, collision_positions, calculated_ts, geometry, phonon):

        print('In roughness!!!!')
        # get the normal of the collision facet
        facet_normals = geometry.mesh.facets_normal[collision_facets.astype(int), :]

        # particles already scattered this timestep are calculated from their current position (at a boundary)
        previous_positions = positions

        # first scattering particles start from their position at the beginning of the timestep
        first = calculated_ts == 0
        previous_positions[first, :] -= group_velocities[first, :]*self.dt

        # the calculated timestep is up to the next scattering event
        new_calculated_ts = calculated_ts + np.linalg.norm(collision_positions - previous_positions, axis = 1)/(np.linalg.norm(group_velocities, axis = 1)*self.dt)
        # print('calc ts', calculated_ts)
        # print('delta calc ts', np.linalg.norm(collision_positions - previous_positions, axis = 1)/(np.linalg.norm(group_velocities, axis = 1)*self.dt))

        # update particle positions to the collision posiiton
        new_positions = collision_positions

        # select new modes and get their properties
        new_modes = self.select_reflected_modes(incident_modes, facet_normals, phonon)

        new_group_vel = phonon.group_vel[new_modes[:, 0], new_modes[:, 1], :]
        new_omega     = phonon.omega[new_modes[:, 0], new_modes[:, 1]]

        # extra step to avoid the particle to be slightly out of the domain
        extra_ts = np.where(1-calculated_ts > 1e-3, 1e-3, 0)

        new_calculated_ts += extra_ts
        new_positions = new_positions + new_group_vel*self.dt*extra_ts.reshape(-1, 1)

        # find next scattering event
        new_ts_to_boundary, new_collision_facets, new_collision_positions, new_positions = self.timesteps_to_boundary(new_positions, new_group_vel, geometry)

        # plt.plot(collision_facets, new_collision_facets, 'x')

        new_collision_cond = self.bound_cond[new_collision_facets.astype(int)]

        ################## DEBUGGING ###############
        # plotting
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection = '3d')
        # ax.scatter(previous_positions[:, 0] ,
        #            previous_positions[:, 1] ,
        #            previous_positions[:, 2] , c = 'b')
        # ax.scatter(collision_positions[:, 0],
        #            collision_positions[:, 1],
        #            collision_positions[:, 2], c = 'r')
        # ax.scatter(new_positions[:, 0]      ,
        #            new_positions[:, 1]      ,
        #            new_positions[:, 2]      , c = 'g')

        # # first drift
        # t = self.dt
        # start_points = previous_positions
        # end_points = start_points + t * group_velocities

        # for i in range(new_omega.shape[0]):
        #     ax.plot([start_points[i, 0], end_points[i, 0]],
        #             [start_points[i, 1], end_points[i, 1]],
        #             [start_points[i, 2], end_points[i, 2]], color = 'k', linewidth = 1)
        
        # # second drift
        # t = self.dt
        # start_points = collision_positions
        # end_points = start_points + t * new_group_vel

        # for i in range(new_omega.shape[0]):
        #     ax.plot([start_points[i, 0], end_points[i, 0]],
        #             [start_points[i, 1], end_points[i, 1]],
        #             [start_points[i, 2], end_points[i, 2]], color = 'k', linewidth = 1)
        
        # ax.set_title('Scattered particles')

        # plt.show()

        ###########################################################################
        
        return new_modes, new_positions, new_group_vel, new_omega, new_ts_to_boundary, new_calculated_ts, new_collision_positions, new_collision_facets, new_collision_cond

    def boundary_scattering(self, geometry, phonon):
        '''Applies boundary scattering or other conditions to the particles where it happened, given their indexes.'''

        indexes_all = self.n_timesteps < 0                      # find scattering particles (N_p,)

        calculated_ts              = np.ones(indexes_all.shape) # start the tracking of calculation as 1 (N_p,)
        calculated_ts[indexes_all] = 0                          # set it to 0 for scattering particles   (N_p,)

        new_n_timesteps = copy.copy(self.n_timesteps) # (N_p,)

        while np.any(calculated_ts < 1): # while there are any particles with the timestep not completely calculated

            # I. Deleting particles entering reservoirs

            # identifying particles hitting rough facets
            indexes_del = np.logical_and(calculated_ts       <   1 ,
                                         np.in1d(self.collision_cond, ['T', 'F']))
            
            if np.any(indexes_del):
                self.delete_particles(indexes_del)
                calculated_ts   = calculated_ts[~indexes_del]
                new_n_timesteps = new_n_timesteps[~indexes_del]
                indexes_all     = indexes_all[~indexes_del]
            
            # II. Applying Periodicities:
            
            # identifying particles hitting facets with periodic boundary condition
            indexes_per = np.logical_and(calculated_ts       <   1 ,
                                        self.collision_cond == 'P')

            if np.any(indexes_per):

                # enforcing periodic boundary condition
                (self.positions[indexes_per, :]         ,
                new_n_timesteps[indexes_per]            ,
                self.collision_facets[indexes_per]      ,
                self.collision_positions[indexes_per, :],
                self.collision_cond[indexes_per]        ,
                calculated_ts[indexes_per]              ) = self.periodic_boundary_condition(self.positions[indexes_per, :]          ,
                                                                                             self.group_vel[indexes_per, :]          ,
                                                                                             self.collision_facets[indexes_per]      ,
                                                                                             self.collision_positions[indexes_per, :],
                                                                                             calculated_ts[indexes_per]              ,
                                                                                             geometry                                )

            # III. Performing scattering:

            # identifying particles hitting rough facets
            indexes_ref = np.logical_and(calculated_ts       <   1 ,
                                         self.collision_cond == 'R')
            
            if np.any(indexes_ref):
                (self.modes[indexes_ref, :]              ,
                 self.positions[indexes_ref, :]          ,
                 self.group_vel[indexes_ref, :]          ,
                 self.omega[indexes_ref]                 ,
                 self.n_timesteps[indexes_ref]           ,
                 calculated_ts[indexes_ref]              ,
                 self.collision_positions[indexes_ref, :],
                 self.collision_facets[indexes_ref]      ,
                 self.collision_cond[indexes_ref]        ) = self.roughness_boundary_condition(self.positions[indexes_ref, :]       ,
                                                                                               self.group_vel[indexes_ref, :]       ,
                                                                                               self.modes[indexes_ref, :]           ,
                                                                                               self.collision_facets[indexes_ref]   ,
                                                                                               self.collision_positions[indexes_ref],
                                                                                               calculated_ts[indexes_ref]           ,
                                                                                               geometry                             ,
                                                                                               phonon                               )

            # IV. Drifting those who will not be scattered again in this timestep

            # identifying drifting particles
            indexes_drift = np.logical_and(indexes_all                        ,
                                           (1-calculated_ts) < new_n_timesteps)

            if np.any(indexes_drift):
                self.positions[indexes_drift, :] += self.group_vel[indexes_drift, :]*self.dt*(1-calculated_ts[indexes_drift]).reshape(-1, 1)

                new_n_timesteps[indexes_drift]   -= (1-calculated_ts[indexes_drift])
                calculated_ts[indexes_drift]     = 1
            
        self.n_timesteps[indexes_all] = new_n_timesteps[indexes_all]

    def lifetime_scattering(self, phonon):
        '''Performs lifetime scattering.'''

        # N_as = N_ad + dt/tau (N_BE(T*) - N_ad)
        
        Tqj = np.hstack((self.temperatures.reshape(-1, 1), self.modes))
        tau = phonon.lifetime_function(Tqj)

        occupation_ad = copy.deepcopy(self.occupation)             # getting current occupation
        occupation_BE = phonon.calculate_occupation(self.temperatures, self.omega, reference = True) # calculating Bose-Einstein occcupation

        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            occupation_as = np.where(tau>0, occupation_ad + (self.dt/tau) *(occupation_BE - occupation_ad), occupation_ad)
            self.occupation = occupation_as
    
    def run_timestep(self, geometry, phonon):

        # start = time.time()

        self.drift()                                    # drift particles

        # print('Drift: {:.2f} s'.format(time.time()-start))
        # start = time.time()
        
        self.fill_reservoirs(geometry, phonon)          # refill reservoirs

        # print('Fill reservoirs: {:.2f} s'.format(time.time()-start))
        # start = time.time()
        
        self.add_reservoir_particles(geometry, phonon)  # add reservoir particles that come in the domain
        
        # print('Add reservoirs particles: {:.2f} s'.format(time.time()-start))
        # start = time.time()

        self.boundary_scattering(geometry, phonon)      # perform boundary scattering/periodicity and particle deletion

        # print('Boundary scattering: {:.2f} s'.format(time.time()-start))
        # start = time.time()

        self.refresh_temperatures(geometry, phonon)     # refresh cell temperatures
        
        # print('Refresh T: {:.2f} s'.format(time.time()-start))
        # start = time.time()

        self.lifetime_scattering(phonon)                # perform lifetime scattering

        # print('Lifetime scattering: {:.2f} s'.format(time.time()-start))
        # start = time.time()

        self.current_timestep += 1                      # +1 timestep index

        self.t = self.current_timestep*self.dt          # 1 dt passed
        
        if  ( self.current_timestep % 100) == 0:
            print('Timestep {:>5d}'.format( int(self.current_timestep) ), self.subvol_temperature )
        
        if ( self.current_timestep % self.n_dt_to_conv) == 0:
            self.calculate_heat_flux(geometry, phonon)
            self.write_convergence(phonon)      # write data on file

        if (len(self.rt_plot) > 0) and (self.current_timestep % self.n_dt_to_plot == 0):
            self.plot_real_time(geometry)

        # print('The rest: {:.2f} s'.format(time.time()-start))

        # saving particle data to analyse
        # if self.current_timestep % 100 == 0 or self.current_timestep == 1:
        #     filename = self.results_folder_name + 'particle_data_{:d}.txt'.format(self.current_timestep)

        #     time = datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f')

        #     header ='Particles final state data \n' + \
        #             'Date and time: {}\n'.format(time) + \
        #             'hdf file = {}, POSCAR file = {}\n'.format(self.args.hdf_file, self.args.poscar_file) + \
        #             'q-point, branch, pos x [angs], pos y [angs], pos z [angs], subvol, occupation'

        #     data = np.hstack( (self.modes,
        #                     self.positions,
        #                     self.occupation.reshape(-1, 1)) )
            
        #     # comma separated
        #     np.savetxt(filename, data, '%d, %d, %.3f, %.3f, %.3f, %.6e', delimiter = ',', header = header)
            
    def init_plot_real_time(self, geometry, phonon):
        '''Initialises the real time plot to be updated for each timestep.'''

        n_dt_to_plot = np.floor( np.log10( self.args.iterations[0] ) ) - 1    # number of timesteps to save new animation frame
        n_dt_to_plot = int(10**n_dt_to_plot)
        n_dt_to_plot = 1 #max([1, n_dt_to_plot])

        self.n_dt_to_plot = n_dt_to_plot

        if self.rt_plot[0] in ['T', 'temperature']:
            colors = self.temperatures
            T_bound = self.bound_values[self.bound_cond == 'T']
            vmin = np.floor(self.T_bound.min()-10)
            vmax = np.ceil(self.T_bound.max()+10)
        elif self.rt_plot[0] in ['e', 'energy']:
            colors = self.energies
            T_bound = self.bound_values[self.bound_cond == 'T']
            vmin = phonon.calculate_energy(T_bound.min()-3, phonon.omega, reference = True).min()
            vmax = phonon.calculate_energy(T_bound.max()+3, phonon.omega, reference = True).max()
        elif self.rt_plot[0] in ['omega', 'angular_frequency']:
            colors = self.omega
            vmin = self.omega[self.omega>0].min()
            vmax = self.omega.max()
        elif self.rt_plot[0] in ['n', 'occupation']:
            colors = self.occupation
            order = [np.floor( np.log10( self.occupation.min()) ), np.floor( np.log10( self.occupation.max()) )]
            vmin = (10**order[0])*np.ceil(self.occupation.min()/(10**order[0]))
            vmax = (10**order[1])*np.ceil(self.occupation.max()/(10**order[1]))
        elif self.rt_plot[0] in ['qpoint']:
            colors = self.modes[:, 0]
            vmin = 0
            vmax = phonon.number_of_qpoints
        elif self.rt_plot[0] in ['branch']:
            colors = self.modes[:, 1]
            vmin = 0
            vmax = phonon.number_of_branches
        elif self.rt_plot[0] in ['ts_to_boundary']:
            colors = self.n_timesteps

            vel  = np.sqrt( (phonon.group_vel**2).sum(axis = 2) )
            min_vel = vel[vel>0].min()
            max_path = np.sqrt( (geometry.bounds[1, :]**2).sum() ).min()
            
            vmin = 0
            vmax = 100 # max_path/(min_vel*self.dt)

        plt.ion()
        

        # Think about a way to make the animation "pretty" with a proper aspect ratio depending on dimensions of the domain

        # box_size = np.ptp(geometry.bounds, axis = 0)
        # figsize = np.array([box_size.max()/2, box_size.min()])
        # figsize = figsize*8/(box_size.max()/2)

        figsize = (15,15)

        fig = plt.figure(figsize = figsize, dpi = 100)
        ax = fig.add_subplot(111, projection='3d')
        ax.set_box_aspect( np.ptp(geometry.bounds, axis = 0) )
        ax.set_xlim(geometry.bounds[:, 0])
        ax.set_ylim(geometry.bounds[:, 1])
        ax.set_zlim(geometry.bounds[:, 2])

        graph = ax.scatter(self.positions[:, 0],
                           self.positions[:, 1],
                           self.positions[:, 2],
                           s    = 1            ,
                           vmin = vmin         ,
                           vmax = vmax         ,
                           c    = colors       ,
                           cmap = self.colormap)
        # fig.colorbar(graph)

        plt.tight_layout()

        fig.canvas.draw()

        plt.show()

        self.plot_images = [np.fromstring(fig.canvas.tostring_rgb(), dtype = np.uint8, sep = '').reshape(fig.canvas.get_width_height()[::-1]+(3,))]

        return graph, fig

    def plot_real_time(self, geometry, property_plot = None, colormap = 'viridis'):
        '''Updates data on the real time plot at the end of the timestep.'''
        if property_plot == None:
            property_plot = self.rt_plot[0]
        
        if colormap == None:
            colormap = self.colormap
        
        if property_plot in ['T', 'temperature']:
            colors = self.temperatures
        elif property_plot in ['e', 'energy']:
            colors = self.energies
        elif property_plot in ['omega', 'angular_frequency']:
            colors = self.omega
        elif property_plot in ['n', 'occupation']:
            colors = self.occupation
        elif property_plot in ['qpoint']:
            colors = self.modes[:, 0]
        elif property_plot in ['branch']:
            colors = self.modes[:, 1]
        elif property_plot in ['ts_to_boundary']:
            colors = self.n_timesteps
        
        self.rt_graph._offsets3d = [self.positions[:,0], self.positions[:,1], self.positions[:,2]]

        self.rt_graph.set_array(colors)

        self.rt_fig.canvas.draw()
        self.rt_fig.canvas.flush_events()

        self.plot_images += [np.fromstring(self.rt_fig.canvas.tostring_rgb(), dtype = np.uint8, sep = '').reshape( self.rt_fig.canvas.get_width_height()[::-1]+(3,) ) ]

        plt.figure(self.rt_fig)
        plt.savefig(self.results_folder_name+'last_anim_frame.png')
        
        return

    def save_plot_real_time(self):
        if len(self.rt_plot) > 0:
            plt.close(fig = self.rt_fig)
            print('Saving animation...')
            imageio.mimsave(self.results_folder_name+'simulation.gif', self.plot_images, fps=10)
            print('Saved!')

    def plot_figures(self, geometry, property_plot=['temperature'], colormap = 'viridis'):

        fig = plt.figure( figsize = (8, 8), dpi = 150 )
        n = len(property_plot)

        for i in range(n):
            if property_plot[i] in ['T', 'temperature']:
                data = self.temperatures
                figname = 'temperature'
                title = 'Temperature [K]'
            elif property_plot[i] in ['omega', 'angular_frequency']:
                data = self.omega
                figname = 'angular_frequency'
                title = 'Angular Frequency [rad/s]'
            elif property_plot[i] in ['n', 'occupation']:
                data = np.log(self.occupation)
                figname = 'occupation'
                title = 'Occupation Number'
            elif property_plot[i] in ['e', 'energy']:
                data = self.energies
                figname = 'energy'
                title = 'Energy [eV]'

            plt.clf()
            ax = fig.add_subplot(111, projection='3d')
            ax.set_box_aspect( np.ptp(geometry.bounds, axis = 0) )
            graph = ax.scatter(self.positions[:, 0], self.positions[:, 1], self.positions[:, 2], cmap = colormap, c = data, s = 1)
            fig.colorbar(graph, orientation='horizontal')
            plt.title(title, {'size': 15} )

            plt.tight_layout()
            
            plt.savefig(self.results_folder_name+figname+'.png')

    def open_convergence(self):

        n_dt_to_conv = np.floor( np.log10( self.args.iterations[0] ) ) - 2    # number of timesteps to save convergence data
        n_dt_to_conv = int(10**n_dt_to_conv)
        n_dt_to_conv = max([10, n_dt_to_conv])

        self.n_dt_to_conv = n_dt_to_conv


        filename = self.results_folder_name+'convergence.txt'

        self.f = open(filename, 'a+')

        line = '# '
        line += 'Real Time                  '
        line += 'Timest. '
        line += 'Simula. Time '
        line += 'Average Ene. '
        line += 'Energy Res 1 Energy Res 2 '
        line += 'HFlux Res 1 HFlux Res 2 '
        line += 'No. Part. '
        for i in range(self.n_of_subvols):
            line += 'T Slc {:>3d} '.format(i)
        for i in range(self.n_of_subvols):
            line += 'Energ Sl {:>2d} '.format(i) # temperature per subvol
        for i in range(self.n_of_subvols):
            line += 'Hflux Sl {:>2d} '.format(i)
        for i in range(self.n_of_subvols):
            line += 'Np Sl  {:>2d} '.format(i)
        
        line += '\n'

        self.f.write(line)
        self.f.close()

    def write_convergence(self, phonon):
        
        line = ''
        line += datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f ')  # real time in ISO format
        line += '{:>8d} '.format( int(self.current_timestep) )   # timestep
        line += '{:>11.5e} '.format( self.t )   # time
        if self.energies.shape[0]>0:
            line += '{:>12.5e} '.format( self.energies.mean() )  # average energy
        else:
            line += '{:>11.5e} '.format( 0 )

        line += '{:>10d} '.format( self.energies.shape[0] )  # number of particles
        line += np.array2string(self.subvol_temperature, formatter = {'float_kind':'{:>9.3f}'.format} ).strip('[]') + ' '
        line += np.array2string(self.subvol_energy     , formatter = {'float_kind':'{:>11.5e}'.format}).strip('[]') + ' '
        # line += np.array2string(self.subvol_heat_flux  , formatter = {'float_kind':'{:>12.5e}'.format}).strip('[]') + ' '
        line += np.array2string(self.subvol_N_p        , formatter = {'int'       :'{:>10d}'.format}  ).strip('[]') + ' '
        
        line += '\n'

        filename = self.results_folder_name+'convergence.txt'

        self.f = open(filename, 'a+')

        self.f.writelines(line)

        self.f.close()

    def write_final_state(self):

        print('Saving end of run particle data...')
        
        time = datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f')
        
        # saving final particle states: modes, positions, subvol id and occupation number.
        # Obs.: phonon properties of particles can be retrieved by mode information and subvol temperature

        filename = self.results_folder_name + 'particle_data.txt'

        header ='Particles final state data \n' + \
                'Date and time: {}\n'.format(time) + \
                'hdf file = {}, POSCAR file = {}\n'.format(self.args.hdf_file, self.args.poscar_file) + \
                'q-point, branch, pos x [angs], pos y [angs], pos z [angs], subvol, occupation'

        data = np.hstack( (self.modes,
                           self.positions,
                           self.occupation.reshape(-1, 1)) )
        
        # comma separated
        np.savetxt(filename, data, '%d, %d, %.3f, %.3f, %.3f, %.6e', delimiter = ',', header = header)

        # saving final subvol states: temperatures and heat flux

        filename = self.results_folder_name + 'subvol_data.txt'

        header ='subvols final state data \n' + \
                'Date and time: {}\n'.format(time) + \
                'hdf file = {}, POSCAR file = {}\n'.format(self.args.hdf_file, self.args.poscar_file) + \
                'subvol id, subvol volume, temperature [K], heat flux x, y, and z [W/m^2]'

        data = np.hstack( (np.arange(self.n_of_subvols).reshape(-1, 1),
                           self.subvol_volume.reshape(-1, 1),
                           self.subvol_temperature.reshape(-1, 1),
                           self.subvol_heat_flux) )
        
        # comma separated
        np.savetxt(filename, data, '%d, %.3e, %.3f, %.3e, %.3e, %.3e', delimiter = ',', header = header)

    def print_sim_info(self, geometry, phonon):
        print('---------- o ----------- o ------------- o ------------')
        print('Simulation Info:')

        T_max = self.res_facet_temperature.max()

        Tqj = np.hstack((np.ones(phonon.number_of_modes).reshape(-1, 1)*T_max, phonon.unique_modes))
        tau = phonon.lifetime_function(Tqj)
        v = np.absolute(phonon.group_vel[phonon.unique_modes[:, 0], phonon.unique_modes[:, 1], self.slice_axis])

        kn = v*tau/self.slice_length

        maximum_n_slices = int(np.floor(geometry.bounds[:, self.slice_axis].ptp()/(v.max()*self.dt)))

        prob_enter = 100*v*self.dt/self.slice_length
        prob_enter = np.where(prob_enter>100, 100, prob_enter)

        print('Slice Knudsen number: Max = {:.3e}, Min = {:.3e}'.format(kn.max(), kn.min()))
        print('Maximum number of slices: {}'.format(maximum_n_slices))
        print('Probability of entering: Max = {:.3f}%, Min = {:.3f}%'.format(prob_enter.max(), prob_enter.min()))
        print('---------- o ----------- o ------------- o ------------')
        
 # def remove_out_points(self, points, mesh):
    #     '''Remove points outside the geometry mesh.'''
    #     in_points = mesh.contains(points)     # boolean array
    #     out_points = ~in_points
    #     return np.delete(points, out_points, axis=0), out_points.sum()   # new points, number of particles

# def nunique(a, dec = 4):
#     a = np.around(a, decimals = dec)
#     return np.unique(a).shape[0]

    # def write_modes_data(self):
    #     '''Write on file the mode distribution in a given timestep (qpoint, branch, slice).'''

    #     filename = self.results_folder_name + 'time_{:>.3e}'.format(self.t)+'.txt'

    #     data = np.hstack((self.modes, self.slice_id.reshape(-1, 1)))

    #     np.savetxt(filename, data, '%4d %2d %2d')
        
# OLD CHOICE OF MODE IN BOUNDARY SCATTERING

        # periodic basis
        # b = np.sin((phonon.q_points-phonon.q_points[incident_mode[0]])*np.pi)**2
        # b = b.mean(axis = 1)

        # omega factor
        # incident_omega = phonon.omega[incident_mode[0], incident_mode[1]]
        # omega_factor   = ((phonon.omega-incident_omega)/incident_omega)**2

        # sign of reflected waves
        # sign = np.dot(phonon.group_vel, -facet_normal.reshape(1, -1, 1)).squeeze()

        # weights and probabilities
        # W = np.exp(-(b.reshape(-1, 1)+omega_factor)/-np.log(specularity))
        # W = np.where(sign>0, W, 0)
        
        # P = W/W.sum()

        # roll the dice and choose the mode

        # r = np.random.rand()

        # mask = (P.cumsum()<=r).reshape(phonon.number_of_qpoints, phonon.number_of_branches)
        # new_mode = np.array([np.where(mask)[0][-1], np.where(mask)[1][-1]])
                
# def plot_modes_histograms(self, phonon):

#         columns = 5
#         rows = int(np.ceil(self.n_of_slices/columns))

#         self.hist_fig = plt.figure(figsize = (columns*3, rows*3) )
        
#         ax = []

#         for i in range(self.n_of_slices):
#             ax += [self.hist_fig.add_subplot(rows, columns, i+1)]

#             data = []

#             for j in range(phonon.number_of_branches):
        
#                 indexes = (self.modes[:,1] == j) & (self.slice_id == i)

#                 data += [self.modes[indexes, 0]]

#             ax[i].hist(data, stacked = True, histtype = 'bar', density = True, bins = int(phonon.number_of_qpoints/20) )
    
#         plt.tight_layout()
#         plt.savefig(self.results_folder_name+'time_{:>.3e}'.format(self.t)+'.png')
#         # plt.show()

#         plt.clf()