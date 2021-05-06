# calculations
import numpy as np

# plotting and animation
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Normalize, NoNorm
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec

# other
import os
from functools import partial

# simulation
from classes.Constants import Constants

matplotlib.use('Agg')

class Visualisation(Constants):
    def __init__(self, args, geometry, phonon):
        super(Visualisation, self).__init__()
        print('Initialising visualisation class...')

        self.args = args
        self.phonon = phonon
        self.geometry = geometry

        self.folder = self.args.results_folder

        self.convergence_file = self.args.results_folder+'convergence.txt'
        self.particle_file    = self.args.results_folder+'particle_data.txt'
        self.slice_file       = self.args.results_folder+'slice_data.txt'

        self.dt         = self.args.timestep[0]
        self.T_boundary = np.array(self.args.temperatures)

        self.unique_modes = np.stack(np.meshgrid( np.arange(phonon.number_of_qpoints), np.arange(phonon.number_of_branches) ), axis = -1 ).reshape(-1, 2).astype(int)

    
    def preprocess(self):
        print('Generating preprocessing plots...')
        
        self.scattering_probability()
        self.density_of_states()

    def postprocess(self):
        print('Generating postprocessing plots...')
        
        # convergence data
        
        self.read_convergence()
        self.convergence_temperature()
        self.convergence_particles()
        self.convergence_heat_flux()
        self.convergence_energy()

        # final particles states

        self.read_particles()
        # self.mode_histogram()
        self.velocity_histogram()
        self.energy_histogram()
        self.flux_contribution()

        
    def scattering_probability(self):
        
        T = [self.T_boundary.min(), self.T_boundary.mean(), self.T_boundary.max()]
        
        fig = plt.figure(figsize = (20,8), dpi = 120)
        
        x_data = self.phonon.omega[self.unique_modes[:, 0], self.unique_modes[:, 1]]

        axes = []
        y_data = []

        criteria = 0.1

        n_plots = 3

        min_lt = np.zeros(3)
        
        # calculating y data for each temperature
        for i in range(n_plots):
            ax = fig.add_subplot(1, n_plots, i+1)
            axes.append(ax)

            function = partial(self.get_lifetime, T = T[i])
            
            lifetime = np.array(list(map(function, self.unique_modes)))

            min_lt[i] = lifetime[lifetime > 0].min()

            with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                scat_prob = np.where(lifetime>0, 1-np.exp(-self.dt/lifetime), 0)

            y_data.append(scat_prob)

            # colors = np.where(y_data[i] > criteria, 'orangered', 'royalblue')

        # setting limits and colors based on y data
        y_data = np.array(y_data)

        min_y = 0

        max_y = y_data.max()
        max_y = np.ceil(max_y*10)/10

        max_dt = -np.log(1-criteria)*min_lt

        out_points = np.where(y_data > criteria, 1, 0).sum(axis = 1)/y_data.shape[1]
        out_points *= 100

        colors = ['royalblue', 'blue', 'gold', 'crimson', 'crimson']
        nodes  = np.array([0.0, criteria-0.01, criteria, criteria+0.01, max([criteria+0.05, max_y])])/max([criteria+0.05, max_y])
        cmap = LinearSegmentedColormap.from_list('byr', list(zip(nodes, colors)))

        for i in range(n_plots):
            ax = axes[i]

            ax.scatter(x_data, y_data[i, :], s = 1,
                       c = y_data[i, :], cmap = cmap,
                       vmin = min(nodes)*max([criteria+0.05, max_y]),
                       vmax = max(nodes)*max([criteria+0.05, max_y]))

            ax.set_xlabel(r'Angular frequency $\omega$ [rad THz]',
                          fontsize = 'large')
            ax.set_ylabel(r'Scattering probability $P = 1-\exp(-dt/\tau)$',
                          fontsize = 'large')

            ax.set_ylim(min_y, max_y)

            ax.text(x = 0, y = max_y*0.97,
                    s = 'T = {:.1f} K\nOut points = {:.2f} %\nMax allowed dt = {:.3f} ps'.format(T[i], out_points[i], max_dt[i]),
                    va = 'top',
                    ha = 'left',
                    fontsize = 'x-large',
                    linespacing = 1.1)

        fig.suptitle(r'Scattering probability for $T_{{cold}}$, $T_{{mean}}$ and $T_{{hot}}$. Criteria: $P < {:.2f}$'.format(criteria),
                     fontsize = 'xx-large')

        plt.tight_layout()
        plt.savefig(self.folder+'scattering_prob.png')

        if (y_data > criteria).sum() > 0:
            print('MCPhonon Warning: Timestep is too large! Check scattering probability plot for details.')
    
    def get_lifetime(self, mode, T):
        return self.phonon.lifetime_function[mode[0]][mode[1]](T)

    def density_of_states(self):

        omega = self.phonon.omega[self.unique_modes[:, 0], self.unique_modes[:, 1]]

        n_bins    = 1000
        
        d_omega   = omega.max()/n_bins

        intervals = np.arange(0, omega.max()+d_omega, d_omega)
        
        below = (omega.reshape(-1, 1) < intervals)[:, 1:]
        above = (omega.reshape(-1, 1) >= intervals)[:, :-1]
        
        dos = below & above

        dos = dos.sum(axis = 0)
        dos = dos/d_omega

        x_data = intervals[:-1]+d_omega
        y_data = dos

        fig = plt.figure(figsize = (10,8), dpi = 120)
        ax = fig.add_subplot(111)

        # ax.plot(x_data, y_data, color = 'k')
        ax.fill_between(x_data, 0, y_data, facecolor = 'slategray')

        ax.set_ylim(0)

        ax.set_xlabel(r'Angular Frequency $\omega$ [THz]', fontsize = 'x-large')
        ax.set_ylabel(r'Density of states $g(\omega)$ [THz$^{-1}$]', fontsize = 'x-large')

        plt.title(r'Density of states. {:d} bins, $d\omega = $ {:.3f} THz'.format(n_bins, d_omega), fontsize = 'xx-large')

        plt.tight_layout()
        plt.savefig(self.folder+'density_of_states.png')

    def read_convergence(self):
        f = open(self.convergence_file, 'r')

        lines = f.readlines()

        f.close()

        data = [ i.strip().split(' ') for i in lines]
        data = data[1:]
        data = [ list(filter(None, i)) for i in data]
        data = np.array(data)

        self.number_of_slices = self.args.slices[0]

        self.datetime = data[:, 0].astype('datetime64[us]')
        self.timestep = data[:, 1].astype(int)
        self.sim_time = data[:, 2].astype(float)
        self.avg_en   = data[:, 3].astype(float)
        self.N_p      = data[:, 4].astype(int)
        self.T        = data[:, 5:self.number_of_slices+5].astype(float)
        self.slice_en = data[:,   self.number_of_slices+5:2*self.number_of_slices+5].astype(float)
        self.phi      = data[:, 2*self.number_of_slices+5:3*self.number_of_slices+5].astype(float)
        self.slice_Np = data[:, 3*self.number_of_slices+5:4*self.number_of_slices+5].astype(float)

    def read_particles(self):
        filename = self.folder+'particle_data.txt'

        f = open(filename, 'r')

        lines = f.readlines()

        f.close()

        data = [ i.strip().split(',') for i in lines]
        data = data[4:]
        data = [ list(filter(None, i)) for i in data]
        data = np.array(data)

        self.q_point    = data[:, 0].astype(int)
        self.branch     = data[:, 1].astype(int)
        self.position   = data[:, 2:5].astype(float)
        self.slice_id   = data[:, 5].astype(int)
        self.occupation = data[:, 6].astype(float)
    
    def convergence_temperature(self):
        fig = plt.figure( figsize = (15, 5) )
        slice_axis = np.arange(self.number_of_slices)+1
        x_axis = self.sim_time

        ax1 = fig.add_subplot(121)

        # Temperature
        for i in range(self.number_of_slices):
            y_axis = self.T[:, i]
            ax1.plot(x_axis, y_axis)

        ax1.legend(np.arange(self.number_of_slices)+1)

        ax1.set_xlabel('Simulation time [ps]', fontsize = 12)
        ax1.set_ylabel('Temperature [K]', fontsize = 12)
        
        labels = ['Slice {:d}'.format(i) for i in slice_axis]
        
        ax1.legend(labels)

        # Temperature profile
        ax2 = fig.add_subplot(122, sharey = ax1)
        n_timesteps = self.T.shape[0]

        labels = []

        n = 5
        color = 'royalblue'

        alphas = np.linspace(0.1, 1.0, n)
        rgba_colors = np.zeros((n, 4))
        rgba_colors[:, :3] = matplotlib.colors.to_rgb(color)
        rgba_colors[:,  3] = alphas

        for i in range(n):
            index = int(np.floor(i*n_timesteps/(n-1)))
            if i == n-1:
                labels.append('{:.2f} ps'.format(self.sim_time[-1]))
                ax2.plot( slice_axis, self.T[-1, :], '-+', color = rgba_colors[i, :])
            else:
                labels.append('{:.2f} ps'.format(self.sim_time[index]))
                ax2.plot( slice_axis, self.T[index, :], '-+', color = rgba_colors[i, :])

        ax2.plot( [0, self.number_of_slices+1], self.args.temperatures, '--', color = 'k')

        labels.append('Linear')

        ax2.legend(labels)

        ax2.set_xlabel('Slice', fontsize = 12)
        ax2.set_ylabel('Temperature [K]', fontsize = 12)
        ax2.set_xticks(slice_axis)
        ax2.set_yticks(ax1.get_yticks())

        ax1.grid(True, ls = '--', lw = 1, color = 'slategray')
        ax2.grid(True, ls = '--', lw = 1, color = 'slategray')

        ax1.ticklabel_format(axis = 'y', style = 'sci', scilimits=(0,0), useOffset = False)
        ax2.ticklabel_format(axis = 'y', style = 'sci', scilimits=(0,0), useOffset = False)

        plt.suptitle('Temperatures for each slice: evolution over time and profiles.', fontsize = 'xx-large')

        plt.tight_layout()
        plt.savefig(self.folder+'convergence_T.png')
        
    def convergence_particles(self):
        fig = plt.figure( figsize = (15, 5) )
        slice_axis = np.arange(self.number_of_slices)+1
        x_axis = self.sim_time

        ax1 = fig.add_subplot(121)
        
        for i in range(self.number_of_slices):
            y_axis = self.slice_Np[:, i]
            ax1.plot(x_axis, y_axis)
        
        mean = self.slice_Np.mean(axis = 1)
        std  = self.slice_Np.std(axis = 1)

        ax1.plot(x_axis, mean, '--', color = 'k')
        ax1.fill_between(x_axis, mean-std, mean+std, color = 'lightsteelblue')

        labels = ['Slice {:d}'.format(i) for i in slice_axis]
        labels.append('Mean')
        labels.append(r'$\pm \sigma$')
        ax1.legend(labels)

        ax1.set_xlabel('Simulation time [ps]', fontsize = 12)
        ax1.set_ylabel('Number of Particles in Slice', fontsize = 12)

        ax2 = fig.add_subplot(122, sharey = ax1)
        n_timesteps = self.slice_Np.shape[0]

        labels = []

        n = 5
        color = 'royalblue'

        alphas = np.linspace(0.1, 1.0, n)
        rgba_colors = np.zeros((n, 4))
        rgba_colors[:, :3] = matplotlib.colors.to_rgb(color)
        rgba_colors[:,  3] = alphas

        for i in range(n):
            index = int(np.floor(i*n_timesteps/(n-1)))
            if i == n-1:
                labels.append('{:.2f} ps'.format(self.sim_time[-1]))
                ax2.plot( slice_axis, self.slice_Np[-1, :], '-+', color = rgba_colors[i, :])
            else:
                labels.append('{:.2f} ps'.format(self.sim_time[index]))
                ax2.plot( slice_axis, self.slice_Np[index, :], '-+', color = rgba_colors[i, :])

        ax2.plot( [0, self.number_of_slices+1], np.ones(2)*self.args.particles*self.phonon.number_of_modes, '--', color = 'k')

        labels.append('Expected')
        
        ax2.legend(labels)

        ax2.set_xlabel('Slice', fontsize = 12)
        ax2.set_ylabel('Number of Particles in Slice', fontsize = 12)
        ax2.set_xticks(slice_axis)
        ax2.set_yticks(ax1.get_yticks())

        plt.suptitle('Number of particles in each slice: evolution over time and profiles.', fontsize = 'xx-large')

        ax1.grid(True, ls = '--', lw = 1, color = 'slategray')
        ax2.grid(True, ls = '--', lw = 1, color = 'slategray')

        ax1.ticklabel_format(axis = 'y', style = 'sci', scilimits=(0,0), useOffset = False)
        ax2.ticklabel_format(axis = 'y', style = 'sci', scilimits=(0,0), useOffset = False)

        plt.tight_layout()
        plt.savefig(self.folder+'convergence_Np.png')

    def convergence_heat_flux(self):
        fig = plt.figure( figsize = (15, 5) )
        slice_axis = np.arange(self.number_of_slices)+1
        x_axis = self.sim_time

        ax1 = fig.add_subplot(121)

        for i in range(self.number_of_slices):
            y_axis = self.phi[:, i]
            ax1.plot(x_axis, y_axis)
        
        mean = self.phi.mean(axis = 1)
        std  = self.phi.std(axis = 1)

        ax1.plot(x_axis, mean, '--', color = 'k')
        ax1.fill_between(x_axis, mean-std, mean+std, color = 'lightsteelblue')

        labels = ['Slice {:d}'.format(i) for i in slice_axis]
        labels.append('Mean')
        labels.append(r'$\pm \sigma$')
        ax1.legend(labels)

        ax1.set_xlabel('Simulation time [ps]', fontsize = 12)
        ax1.set_ylabel('Heat Flux for Slice [W/m²]', fontsize = 12)

        # Temperature profile
        ax2 = fig.add_subplot(122, sharey = ax1)
        n_timesteps = self.phi.shape[0]

        labels = []

        n = 5
        color = 'royalblue'

        alphas = np.linspace(0.1, 1.0, n)
        rgba_colors = np.zeros((n, 4))
        rgba_colors[:, :3] = matplotlib.colors.to_rgb(color)
        rgba_colors[:,  3] = alphas

        for i in range(n):
            index = int(np.floor(i*n_timesteps/(n-1)))
            if i == n-1:
                labels.append('{:.2f} ps'.format(self.sim_time[-1]))
                ax2.plot( slice_axis, self.phi[-1, :], '-+', color = rgba_colors[i, :])
            else:
                labels.append('{:.2f} ps'.format(self.sim_time[index]))
                ax2.plot( slice_axis, self.phi[index, :], '-+', color = rgba_colors[i, :])

        ax2.legend(labels)

        ax2.set_xlabel('Slice', fontsize = 12)
        ax2.set_ylabel('Heat Flux for Slice [W/m²]', fontsize = 12)
        ax2.set_xticks(slice_axis)
        ax2.set_yticks(ax1.get_yticks())

        plt.suptitle('Heat flux for each slice: evolution over time and profiles.', fontsize = 'xx-large')

        ax1.grid(True, ls = '--', lw = 1, color = 'slategray')
        ax2.grid(True, ls = '--', lw = 1, color = 'slategray')

        ax1.ticklabel_format(axis = 'y', style = 'sci', scilimits=(0,0), useOffset = False)
        ax2.ticklabel_format(axis = 'y', style = 'sci', scilimits=(0,0), useOffset = False)

        plt.tight_layout()
        plt.savefig(self.folder+'convergence_heat_flux.png')

    def convergence_energy(self):
        fig = plt.figure( figsize = (15, 5) )
        slice_axis = np.arange(self.number_of_slices)+1
        x_axis = self.sim_time

        ax1 = fig.add_subplot(121)

        for i in range(self.number_of_slices):
            y_axis = self.slice_en[:, i]
            ax1.plot(x_axis, y_axis)
        
        mean = self.slice_en.mean(axis = 1)
        std  = self.slice_en.std(axis = 1)

        ax1.plot(x_axis, mean, '--', color = 'k')
        ax1.fill_between(x_axis, mean-std, mean+std, color = 'lightsteelblue')

        labels = ['Slice {:d}'.format(i) for i in slice_axis]
        labels.append('Mean')
        labels.append(r'$\pm \sigma$')
        ax1.legend(labels)

        ax1.set_xlabel('Simulation time [ps]', fontsize = 12)
        ax1.set_ylabel('Energy density for slice [eV/angstrom³]', fontsize = 12)

        # Temperature profile
        ax2 = fig.add_subplot(122, sharey = ax1)
        n_timesteps = self.slice_en.shape[0]

        labels = []

        n = 5
        color = 'royalblue'

        alphas = np.linspace(0.1, 1.0, n)
        rgba_colors = np.zeros((n, 4))
        rgba_colors[:, :3] = matplotlib.colors.to_rgb(color)
        rgba_colors[:,  3] = alphas

        for i in range(n):
            index = int(np.floor(i*n_timesteps/(n-1)))
            if i == n-1:
                labels.append('{:.2f} ps'.format(self.sim_time[-1]))
                ax2.plot( slice_axis, self.slice_en[-1, :], '-+', color = rgba_colors[i, :])
            else:
                labels.append('{:.2f} ps'.format(self.sim_time[index]))
                ax2.plot( slice_axis, self.slice_en[index, :], '-+', color = rgba_colors[i, :])

        ax2.legend(labels)

        ax2.set_xlabel('Slice', fontsize = 12)
        ax2.set_ylabel('Energy Density for slice [W/m²]', fontsize = 12)
        ax2.set_xticks(slice_axis)
        ax2.set_yticks(ax1.get_yticks())

        plt.suptitle('Energy density for each slice: evolution over time and profiles.', fontsize = 'xx-large')

        ax1.grid(True, ls = '--', lw = 1, color = 'slategray')
        ax2.grid(True, ls = '--', lw = 1, color = 'slategray')

        ax1.ticklabel_format(axis = 'y', style = 'sci', scilimits=(0,0), useOffset = False)
        ax2.ticklabel_format(axis = 'y', style = 'sci', scilimits=(0,0), useOffset = False)

        plt.tight_layout()
        plt.savefig(self.folder+'convergence_energy.png')

    def mode_histogram(self):

        # calculating how many axes are needed to divide the histograms

        q_per_hist = 50        # how many rows each histogram will have

        n_of_hist = int(np.ceil(self.phonon.number_of_qpoints/q_per_hist))    # how many histograms will be plotted

        n_columns = max(int(np.ceil(10-0.3*self.phonon.number_of_branches)), 1)    # how many columns - more branches imply less columns to keep a fair aspect ratio
        n_rows    = int(np.ceil(n_of_hist/n_columns))   # how many rows
        
        # fig size

        height = n_rows*5
        width  = 0.2*self.phonon.number_of_branches*n_columns
        figsize = (width, height)
        
        fig = plt.figure(figsize = figsize, dpi = 100)  # genereating figure

        heigth_ratios = np.ones(n_rows+1)
        heigth_ratios[0] = 0.1
        gs = gridspec.GridSpec(n_rows+1, n_columns, height_ratios = heigth_ratios)

        # calculating modes

        # whole_data, bins_q, bins_j, _ = plt.hist2d(self.q_point, self.branch,
        #                                            bins = (self.phonon.number_of_qpoints, self.phonon.number_of_branches),
        #                                            vmin = -0.5,
        #                                            vmax = n+0.5,
        #                                            cmap = cmap)

        # setting colors
        present_modes, counts = np.unique(np.vstack( (self.q_point, self.branch) ), axis = 1, return_counts = True)

        n = counts.max()
        color = 'royalblue'
        alphas = np.linspace(0.1, 1.0, n)
        rgba_colors = np.zeros((n, 4))
        rgba_colors[:, :3] = matplotlib.colors.to_rgb(color)
        rgba_colors[:,  3] = alphas

        cmap =  ListedColormap(rgba_colors)

        # generating axes
        # axes = [fig.add_subplot(n_rows, n_columns, i+1) for i in range(n_of_hist)]  

        # plotting        
        for i in range(n_of_hist):
            start = i*q_per_hist
            stop  = (i+1)*q_per_hist

            q_points = np.arange(start, stop)
            
            indexes  = np.isin(self.q_point, q_points)
            
            q      = self.q_point[indexes]
            branch = self.branch[indexes]
            
            if i == n_of_hist-1:               
                q_bins = self.phonon.number_of_qpoints-start
            else:
                q_bins = q_per_hist

            gridcell = gs[i+n_columns]

            ax = fig.add_subplot(gridcell)
            
            data, _, _, _ = ax.hist2d(branch, q,
                      bins = (self.phonon.number_of_branches, q_bins),
                      vmin = -0.5,
                      vmax = n+0.5,
                      cmap = cmap,
                      edgecolors='w',
                      linewidths=0.2)
            
            ax.invert_yaxis()
            ax.set_xlabel('Branch', fontsize = 'x-large')
            ax.set_ylabel('Q-point', fontsize = 'x-large')
        
        ax = fig.add_subplot(gs[0, :])
        # ax.axis(False)
        fig.colorbar(mappable = cm.ScalarMappable(norm = NoNorm(), cmap = cmap),
                     cax = ax,
                    #  aspect = width/0.01,
                     fraction = 1,
                     orientation = 'horizontal',
                     label = 'Number of Particles',
                     ticks = np.arange(n+1).astype(int))

        plt.suptitle('End of run modes histogram (domain)', fontsize = 'xx-large')

        plt.tight_layout(pad = 1, rect = [0.02, 0.02, 0.98, 0.97])
        plt.savefig(self.folder + 'mode_histogram.png')
    
    def velocity_histogram(self):

        n_of_slices = self.args.slices[0] 
        slice_axis = self.args.slices[1]

        velocities = self.phonon.group_vel[self.q_point, self.branch, slice_axis]

        sliced_velocities = []

        for i in range(n_of_slices):
            indexes = self.slice_id == i
            sliced_velocities.append(velocities[indexes])

        phonon_velocity = self.phonon.group_vel[:, :, slice_axis].reshape(-1)
        
        sliced_velocities.append(phonon_velocity)

        # histogram plot

        fig = plt.figure(figsize = (30, 10), dpi = 100)
        
        gs = gridspec.GridSpec(1, 4)
        
        ax = fig.add_subplot(gs[0, :-1])

        data, bins, _ = ax.hist(sliced_velocities, density = True, bins = 50)

        ax.set_xlabel('Group velocity [angstrom / ps]', fontsize = 'x-large')
        ax.set_ylabel('Frequency of appearance', fontsize = 'x-large')

        labels = ['Slice {:d}'.format(i+1) for i in range(n_of_slices)]
        labels.append('Phonon Data')
        
        ax.legend(labels)

        # ratio plot
        
        bins = (bins[:-1]+bins[1:])/2
        data = np.array(data)

        ax1 = fig.add_subplot(gs[0, -1])
        
        for i in range(n_of_slices):
            variation = data[i, :]/data[-1, :]
            ax1.scatter(bins, variation, marker = '+')
        
        ax1.plot( [phonon_velocity.min(), phonon_velocity.max()], [1, 1], '--', color = 'k')

        ax1.set_xlabel('Group velocity [angstrom / ps]', fontsize = 'x-large')
        ax1.set_ylabel('Ratio of slice data to phonon data', fontsize = 'x-large')

        labels = ['Slice {:d}'.format(i+1) for i in range(n_of_slices)]
        labels = ['Expected']+labels

        ax1.legend(labels)
        
        plt.suptitle('Histogram (density) of group velocity along slice axis, per slice and according phonon data', fontsize = 'xx-large')

        fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.98])
        plt.savefig(self.folder + 'velocity_histogram.png')

    def flux_contribution(self):

        # getting data

        n_of_slices = self.args.slices[0] 
        slice_axis = self.args.slices[1]
        
        omega    = self.phonon.omega[    self.q_point, self.branch]
        velocity = self.phonon.group_vel[self.q_point, self.branch, slice_axis]
        volumes = self.geometry.slice_volume[self.slice_id]

        slice_res_T       = np.zeros(n_of_slices+2)
        slice_res_T[ 0]   = self.args.temperatures[0]
        slice_res_T[1:-1] = self.T[-1, :]
        slice_res_T[-1]   = self.args.temperatures[1]

        dX = 2*self.geometry.slice_length*self.a_in_m     # m

        # calculating contributions

        particle_flux = self.phonon.hbar*self.occupation*omega*velocity/volumes # eV/ps a²

        particle_flux = particle_flux*self.eVpsa2_in_Wm2    # W/m²

        dT = slice_res_T[self.slice_id+2] - slice_res_T[self.slice_id]  # K
        
        particle_k = particle_flux*(-dX/dT)     # (W/m²) * (m/K) = W/m K ; Central difference --> [T(+1) - T(-1)]/[x(1) - x(-1)]

        # defining bins

        n_bins = 100

        step = self.phonon.omega.max()/(n_bins-1)

        omega_bins = np.linspace(0-step/2, self.phonon.omega.max()+step/2, n_bins+1)
        omega_center = (omega_bins[:-1] + omega_bins[1:])/2

        # generating figure

        fig = plt.figure(figsize = (15, 10), dpi = 100)
        ax = fig.add_subplot(111)

        k_omega = np.zeros((n_of_slices, n_bins))

        for slc in range(n_of_slices):
            
            slice_indexes = (self.slice_id == slc)  # getting which particles are in slice
            
            for b in range(n_bins):
                bin_indexes = (omega >= omega_bins[b]) & (omega < omega_bins[b+1])    # identifying omega bin
                
                indexes = slice_indexes & bin_indexes   # getting particles of that bin in that slice
            
                k_omega[slc, b] = particle_k[indexes].sum()
            
        for slc in range(n_of_slices):
            
            ax.plot(omega_center, k_omega[slc, :], alpha = 0.5, linewidth = 3)
            # ax.fill_between(omega_center, 0, k_omega[slc, :], alpha = 0.5)
            # ax.bar(omega_center, k_omega[slc, :], alpha = 0.5, width = 0.8*step)
        
        labels = ['Slice {:d}'.format(i+1) for i in range(n_of_slices)]

        ax.legend(labels, fontsize = 'x-large')
        ax.set_xlabel(r'Angular Frequency $\omega$ [rad THz]', fontsize = 'x-large')
        ax.set_ylabel(r'Thermal conductivity in band $k(\omega)$ [W/mK]', fontsize = 'x-large')

        ax.ticklabel_format(axis = 'y', style = 'sci', scilimits=(0,0), useOffset = False)
        ax.ticklabel_format(axis = 'x', useOffset = False)

        plt.yticks(fontsize = 'x-large')
        plt.xticks(fontsize = 'x-large')

        plt.suptitle('Contribution of each frequency band to thermal conductivity. {:d} bands.'.format(n_bins), fontsize = 'xx-large')

        plt.tight_layout()
        plt.savefig(self.folder + 'k_contribution.png')

    def energy_histogram(self):
        
        # getting data

        n_of_slices = self.args.slices[0] 
        slice_axis = self.args.slices[1]
        
        omega    = self.phonon.omega[    self.q_point, self.branch]

        th_occupation = self.phonon.threshold_occupation[self.q_point, self.branch]

        energies = self.hbar*omega*self.occupation

        n_bins = 100

        data = []

        for slc in range(n_of_slices):

            indexes = (self.slice_id == slc) & (energies > 0)

            data.append(energies[indexes])

        # creating figure and plotting

        fig = plt.figure(figsize=(15, 5), dpi = 120)
        ax = fig.add_subplot(111)

        _, bins, _ = ax.hist(data, bins = n_bins, density = True, stacked = True)

        # expected energy at slice temperature

        T = self.T[-1, :].reshape(-1, 1, 1)

        exp_energy = self.phonon.calculate_energy(T, self.phonon.omega, threshold = True)

        # data = []
        
        data = np.zeros(0)

        for i in range(n_of_slices):
            indexes = (exp_energy[i, :, :] > 0)
            data = np.append(data, exp_energy[i, :, :][indexes].reshape(-1))
        
        ax.hist(data, bins = bins, density = True, # stacked = True,
                histtype  = 'step',
                color     = 'black',# for _ in range(n_of_slices)],
                linestyle = '--',
                linewidth = 1)

        labels = [r'Bose-Einstein at local $T$s']
        labels += ['Slice {:d}'.format(i+1) for i in range(n_of_slices)]
        ax.legend(labels)

        ax.set_xlabel(r'Energy level above threshold: $e = \hbar \omega n$ [eV]')
        ax.set_ylabel('Frequency of appearance of energy level')

        ax.ticklabel_format(axis = 'x', style = 'sci', scilimits=(0,0))
        ax.ticklabel_format(axis = 'y', style = 'sci', scilimits=(0,0))
        
        plt.title('Histogram (density) of particle energy level above threshold in each slice (only particles with energy > 0)', pad = 10)

        plt.tight_layout()
        plt.savefig(self.folder + 'energy_histogram.png')
            
        

        



