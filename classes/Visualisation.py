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

# matplotlib.use('Agg')

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
        
        print('Scattering probability...')
        self.scattering_probability()
        print('Density of states...')
        self.density_of_states()

    def postprocess(self):
        # convergence data

        print('Reading convergence data')
        self.read_convergence()
        print('Plotting temperature convergence...')
        self.convergence_temperature()
        print('Plotting number of particles convergence...')
        self.convergence_particles()
        print('Plotting heat flux convergence...')
        self.convergence_heat_flux()
        print('Plotting energy density convergence...')
        self.convergence_energy()
        print('Plotting energy balance convergence...')
        self.convergence_energy_balance()

        # final particles states

        print('Reading particle data...')
        self.read_particles()
        # self.mode_histogram()
        print('Plotting group velocity histogram...')
        self.velocity_histogram()
        print('Plotting energy above threshold histogram...')
        self.energy_histogram()
        print('Plotting thermal conductivity with frequency...')
        self.flux_contribution()
        print('Plotting energy dispersion...')
        self.energy_dispersion()

    def scattering_probability(self):
        '''Plots the scattering probability of the maximum temperature (in which scattering mostly occurs)
        and gives information about simulation instability due to de ratio dt/tau.'''

        T = self.T_boundary.max()
        
        fig = plt.figure(figsize = (8,8), dpi = 120)
        
        x_data = self.phonon.omega[self.unique_modes[:, 0], self.unique_modes[:, 1]]

        # calculating y data
        ax = fig.add_subplot(111)

        Tqj = np.hstack( ( ( np.ones(self.unique_modes.shape[0])*T).reshape(-1, 1), self.unique_modes ) )
        
        lifetime = self.phonon.lifetime_function(Tqj)

        min_lt = lifetime[lifetime > 0].min()

        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            scat_prob = np.where(lifetime>0, 1-np.exp(-self.dt/lifetime), 0)

        y_data = scat_prob

        # colors = np.where(y_data[i] > criteria, 'orangered', 'royalblue')

        # setting limits and colors based on y data
        min_y = 0

        # max_y = y_data.max()
        # max_y = np.ceil(max_y*10)/10
        max_y = 1

        unstable_scat_prob    = 1-np.exp(-2)
        oscillating_scat_prob = 1-np.exp(-1)

        # P = 1-e^(-dt/tau)
        # max P = 1-e^(-max_dt/min_tau)
        # ln(1-max P) = -max_dt/min_tau
        # max_dt = -min_tau*ln(1-max P)

        max_dt = -min_lt*np.log(1-unstable_scat_prob)
        max_non_oscil_dt = -min_lt*np.log(1-oscillating_scat_prob)

        unstable_modes = np.where(y_data > unstable_scat_prob, 1, 0).sum()/y_data.shape[0]
        unstable_modes *= 100

        colors = ['royalblue', 'royalblue', 'gold', 'gold', 'crimson', 'crimson']
        nodes  = np.array([0.0, oscillating_scat_prob-0.01, oscillating_scat_prob+0.01, unstable_scat_prob-0.01, unstable_scat_prob+0.01, 1])
        cmap = LinearSegmentedColormap.from_list('byr', list(zip(nodes, colors)))

        ax.scatter(x_data, y_data, s = 1,
                    c = y_data, cmap = cmap,
                    vmin = 0,
                    vmax = 1)
        ax.plot([0, self.phonon.omega.max()], [   unstable_scat_prob,    unstable_scat_prob], color = 'gray', linestyle='--', alpha = 0.5)
        ax.text(x = 0, y = unstable_scat_prob - 0.01,
                s = r'Instability limit - $dt/\tau = 2$',
                va = 'top',
                ha = 'left',
                fontsize = 'large',
                color = 'gray')
        ax.plot([0, self.phonon.omega.max()], [oscillating_scat_prob, oscillating_scat_prob], color = 'gray', linestyle='--', alpha = 0.5)
        ax.text(x = 0, y = oscillating_scat_prob - 0.01,
                s = r'Oscillations limit - $dt/\tau = 1$',
                va = 'top',
                ha = 'left',
                fontsize = 'large',
                color = 'gray')
        
        ax.set_xlabel(r'Angular frequency $\omega$ [rad THz]',
                        fontsize = 'large')
        ax.set_ylabel(r'Scattering probability $P = 1-\exp(-dt/\tau)$',
                        fontsize = 'large')

        ax.set_ylim(min_y, max_y)

        ax.text(x = 0, y = max_y*0.97,
                s = 'T = {:.1f} K\nUnstable modes = {:.2f} %\nMax stable dt = {:.3f} ps\nMax non-oscillatory dt = {:.3f} ps'.format(T, unstable_modes, max_dt, max_non_oscil_dt),
                va = 'top',
                ha = 'left',
                fontsize = 'large',
                linespacing = 1.1)

        fig.suptitle(r'Scattering probability for $T_{{hot}}$ with $dt = ${:.2f} ps'.format(self.dt),
                     fontsize = 'xx-large')

        plt.tight_layout()
        plt.savefig(self.folder+'scattering_prob.png')

        if np.any(y_data > unstable_scat_prob):
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

        self.datetime    = data[:, 0].astype('datetime64[us]')
        self.timestep    = data[:, 1].astype(int)
        self.sim_time    = data[:, 2].astype(float)
        self.avg_en      = data[:, 3].astype(float)
        self.en_balance  = data[:, 4:6].astype(float)
        self.phi_balance = data[:, 6:8].astype(float)
        self.N_p         = data[:, 8].astype(int)
        self.T           = data[:, 9:self.number_of_slices+9].astype(float)
        self.slice_en    = data[:,   self.number_of_slices+9:2*self.number_of_slices+9].astype(float)
        self.phi         = data[:, 2*self.number_of_slices+9:3*self.number_of_slices+9].astype(float)
        self.slice_Np    = data[:, 3*self.number_of_slices+9:4*self.number_of_slices+9].astype(float) 

    def read_particles(self):
        filename = self.folder+'particle_data.txt'

        print('Start reading particle data')

        data = np.loadtxt(filename, delimiter = ',')

        print('Finished reading particle data')

        # f = open(filename, 'r')

        # lines = f.readlines()

        # f.close()

        # data = [ i.strip().split(',') for i in lines]
        # data = data[4:]
        # data = [ list(filter(None, i)) for i in data]
        # data = np.array(data)

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
        
        ax1.set_xlabel('Simulation time [ps]', fontsize = 12)
        ax1.set_ylabel('Temperature [K]', fontsize = 12)
        
        labels = ['Slice {:d}'.format(i) for i in slice_axis]

        if self.number_of_slices <=10:
            ax1.legend(labels)

        # Temperature profile
        ax2 = fig.add_subplot(122, sharey = ax1)
        n_timesteps = self.T.shape[0]

        labels = []

        n = 5
        color = 'royalblue'

        alphas = np.linspace(0.2, 1.0, n)
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
        if self.number_of_slices <=10:
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
        if self.number_of_slices <=10:
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
        slice_axis  = self.args.slices[1]
        
        omega    = self.phonon.omega[self.q_point, self.branch]
        velocity = self.phonon.group_vel[self.q_point, self.branch, slice_axis]

        n_dt_to_conv = np.floor( np.log10( self.args.iterations[0] ) ) - 2    # number of timesteps for each convergence datapoints
        n_dt_to_conv = int(10**n_dt_to_conv)
        n_dt_to_conv = max([10, n_dt_to_conv])

        n = int(200/(n_dt_to_conv*self.args.timestep[0]))
        slice_res_T       = np.zeros(n_of_slices+2)
        slice_res_T[ 0]   = self.args.temperatures[0]
        slice_res_T[1:-1] = self.T[-n:, :].mean(axis = 0)
        slice_res_T[-1]   = self.args.temperatures[1]

        # calculating contributions

        particle_flux = self.phonon.normalise_to_density(self.phonon.hbar*self.occupation*omega*velocity) # eV/ps a²

        particle_flux = particle_flux*self.eVpsa2_in_Wm2    # W/m²

        dX = 2*self.geometry.slice_length*self.a_in_m     # m

        dT = slice_res_T[self.slice_id+2] - slice_res_T[self.slice_id]  # K
        
        particle_k = particle_flux*(-dX/dT)     # (W/m²) * (m/K) = W/m K ; Central difference --> [T(+1) - T(-1)]/[x(1) - x(-1)]

        # flux considering all the domain
        dX_total = self.geometry.slice_length*(n_of_slices-1)*self.a_in_m   # m
        dT_total = slice_res_T[-2] - slice_res_T[1]                         # K

        particle_k_total = particle_flux*(-dX_total/dT_total)    # W/m²
        
        # defining bins

        n_bins = 100

        step = self.phonon.omega.max()/(n_bins-1)

        omega_bins = np.linspace(0-step/2, self.phonon.omega.max()+step/2, n_bins+1)
        omega_center = (omega_bins[:-1] + omega_bins[1:])/2

        # generating figure

        fig = plt.figure(figsize = (15, 20), dpi = 100)
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)

        k_omega = np.zeros((n_of_slices, n_bins))
        k_omega_total = np.zeros(n_bins)

        for b in range(n_bins):
        
            bin_indexes = (omega >= omega_bins[b]) & (omega < omega_bins[b+1])    # identifying omega bin

            k_omega_total[b] = particle_k_total[bin_indexes].sum()/(self.args.particles[0]*n_of_slices)
            
            for slc in range(n_of_slices):    
                slice_indexes = (self.slice_id == slc)  # getting which particles are in slice
                
                indexes = slice_indexes & bin_indexes   # getting particles of that bin in that slice
            
                k_omega[slc, b] = particle_k[indexes].sum()/(self.args.particles[0])
                
        for slc in range(n_of_slices):
            
            ax1.plot(omega_center, k_omega[slc, :], alpha = 0.5, linewidth = 3)
            # ax.fill_between(omega_center, 0, k_omega[slc, :], alpha = 0.5)
            # ax.bar(omega_center, k_omega[slc, :], alpha = 0.5, width = 0.8*step)

            ax2.plot(omega_center, np.cumsum(k_omega[slc, :]), alpha = 0.5, linewidth = 3)
        
        ax1.plot(omega_center, k_omega_total           , color = 'k', linestyle = '--')
        ax2.plot(omega_center, np.cumsum(k_omega_total), color = 'k', linestyle = '--')
        
        labels = ['Slice {:d}'.format(i+1) for i in range(n_of_slices)]
        labels += ['Domain']
        
        ax1.legend(labels, fontsize = 'x-large')
        ax1.set_xlabel(r'Angular Frequency $\omega$ [rad THz]', fontsize = 'x-large')
        ax1.set_ylabel(r'Thermal conductivity in band $k(\omega)$ [W/mK]', fontsize = 'x-large')

        ax1.ticklabel_format(axis = 'y', style = 'sci', scilimits=(0,0), useOffset = False)
        ax1.ticklabel_format(axis = 'x', useOffset = False)

        ax2.legend(labels, fontsize = 'x-large')
        ax2.set_xlabel(r'Angular Frequency $\omega$ [rad THz]', fontsize = 'x-large')
        ax2.set_ylabel(r'Cumulated Thermal conductivity in band $k(\omega)$ [W/mK]', fontsize = 'x-large')

        ax2.ticklabel_format(axis = 'y', style = 'sci', scilimits=(0,0), useOffset = False)
        ax2.ticklabel_format(axis = 'x', useOffset = False)

        text_x = ax2.get_xlim()[1]-np.array(ax2.get_xlim()).ptp()*0.05
        text_y = ax2.get_ylim()[0]+np.array(ax2.get_ylim()).ptp()*0.05

        ax2.text(text_x, text_y, r'$\kappa$ = {:.3e} W/mK'.format(k_omega_total.sum()),
                 verticalalignment   = 'bottom',
                 horizontalalignment = 'right',
                 fontsize = 'xx-large')
        
        plt.sca(ax1)
        plt.yticks(fontsize = 'x-large')
        plt.xticks(fontsize = 'x-large')
        plt.grid(True)
        plt.sca(ax2)
        plt.yticks(fontsize = 'x-large')
        plt.xticks(fontsize = 'x-large')
        plt.grid(True)

        plt.suptitle('Contribution of each frequency band to thermal conductivity. {:d} bands.'.format(n_bins), fontsize = 'xx-large')

        plt.tight_layout(pad = 3)
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
            # indexes = (exp_energy[i, :, :] > 0)
            data = np.append(data, exp_energy[i, :, :].reshape(-1))
        
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
            
    def energy_dispersion(self):

        omega       = self.phonon.omega[self.q_point, self.branch]  # particles omegas
        energies    = self.hbar*omega*self.occupation               # particles energies
        temperature = self.T[-1, :][self.slice_id]                  # particles temperatures

        exp_energy  = self.phonon.calculate_energy(temperature, omega, threshold = True)    # particle expected energy at T

        omega_order = np.ceil(np.log10(omega.max()))

        vmin = 0
        vmax = np.ceil(omega.max()/10**omega_order)*10**omega_order

        # plotting
        rows = 1 # int(np.ceil(self.number_of_slices**0.5))
        columns = self.number_of_slices #int(np.ceil(self.number_of_slices/rows))

        width_ratios = np.ones(columns)
        height_ratios = np.ones(rows+1)
        height_ratios[0] = 0.1

        fig = plt.figure(figsize = (columns*4, rows*4), dpi = 100)
        gs = gridspec.GridSpec(rows+1, columns, width_ratios = width_ratios, height_ratios = height_ratios)

        for i in range(self.number_of_slices):

            indexes = self.slice_id == i

            ax = plt.subplot(gs[columns+i])

            scat = ax.scatter(exp_energy[indexes], energies[indexes], marker='.', s = 1, c = omega[indexes], cmap = 'viridis', vmin = vmin, vmax = vmax)
            
            line = ax.get_xlim()

            ax.plot(line, line, linestyle = '--', linewidth = 1, color = 'k')

            ax.set_xlabel(r'$\hbar \omega [n^0(T) - n^0(T_{th})]$ [eV]')
            ax.set_ylabel(r'$\hbar \omega [n - n^0(T_{th})]$ [eV]')
            ax.set_title(r'$T = {}$ K'.format(self.T[-1, :][i]))

            ax.ticklabel_format(axis = 'x', style = 'sci', scilimits=(0,0))
            ax.ticklabel_format(axis = 'y', style = 'sci', scilimits=(0,0))
        
        fig.colorbar(scat, label = r'$\omega$ [THz]', fraction = 1, cax = plt.subplot(gs[0, :]), orientation = 'horizontal')
       
        plt.suptitle('Energy difference from threshold in final state', fontsize = 'large')

        plt.tight_layout(pad = 2)

        plt.savefig(self.folder + 'energy_distribution.png')
        
    def final_temperature_profile(self):
        '''Mean of the last 20 data points.'''

        n = 20

        mean_T  = self.T[-n:, :].mean(axis = 0)
        stdev_T = self.T[-n:, :].std( axis = 0)

        n_of_slices = self.geometry.n_of_slices
        slice_length = self.geometry.slice_length

        space_axis = np.arange(n_of_slices)*slice_length+slice_length/2

        fig = plt.figure(figsize = (8, 8), dpi = 120)
        ax = fig.add_subplot(111)
        
        ax.errorbar(space_axis, mean_T, xerr = stdev_T)

    def convergence_energy_balance(self):
        fig = plt.figure( figsize = (15, 5) )
        x_axis = self.sim_time

        ax1 = fig.add_subplot(121)

        # Energy balance
        y_axis = self.en_balance
        ax1.plot(x_axis, y_axis)
        ax1.plot(x_axis, y_axis.sum(axis = 1), linestyle = '--', color = 'k')
        
        ax1.set_xlabel('Simulation time [ps]', fontsize = 12)
        ax1.set_ylabel('Energy balance on surface [eV]', fontsize = 12)
        
        labels = ['Surface {}'.format(i+1) for i in range(self.en_balance.shape[1])]
        labels.append('Balance')

        if self.en_balance.shape[1] <=10:
            ax1.legend(labels)

        ax1.grid(True, ls = '--', lw = 1, color = 'slategray')

        ax1.ticklabel_format(axis = 'y', style = 'sci', scilimits=(0,0), useOffset = False)

        ax2 = fig.add_subplot(122)

        # Flux balance
        y_axis = self.phi_balance
        for i in range(y_axis.shape[1]):
            ax2.plot(x_axis, y_axis[:, i])
        ax2.plot(x_axis, y_axis[:, 0]-y_axis[:, 1], linestyle = '--', color = 'k')
        
        ax2.set_xlabel('Simulation time [ps]', fontsize = 12)
        ax2.set_ylabel('Heat flux balance on surface [W/m²]', fontsize = 12)
        
        labels = ['Surface {}'.format(i+1) for i in range(self.en_balance.shape[1])]
        labels.append('Balance')

        if self.en_balance.shape[1] <=10:
            ax2.legend(labels)

        ax2.grid(True, ls = '--', lw = 1, color = 'slategray')

        ax2.ticklabel_format(axis = 'y', style = 'sci', scilimits=(0,0), useOffset = False)

        plt.suptitle('Energy and Heat Flux balance over time.', fontsize = 'xx-large')

        plt.tight_layout()
        plt.savefig(self.folder+'convergence_en_balance.png')
