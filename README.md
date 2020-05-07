# twostate
calculate absorption and raman spectra

original fortran source code
by anne myers-kelley
university of california, merced
see https://doi.org/10.6071/M32Q0X
original source licensed under cc by 4.0

to do: 
version 2: vectorization of most for loops
           replace unecessary functions (e.g., 'fact') with numpy versions
           move 'dukomp', 'simpov', and 'kompov' into main text
version 1: direct conversion of fortran source to python script with minimal changes

##### original twostate readme #####

Notes on running program TWOSTATE to calculate absorption spectra and resonance Raman excitation profiles

This program calculates resonance Raman and absorption spectra using the time-dependent wavepacket method with two excited electronic states for a molecule with up to 30 vibronically active vibrations.  The ground and excited state vibrational frequencies may be different and coordinate dependence of the electronic transition moment may be included in any number of modes, but the ground and excited state normal modes are assumed to be parallel (no Duschinsky rotation).  Up to three of the vibrational modes may be thermally populated in the initial state up to n = 2.  It uses Mukamel's Brownian oscillator model for the solvent induced
broadening in the overdamped limit.  All deltas and dudqs are in ground state dimensionless coordinates.  

The input file TWOSTATE.IN and the executable program file TWOSTATE.EXE must be in the same folder.

The input file has the following format:
Line 1
	# of vibrational modes to consider (30 max).
	# of Raman transitions to calculate (40 max).
	# of time steps in Fourier transforms.  2500 is typical, 5000 max.
	Smallest Boltzmann factor to consider in thermal sum.  Set to 1 if no thermal occupation is to be considered.  Thermal occupation is included in only the last three modes listed, and only up to n = 2.
Line 2
	Electronic inhomogeneous broadening standard deviation in cm-1.
	Cutoff parameter for Brownian oscillator calculation.  Usually use 1.e-8.
	Temperature in K.  This enters into not only the thermal populations of the quantized normal modes but also the calculation of the Brownian oscillator lineshape.
	Angle in radians between transition dipole moments for the two electronic states.
Line 3
	Zero-zero energy in cm-1 for first excited electronic state.
	Homogeneous (Brownian oscillator) linewidth in cm-1 for first electronic state.
	Brownian oscillator lineshape parameter for first electronic state.  Usually set to 0.1.
	Transition length for first electronic state, in Å.
Line 4
	Repeat everything in line 3, for second electronic state.  Set transition length = 0 if only one state is to be considered.
Line 5
	Starting wavenumber for absorption spectrum calculation.
	Ending wavenumber for absorption spectrum calculation.
	Time step in fs for Fourier transform.  Usually set to 0.5.
	Solvent refractive index.
Line 6
	Ground state vibrational frequency in cm-1 for first vibrational mode.
	Vibrational frequency of this mode in first excited state in cm-1.  Usually set equal to ground state frequency.
	Delta in first excited state.  Huang-Rhys parameter is S = delta**2/2.
	Coordinate dependence of transition moment along this mode, (du/dq)/u0, in first excited state.  Usually set to 0.
Repeat line 6 for each of the other vibrational modes
Next line
	Vibrational frequency for first vibrational mode in second excited state
	Delta in second excited state
	Coordinate dependence of transition moment in second excited state
Repeat this line for each of the other vibrational modes
Next line
	Number of quanta excited in each mode for the first Raman transition to calculate.  For a vibrational fundamental, there will be one 1 and the rest 0.  For a combination band, there will be two or more modes with 1.  For an overtone, one mode with 2 and the rest 0.  Maximum allowed is 2.
Repeat this line for each of the other Raman transitions


TWOSTATE.F is Fortran source code that should be able to be compiled on most standard Fortran compilers.  No special libraries are required.  “Typical” calculations on “typical” PCs take between a few seconds and a few minutes to run.

Several output files are written:

TWOSTATE.TXT contains the calculated absorption spectrum in two columns, wavenumber and absorption cross section in Å2/molecule.  The absorption cross section is the molar absorptivity in L mole-1 cm-1 divided by 26149.  This file is in convenient format for importing into spreadsheet programs to be plotted with the experimental spectrum.

TWOSTATE.OUT contains a restatement of most of the input parameters and the calculated solvent reorganization energies.

PROFXX.TXT contains the Raman excitation profile for mode xx in two columns, wavenumber and Raman cross section in units of Å2 sr-1 molecule-1 divided by 10-11.  The scaling by 10-11 makes the cross sections have convenient values for display (typically in the range from 10 to 10000).  One file is written for each Raman transition calculated.
