#     program twostate
#
#     Program to calculate Raman and absorption intensities using time-dep
#     method with two excited electronic states, including excited
#     state frequency changes or coordinate dependence of the transition
#     moment, and three thermally populated modes which may also have
#     both frequency changes and coord dependence.
#     Uses Mukamel's Brownian oscillator model for the solvent induced
#     broadening (identical to stochstic model for real part of broadening
#     function, but with imaginary part with gives solvent Stokes shift).
#     The formula correct when not in the high temperature limit is used.
#     All deltas and dudqs are in ground state dimensionless coordinates.
#
#         Raman intensities are reported in units of differential cross section,
#         (A**2/molecule-sr)*1.e11
#
#         Absorption spectrum is preinted in 'twostate.txt'
#         Input parameters, and Raman intensities at discrete excitation
#         wavelengths, are printed in 'twostate.out'
#         Raman profile for line x is printed in 'profx.txt'
#
#     Inputs (read from file 'twostate.in'):
#     nmode = total # vibrational modes (30 max)
#     nline = # Raman lines to calculate (40 max)
#     ntime = # time steps in Fourier transform (5000 max)
#     bcut = minimum Boltzmann factor to consider in thermal sum
#     cutoff = cutoff parameter in the sum over n in Brownian oscillator
#       calc, usually 10-6 to 10-8 range (check convergence by
#       reducing cutoff and re-running)
#     e0(i) = electronic zero-zero energy for state i
#     gamm(i) = electronic homogeneous linewidth (FWHM in cm-1) in state i
#     rkappa(i) = lineshape parameter in stochastic model for state i
#     sig = electronic inhomogeneous width (Gaussian standard dev. in cm-1)
#     u(i) = electonic transition length (A) for state i
#     alow,ahigh = lowest and highest energies to calc. absorption
#     delt = time step in Fourier transform (fs), typically around 0.5
#     refrac = solvent refractive index
#     efreq = array of energies (cm-1) to calculate Raman spectrum
#     wg(j) = ground state vib. freq (cm-1) of mode j
#     we(i,j) = ex. state freq. of mode j in state i
#     delta(i,j) = dimensionaless displacement of mode j in state i
#     du(i,j) = (du/dq)/u0 for mode j in state i
#       (i.e. ratio of vibronically induced to allowed transition mooment)
#     temp = temperature in Kelvin
#     nquanta(i,j) = # of quanta excited in mode j in Raman line i
#         angle = angle between transition moments of two state (in radians)

import os
import numpy as np
from scipy.constants import c, h, k, pi
from scipy.special import factorial as fact

file_path = os.path.abspath('twostate v2.py')

kB = k / h / c / 100 # /cm K
hbar = 5e12 / pi / c # units of fs /cm

def convl(f, p, nf, np2, sig):
    # calculates the convolution of f with p and returns result in g
    
    g = np.zeros(nf)
    
    if sig == 0:
        g = f
        return g
    pmult = p[0]
    psum = pmult
    g = f * pmult
    for j in range(1, np2):
        pmult = p[j]
        if pmult < 1e-5:
            g /= psum
            return g
        for k in range(1, nf + 1):
            kmj = k - j
            if kmj > 0:
                g[k - 1] += pmult * f[kmj]
            kpj = k + j
            if kpj <= nf:
                g[k - 1] += pmult * f[kpj - 1]
        psum += 2 * pmult
    g /= psum
    return g
    
def eta(jj, ii, kk):
    eta = 0
    for i in range(2 * kk + 1):
        if 2 * kk - i > jj:
            comb1 = 0
        if 2 * kk - i <= jj:
            comb1 = fact(jj) / fact(2 * kk - i) / fact(jj - (2 * kk - i))
        if i > ii:
            comb2 = 0
        if i <= ii:
            comb2 = fact(ii) / fact(i) / fact(ii - i)
        eta += comb1 * comb2 * (-1) ** i
    return eta

############################
##### input parameters #####
############################

with open(file_path + 'twostate.in', 'r') as twostate_in:
    params_in = twostate_in.read().split()
    params_in = [line.split(',') for line in params_in]

nmode = int(params_in[0][0])
nline = int(params_in[0][1])
ntime = int(params_in[0][2])
bcut = float(params_in[0][3])
sig = float(params_in[1][0])
cutoff = float(params_in[1][1])
temp = float(params_in[1][2])
angle = float(params_in[1][3])

if nmode > 30:
    nmode = 30
if nline > 40:
    nline = 40
if ntime > 5000:
    ntime = 5000
if bcut >= 1:
    bcut = 1

e0 = np.array([params_in[2][0], params_in[3][0]], dtype = float)
gamma = np.array([params_in[2][1], params_in[3][1]], dtype = float)
rkappa = np.array([params_in[2][2], params_in[3][2]], dtype = float)
u = np.array([params_in[2][3], params_in[3][3]], dtype = float)

alow = float(params_in[4][0])
ahigh = float(params_in[4][1])
delt = float(params_in[4][2])
refrac = float(params_in[4][3])

wg = np.zeros(nmode)
we = np.zeros((2, nmode))
delta = np.copy(we)
du = np.copy(we)

vib_count = 5 + nmode
for i in range(nmode):
    j = i + 5
    wg[i] = float(params_in[j][0])
    we[0, i] = float(params_in[j][1])
    delta[0, i] = float(params_in[j][2])
    du[0, i] = float(params_in[j][3])
    we[1, i] = float(params_in[vib_count + i][0])
    delta[1, i] = float(params_in[vib_count + i][1])
    du[1, i] = float(params_in[vib_count + i][2])

nquanta = np.zeros((nline, nmode), dtype = int)

R_count = 5 + 2 * nmode
total_count = 0
for i in range(nline):
    for j in range(nmode):
        nquanta[i, j] = params_in[R_count + total_count][0]
        total_count += 1

# write parameters to formatted output

with open(file_path + 'twostate.out', 'w') as twostate_out:
    twostate_out.write(('  # time steps, Boltzmann cutoff, T'
                       '\n {:11}{:15}{:11.4f}'
                       '\n  Inhom. width, Brownian cutoff, angle in radians'
                       '\n {:12.3f}{:12.3f}{:12.3f}'
                       '\n  E0, gamma, kappa, trans. length for each state'.format(ntime, bcut, temp, sig, cutoff, angle)))
    for i in range(2):
        twostate_out.write('\n {:12.3f}{:12.3f}{:12.3f}{:12.3f}'.format(e0[i], gamma[i], rkappa[i], u[i]))
    twostate_out.write(('\n  Time step, refractive index'
                       '\n {:12.3f}{:12.3f}'
                       '\n w(g), w(e), delta, du/dq, in first state'.format(delt, refrac)))
    for i in range(nmode):
        twostate_out.write('\n {:12.3f}{:12.3f}{:12.3f}{:12.3f}'.format(wg[i], we[0, i], delta[0, i], du[0, i]))
    twostate_out.write('\n  w(g), w(e), delta, du/dq in second state')
    for i in range(nmode):
        twostate_out.write('\n {:12.3f}{:12.3f}{:12.3f}{:12.3f}'.format(wg[i], we[1, i], delta[1, i], du[1, i]))
    twostate_out.write('\n  # of quanta in each mode in each line')
    for i in range(nline):
        twostate_out.write('\n{}'.format(nquanta[i, 0:nmode]))

######################
##### prefactors #####
######################
                           
raman_pre = 6.24e-27 / pi * delt ** 2
du[:, -3:] *= -1
delta[:, -3:] *= np.sqrt(we[:, -3:] / np.array([wg[-3:],] * 2))
delth = delt / hbar
beta = 1 / (kB * temp)
time_points = np.arange(ntime, dtype = float)
th = time_points * delth

#############################
##### vibrational terms #####
#############################

##### brownian oscillators #####

rlambt = np.zeros((2, ntime), dtype = float)

rk2 = rkappa ** 2
a = (2.355 + 1.76 * rkappa)/(1 + 0.85 * rkappa + 0.88 * rk2)
rlamb = rkappa * gamma / a
reorg = beta * (rlamb / rkappa) ** 2 / 2
vpre = 4 * reorg * rlamb / beta
rpre = reorg / rlamb / np.tan(rlamb * beta / 2)
rlambt = rlamb[:, None] * th
damp = 1 - np.exp(-rlambt) - rlambt

with open(file_path + 'twostate.out', 'a') as twostate_out:
    twostate_out.write(('\n  solvent reorg = {:12.5f}      cm-1 in state{:12}'
                       '\n  solvent reorg = {:12.5f}      cm-1 in state{:12}'.format(reorg[0], 1, reorg[1], 2)))

vs = np.zeros((2, ntime))
for i in range(2):
    v = 2 * pi / beta
    ainc = v * delth
    arg = time_points[1:] * ainc - 1
    einc = np.exp(-ainc)
    dex = einc ** time_points[1:]
    denom = v * (v ** 2 - rlamb[i] ** 2)

    vs[i, 1:] += (arg + dex) / denom

    ii = 0
    n = 1
    while ii != 0 or n == 1:
        n = n + 1
        ii = 0
        vn = v * n
        ainc = vn * delth
        arg = time_points[1:] * ainc - 1
        einc = np.exp(-ainc)
        dex = einc ** time_points[1:]
        denom = vn * (vn ** 2 - rlamb[i] ** 2)

        vsi = (arg + dex) / denom
    
        if np.any(np.abs(vsi / vs[i, 1:]) > cutoff):
            ii = 1
        
        vs[i, 1:] += vsi

rdamp = (vpre[:, None] * vs - rpre[:, None] * damp)
cdamp = np.conj(np.exp(1j * reorg[:, None] * damp / rlamb[:, None] - rdamp))

##### undamped oscillators #####

# dukomp subroutine in fortran source code
rat = np.sqrt(we[:,-3:] / wg[-3:])
wp = we[:,-3:] + wg[-3:]
wm = we[:,-3:] - wg[-3:]
rat2 = 2 * we[:,-3:] / wg[-3:]
crat4 = 1j * ((we[:,-3:] / wg[-3:]) ** 2 - 1)
psipre = 0.25 * wp ** 2 / wg[-3:] / we[:,-3:]
psirat = (wm / wp) ** 2
cinc1 = np.exp(-1j * delth) ** we[:, -3:]
cinc3 = cinc1 ** 2
delta2 = delta ** 2
ceiwt = cinc1[:, :, None] ** time_points
ceiwt2 = cinc3[:, :, None] ** time_points
ssinwt = np.imag(ceiwt)
cg = (rat2[:, :, None] + crat4[:, :, None] * ssinwt) / (rat2[:, :, None] - crat4[:, :, None] * ssinwt)
ca = np.sqrt((wm[:, :, None] - wp[:, :, None] * ceiwt) / 2 / (wp[:, :, None] - wm[:, :, None] * ceiwt))
cpsi = np.sqrt(1 / psipre[:, :, None] / (1 - psirat[:, :, None] * ceiwt2))
cf = -(wg[-3:, None] * (1 - ceiwt)) / (wp[:, :, None] - wm[:, :, None] * ceiwt)
cesum = rat[:, :, None] * cf * delta[:, -3:, None] / ca

cherm = np.ones((2, 3, 9, ntime))
cherm[:, :, 1] = 2 * np.real(cesum)
for i in range(2):
    for j in range(3):
        for k in range(2, 9):
            cherm[i, j, k] = 2 * (np.real(cesum[i, j]) * cherm[i, j, k - 1] - (k - 1) * cherm[i, j, k - 2])

c0 = cpsi * np.exp(delta2[:, -3:, None] * cf)
covlp = np.zeros((2, 3, 6, 4, ntime), dtype = complex)
pre = 1 / np.sqrt(fact(time_points[:4])[:, None] * fact(time_points[:6]) * 2 ** (time_points[:4, None] + time_points[:6]))
kstar = (time_points[:4, None] + time_points[:6]) // 2
kstar = np.rint(kstar).astype(int)

d1 = du[-3:] / np.sqrt(2)
d2 = d1 ** 2
si = np.sqrt(time_points[:3])
sip = np.sqrt(time_points[1:4])
jj = np.arange(1, 4)
jj = jj[:, None] + jj - 1
sf = np.sqrt(jj - 1)
sfp = np.sqrt(jj)

covlp2 = np.zeros((2, 3, 3, 3, ntime), dtype = complex)

for i in range(2):
    for j in range(3):
        for k in range(4):
            for l in range(6):
                for m in range(kstar[k, l] + 1):
                    kl = k + l
                    prek = fact(2 * m) / fact(m)
                    prek *= pre[k, l] * eta(l, k, m)
                    cadd = prek * c0[i, j] * ca[i, j] ** kl * cg[i, j] ** m * cherm[i, j, kl - 2 * m]
                    covlp[i, j, l, k] += cadd
        for k in range(3):
            for l in range(3):
                cadd = np.zeros(ntime, dtype = complex)
                cad2 = np.copy(cadd)
                if jj[k, l] > 1:
                    cadd = sf[k, l] * covlp[i, j, jj[k, l] - 2, k]
                if k > 0:
                    cadd += si[k] * covlp[i, j, jj[k, l] - 1, k - 1]
                cadd += sfp[k, l] * covlp[i, j, jj[k, l], i] + sip[k] * covlp[i, j, jj[k, l] - 1, k + 1]
                if k > 0 and jj[k, l] > 1:
                    cad2 = sf[k, l] * si[k] * covlp[i, j, jj[k, l] - 2, k - 1]
                if jj[k, l] > 1:
                    cad2 += sf[k, l] * sip[k] * covlp[i, j, jj[k, l] - 2, k + 1]
                if k > 0:
                    cad2 += si[k] * sfp[k, l] * covlp[i, j, jj[k, l], k - 1]
                cad2 += sfp[k, l] * sip[k] * covlp[i, j, jj[k, l], k + 1]
                covlp2[i, j, k, l] = covlp[i, j, jj[k, l] - 1, k] + d1[i, j] * cadd + d2[i, j] * cad2

# simpov subroutine in fortran source code
    
# calculates time-dependent overlap for mode
# with equal ground and excited state frequency

S = delta2[:, :-3] / 2
sqrtS = -delta[:, :-3] / np.sqrt(2)
cinc = np.exp(-1j * delth) ** wg[:-3]
ceiwt = cinc[:, None] ** time_points

# calculate <0|0(t)> through <4|0(t)>
covlp = np.zeros((2, nmode - 3, 5, ntime), dtype = complex)
covlpq = np.zeros((2, nmode - 3, 3, ntime), dtype = complex)

ce = 1 - ceiwt
ct = np.exp(-S[:, :, None] * ce)

covlp[:, :, 0] = ct
covlp[:, :, 1] = -sqrtS[:, :, None] * ce * ct
covlp[:, :, 2] = S[:, :, None] * (-ce) ** 2 * ct / np.sqrt(2)
covlp[:, :, 3] = -(S * sqrtS)[:, :, None] * ce ** 3 * ct / np.sqrt(6)
covlp[:, :, 4] = S[:, :, None] ** 2 * ce ** 4 / 2 / np.sqrt(6)

covlpq[:, :, 0] = covlp[:, :, 1]
covlpq[:, :, 1] = (S[:, :, None] * ce ** 2 + ceiwt) * ct
covlpq[:, :, 2] = -(S[:, :, None] + ce ** 2 + 2 * ceiwt) * ce * sqrtS[:, :, None] * ct / np.sqrt(2)

for i in range(2):
    for j in range(nmode - 3):
        if wg[j] != we[i, j]:
            # kompov subroutine in frotran source code
    
            # calculates time-dependent overlap for non-duschinsky rotated mode
            # with different ground and excited state frequencies. negative ex.
            # state frequency is interpreted as imaginary freq.

            covlp[i, j] = 0
            covlpq[i, j] = 0

            S = -delta[i, j]
            ceiwt = np.zeros((2, nmode - 3, ntime), dtype = complex)

            if we[i, j] < 0:
                # calculate cos(wt) and sin(wt) for imaginary w
                
                einc = np.exp(-we[i, j] * delth)
                einc2 = np.exp(we[i, j] * delth)

                cqt = einc ** time_points
                ceiwt[i, j] = einc2 ** time_points
                coswt = (cqt + ceiwt[i, j]) / 2
                csinwt = 1j * (cqt - ceiwt[i, j]) / 2
                
                cwwe = -1j * we[i, j]
                
            elif we[i, j] > 0:
                
                # calculate cos(wt) and sin(wt) for real w
                einc = np.cos(we[i, j] * delth)
                einc2 = np.sin(we[i, j] * delth)
            
                coswt = np.ones(ntime, dtype = complex)
                csinwt = np.zeros(ntime, dtype = complex)
                
                for k in range(1, ntime):    
                    coswt[k] = einc * coswt[k - 1] - einc2 * csinwt[k - 1]
                    csinwt[k] = einc2 * coswt[k - 1] + einc * csinwt[k - 1]

                cwwe = we[i, j] + 0j

            # calculate q(t) [cqt], p(t) [cpt], a(t) [cat], part of g(t) [cgt]            
            cqt = S * (1 - coswt)
            cpt = S * cwwe * csinwt / wg[j]
            cat = (coswt + 1j * cwwe * csinwt / wg[j]) / 2
            cat /= coswt + 1j * csinwt * wg[j] / cwwe
            cgt = 1j * cat * cqt ** 2 - cpt * (cqt + S) / 2

            if we[i, j] > 0:
                cgt += th * cwwe / 2

            # put det(z) into coswt array
            coswt += 1j * csinwt * wg[j] / cwwe

            # evaluate ln(det z) and add to g(t)
            nphase = 0
            rxold = 1

            realx = np.real(coswt)
            realy = np.imag(coswt)
            theta = np.arctan(realy / realx)
      
            cinc = 1j * np.log(realx ** 2 + realy ** 2) / 4
            for k in range(ntime):
                if realx[k] / rxold < 0:
                    nphase += 1
                rxold = realx[k]
                cinc[k] -= (theta[k] + nphase * pi) / 2
                cgt[k] += cinc[k]

            # calculate p prime and a
            cpt -= 2j * cat * cqt
            cat += 0.5

            # calculate overlaps <0|0(t)> through <4|0(t)>
            cinc = np.exp(1j * cgt - cpt ** 2 / cat / 4)
            covlp[i, j, 0] = cinc / np.sqrt(cat)
            covlp[i, j, 1] = -1j * cpt * covlp[i, j, 0] / (cat * np.sqrt(2))
            cinc = (cpt / cat) ** 2 + 2 * (1 - 1 / cat)
            covlp[i, j, 2] = -cinc * covlp[i, j, 0]/ 2 / np.sqrt(2)
            cinc = cpt / cat
            cinc *= cinc ** 2 / 6 + 1 - 1 / cat
            covlp[i, j, 3] = 1j * np.sqrt(3) * cinc * covlp[i, j, 0] / 2
            cinc = (cpt / cat) ** 2
            cinc *= cinc + 12 * (1 - 1 / cat)
            cinc += 12 * (1 - 1 / cat) ** 2
            covlp[i, j, 4] = covlp[i, j, 0] * cinc / 8 / np.sqrt(6)

ct1 = du[:, :-3, None] * (covlp[:, :, 1] + covlpq[:, :, 0]) / np.sqrt(2)
ct2 = du[:, :-3, None] * (covlp[:, :, 2] + (covlpq[:, :, 0] + covlpq[:, :, 1]) / np.sqrt(2))
ct3 = du[:, :-3, None] * (np.sqrt(3 / 2) * covlp[:, :, 3] + covlp[:, :, 1] + covlpq[:, :, 2] / np.sqrt(2))

covlp[:, :, 0] += ct1
covlp[:, :, 1] += ct2
covlp[:, :, 2] += ct3

##### set up the time integrals #####

cat = u[:, None] ** 2 * np.exp(-1j * (e0 + reorg)[:, None] * th) * cdamp

################################
##### raman cross sections #####
################################

spectrum = np.zeros((1000, nline))
xfreq = np.linspace(alow, ahigh, num = 1000, endpoint = True)

part = 0

for i1 in range(3):
    for i2 in range(3):
        for i3 in range(3):
            ei = i1 * wg[-3] + i2 * wg[-2] + i3 * wg[-1]
            weight = np.exp(-ei * beta)

            if weight < bcut:
                break
            
            part += weight
            rshft = np.zeros(nline)
            for j in range(nline):
                ceiwt = np.ones((2, ntime), dtype = complex)
                for k in range(nmode - 3):
                    ceiwt[0] *= covlp[0, k, nquanta[j, k]]
                    ceiwt[1] *= covlp[1, k, nquanta[j, k]]
                    rshft[j] += nquanta[j, k] * wg[k]
                rshft[j] += nquanta[j, -3] * wg[-3] + nquanta[j, -2] * wg[-2] + nquanta[j, -1] * wg[-1]
                ceiwt[0] *= cat[0] * covlp2[0, 0, i1, nquanta[j, -3]] * covlp2[0, 1, i2, nquanta[j, -2]] * covlp2[0, 2, i3, nquanta[j, -1]]
                ceiwt[1] *= cat[1] * covlp2[1, 0, i1, nquanta[j, -3]] * covlp2[1, 1, i2, nquanta[j, -2]] * covlp2[1, 2, i3, nquanta[j, -1]]

                # loop over 1000 excitation frequencies

                for k in range(1000):
                    csum1 = ceiwt[0, 0] / 2
                    csum2 = ceiwt[1, 0] / 2
                    cinc = np.exp(1j * (ei + xfreq[k]) * delth)

                    # do time integral by simple sum (rectangle rule)
                    cold = cinc ** time_points
                    csum1 += np.sum(cold[1:] * ceiwt[0, 1:])
                    csum2 += np.sum(cold[1:] * ceiwt[1, 1:])
                    sigma1 = np.real(csum1 * np.conj(csum1))
                    sigma2 = np.real(csum2 * np.conj(csum2))
                    terf = np.real(csum1 * np.conj(csum2) + csum2 * np.conj(csum1))

                    # calculate total cross sections (k-th excitation freq,
                    # j-th raman line)

                    sigma = sigma1 + sigma2 + terf * (np.cos(angle) ** 2 + np.sin(angle) ** 2 / 8)
                    sig0 = (sigma1 + sigma2 + terf) / 3
                    sig2 = 2 / 3 * (sigma1 + sigma2 + terf) * (np.cos(angle) ** 2 - np.sin(angle) ** 2 / 2)
                    spectrum[k, j] += sigma * weight
        else:
            continue
        break
    else:
        continue
    break

####################################
##### inhomogeneous broadening #####
####################################

pfunc = xfreq[:501] - alow
if sig != 0:
    pfunc = np.exp(-pfunc ** 2 / 2 / sig ** 2)
    
raman_pre /= part
for i in range(nline):
    ffunc = spectrum[:, i]

    # convolve with gaussian inhomogeneous distribution function
    gfunc = convl(ffunc, pfunc, 1000, 501, sig)

    spectrum[:, i] = raman_pre * gfunc * xfreq * (xfreq - rshft[i]) ** 3
    

# raman output profiles

spectrum *= 1e11
for i in range(nline):
    with open(file_path + 'prof{:02d} .txt'.format(i + 1), 'w') as prof:
        for j in range(1000):
            prof.write(' {:11.3f}     {:11.4f}\n'.format(xfreq[j], spectrum[j, i]))

##############################
##### absorption spectrum ####
##############################

xs = np.zeros(1000)
abs_pre = 5.745e-6 * delt / refrac / part

for i1 in range(3):
    for i2 in range(3):
        for i3 in range(3):
            ei = i1 * wg[-3] + i2 * wg[-2] + i3 * wg[-1]
            weight = np.exp(-ei * beta)
            
            if weight < bcut:
                break

            weight *= abs_pre
            
            ceiwt = np.zeros((2, ntime), dtype = complex)

            ceiwt = cat * covlp2[:, 0, i1, 0] * covlp2[:, 1, i2, 0] * covlp2[:, 2, i3, 0]
            ceiwt *= np.prod(covlp[:, :, 0], axis = 1)
            csum = (ceiwt[0, 0] + ceiwt[1, 0]) / 2
            cinc = np.exp(1j * (ei + xfreq) * delth)
            cold = cinc[:, None] ** time_points
            csum += np.sum(cold[:, 1:] * np.sum(ceiwt, axis = 0)[1:], axis = 1)
            sigma = np.abs(np.real(csum)) * weight
            xs += sigma
        else:
            continue
        break
    else:
        continue
    break

# convolve with gaussian inhomogeneous distribution function
gfunc = convl(xs, pfunc, 1000, 501, sig)

xs = gfunc * xfreq

# absorption output file

with open(file_path + 'twostate.txt', 'w') as twostate:
    for i in range(1000):
        twostate.write('{:21.14f}{:21.14f}\n'.format(xfreq[i], xs[i]))
