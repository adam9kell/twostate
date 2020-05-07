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

file_path = os.path.abspath('twostate_v1.py')
index = [ind for ind, char in enumerate(file_path) if char == '\\']
file_path = file_path[:index[-1] + 1]


def convl(f,p,nf,np2,sig):
#
#   calculates the convolution of f with p and returns result in g
#
    g = np.zeros_like(f)
    
    if sig ==0:
        for i in range(nf):
            g[i] = f[i]
        return g
    pmult = p[0]
    psum = pmult
    for i in range(nf):
        g[i] = f[i]*pmult
    for j in range(1,np2):
        pmult = p[j]
        if pmult < 1.e-5:
            for i in range(nf):
                g[i] = g[i]/psum
            return g
        for k in range(1,nf+1):
            kmj = k - j
            if kmj > 0:
                g[k-1]=g[k-1]+pmult*f[kmj]
            kpj = k + j
            if kpj <= nf:
                g[k-1]=g[k-1]+pmult*f[kpj-1]
        psum = psum + 2.*pmult
    for i in range(nf):
        g[i] = g[i]/psum
    return g

def simpov(IS,n,delth,ntime,wg,delta,covlp,ceiwt,covlpq):
#
#   Calculates time-dependent overlap for mode with equal ground and
#   excited state frequency
#
    ci = 0 + 1j
    sq2 = 0.7071
    sq6 = 2.44949
    
    s = delta[IS,n]**2/2.
    sqrts = -delta[IS,n]*sq2
    cinc = np.exp(-ci*wg[n]*delth)
    ceiwt[IS,0] = 1 + 0j
    for i in range(1,ntime):
        ceiwt[IS,i] = ceiwt[IS,i-1]*cinc
#   Calculate <0|0(t)> through <4|0(t)>
#
    for i in range(ntime):
        ce = ceiwt[IS,i]
        ce1 = 1. - ce
        ct = np.exp(-s*ce1)
        covlp[IS,n,0,i]=np.exp(-s*(1.-ceiwt[IS,i]))
        covlp[IS,n,1,i]=sqrts*(ceiwt[IS,i]-1.)*covlp[IS,n,0,i]
        covlp[IS,n,2,i]=sq2*s*(ceiwt[IS,i]-1.)**2*covlp[IS,n,0,i]
        covlp[IS,n,3,i]=-s*sqrts*(1.-ceiwt[IS,i])**3*covlp[IS,n,0,i]/sq6
        covlpq[IS,n,0,i]=covlp[IS,n,1,i]
        covlpq[IS,n,1,i]=(ce+s*ce1**2)*ct
        covlpq[IS,n,2,i]=-(s+ce1**3+2.*ce*ce1)*ct*sqrts*sq2
        covlp[IS,n,4,i]=s**2*(1.-ceiwt[IS,i])**4/(2.*sq6)
    return covlp, covlpq

def kompov(IS,n,delth,ntime,wg,we,delta,covlp,ceiwt,coswt,csinwt,cqt,cpt,cat,cgt,toverh):
#
#   Calculates time-dependent overlap for non-Duschinsky rotated mode
#   with different ground and excited state frequencies. Negative ex.
#   state frequency is interpreted as imaginary freq.
#
    ci = 0 + 1j
    pi = 3.14159
    sq2 = 1.414
    sq3 = 1.73205
    sq6 = 2.44949
    
    s = -delta[IS,n]
    wwg = wg[n]
    ww = we[IS,n]
    if ww < 0:
#
#   Calculate coswt and sinwt for imaginary w
#
        einc = np.exp(-ww*delth)
        einc2 = np.exp(ww*delth)
        cqt[0] = 1 + 0j
        ceiwt[IS,0] = 1 + 0j
        for i in range(1,ntime):
            cqt[i] = cqt[i-1]*einc
            ceiwt[IS,i] = ceiwt[IS,i-1]*einc2
        for i in range(ntime):
            coswt[i] = 0.5*(cqt[i]+ceiwt[IS,i])
            csinwt[i] = ci*0.5*(cqt[i]-ceiwt[IS,i])
        cwwe = -ci*ww
#
#   Calculate coswt and sinwt for real w
#
    if ww >= 0:
        einc = np.cos(ww*delth)
        einc2 = np.sin(ww*delth)
        coswt[0] = 1 + 0j
        csinwt[0] = 0 + 0j
        for i in range(1,ntime):
            coswt[i] = einc*coswt[i-1]-einc2*csinwt[i-1]
            csinwt[i] = einc2*coswt[i-1]+einc*csinwt[i-1]
        cwwe = (1 + 0j)*ww
#
#   Calculate q(t) [cqt], p(t) [cpt], a(t) [cat], part of g(t) [cgt]
#
    for i in range(ntime):
        cqt[i] = s*(1.-coswt[i])
        cpt[i] = s*cwwe*csinwt[i]/wwg
        cat[IS,i] = -0.5*(ci*coswt[i]-cwwe*csinwt[i]/wwg)*ci
        cat[IS,i] = cat[IS,i]/(ci*csinwt[i]*wwg/cwwe + coswt[i])
        cgt[i] = ci*cat[IS,i]*cqt[i]**2 - 0.5*cpt[i]*(cqt[i]+s)
        if ww > 0:
            cgt[i] = cgt[i]+0.5*toverh[i]*cwwe
#
#   Put det(z) into coswt array
#
        coswt[i] = coswt[i]+ci*csinwt[i]*wwg/cwwe
    nphase = 0
    rxold = 1.
    for i in range(ntime):
#
#   Evaluate ln(det z) and add to g(t)
#
        realx = np.real(coswt[i])
        realy = np.imag(coswt[i])
        thet = np.arctan(realy/realx)
        if realx/rxold < 0:
            nphase=nphase+1
        rxold=realx
        cinc = ci*0.25*(np.log(realx**2+realy**2))
        cinc = cinc - 0.5*thet - 0.5*pi*nphase
        cgt[i]= cgt[i] + cinc
#
#   Calculate p prime and a
#
        cpt[i] = cpt[i] - 2.*ci*cat[IS,i]*cqt[i]
        cat[IS,i] = cat[IS,i] + 0.5
#
#   Calculate overlaps <0|0(t)> through <4|0(t)>
#
        cinc = np.exp(ci*cgt[i]-cpt[i]**2*0.25/cat[IS,i])
        covlp[IS,n,0,i] = cinc/np.sqrt(cat[IS,i])
        covlp[IS,n,1,i] = -ci*cpt[i]*covlp[IS,n,0,i]/(cat[IS,i]*sq2)
        cinc = (cpt[i]/cat[IS,i])**2
        cinc = cinc + 2.*(1.-1./cat[IS,i])
        covlp[IS,n,2,i] = -0.5*cinc*covlp[IS,n,0,i]/sq2
        cinc = cpt[i]/cat[IS,i]
        cinc = cinc*(cinc**2/6. + 1. - 1./cat[IS,i])
        covlp[IS,n,3,i] = ci*sq3*cinc*covlp[IS,n,0,i]/2
        cinc = (cpt[i]/cat[IS,i])**2
        cinc = cinc**2 + 12.*cinc*(1.-1./cat[IS,i])
        cinc = cinc + 12.*(1.-1./cat[IS,i])**2
        covlp[IS,n,4,i] = covlp[IS,n,0,i]*cinc/(8.*sq6)
    return covlp
    
def dukomp(IS,wg,we,delta,delth,ntime,nmax,nocc,dudq,covlp,ceiwt,covlp2,iw):
    cherm = np.zeros((12,5000), dtype = complex)
    ceiwt2 = np.zeros(5000, dtype = complex)
    cf = np.copy(ceiwt2)
    cg = np.copy(ceiwt2)
    cpsi = np.copy(ceiwt2)
    ca = np.copy(ceiwt2)
    c0 = np.copy(ceiwt2)
    cesum = np.copy(ceiwt2)
    ssinwt = np.zeros(5000)
    ci = 0 + 1j
    
    rat = np.sqrt(we/wg)
    wp = we + wg
    wm = we - wg
    rat2 = 2.*we/wg
    crat4 = ci*((we/wg)**2 - 1.)
    psipre = 0.25*wp**2/(we*wg)
    psirat = (wm/wp)**2
    
    cinc1 = np.exp(-ci*we*delth)
    cinc3 = cinc1**2
    ceiwt[IS,0] = 1 + 0j
    ceiwt2[0] = 1 + 0j
    delta2 = delta**2
    for i in range(1,ntime):
        ceiwt[IS,i] = ceiwt[IS,i-1]*cinc1
        ceiwt2[i] = ceiwt2[i-1]*cinc3
    for i in range(ntime):
        ssinwt[i] = np.imag(ceiwt[IS,i])
    
    for i in range(ntime):
        cf[i] = -(wg*(1.-ceiwt[IS,i]))/(wp-wm*ceiwt[IS,i])
        cg[i] = (rat2+crat4*ssinwt[i])/(rat2-crat4*ssinwt[i])
        cal = (wm-wp*ceiwt[IS,i])/(2.*wp-2.*wm*ceiwt[IS,i])
        cph = psipre*(1.-psirat*ceiwt2[i])
        cph = 1./cph
        cpsi[i] = np.sqrt(cph)
        ca[i] = np.sqrt(cal)
        c0[i] = cpsi[i]*np.exp(delta2*cf[i])
        cesum[i] = rat*cf[i]*delta/ca[i]
    
    cherm = hermit(cesum,cherm,nmax+nocc+1,ntime)
    for i in range(4):
        for j in range(6):
            for l in range(ntime):
                covlp[IS,j,i,l] = 0 + 0j
            ij = i + j
            pre = 1./np.sqrt(fact(i)*fact(j)*2.**ij)
            kstar = ij//2
            kstar = np.rint(kstar).astype(int)
            for k in range(kstar+1):
                prek = fact(2*k)/fact(k)
                prek = prek*pre*eta(j,i,k)
                for l in range(ntime):
                    cadd = prek*c0[l]
                    cadd = cadd*ca[l]**ij
                    cadd = cadd*cg[l]**k
                    cadd = cadd*cherm[ij-2*k,l]
                    covlp[IS,j,i,l]=covlp[IS,j,i,l]+cadd
    d1 = dudq/np.sqrt(2.)
    d2 = d1**2
    for i in range(1,4):
        si = np.sqrt(float(i-1))
        sip = np.sqrt(float(i))
        for jj in range(1,4):
            j = jj + i - 1
            sf = np.sqrt(float(j-1))
            sfp = np.sqrt(float(j))
            for k in range(ntime):
                cadd = 0 + 0j
                cad2 = 0 + 0j
                if j > 1:
                    cadd=sf*covlp[IS,j-2,i-1,k]
                if i > 1:
                    cadd=cadd+si*covlp[IS,j-1,i-2,k]
                cadd=cadd+sfp*covlp[IS,j,i-1,k]+sip*covlp[IS,j-1,i,k]
                if i > 1 and j > 1:
                    cad2=sf*si*covlp[IS,j-2,i-1,k]
                if j > 1:
                    cad2=cad2+sf*sip*covlp[IS,j-2,i,k]
                if i > 1:
                    cad2=cad2+si*sfp*covlp[IS,j,i-2,k]
                cad2=cad2+sfp*sip*covlp[IS,j,i,k]
                covlp2[IS,iw-1,i-1,jj-1,k]=covlp[IS,j-1,i-1,k]+d1*cadd+d2*cad2
    return covlp, covlp2
    
def fact(n):
    
    rn = float(n)
    fact = 1.
    if n <= 1:
        return fact
    for i in range(2,n+1):
        fact = fact*i
    return fact

def eta(jj,ii,kk):
    
    eta = 0.
    for i in range(2*kk+1):
        eta = eta + comb(jj,2*kk-i)*comb(ii,i)*(-1)**i
    return eta

def comb(m,l):
    
    if l > m:
        comb = 0.
    if l <= m:
        comb = fact(m)/(fact(l)*fact(m-l))
    return comb

def hermit(carg,cherm,n,ntime):
    
    for i in range(ntime):
        ca = carg[i]
        cherm[0,i] = 1 + 0j
        cherm[1,i] = 2.*np.real(ca)
        for j in range(2,n):
            cherm[j,i] = 2.*np.real(ca)*cherm[j-1,i] - 2.*(j-1)*cherm[j-2,i]
    return cherm

wg = np.zeros(30)
we = np.zeros((2, 30))
delta = np.copy(we)
du = np.copy(we)
covlp = np.zeros((2, 30, 5, 5000), dtype = complex)
ceiwt = np.zeros((2, 5000), dtype = complex)
covlp2 = np.zeros((2, 3, 3, 3, 5000), dtype = complex)
covlpq = np.zeros((2, 30, 3, 5000), dtype = complex)
coswt = np.zeros(5000, dtype = complex)
csinwt = np.copy(coswt)
cqt = np.copy(coswt)
cpt = np.copy(coswt)
cat = np.copy(ceiwt)
cgt = np.copy(coswt)
toverh = np.zeros(5000)
spectrum = np.zeros((1000, 40))
xfreq = np.zeros(1000)
nquanta = np.zeros((40, 30), dtype = int)
xs = np.copy(xfreq)
ffunc = np.copy(xfreq)
pfunc = np.zeros(501)
rshft = np.zeros(40)
ifreq = np.zeros(10)
cdamp = np.copy(ceiwt)
vs = np.copy(toverh)
pi = 3.14159
ci = 0 + 1j
hbar = 5308.8
boltz = 0.695
sq2 = 0.707107
sq32 = 1.224745
base = 'prof'
ext = '.txt'

#input parameters

twostate_in = open(file_path + 'twostate.in', 'r')
params_in = twostate_in.read().split()
params_in = [line.split(',') for line in params_in]
twostate_in.close()

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

e0 = np.array([float(params_in[2][0]), float(params_in[3][0])])
gamma = np.array([float(params_in[2][1]), float(params_in[3][1])])
rkappa = np.array([float(params_in[2][2]), float(params_in[3][2])])
u = np.array([float(params_in[2][3]), float(params_in[3][3])])

alow = float(params_in[4][0])
ahigh = float(params_in[4][1])
delt = float(params_in[4][2])
refrac = float(params_in[4][3])

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

R_count = 5 + 2 * nmode
total_count = 0
for i in range(nline):
    for j in range(nmode):
        nquanta[i, j] = params_in[R_count + total_count][0]
        total_count += 1
#
#     print parameters
#
twostate_out = open(file_path + 'twostate.out', 'w')
twostate_out.write('  # time steps, Boltzmann cutoff, T\n')
twostate_out.write(' {:11}{:15}{:11.4f}\n'.format(ntime, bcut, temp))
twostate_out.write('  Inhom. width, Brownian cutoff, angle in radians\n')
twostate_out.write(' {:12.3f}{:12.3f}{:12.3f}\n'.format(sig, cutoff, angle))
twostate_out.write('  E0, gamma, kappa, trans. length for each state\n')
for i in range(2):
    twostate_out.write(' {:12.3f}{:12.3f}{:12.3f}{:12.3f}\n'.format(e0[i], gamma[i], rkappa[i], u[i]))
twostate_out.write('  Time step, refactive index\n')
twostate_out.write(' {:12.3f}{:12.3f}\n'.format(delt, refrac))
twostate_out.write('  w(g), w(e), delta, du/dq in first state\n')
for i in range(nmode):
    twostate_out.write(' {:12.3f}{:12.3f}{:12.3f}{:12.3f}\n'.format(wg[i], we[0, i], delta[0, i], du[0, i]))
twostate_out.write('  w(g), w(e), delta, du/dq in second state\n')
for i in range(nmode):
    twostate_out.write(' {:12.3f}{:12.3f}{:12.3f}{:12.3f}\n'.format(wg[i], we[1, i], delta[1, i], du[1, i]))
twostate_out.write('  # of quanta in each mode in each line\n')
for i in range(nline):
    twostate_out.write('{}\n'.format(nquanta[i, :nmode]))
#
#
pre = 2.08e-20*1.e-6*delt**2*0.3/pi
for j in range(nmode-2,nmode + 1):
    for i in range(2):
        du[i,j - 1] = -du[i,j - 1]
        delta[i,j - 1] = delta[i,j - 1]*np.sqrt(we[i,j - 1]/wg[j - 1])
delth = delt/hbar
beta = 1./(boltz*temp)
for i in range(1,ntime + 1):
    toverh[i - 1] = (i-1)*delth
#
#         Calculate vibrational stuff for both excited states
#	      Brownian oscillator first
#
#
for l in range(2):
    for i in range(ntime):
        vs[i] = 0.
    rk2 = rkappa[l]**2
    a = (2.355+1.76*rkappa[l])/(1.+0.85*rkappa[l]+0.88*rk2)
    rlamb = rkappa[l]*gamma[l]/a
    reorg = beta*(rlamb/rkappa[l])**2/2.
    twostate_out.write('  solvent reorg = {:12.5f}      cm-1 in state{:12}\n'.format(reorg, l + 1))
    v = 2.*pi/beta
    n = 0
    n = n + 1
    vn = v*n*1.
    ainc=vn*delth
    arg=-1.
    einc=np.exp(-ainc)
    dex=1.
    denom = vn*(vn**2-rlamb**2)
    ii = 0
    for i in range(1,ntime):
        arg = arg+ainc
        dex = dex*einc
        vsi = (dex+arg)/denom
        vs[i] = vs[i] + vsi
    while ii!=0 or n==1:
        n = n + 1
        vn = v*n*1.
        ainc=vn*delth
        arg=-1.
        einc=np.exp(-ainc)
        dex=1.
        denom = vn*(vn**2-rlamb**2)
        ii = 0
        for i in range(1,ntime):
            arg = arg+ainc
            dex = dex*einc
            vsi = (dex+arg)/denom
            if np.abs(vsi/vs[i]) > cutoff:
                ii=1
            vs[i] = vs[i] + vsi
    vpre = 4.*reorg*rlamb/beta
    rpre = (reorg/rlamb)/np.tan(rlamb*beta/2.)
    for i in range(ntime):
        rlambt = rlamb*toverh[i]
        damp = 1. - np.exp(-rlambt) - rlambt
        rdamp = -rpre*damp + vpre*vs[i]
        cdamp[l,i] = np.conj(np.exp(-rdamp+ci*reorg*damp/rlamb))

#    Now undamped oscillators
#
    for i in range(1,4):
        im = nmode-3+i
        covlp, covlp2 = dukomp(l,wg[im-1],we[l,im-1],delta[l,im-1],delth,ntime,5,3,du[l,im-1],covlp,ceiwt,covlp2,i)
    for i in range(nmode-3):
        if wg[i] == we[l,i]:
            covlp, covlpq = simpov(l,i,delth,ntime,wg,delta,covlp,ceiwt,covlpq)
        if wg[i] != we[l,i]:
            covlp = kompov(l,i,delth,ntime,wg,we,delta,covlp,ceiwt,coswt,csinwt,cqt,cpt,cat,cgt,toverh)
        for j in range(ntime):
            ct1 = du[l,i]*sq2*(covlp[l,i,1,j]+covlpq[l,i,0,j])
            ct2 = du[l,i]*(covlp[l,i,2,j]+sq2*covlp[l,i,0,j]+sq2*covlpq[l,i,1,j])
            ct3 = du[l,i]*(sq32*covlp[l,i,3,j]+covlp[l,i,1,j]+sq2*covlpq[l,i,2,j])
            covlp[l,i,0,j] = covlp[l,i,0,j] + ct1
            covlp[l,i,1,j] = covlp[l,i,1,j] + ct2
            covlp[l,i,2,j] = covlp[l,i,2,j] + ct3
#
#   Set up the time integrals
#
    for i in range(ntime):
        cat[l,i]=u[l]**2*np.exp(-ci*(e0[l]+reorg)*toverh[i])*cdamp[l,i]
twostate_out.close()
#
#   Set up inhomogeneous broadening
#
part = 0.
xfreq[0] = alow
xinc = (ahigh-alow)/999.
for k in range(1,1000):
    xfreq[k] = xfreq[k-1] + xinc
pfunc[0] = 0.
for i in range(1,501):
    pfunc[i] = pfunc[i-1] + xinc
if sig != 0:
    deno = 2.*sig**2
    for i in range(501):
        pfunc[i] = np.exp(-pfunc[i]**2/deno)
#   Loop over all Raman lines to calculate. ceiwt is product of overlaps
#
for i1 in range(3):
    for i2 in range(3):
        for i3 in range(3):
            ei = i1*wg[nmode-3]+i2*wg[nmode-2]+i3*wg[nmode-1]
            weight = np.exp(-ei*beta)
            if weight < bcut:
                break
            part = part + weight
            for j in range(nline):
                for k in range(ntime):
                    ceiwt[0,k] = 1 + 0j
                    ceiwt[1,k] = 1 + 0j
                rshft[j] = 0.
                for k in range(nmode-3):
                    kk = nquanta[j,k]+int(1)
                    for l in range(ntime):
                        ceiwt[0,l]=ceiwt[0,l]*covlp[0,k,kk-1,l]
                        ceiwt[1,l]=ceiwt[1,l]*covlp[1,k,kk-1,l]
                    rshft[j] = rshft[j] + (kk-1)*wg[k]
                kk1 = nquanta[j,nmode-3]+int(1)
                kk2 = nquanta[j,nmode-2]+int(1)
                kk3 = nquanta[j,nmode-1]+int(1)
                rshft[j]=rshft[j]+(kk1-1)*wg[nmode-3]+(kk2-1)*wg[nmode-2]+(kk3-1)*wg[nmode-1]
                for k in range(ntime):
                    ceiwt[0,k]=ceiwt[0,k]*cat[0,k]*covlp2[0,0,i1,kk1-1,k]*covlp2[0,1,i2,kk2-1,k]*covlp2[0,2,i3,kk3-1,k]
                    ceiwt[1,k]=ceiwt[1,k]*cat[1,k]*covlp2[1,0,i1,kk1-1,k]*covlp2[1,1,i2,kk2-1,k]*covlp2[1,2,i3,kk3-1,k]
#
#   Loop over 1000 excitation frequencies
#
                for k in range(1000):
                    csum1=0.5*ceiwt[0,0]
                    csum2=0.5*ceiwt[1,0]
                    cinc = np.exp(ci*(ei+xfreq[k])*delth)
                    cold = 1 + 0j
#
#   Do time integral by simple sum (rectangle rule)
#
                    for l in range(1,ntime):
                        cold = cold*cinc
                        csum1= csum1 + cold*ceiwt[0,l]
                        csum2= csum2 + cold*ceiwt[1,l]
                    sigma1 = np.real(csum1*np.conj(csum1))
                    sigma2 = np.real(csum2*np.conj(csum2))
                    terf=np.real(csum1*np.conj(csum2)+csum2*np.conj(csum1))
#
#   Calculate total cross sections (k-th excitation freq,
#            j-th Raman mode)
#
                    sigma=sigma1+sigma2+terf*((np.cos(angle))**2+0.125*(np.sin(angle))**2)
                    sig0 = 0.33*(sigma1+sigma2+terf)
                    sig21 = 0.66*(sigma1+sigma2+terf)
                    sig22 = (np.cos(angle))**2-0.5*(np.sin(angle))**2
                    sig2 = sig21 * sig22
                    spectrum[k,j]=sigma*weight + spectrum[k,j]
        else:
            continue
        break
    else:
        continue
    break
#
#   Convolve with Gaussian inhomogeneous distribution
#
pre = pre/part
for i in range(nline):
    for j in range(1000):
        ffunc[j] = spectrum[j,i]
    gfunc = convl(ffunc,pfunc,1000,501,sig)
    for j in range(1000):
        spectrum[j,i]=pre*gfunc[j]*xfreq[j]*(xfreq[j]-rshft[i])**3
#
#
#   Print out profiles
#
for i in range(nline):
    for j in range(1000):
        spectrum[j,i]=spectrum[j,i]*1.e11
for i in range(nline):
    prof = open(file_path + '{}{:02d} {}'.format(base, i + 1, ext), 'w')
    for j in range(1000):
        prof.write(' {:11.3f}     {:11.4f}\n'.format(xfreq[j], spectrum[j, i]))
    prof.close()
#
#   Calculate absorption spectrum
#
pre2 = 5.745e-3*1.e-3*delt/(refrac*part)
for i1 in range(3):
    for i2 in range(3):
        for i3 in range(3):
            ei = i1*wg[nmode-3]+i2*wg[nmode-2]+i3*wg[nmode-1]
            weight = np.exp(-ei*beta)
            if weight < bcut:
                break
            weight = weight*pre2
            for k in range(ntime):
                ceiwt[0,k]=cat[0,k]*covlp2[0,0,i1,0,k]*covlp2[0,1,i2,0,k]*covlp2[0,2,i3,0,k]
                ceiwt[1,k]=cat[1,k]*covlp2[1,0,i1,0,k]*covlp2[1,1,i2,0,k]*covlp2[1,2,i3,0,k]
            for k in range(nmode-3):
                for l in range(ntime):
                    ceiwt[0,l] = ceiwt[0,l]*covlp[0,k,0,l]
                    ceiwt[1,l] = ceiwt[1,l]*covlp[1,k,0,l]
            for k in range(1000):
                csum = 0.5*(ceiwt[0,0]+ceiwt[1,0])
                cinc = np.exp(ci*(ei+xfreq[k])*delth)
                cold = 1 + 0j
                for l in range(1,ntime):
                    cold = cold*cinc
                    csum = csum+cold*(ceiwt[0,l]+ceiwt[1,l])
                sigma = np.abs(np.real(csum))
                sigma = sigma*weight
                xs[k] = sigma + xs[k]
        else:
            continue
        break
    else:
        continue
    break
gfunc = convl(xs,pfunc,1000,501,sig)
for j in range(1000):
    xs[j] = gfunc[j]*xfreq[j]
twostate = open(file_path + 'twostate.txt', 'w')
for i in range(1000):
    twostate.write('{:21.14f}{:21.14f}\n'.format(xfreq[i], xs[i]))
twostate.close()
