function [dt, T, tauM, tauM2, tauUp, tauS, tauDn, Imax, I0, I0n, Ib, C, gL] = initVals(Ith, tau)

base_tau = 300e-12/30e-9;
delV = 90e-3;
base_dt = 0.2e-3;
base_T = 100e-3;
base_tauM = 10e-3;
base_tauM2 = 50e-3;
base_tauUp = 10e-3;


base_Ith = 2.7e-9;
base_Imax = 4e-9;
base_I0 = 10e-12;     % base current for synapses
base_I0n = 1e-10;
base_Ib = 15e-9;

dt = base_dt*(tau/base_tau);
T = base_T*(tau/base_tau);
tauM = base_tauM*(tau/base_tau);
tauM2 = base_tauM2*(tau/base_tau);
tauUp = base_tauUp*(tau/base_tau);
tauS = tauM/4;
tauDn = 2*tauUp;

Imax = base_Imax*(Ith/base_Ith);
I0 = base_I0*(Ith/base_Ith);
I0n = base_I0n*(Ith/base_Ith);
Ib = base_Ib;
%Ib = base_Ib*(Ith/base_Ith);

C = Ith*tau/delV;
gL = C/tau;
end