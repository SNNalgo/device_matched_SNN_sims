function [ neuronVoltage,spikes ] = LIF( neuronCurrents, t, Rp, gL, C )
%LIFNEURONSIMULATOR performs LIF neuron modeling numerically
%   neuronCurrents and time resolution are inputted in A and s (with proper orders of magnitude)
%   respectively, Rp is the refractory period
[N,M] = size(neuronCurrents);   % N = number of neurons, M = number of time instances
%gL = 30e-9;    % nS to S
%C = 300e-12;   % pF to F
Vt = 20e-3;    % mV to V
El = -70e-3;   % mV to V
neuronVoltage = El.*ones(N,M);  % initial steady state with I = 0
refractoryCount = zeros(N,1);
spikes = zeros(N,M);
for i = 2:M
    k1 = (1/C)*(-gL*(neuronVoltage(:,i-1)-El.*ones(N,1)) + neuronCurrents(:,i-1));
    Vph = neuronVoltage(:,i-1) + t.*k1;
    k2 = (1/C)*(-gL*(Vph-El.*ones(N,1)) + neuronCurrents(:,i));
    Vnew = neuronVoltage(:,i-1) + t.*((k1+k2)./2);
    [r,~] = find(Vnew>Vt);
    for j = 1:size(r)
        Vnew(r(j),1) = El;
        spikes(r(j),i) = 1;
        refractoryCount(r(j),1) = Rp+t;
    end
    Vnew(refractoryCount>0) = El;
    refractoryCount(refractoryCount>0) = refractoryCount(refractoryCount>0)-t;
    neuronVoltage(:,i) = Vnew;
end

end

