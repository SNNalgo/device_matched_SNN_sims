function [success,failtype,failspikes,num_spikes] = test_iris3( w1,w2,level1_spikes,type,gL,C,dt,tauM,I0,I0n )
%TEST_IRIS Summary of this function goes here
%   Detailed explanation goes here
success = 0;
failtype = [];
failspikes = [];
num_spikes = [];

[~,M,Nsamp] = size(level1_spikes);
%cell_failspikes = cell(Nsamp,1);
[N1,N2] = size(w1);

%gL = 30e-9;    % nS to S
%C = 300e-12;   % pF to F
Vt = 20e-3;    % mV to V
El = -70e-3;   % mV to V

% T = 100e-3;     % simulation time per input
%dt = 0.2e-3;    % time-step

%I0 = 10e-12;     % base current for synapses
%I0n = 1e-8;
%tauM = 10e-3;    % time constants for current
tauS = tauM/4;

% bus2 = -10000*ones(N2,1);
s2 = zeros(N2,M,Nsamp);
v2 = zeros(N2,M,Nsamp);

for i = 1:Nsamp
    s1 = level1_spikes(:,:,i);
    s2(:,:,i) = zeros(N2,M);
    v2(:,1,i) = El;     % initialising the voltage values
    
    Isyn12 = zeros(N2,M);        % current from first layer to second layer
    Isyn22 = zeros(N2,M);        % lateral synaptic current
    Iapp2 = zeros(N2,M);         % applied current to second layer
    
%     bus1 = -10000*ones(N1,1);
%     bus2 = bus2-T;
    spike_times1 = cell(N1,1);
    spike_times2 = cell(N2,1);
    
    for j = 2:M
        t = double(j)*dt;
%         bus1(s1(:,j)>0) = t;
        [r1,~] = find(s1(:,j)>0);
        for k = 1:size(r1)
            spike_times1{r1(k)} = [spike_times1{r1(k)};t];
        end
        
        %%% Simulate the voltages first (of layer 2) %%%
        k1 = (1/C)*(-gL*(v2(:,j-1,i)-El.*ones(N2,1)) + Isyn12(:,j-1)+Isyn22(:,j-1)+Iapp2(:,j-1));
        Vph = v2(:,j-1,i) + dt.*k1;
        k2 = (1/C)*(-gL*(Vph-El.*ones(N2,1)) + Isyn12(:,j-1)+Isyn22(:,j-1)+Iapp2(:,j));
        Vnew = v2(:,j-1,i) + dt.*((k1+k2)./2);
        v2(:,j,i) = Vnew;
        [r2,~] = find(Vnew>Vt);
        for k = 1:size(r2)
            spike_times2{r2(k)} = [spike_times2{r2(k)};t];
            %RefractoryCount(r2(k),1) = Rp+dt;
        end
        spind=sign(Vnew-Vt);
        if max(spind)>0
            ind = find(spind>0);
%             bus2(ind) = t;
            v2(ind,j-1,i) = 0.1;
            s2(ind,j,i) = 1;
            v2(ind,j,i) = El;
        end
        %%% Voltage simulation end %%%
        
        %%% Simulate Currents next %%%
        for k = 1:N1
            if ~isempty(spike_times1{k})
                Isyn12(:,j) = Isyn12(:,j)+I0*(w1(k,:)')*(sum(exp((spike_times1{k}-t)/tauM)-exp((spike_times1{k}-t)/tauS)));
            end
        end
        for k = 1:N2
            if ~isempty(spike_times2{k})
                %Isyn22(:,j) = Isyn22(:,j)+I0n*(w2(k,:)')*(sum(exp((spike_times2{k}-t)/tauM)-exp((spike_times2{k}-t)/tauS)));
                Isyn22(:,j) = Isyn22(:,j)+I0n*(w2(k,:)');
            end
        end
        %%% Current simulation ends %%%
        
        %%% no STDP while testing (for now) %%%
        
    end
    
    %%% check the output and see if it is correct %%
    c1 = 0;
    sp_num = 0;
    for j = 1:N2
        if (j==type(i) && ~isempty(spike_times2{j}))||(j~=type(i) && isempty(spike_times2{j}))
            c1=c1+1;
        end
        if (~isempty(spike_times2{j}))
            sp_num =sp_num+1;
        end
    end
    num_spikes = [num_spikes sp_num];
    if c1==N2
        success=success+1;
    else
        failtype = [failtype type(i)];
        failspikes = [failspikes sp_num];
    end
%     if sp_num == 0 && type(i) == 2
%         success=success+1;
%     end
end

%success = success/Nsamp;

end

