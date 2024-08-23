
clear
load('iris_data3.mat')

%load('iris_data_3_sensors.mat')

gL = 30e-9;    % nS to S
C = 300e-12;   % pF to F
Vt = 20e-3;    % mV to V
El = -70e-3;   % mV to V

N2 = 3;         % no. of level 2 neurons
[~,N1] = size(sorted_data);
Imax = 4e-9;    % max current - 4nA for 3/4 sensor linear, 4.4nA for 5 sensor, 5.2nA for 3 sensor gaussian 

T = 100e-3;     % simulation time per input
dt = 0.2e-3;    % time-step
M = int32(T/dt);
Rp = 5e-3;

I0 = 10e-12;     % base current for synapses
I0n = 1e-10;
tauM = 10e-3;    % time constants for current
tauM2 = 50e-3;
tauS = tauM/4;

gammaUp = 9;   % learning rates up - 1.5/-2 for 5 sensor, 9/-15 otherwise
gammaDn = -15;
tauUp = 10e-3;   % time constants for learning
tauDn = 2*tauUp;
mu = 1.7;

w1 = randn(N1,N2)*20+250;   % initializing weights
w2 = -randn(N2,N2)*100-4000;  % initializing inhibitory weights

wmax = 700;     % 900 for 5 sensor, 700 for linear 4 sensor, 750 for linear 3 sensor, 600 for 3 sensor gaussian
wmin = 0;

for i = 1:N2
    w2(i,i) = 0;
end

I = Imax*sorted_data';  % to be consistent with the LIF function

loop = 1;
epoch = 0;

% first generate all the first level responses
level1_potential = zeros(N1,M,150);
level1_spikes = zeros(N1,M,150);

for i = 1:150
    Iin_level1 = I(:,i)*ones(1,M);
    [nv1,spikes1] = LIF(Iin_level1,dt,Rp);
    level1_potential(:,:,i) = nv1;
    level1_spikes(:,:,i) = spikes1;
end

s2 = zeros(N2,M,150);
v2 = zeros(N2,M,150);

Ib = 15e-9;     % starting bias current
bus2 = -10000*ones(N2,1);
success = [];
diff = [];
prev_w1 = w1;
while loop>0
    % one showing of the data
    for i = 1:45
        s1 = level1_spikes(:,:,i);
        s2(:,:,i) = zeros(N2,M);
        v2(:,1,i) = El;     % initialising the voltage values
        
        Isyn12 = zeros(N2,M);        % current from first layer to second layer
        Isyn22 = zeros(N2,M);        % lateral synaptic current
        Iapp2 = zeros(N2,M);         % applied current to second layer
        for j = 1:N2
            if j~=type(i)
                Iapp2(j,:) = -Ib;    % bias current applied
            end
        end
        
        bus1 = -10000*ones(N1,1);
        bus2 = bus2-T;
        spike_times1 = cell(N1,1);
        spike_times2 = cell(N2,1);
        
        for j = 2:M
            t = double(j)*dt;
            bus1(s1(:,j)>0) = t;
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
                bus2(ind) = t;
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
                    Isyn22(:,j) = Isyn22(:,j)+I0n*(w2(k,:)')*(sum(exp((spike_times2{k}-t)/tauM2)));
                    %Isyn22(:,j) = Isyn22(:,j)+I0n*(w2(k,:)');
                end
            end
            %%% Current simulation ends %%%
            
            %%% STDP %%%
            [sp,~] = find(s2(:,j,i)>0);       % sp stores the indices of the level 2 neurons spiking at the current time instance
            
            for k = 1:size(sp)
                dw1 = gammaUp*((1-w1(:,sp(k))/wmax).^mu).*(exp((bus1-t)/tauUp));
                %                 dw1 = gammaUp*(exp((bus1-t)/tauUp));
                w1(:,sp(k)) = w1(:,sp(k))+dw1;
            end
            
            [sp1,~] = find(s1(:,j)>0);          % sp1 stores indices of level 1 neurons spiking at the current time instance
            
            for k = 1:size(sp1)
                dw_dn = gammaDn*((w1(sp1(k),:)/wmax).^mu).*(exp((bus2-t)/tauDn))';
                %                 dw_dn = gammaDn*(exp((bus2-t)/tauDn))';
                w1(sp1(k),:) = w1(sp1(k),:)+dw_dn;
            end
            
            w1(w1>wmax) = wmax;
            w1(w1<wmin) = wmin;
            %%% STDP ends %%%
        end
    end
    epoch = epoch+1;
    current_diff = sum(sum((w1-prev_w1).^2));
    diff =[diff current_diff];
    prev_w1 = w1;
    %Ib = 0.7*Ib;
    [ss,failtype,failspikes,num_spikes] = test_iris3(w1,w2,level1_spikes,type);
    success = [success ss];
    type_2_fail = length(find(failtype == 2));
    disp('success')
    disp(ss)
    disp('type_2_fail')
    disp(type_2_fail)
    zero_spike_fail = length(find(failspikes == 0));
    disp('zero spike')
    disp(zero_spike_fail)
    disp(epoch)
    disp(w1)
    
    if epoch > 24
        loop = -1;
    end
end