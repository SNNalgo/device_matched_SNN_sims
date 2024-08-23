clear
load fisheriris

data = meas;
for i = 1:4
    data(:,i) = (data(:,i)-min(data(:,i)))/(max(data(:,i))-min(data(:,i)));
end

numSensors = 7;     % assume number of sensors
centers =[];

for i = 0:(numSensors-1)
    centers = [centers i/(numSensors-1)];
end

width = 1/(numSensors-1);
sigma = 0.5*width;

extended_data = zeros(150,numSensors*4);

for i = 1:150
    for j = 1:numSensors
        %%% try different functions here %%%
%         extended_data(i,(j*4-3):(j*4)) = ((data(i,:)>(centers(j)-(width/2))).*(data(i,:)<=(centers(j)+(width/2))));
        extended_data(i,(j*4-3):(j*4)) = exp(-((data(i,:)-centers(j)).^2)/(2*(sigma.^2)));
    end
end

sorted_data = zeros(150,numSensors*4);
pp = randperm(50);
type = zeros(150,1);
type_in = [1 2 3];
offset = [0 50 100];

for i = 1:50
    rrn = randperm(3);
    type(3*i-2,1) = type_in(rrn(1));
    sorted_data(3*i-2,:) = extended_data(pp(i)+offset(rrn(1)),:);
    type(3*i-1,1) = type_in(rrn(2));
    sorted_data(3*i-1,:) = extended_data(pp(i)+offset(rrn(2)),:);
    type(3*i,1) = type_in(rrn(3));
    sorted_data(3*i,:) = extended_data(pp(i)+offset(rrn(3)),:);
end

save('iris_data_7_sensors','sorted_data','type')