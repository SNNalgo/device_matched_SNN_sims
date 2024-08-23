clear
load fisheriris

data = meas;
for i = 1:4
    data(:,i) = (data(:,i)-min(data(:,i)))/(max(data(:,i))-min(data(:,i)));
end
params = 12;
extended_data = zeros(150,params);

for i = 1:150
    extended_data(i,1:4) = (data(i,:));
    extended_data(i,5:8) = 1-data(i,:);
%     extended_data(i,9:end) = 4*data(i,:).*(1-data(i,:));
    extended_data(i,9:12) = 1-2*abs(data(i,:)-0.5);
%     extended_data(i,13:end) = 2*abs(data(i,:)-0.5);
end
extended_data(extended_data<0) = 0;

sorted_data = zeros(150,params);
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

save('iris_data2','sorted_data','type')