clear
load fisheriris

data = meas;
for i = 1:4
    data(:,i) = (data(:,i)-min(data(:,i)))/(max(data(:,i))-min(data(:,i)));
end

extended_data = zeros(150,8);

for i = 1:150
    extended_data(i,1:4) = data(i,:);
    extended_data(i,5:end) = 1-data(i,:);
end

sorted_data = zeros(150,8);
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

save('iris_data','sorted_data','type')
