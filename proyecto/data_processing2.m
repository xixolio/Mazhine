function [final_data,data,norm_data,max_values,min_values] = data_processing(data_name,lag)

data = csvread('serie1.e01C2.csv',1,1);
angles = data(:,2)*2*pi/360.;
data = data(:,1)';
[norm_data,max_values,min_values] = normalize(data);
x_vector = cos(angles').*norm_data(1,:);
y_vector = sin(angles').*norm_data(1,:);
data = [x_vector; y_vector];
final_data = zeros(2*lag,size(data,2)-lag+1);

for i=1:size(data,2)-lag+1
    for j=1:lag
        final_data(2*j-1:2*j,i) = norm_data(:,i+j-1);
    end
end

end
