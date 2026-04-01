clc; close all; clear all;
% 1D Fokker Plank implemented with velocity as a function of position
% Portions of code used from Dr. Uri Rogers

%% 1D Fokker Planck Equation and Variables

time_stop = 100E-3;     %Simulation Stop time in Seconds
time_step = 1E-4;       %Step size
time_measure = 50E-3;   %Estimate position pdf time in seconds
t = 0:time_step:time_stop;
delta_time = t(2)-t(1);

iterations = 80000;      %Number of experiements to run
number_Bins = 50;       %Number of histogram bins

Db = 1E-10;%2.10E10-5;     % cm^2/s diffusion constannt (http://link.springer.com/chapter/10.1007/978-1-4757-4748-5_5#page-1)
weinerX = randn(iterations,length(t)); 
weinerY = randn(iterations,length(t)); 
weinerV = randn(iterations,length(t)); 

radius = 10E6;
maxV = 10E4;

x_pos = zeros(iterations+1,length(t)+1); %starting position
y_pos = x_pos;
vel = ones(1, length(t + 1))*1E-4;
tempX = zeros(1, length(t + 1));
tempY = zeros(1,length(t + 1));


for ii = 1:iterations
    for jj = 2:(length(t) + 1) %recursive calls vs loop?

        tempX(1, jj) = ( tempX(jj-1) + (vel(1, jj - 1)*delta_time) + (sqrt(Db*delta_time)*weinerX(ii,jj-1)) ); %x_n+1 = x_n + mew(x)*deltaT + sqrt(Db*time)*weiner process
        tempY(1, jj) = ( tempY(jj-1) + (sqrt(Db*delta_time)*weinerY(ii,jj-1)) );
        vel(1, jj) = (abs(tempY(jj - 1)) - radius)^2 *(maxV/radius); %velocity must be positive 

    end
    x_pos(ii,:) = tempX(1, :);
    y_pos(ii,:) = tempY(1, :);
end
%%

figure()
plot(1E6*x_pos(1,:),1E6*y_pos(1,:),'k','linewidth',1)

% figure()
% plot(vel)
%axis([10 -10 10 -10]);
