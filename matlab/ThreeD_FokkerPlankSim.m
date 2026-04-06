% This script simulates a 3-D pdf, for the solution to the
% "ideal" Fokker-Plank Equation
%
% Written by Uri Rogers;  
% 5/7/16

close all; clear all; clc;
tic
time_stop = 100E-3;%0.005;     %Simulation Stop time in Seconds
time_step = 1E-4;       %Step size
time_measure = 50E-3;   %Estimate position pdf time in seconds

iterations = 10;      %Number of experiements to run
number_Bins = 50;       %Number of histogram bins.

%Sphere for 3-D plot
sensor_radius = 1;
%Black Sensor Location
x_black = 9;  %Dimension are in micrometers
y_black = -11;
z_black = -2;
% Now the Red Sensor
x_red = 11;  %Dimension are in micrometers
y_red = 0;
z_red = 0;
% Now the Blue Sensor
x_blue = 5;  %Dimension are in micrometers
y_blue = -3;
z_blue = 6;

color = ['r-.'; 'b--'; 'k- '];
tickFontSize = 14;  %Tick Mark Font Size
legendSize = 14;    %Legend Font Size
labelSize = 18;

rng(12)   %Spheres located for 80,000 iterations, and rng = 12

% System Configuration
Db = 1E-10;     % m^2/s,  Assume anisotropic diffusion coefficient
Vel = 1E-4;%3E-4; % m/s

t = 0:time_step:time_stop;
delta_time = t(2)-t(1);

measure_pdf_index = find(t>=time_measure,1)

x_pos_Drift = repmat(Vel*t,iterations,1);

x_rand = randn(iterations,length(t));
y_rand = randn(iterations,length(t));
z_rand = randn(iterations,length(t));

x_pos_Diff = sqrt(Db*delta_time)*x_rand;
y_pos_Diff = sqrt(Db*delta_time)*y_rand;
z_pos_Diff = sqrt(Db*delta_time)*z_rand;

% Now calculate the actual position of the biomarker for each iteration
% Since the flow is 1-D laminar and velocity is assumed independent of
% position, this is a straightforward cummulative summation process
x_pos = x_pos_Drift+cumsum(x_pos_Diff,2);
y_pos = cumsum(y_pos_Diff,2);               % A weiner process in y-dimension
z_pos = cumsum(z_pos_Diff,2);               % A weiner process in y-dimension

figure
subplot(2,2,1);plot(x_pos(1,:));title('X');
subplot(2,2,2);plot(y_pos(1,:));title('Y');
subplot(2,2,3);plot(z_pos(1,:));title('Z');
subplot(2,2,4);plot(x_pos_Drift(1,:));title('xposDrift');

%%
% Now plot the x-y location of a single particle
figure
plot(1E6*x_pos(1,:),1E6*y_pos(1,:),'k','linewidth',2)
xlabel('x-axis position \mum','fontsize',labelSize)
ylabel('y-axis position \mum','fontsize',labelSize)

% Now plot the x-y-z location of a single particle
figure
plot3(1E6*x_pos(1,:),1E6*y_pos(1,:),1E6*z_pos(1,:),'k','linewidth',3);
hold on
plot3(1E6*x_pos(2,:),1E6*y_pos(2,:),1E6*z_pos(2,:),'r-.','linewidth',3);
plot3(1E6*x_pos(3,:),1E6*y_pos(3,:),1E6*z_pos(3,:),'b:','linewidth',3);
xlabel('x-axis \mum','fontsize',labelSize)
ylabel('y-axis \mum','fontsize',labelSize)
zlabel('z-axis \mum','fontsize',labelSize)
set(gca,'fontsize',tickFontSize)
grid on
view([-8,10])

% Now add a sphere to the graph to model the sensor
[x,y,z]=sphere(25);
x =x*sensor_radius;
y =y*sensor_radius;
z =z*sensor_radius;
lightGrey = 0.8*[1 1 1]; % It looks better if the lines are lighter
surface(x+x_black,y+y_black,z+z_black,'FaceColor', 'k','facealpha',0.1,'EdgeColor','k')
surface(x+x_blue,y+y_blue,z+z_blue,'FaceColor', 'b','facealpha',0.1,'EdgeColor','b')
surface(x+x_red,y+y_red,z+z_red,'FaceColor', 'r','facealpha',0.1,'EdgeColor','r')

% print -depsc Three_D_threeSensorPath_Joural.eps

%%
% Now determine the pdf at a certain point in time and compare it to theory
% to ensure the path simulation model matches theory
x_at_time_t = x_pos(:,measure_pdf_index);
y_at_time_t = y_pos(:,measure_pdf_index);

[muhat_x,sigmahat_x] = normfit(x_at_time_t)
sigma_ideal = sqrt(Db*time_measure)
mu_ideal_x = Vel*time_measure
[muhat_y,sigmahat_y] = normfit(y_at_time_t)

bins_x = [-5E-6:5E-7:20E-6];
bins_y = [-10E-6:5E-7:10E-6];

[counts_x,centers_x]=hist(x_at_time_t,bins_x);
[counts_y,centers_y]=hist(y_at_time_t,bins_y);
bin_Width_x = centers_x(2)-centers_x(1);
bin_Width_y = centers_y(2)-centers_y(1);
%%
xx = bins_x;
yy = bins_y;
x_pdf_at_time_t =1/sqrt(2*pi*Db*time_measure)* exp(-(xx-(Vel*time_measure)).^2/(2*Db*time_measure));
y_pdf_at_time_t =1/sqrt(2*pi*Db*time_measure)* exp(-(yy).^2/(2*Db*time_measure));
figure
plot(1E6*centers_x,counts_x/sum(counts_x)/bin_Width_x,'b','linewidth',2); hold on;
plot(1E6*xx,x_pdf_at_time_t,'r--','linewidth',2)

plot(1E6*centers_y,counts_y/sum(counts_y)/bin_Width_y,'k','linewidth',2); hold on;
plot(1E6*yy,y_pdf_at_time_t,'r--','linewidth',2)

 ylabel('PDF $$f(x)$$','interpreter','latex','fontsize',labelSize)
 xlabel('x and y location  $$({\mu}m)$$','interpreter','latex','fontsize',labelSize)
 pbaspect([1.618 1 1]);      %This sets the aspect ratio to the "Golden Ratio" vs 4/3
 set(gca,'fontsize',tickFontSize)
 grid on
 
  I = legend('x-axis est.','x-axis theory','y-axis est.','y-axis theory');
  c = get(I,'children');
  set(I,'interpreter','latex');
  set(I,'FontSize',legendSize)
  set(I,'Location','Northeast')
  
  %%%% Use this for creating figure for paper
   % print -depsc Two_D_FokkerPlankSimmulation_Joural.eps
toc
