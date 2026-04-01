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

Db = 5E-10;%2.10E10-5;     % cm^2/s diffusion constannt (http://link.springer.com/chapter/10.1007/978-1-4757-4748-5_5#page-1)
weinerX = randn(iterations,length(t)); 
weinerY = randn(iterations,length(t)); 
weinerZ = randn(iterations,length(t)); 

radius = 10E6;
maxVelocity = 10E4;

x_position = zeros(iterations+1,length(t)+1); %starting position
y_position = x_position;
z_position = x_position;
velocity = ones(1, length(t + 1))*1E-4;
tempX = zeros(1, length(t + 1));
tempY = zeros(1,length(t + 1));
tempZ = zeros(1,length(t + 1));

tic
h = waitbar(0,'Working');
for ii = 1:iterations
    waitbar(ii/iterations);
    for jj = 2:(length(t) + 1) %recursive calls vs loop?

        tempX(1, jj) = ( tempX(jj-1) + (velocity(1, jj - 1)*delta_time) + (sqrt(Db*delta_time)*weinerX(ii,jj-1)) ); %x_n+1 = x_n + mew(x)*deltaT + sqrt(Db*time)*weiner process
        tempY(1, jj) = ( tempY(jj-1) + (sqrt(Db*delta_time)*weinerY(ii,jj-1)) );
        tempZ(1, jj) = ( tempZ(jj-1) + (sqrt(Db*delta_time)*weinerZ(ii,jj-1)) );
        velocity(1, jj) = (abs(tempY(jj - 1)) - radius)^2 *(maxVelocity/radius); %velocity must be positive 

    end
    x_position(ii,:) = tempX(1, :);
    y_position(ii,:) = tempY(1, :);
    z_position(ii,:) = tempZ(1, :);
end
close(h);
toc
%% plot single execution

figure()
plot3(1E6*x_position(1,:),1E6*y_position(1,:),1E6*z_position(1,:),'k','linewidth',1)
%plot3(1E6*x_position,1E6*y_position,1E6*z_position,'k','linewidth',1) %plot all points

%% PDF

measure_pdf_index = find(t>=time_measure,1);
x_at_time_t = x_position(:,measure_pdf_index);
y_at_time_t = y_position(:,measure_pdf_index);

[muhat_x,sigmahat_x] = normfit(x_at_time_t)
sigma_ideal = sqrt(Db*time_measure)
mu_ideal_x = velocity(x_at_time_t)*time_measure
[muhat_y,sigmahat_y] = normfit(y_at_time_t)

bins_x = [-5E-6:5E-7:20E-6];
bins_y = [-10E-6:5E-7:10E-6];

[counts_x,centers_x]=hist(x_at_time_t,bins_x);
[counts_y,centers_y]=hist(y_at_time_t,bins_y);
bin_Width_x = centers_x(2)-centers_x(1);
bin_Width_y = centers_y(2)-centers_y(1);
%% PDF Plot

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

