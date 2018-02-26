h = tools;
m = 20; % number of weights
N = 1000; % number of samples
s = rng;
theta_true = randn(m,1); % true weights
noise = randn(N,1);
H = randn(N,m);
y = H*theta_true + noise;
theta_0 = pinv(H'*H)*H'*y;
theta_i = h.estimate(H,y,theta_0,0.1);

hold on 
plot(theta_true)
plot(theta_0)
plot(theta_i)
legend('\theta','\theta_0','\theta_{i}')