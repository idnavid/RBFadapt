function h = tools()
    h.steering = @steering;
    h.source = @source;
    h.gen_noise = @gen_noise;
    h.bi_noise = @bi_noise;
    h.bi_noise_skewed = @bi_noise_skewed;
    h.mse = @mse;
    h.error = @estimation_error; 
    h.estimate = @estimate_iter;
end
function a = steering(M,phi)
% Estimates steering vector of dim M at angle phi
% For simplicity, this function assumes half-wavelength
% spacing between antennas.
a = 0:M-1;
a = exp(-1i*a*pi*sin(phi)); 
a = a(:);
end

function b = source(f,L)
b = 0:L-1;
b = exp(2*pi*1i*f*b);
b = b(:);
end


function n = gen_noise(dim,L,pi1,mu1,mu2,sigma1,sigma2)
mu_n = [mu1+zeros(1,dim);mu2+zeros(1,dim)];
sigma_n = cat(3,sigma1*eye(dim),sigma2.*eye(dim));
delta = pi1;
prior_n = [delta, 1-delta];
obj = gmdistribution(mu_n,sigma_n,prior_n);
n = random(obj,L); 
n = reshape(n,dim,L);
n = n(:);
% figure
% hist(n(1,:),100)
% pause
end

function n = bi_noise(dim,L,pi1,mu1,mu2,sigma1,sigma2)
n1 = mu1+sigma1*randn(dim,floor(L*pi1)); 
pi2 = 1 - pi1;
n2 = mu2+sigma2*randn(dim,floor(L*pi2));
n = zeros(dim,L);
n(:,1:length(n1)+length(n2)) = [n1, n2]; 
n=n(:,randperm(length(n)));
n = n(:);
end

function n = bi_noise_skewed(dim,L,pi1,mu1,mu2,sigma1,sigma2)
n1 = mu1+sigma1*randn(dim,floor(L*pi1)); 
pi2 = 1 - pi1;
n2 = mu2+sigma2*randn(dim,floor(L*pi2));
n = zeros(dim,L);
n(:,1:length(n1)+length(n2)) = [n1, abs(n2)]; 
n=n(:,randperm(length(n)));
n = n(:);
end

function err = estimation_error(param_hat,param_true)
m = length(param_hat);
err = (param_hat - param_true);
err = err'*err/m;
end

function param_hat = estimate_iter(H,y,param_0,alpha)
% H: regressor matrix
% y: observation vector
% param_0: initial estimation pinv(H'*H)*H'*y
% alpha: tuning parameter (scalar between 0 and 1). alpha=0.1

param_hat = param_0;
for i = 1:100
w = (y - H*param_hat); 
w = exp(-alpha*w.^2); 
w = w/sum(w);
w = diag(w); 
param_hat = inv(H'*w*H)*H'*w*y;
end 
end

function e_mse = mse(doa_matrix,target_doa)
% doa_matrix contains doa estimation for different 
% trials (rows) across different angles. 
[ntrials,ntargets] = size(doa_matrix); 
avg_doa = mean(doa_matrix,1); 
errors = avg_doa - target_doa; 
errors = errors(:); 
e_mse = sqrt(errors'*errors)/length(errors);
end