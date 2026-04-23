%% RUN_MUTUALISM.M
%  Plant-pollinator NSFD comparison.
%  Compares Forward Euler, classical RK4, and NSFD against ode45 reference
%  on the obligate mutualism model from Honors Capstone.

clear; clc; close all;

%% Parameters (capstone Appendix B, Peponapis-Cucurbita calibration)
p.rP = 0.020;  p.rA = 0.016;
p.KP = 200;    p.KA = 150;
p.muP = 0.0216; p.muA = 0.01696;
p.aP = 0.050;  p.aA = 0.048;
p.hP = 12;     p.hA = 12;
p.Omega = 0.141;

%% Right-hand side and NSFD splitting
f  = @(x) mutualism_rhs(x, p);
F  = @(x) mutualism_F(x, p);
G  = @(x) mutualism_G(x, p);

%% Denominator function tied to dominant linear rate
q = max(p.rP, p.rA) + max(p.muP, p.muA);
phi = @(h) (1 - exp(-q*h)) / q;

%% Simulation settings
t0 = 0; tf = 1000;
x0 = [150; 120];              % inside basin of attraction of coexistence E*
h_values = [1, 10, 50, 100, 200];  % step sizes to compare

%% Reference solution via ode45
opts = odeset('RelTol',1e-10,'AbsTol',1e-12,'NonNegative',[1 2]);
[t_ref, X_ref] = ode45(@(t,x) f(x), [t0 tf], x0, opts);
x_eq = X_ref(end, :);
fprintf('Reference coexistence equilibrium: P* = %.2f, A* = %.2f\n', x_eq(1), x_eq(2));

%% Run all three schemes at each step size
results = struct();
methods = {'Euler', 'RK4', 'NSFD'};

for hi = 1:length(h_values)
    h = h_values(hi);
    N = ceil((tf - t0) / h);
    tgrid = t0 + (0:N)' * h;

    for mi = 1:length(methods)
        method = methods{mi};
        X = zeros(N+1, 2); X(1, :) = x0';
        tic;
        for k = 1:N
            switch method
                case 'Euler'
                    X(k+1, :) = euler_step(f, X(k, :)', h)';
                case 'RK4'
                    X(k+1, :) = rk4_step(f, X(k, :)', h)';
                case 'NSFD'
                    X(k+1, :) = nsfd_step(F, G, X(k, :)', phi(h))';
            end
        end
        walltime = toc;

        neg_count = sum(any(X < 0, 2));
        final_err = norm(X(end, :) - x_eq);

        results(hi, mi).method = method;
        results(hi, mi).h = h;
        results(hi, mi).t = tgrid;
        results(hi, mi).X = X;
        results(hi, mi).walltime = walltime;
        results(hi, mi).neg_count = neg_count;
        results(hi, mi).final_err = final_err;
    end
end

%% Figure 1: trajectories at a challenging step size
fig1 = figure('Position',[100 100 900 350]);
hi_plot = find(h_values == 50);     % step size where Euler struggles

subplot(1,2,1); hold on; grid on;
plot(t_ref, X_ref(:,1), 'k-', 'LineWidth', 2.0, 'DisplayName', 'ode45 reference');
plot(results(hi_plot,1).t, results(hi_plot,1).X(:,1), 'r--o', 'MarkerSize', 3, 'DisplayName', 'Euler');
plot(results(hi_plot,2).t, results(hi_plot,2).X(:,1), 'b:s',  'MarkerSize', 3, 'DisplayName', 'RK4');
plot(results(hi_plot,3).t, results(hi_plot,3).X(:,1), 'g-^',  'MarkerSize', 3, 'DisplayName', 'NSFD');
xlabel('Time (days)'); ylabel('Plant abundance P');
title(sprintf('Plant population, h = %g days', h_values(hi_plot)));
legend('Location','best');

subplot(1,2,2); hold on; grid on;
plot(t_ref, X_ref(:,2), 'k-', 'LineWidth', 2.0, 'DisplayName', 'ode45 reference');
plot(results(hi_plot,1).t, results(hi_plot,1).X(:,2), 'r--o', 'MarkerSize', 3, 'DisplayName', 'Euler');
plot(results(hi_plot,2).t, results(hi_plot,2).X(:,2), 'b:s',  'MarkerSize', 3, 'DisplayName', 'RK4');
plot(results(hi_plot,3).t, results(hi_plot,3).X(:,2), 'g-^',  'MarkerSize', 3, 'DisplayName', 'NSFD');
xlabel('Time (days)'); ylabel('Pollinator abundance A');
title(sprintf('Pollinator population, h = %g days', h_values(hi_plot)));
legend('Location','best');

saveas(fig1, 'fig_mutualism_trajectories.png');

%% Table: convergence to equilibrium and positivity violations
fprintf('\n%-8s %-8s %-12s %-12s %-10s\n', 'Method', 'h', 'Final err', 'Neg steps', 'Wall (s)');
fprintf('%s\n', repmat('-', 1, 55));
for hi = 1:length(h_values)
    for mi = 1:length(methods)
        r = results(hi, mi);
        fprintf('%-8s %-8.1f %-12.3e %-12d %-10.4f\n', ...
                r.method, r.h, r.final_err, r.neg_count, r.walltime);
    end
    fprintf('\n');
end

%% ---- Local functions ----
function dx = mutualism_rhs(x, p)
    P = x(1); A = x(2);
    dP = p.rP*P*(1 - P/p.KP) + p.aP*p.Omega*A/(p.hP + A)*P - p.muP*P;
    dA = p.rA*A*(1 - A/p.KA) + p.aA*p.Omega*P/(p.hA + P)*A - p.muA*A;
    dx = [dP; dA];
end

function Fx = mutualism_F(x, p)
    P = x(1); A = x(2);
    Fx = [p.rP*P + p.aP*p.Omega*A/(p.hP + A)*P;
          p.rA*A + p.aA*p.Omega*P/(p.hA + P)*A];
end

function Gx = mutualism_G(x, p)
    P = x(1); A = x(2);
    Gx = [p.rP*P/p.KP + p.muP;
          p.rA*A/p.KA + p.muA];
end
