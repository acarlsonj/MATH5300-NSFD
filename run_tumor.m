%% RUN_TUMOR.M
%  PD-L1 tumor-immune NSFD comparison.
%  Compares Forward Euler, classical RK4, and NSFD against ode15s reference
%  on the four-compartment model from the UTA RTG Mathematical Oncology project.

clear; clc; close all;

%% Parameters (representative midpoints from RTG draft)
%  muH/muF/muX set to 5e-6 (geometric midpoint of the literature range)
%  to place the system in the immune-surveillance regime.
p.lamC = 0.05;
p.lamF = 0.10;  p.lamX = 0.10;
p.deltaX = 0.01; p.deltaF = 5e-3;
p.b = 100;
p.a = 0.30; p.k = 1.40;
p.muH = 5e-6; p.muF = 5e-6; p.muX = 5e-6;
p.gamma = 0.20;
p.CE = 2.8e6;
p.KH = 1e6;

%% Initial conditions: [C; H; X; F]
x0 = [1e5; 100; 10; 5];

%% Right-hand side and NSFD splitting
f  = @(x) tumor_rhs(x, p);
F  = @(x) tumor_F(x, p);
G  = @(x) tumor_G(x, p);

%% Denominator function
q = p.lamC;
phi = @(h) (1 - exp(-q*h)) / q;

%% Simulation settings
t0 = 0; tf = 200;
h_values = [0.01, 0.1, 0.5, 1.0];

% ----- Reference via ode15s (stiff solver) -----
% Floor absolute tolerance at 1 cell/uL so the solver stops tracking
% biologically meaningless near-zero values; also cap max step.
abstol = [1e3; 1e-2; 1e-2; 1e-2];   % scaled per-variable: C is large, T cells are small
opts = odeset('RelTol', 1e-6, 'AbsTol', abstol, ...
              'NonNegative', [1 2 3 4], 'MaxStep', 5.0);
[t_ref, X_ref] = ode15s(@(t,x) f(x), [t0 tf], x0, opts);

% Clip reference at the absolute tolerance floor for plotting purposes
X_ref_plot = X_ref;
for col = 1:4
    X_ref_plot(:, col) = max(X_ref_plot(:, col), abstol(col));
end

%% Run all three schemes
results = struct();
methods = {'Euler', 'RK4', 'NSFD'};

for hi = 1:length(h_values)
    h = h_values(hi);
    N = ceil((tf - t0) / h);
    tgrid = t0 + (0:N)' * h;

    for mi = 1:length(methods)
        method = methods{mi};
        X = zeros(N+1, 4); X(1, :) = x0';
        failed = false; fail_step = N;

        tic;
        for k = 1:N
            try
                switch method
                    case 'Euler'
                        X(k+1, :) = euler_step(f, X(k, :)', h)';
                    case 'RK4'
                        X(k+1, :) = rk4_step(f, X(k, :)', h)';
                    case 'NSFD'
                        X(k+1, :) = nsfd_step(F, G, X(k, :)', phi(h))';
                end
                if any(~isreal(X(k+1, :))) || any(isnan(X(k+1, :))) || any(isinf(X(k+1, :)))
                    failed = true; fail_step = k;
                    X(k+1:end, :) = NaN;
                    break
                end
            catch
                failed = true; fail_step = k;
                X(k+1:end, :) = NaN;
                break
            end
        end
        walltime = toc;

        real_mask = all(isfinite(X) & imag(X)==0, 2);
        neg_count = sum(any(real(X(real_mask,:)) < 0, 2));

        results(hi, mi).method = method;
        results(hi, mi).h = h;
        results(hi, mi).t = tgrid;
        results(hi, mi).X = X;
        results(hi, mi).walltime = walltime;
        results(hi, mi).failed = failed;
        results(hi, mi).fail_step = fail_step;
        results(hi, mi).neg_count = neg_count;
    end
end

%% Figure: cancer cell time series
fig1 = figure('Position',[100 100 900 600]);
hi_plot = find(h_values == 0.5);

subplot(2,2,1); hold on; grid on;
plot(t_ref, X_ref(:,1), 'k-', 'LineWidth', 2.0, 'DisplayName', 'ode15s reference');
for mi = 1:3
    r = results(hi_plot, mi);
    valid = all(isfinite(r.X) & imag(r.X)==0, 2);
    styles = {'r--', 'b:', 'g-'};
    plot(r.t(valid), real(r.X(valid,1)), styles{mi}, 'LineWidth', 1.3, 'DisplayName', r.method);
end
xlabel('Time (days)'); ylabel('Cancer cells C');
title(sprintf('Cancer cells, h = %g days', h_values(hi_plot)));
legend('Location','best');

subplot(2,2,2); hold on; grid on;
plot(t_ref, X_ref(:,2), 'k-', 'LineWidth', 2.0);
for mi = 1:3
    r = results(hi_plot, mi);
    valid = all(isfinite(r.X) & imag(r.X)==0, 2);
    styles = {'r--', 'b:', 'g-'};
    plot(r.t(valid), real(r.X(valid,2)), styles{mi}, 'LineWidth', 1.3);
end
xlabel('Time (days)'); ylabel('CD4^+ T cells H');
title(sprintf('Helper T cells, h = %g days', h_values(hi_plot)));

subplot(2,2,3); hold on; grid on;
plot(t_ref, X_ref(:,3), 'k-', 'LineWidth', 2.0);
for mi = 1:3
    r = results(hi_plot, mi);
    valid = all(isfinite(r.X) & imag(r.X)==0, 2);
    styles = {'r--', 'b:', 'g-'};
    plot(r.t(valid), real(r.X(valid,3)), styles{mi}, 'LineWidth', 1.3);
end
xlabel('Time (days)'); ylabel('Exhausted CD8^+ X');
title(sprintf('Exhausted T cells, h = %g days', h_values(hi_plot)));

subplot(2,2,4); hold on; grid on;
plot(t_ref, X_ref(:,4), 'k-', 'LineWidth', 2.0);
for mi = 1:3
    r = results(hi_plot, mi);
    valid = all(isfinite(r.X) & imag(r.X)==0, 2);
    styles = {'r--', 'b:', 'g-'};
    plot(r.t(valid), real(r.X(valid,4)), styles{mi}, 'LineWidth', 1.3);
end
xlabel('Time (days)'); ylabel('Effector CD8^+ F');
title(sprintf('Effector T cells, h = %g days', h_values(hi_plot)));

saveas(fig1, 'fig_tumor_trajectories.png');

%% Table
fprintf('\n%-8s %-8s %-10s %-12s %-10s\n', 'Method', 'h', 'Fail step', 'Neg steps', 'Wall (s)');
fprintf('%s\n', repmat('-', 1, 55));
for hi = 1:length(h_values)
    for mi = 1:length(methods)
        r = results(hi, mi);
        fs = '';
        if r.failed; fs = sprintf('fail @%d', r.fail_step); else; fs = 'complete'; end
        fprintf('%-8s %-8.3g %-10s %-12d %-10.4f\n', r.method, r.h, fs, r.neg_count, r.walltime);
    end
    fprintf('\n');
end

%% ---- Local functions ----
function dx = tumor_rhs(x, p)
    C = x(1); H = x(2); X = x(3); F = x(4);
    D = D_term(C, X, p);
    M = H / (p.KH + H);
    dC = p.lamC*C*(1 - C/p.CE) - D*C - p.deltaF*F*C;
    dH = p.b*(D + p.deltaF*F)*C - p.muH*C*H;
    dX = p.lamX*M*X - p.muX*C*X + p.gamma*F;
    dF = p.lamF*M*F - p.muF*C*F - p.gamma*F;
    dx = [dC; dH; dX; dF];
end

function Fx = tumor_F(x, p)
    C = x(1); H = x(2); X = x(3); F = x(4);
    D = D_term(C, X, p);
    M = H / (p.KH + H);
    Fx = [p.lamC*C;
          p.b*(D + p.deltaF*F)*C;
          p.lamX*M*X + p.gamma*F;
          p.lamF*M*F];
end

function Gx = tumor_G(x, p)
    C = x(1); H = x(2); X = x(3); F = x(4);
    D = D_term(C, X, p);
    Gx = [p.lamC*C/p.CE + D + p.deltaF*F;
          p.muH*C;
          p.muX*C;
          p.muF*C + p.gamma];
end

function D = D_term(C, X, p)
% Safe evaluation of the DePillis-Radunskaya fractional death term
% delta_X * (X/C)^k / (a + (X/C)^k).  Floors C and X away from zero
% and caps the ratio to prevent overflow in the power.  Matches the
% formulation used in run_all.m so both drivers agree.
    C_safe = max(C, 1e-12);
    X_safe = max(X, 0);
    ratio = min(X_safe / C_safe, 1e6);
    r = ratio^p.k;
    D = p.deltaX * r / (p.a + r);
end
