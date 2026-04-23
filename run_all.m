%% RUN_ALL.M
%  Unified MATLAB driver for the NSFD comparison paper.
%  Runs both case studies (plant-pollinator mutualism and PD-L1 tumor-immune)
%  end-to-end and generates all figures used in the Results section.
%
%  Requires in the same folder: euler_step.m, rk4_step.m, nsfd_step.m
%
%  Outputs (as PNG files in current directory):
%    fig_mutualism_paper.png
%    fig_mutualism_trajectories.png
%    fig_mutualism_error.png
%    fig_tumor_slides.png
%    fig_tumor_trajectories.png
%    fig_tumor_failure.png
%
%  Author: Austin Carlson, MATH 5300 Final Project

clear; clc; close all;

fprintf('============================================================\n');
fprintf('NSFD Comparison: Mutualism + Tumor-Immune Case Studies\n');
fprintf('============================================================\n');

run_mutualism_case();
run_tumor_case();

fprintf('\nAll done. Figures saved to current directory.\n');

%% ================================================================
%  CASE STUDY 1: PLANT-POLLINATOR MUTUALISM
%  ================================================================
function run_mutualism_case()

fprintf('\n[Case 1/2] Plant-pollinator mutualism\n');
fprintf('%s\n', repmat('-', 1, 60));

% ----- Parameters (capstone Appendix B, Peponapis-Cucurbita) -----
p.rP = 0.020;  p.rA = 0.016;
p.KP = 200;    p.KA = 150;
p.muP = 0.0216; p.muA = 0.01696;
p.aP = 0.050;  p.aA = 0.048;
p.hP = 12;     p.hA = 12;
p.Omega = 0.141;

% ----- Right-hand side and NSFD splitting -----
f_rhs = @(x) mutualism_rhs(x, p);
F_fun = @(x) mutualism_F(x, p);
G_fun = @(x) mutualism_G(x, p);

% Denominator function tied to dominant linear rate
q = max(p.rP, p.rA) + max(p.muP, p.muA);
phi = @(h) (1 - exp(-q*h)) / q;

% ----- Simulation settings -----
t0 = 0; tf = 1000;
x0 = [150; 120];
h_values = [1, 10, 50, 100, 200];
methods = {'Euler', 'RK4', 'NSFD'};

% ----- Reference via ode45 -----
opts = odeset('RelTol',1e-10,'AbsTol',1e-12,'NonNegative',[1 2]);
[t_ref, X_ref] = ode45(@(t,x) f_rhs(x), [t0 tf], x0, opts);
x_eq = X_ref(end, :);
fprintf('Reference coexistence equilibrium: P* = %.3f, A* = %.3f\n', x_eq(1), x_eq(2));

% ----- Run all three schemes at each step size -----
results = cell(length(h_values), length(methods));
for hi = 1:length(h_values)
    h = h_values(hi);
    for mi = 1:length(methods)
        results{hi, mi} = integrate_scheme(methods{mi}, x0, h, t0, tf, ...
                                           f_rhs, F_fun, G_fun, phi);
        results{hi, mi}.final_err = norm(results{hi, mi}.X(end, :) - x_eq);
    end
end

% ----- Results table -----
fprintf('\n%-8s %-8s %-14s %-12s %-10s\n', 'Method', 'h', 'Final err', 'Neg steps', 'Wall (s)');
fprintf('%s\n', repmat('-', 1, 55));
for hi = 1:length(h_values)
    for mi = 1:length(methods)
        r = results{hi, mi};
        if isnan(r.final_err) || ~isfinite(r.final_err)
            err_str = 'diverged';
        else
            err_str = sprintf('%.3e', r.final_err);
        end
        fprintf('%-8s %-8.1f %-14s %-12d %-10.4f\n', ...
                methods{mi}, h_values(hi), err_str, r.neg_count, r.walltime);
    end
    fprintf('\n');
end

% ----- Figure 1: trajectories at h = 50 and h = 100 -----
fig1 = figure('Position',[100 100 900 700],'Color','w');
colors = struct('Euler',[0.85 0.10 0.10], 'RK4',[0.10 0.30 0.85], 'NSFD',[0.10 0.65 0.20]);
styles = struct('Euler','--', 'RK4',':', 'NSFD','-');
var_names = {'Plant P', 'Pollinator A'};
h_plot_list = [50, 100];

for row = 1:2
    h_plot = h_plot_list(row);
    hi = find(h_values == h_plot);
    for col = 1:2
        subplot(2, 2, (row-1)*2 + col); hold on; grid on;

        % y-limits from reference with padding
        ymin_ref = min(X_ref(:, col));
        ymax_ref = max(X_ref(:, col));
        y_range = ymax_ref - ymin_ref;
        y_pad = 0.3 * max(y_range, 50);
        y_lo = ymin_ref - y_pad;
        y_hi = ymax_ref + y_pad;

        plot(t_ref, X_ref(:, col), 'k-', 'LineWidth', 2.0, 'DisplayName', 'ode45 reference');

        for mi = 1:length(methods)
            m = methods{mi};
            r = results{hi, mi};
            valid = all(isfinite(r.X), 2);
            in_view = valid & (r.X(:, col) >= y_lo) & (r.X(:, col) <= y_hi);
            if r.failed
                lbl = sprintf('%s (blew up)', m);
            elseif sum(in_view) < sum(valid)
                lbl = sprintf('%s (out of range)', m);
            else
                lbl = m;
            end
            plot(r.t(in_view), r.X(in_view, col), ...
                 'LineStyle', styles.(m), 'Color', colors.(m), ...
                 'Marker', 'o', 'MarkerSize', 4, 'LineWidth', 1.5, ...
                 'DisplayName', lbl);
        end

        ylim([y_lo, y_hi]); xlim([0 tf]);
        xlabel('Time (days)');
        ylabel(var_names{col});
        title(sprintf('%s population, h = %d days', var_names{col}, h_plot));
        if row == 1 && col == 1
            legend('Location','best','FontSize',9);
        end
    end
end
saveas(fig1, 'fig_mutualism_trajectories.png');
fprintf('Saved fig_mutualism_trajectories.png\n');

% ----- Figure 2: error vs step size -----
fig2 = figure('Position',[100 100 650 400],'Color','w');
hold on; grid on;
for mi = 1:length(methods)
    m = methods{mi};
    errs = zeros(size(h_values));
    for hi = 1:length(h_values)
        errs(hi) = results{hi, mi}.final_err;
    end
    errs(~isfinite(errs)) = NaN;
    loglog(h_values, errs, 'o-', 'Color', colors.(m), ...
           'LineWidth', 1.5, 'MarkerSize', 8, 'DisplayName', m);
end
set(gca, 'XScale','log', 'YScale','log');
xlabel('Step size h (days)');
ylabel('Final error ||x_N - x*||');
title('Mutualism: final error vs step size');
legend('Location','best');
saveas(fig2, 'fig_mutualism_error.png');
fprintf('Saved fig_mutualism_error.png\n');

% ----- Figure 3: paper figure at h = 100 -----
%   Wide 1x2 panel for the term paper.  Reuses the h=100 trajectories
%   from the results loop above so we don't re-integrate anything.
hi_paper = find(h_values == 100);
paper_ylims  = {[-15 170], [-15 140]};
paper_ylabel = {'Plant abundance P', 'Pollinator abundance A'};
paper_title  = {'Plant population, h = 100 d', 'Pollinator population, h = 100 d'};

fig3 = figure('Position',[100 100 1100 360],'Color','w');
for col = 1:2
    subplot(1, 2, col); hold on; grid on; box on;
    plot(t_ref, X_ref(:, col), 'k-', 'LineWidth', 2.0, ...
         'DisplayName', 'ode45 reference');
    for mi = 1:length(methods)
        m = methods{mi};
        r = results{hi_paper, mi};
        plot(r.t, r.X(:, col), ...
             'LineStyle', styles.(m), 'Color', colors.(m), ...
             'Marker', 'o', 'MarkerSize', 5, 'LineWidth', 1.5, ...
             'DisplayName', m);
    end
    plot([t0 tf], [0 0], 'k:', 'LineWidth', 0.8);   % zero reference
    xlim([t0 tf]); ylim(paper_ylims{col});
    xlabel('Time (days)'); ylabel(paper_ylabel{col});
    title(paper_title{col});
    if col == 1
        legend('Location','northeast','FontSize',8);
    end
end
saveas(fig3, 'fig_mutualism_paper.png');
fprintf('Saved fig_mutualism_paper.png\n');

end % run_mutualism_case


%% ================================================================
%  CASE STUDY 2: PD-L1 TUMOR-IMMUNE DYNAMICS
%  ================================================================
function run_tumor_case()

fprintf('\n[Case 2/2] PD-L1 tumor-immune dynamics\n');
fprintf('%s\n', repmat('-', 1, 60));

% ----- Parameters (RTG draft, midpoints of literature ranges) -----
% T-cell death rates set to 5e-6 (geometric midpoint of [1e-7, 1e-4]) to
% place the system in the immune-surveillance regime; see paper Table 2.
p.lamC = 0.05;
p.lamF = 0.10;  p.lamX = 0.10;
p.deltaX = 0.01; p.deltaF = 5e-3;
p.b = 100;
p.a = 0.30; p.k = 1.40;
p.muH = 5e-6; p.muF = 5e-6; p.muX = 5e-6;
p.gamma = 0.20;
p.CE = 2.8e6;
p.KH = 1e6;

% ----- RHS and NSFD splitting -----
f_rhs = @(x) tumor_rhs(x, p);
F_fun = @(x) tumor_F(x, p);
G_fun = @(x) tumor_G(x, p);

q = p.lamC;
phi = @(h) (1 - exp(-q*h)) / q;

% ----- Simulation settings -----
t0 = 0; tf = 200;
x0 = [1e5; 100; 10; 5];
h_values = [0.01, 0.1, 0.5, 1.0];
methods = {'Euler', 'RK4', 'NSFD'};

% ----- Reference via ode15s (stiff solver) -----
% Floor AbsTol at one cell/uL per variable so the solver stops tracking
% biologically meaningless near-zero values; also cap the max step.
abstol = [1e3; 1e-2; 1e-2; 1e-2];   % C is large; T cells are small
opts = odeset('RelTol', 1e-6, 'AbsTol', abstol, ...
              'NonNegative', [1 2 3 4], 'MaxStep', 5.0);
[t_ref, X_ref] = ode15s(@(t,x) f_rhs(x), [t0 tf], x0, opts);
fprintf('Reference final state: C=%.3e, H=%.3e, X=%.3e, F=%.3e\n', ...
        X_ref(end,1), X_ref(end,2), X_ref(end,3), X_ref(end,4));

% ----- Run all three schemes -----
results = cell(length(h_values), length(methods));
for hi = 1:length(h_values)
    h = h_values(hi);
    for mi = 1:length(methods)
        results{hi, mi} = integrate_scheme(methods{mi}, x0, h, t0, tf, ...
                                           f_rhs, F_fun, G_fun, phi);
    end
end

% ----- Results table -----
fprintf('\n%-8s %-8s %-22s %-12s %-10s\n', 'Method', 'h', 'Status', 'Neg steps', 'Wall (s)');
fprintf('%s\n', repmat('-', 1, 62));
for hi = 1:length(h_values)
    for mi = 1:length(methods)
        r = results{hi, mi};
        if r.failed
            status = sprintf('failed at step %d', r.fail_step);
        else
            status = 'complete';
        end
        fprintf('%-8s %-8.3g %-22s %-12d %-10.4f\n', ...
                methods{mi}, h_values(hi), status, r.neg_count, r.walltime);
    end
    fprintf('\n');
end

% ----- Figure 3: 4-panel time series at h = 0.5 -----
fig3 = figure('Position',[100 100 900 650],'Color','w');
colors = struct('Euler',[0.40 0.40 0.40], 'RK4',[0.00 0.392 0.690], 'NSFD',[0.961 0.502 0.149]);
styles = struct('Euler','--', 'RK4',':', 'NSFD','-');
labels = {'Cancer cells C', 'CD4^+ T cells H', 'Exhausted CD8^+ X', 'Effector CD8^+ F'};
h_plot = 0.5;
hi_plot = find(h_values == h_plot);

% Per-panel y-limits: lower bound is the physical floor, upper bound from data.
% Log scale only for panels with wide dynamic range.
y_floors = abstol';                 % [1e3 1e-2 1e-2 1e-2]
use_log  = [false, true, true, true];

for vi = 1:4
    subplot(2, 2, vi); hold on; grid on; box on;

    ref_vals = max(X_ref(:, vi), y_floors(vi));
    ymax_ref = max(ref_vals);
    ymin_ref = y_floors(vi);

    if use_log(vi)
        set(gca, 'YScale', 'log');
        ylim([ymin_ref, ymax_ref * 3]);
    else
        ylim([0, ymax_ref * 1.15]);
    end

    % Reference
    plot(t_ref, ref_vals, 'k-', 'LineWidth', 2.0, 'DisplayName', 'ode15s reference');

    % Three schemes
    for mi = 1:length(methods)
        m = methods{mi};
        r = results{hi_plot, mi};
        valid = all(isfinite(r.X) & imag(r.X)==0, 2);
        if any(valid)
            vals = real(r.X(valid, vi));
            if use_log(vi)
                vals = max(vals, y_floors(vi));
            end
            if r.failed
                lbl = sprintf('%s (fail @ step %d)', m, r.fail_step);
            else
                lbl = m;
            end
            plot(r.t(valid), vals, ...
                 'LineStyle', styles.(m), 'Color', colors.(m), ...
                 'LineWidth', 1.5, 'DisplayName', lbl);
        end
    end

    xlim([t0 100]);
    xlabel('Time (days)');
    ylabel(labels{vi});
    title(sprintf('%s, h = %g days', labels{vi}, h_plot));
    if vi == 1
        legend('Location','best','FontSize',9);
    end
end
saveas(fig3, 'fig_tumor_trajectories.png');
fprintf('Saved fig_tumor_trajectories.png\n');

% ----- Figure 4: Helper T-cell surveillance at h = 0.5 -----
%   The CD4+ compartment mobilizes three orders of magnitude before
%   tumor escape drives it to zero.  Near the extinction boundary,
%   explicit methods overshoot into negative H and diverge.
fig4 = figure('Position',[100 100 800 400],'Color','w');
hold on; grid on; box on;
set(gca, 'YScale', 'log');

hi_fail = find(h_values == 0.5);

% Floor at 1e-4 cells/uL for readability
H_floor = 1e-4;
ref_H = max(X_ref(:, 2), H_floor);
plot(t_ref, ref_H, 'k-', 'LineWidth', 2.0, 'DisplayName', 'ode15s reference');

for mi = 1:length(methods)
    m = methods{mi};
    r = results{hi_fail, mi};
    valid = all(isfinite(r.X) & imag(r.X)==0, 2);
    if any(valid)
        vals = max(abs(real(r.X(valid, 2))), H_floor);
        if r.failed
            lbl = sprintf('%s (diverges @ step %d)', m, r.fail_step);
        else
            lbl = m;
        end
        plot(r.t(valid), vals, ...
             'LineStyle', styles.(m), 'Color', colors.(m), ...
             'LineWidth', 1.8, 'DisplayName', lbl);
    end
end

xlabel('Time (days)');
ylabel('CD4^+ T cells H (cells/\muL)');
title('Helper T-cell population at h = 0.5 days (log scale)');
% Cap y-axis; explicit methods overflow to 1e100+.
ylim([H_floor, 1e7]);
xlim([0 150]);
legend('Location','northeast');
saveas(fig4, 'fig_tumor_failure.png');
fprintf('Saved fig_tumor_failure.png\n');

% ----- Figure 5: 2-panel slide version at h = 0.5 -----
%   Wide 1100x360 layout for the 16:9 presentation.  Cancer cells
%   (linear) on the left, helper T cells (log) on the right.
fig5 = figure('Position',[100 100 1100 360],'Color','w');
hi_slides = find(h_values == 0.5);

% Left panel: cancer cells, linear y
subplot(1, 2, 1); hold on; grid on; box on;
plot(t_ref, X_ref(:,1), 'k-', 'LineWidth', 2.0, ...
     'DisplayName', 'ode15s reference');
for mi = 1:length(methods)
    m = methods{mi};
    r = results{hi_slides, mi};
    valid = all(isfinite(r.X) & imag(r.X)==0, 2);
    if any(valid)
        % Clip divergent values so the axis stays readable
        vals = min(real(r.X(valid, 1)), 3.5e6);
        if r.failed
            lbl = sprintf('%s (diverged)', m);
        else
            lbl = m;
        end
        plot(r.t(valid), vals, ...
             'LineStyle', styles.(m), 'Color', colors.(m), ...
             'LineWidth', 1.8, 'DisplayName', lbl);
    end
end
xlim([0 tf]); ylim([0, 3.2e6]);
xlabel('Time (days)'); ylabel('Cancer cells C');
title(sprintf('Cancer cells, h = %g d', h_values(hi_slides)));
legend('Location','southeast','FontSize',8);

% Right panel: helper H, log y
subplot(1, 2, 2); hold on; grid on; box on;
set(gca, 'YScale', 'log');
H_floor = 1e-4;
ref_H = max(X_ref(:,2), H_floor);
plot(t_ref, ref_H, 'k-', 'LineWidth', 2.0);
for mi = 1:length(methods)
    m = methods{mi};
    r = results{hi_slides, mi};
    valid = all(isfinite(r.X) & imag(r.X)==0, 2);
    if any(valid)
        vals = max(abs(real(r.X(valid, 2))), H_floor);
        vals = min(vals, 1e10);
        plot(r.t(valid), vals, ...
             'LineStyle', styles.(m), 'Color', colors.(m), ...
             'LineWidth', 1.8);
    end
end
xlim([0 tf]); ylim([H_floor, 1e7]);
xlabel('Time (days)'); ylabel('CD4^+ helper H (cells/\muL, log)');
title(sprintf('Helper T cells, h = %g d', h_values(hi_slides)));

saveas(fig5, 'fig_tumor_slides.png');
fprintf('Saved fig_tumor_slides.png\n');

end % run_tumor_case


%% ================================================================
%  GENERIC INTEGRATION DRIVER
%  ================================================================
function r = integrate_scheme(method, x0, h, t0, tf, f_rhs, F_fun, G_fun, phi)
% INTEGRATE_SCHEME  Run one of {Euler, RK4, NSFD} on the given IVP.
%   Returns a struct r with fields:
%     t          - time vector (N+1 x 1)
%     X          - state trajectory (N+1 x d)
%     walltime   - wall-clock time (seconds)
%     failed     - true if integration produced NaN/Inf/complex values
%     fail_step  - step index at which failure occurred (N if success)
%     neg_count  - number of steps at which any state was negative

    N = ceil((tf - t0) / h);
    tgrid = t0 + (0:N)' * h;
    d = length(x0);
    X = zeros(N+1, d);
    X(1, :) = x0';

    failed = false;
    fail_step = N;

    tic;
    for k = 1:N
        xk = X(k, :)';
        try
            switch method
                case 'Euler'
                    xnext = euler_step(f_rhs, xk, h);
                case 'RK4'
                    xnext = rk4_step(f_rhs, xk, h);
                case 'NSFD'
                    xnext = nsfd_step(F_fun, G_fun, xk, phi(h));
                otherwise
                    error('Unknown method: %s', method);
            end

            % Detect NaN, Inf, or complex values
            if any(~isfinite(xnext)) || any(imag(xnext) ~= 0)
                failed = true;
                fail_step = k;
                X(k+1:end, :) = NaN;
                break
            end

            X(k+1, :) = real(xnext)';
        catch
            failed = true;
            fail_step = k;
            X(k+1:end, :) = NaN;
            break
        end
    end
    walltime = toc;

    % Count negative-state steps on the valid portion
    valid_mask = all(isfinite(X) & imag(X)==0, 2);
    neg_count = sum(any(real(X(valid_mask, :)) < 0, 2));

    r.t = tgrid;
    r.X = X;
    r.walltime = walltime;
    r.failed = failed;
    r.fail_step = fail_step;
    r.neg_count = neg_count;
end


%% ================================================================
%  MUTUALISM MODEL DEFINITIONS
%  ================================================================
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


%% ================================================================
%  TUMOR-IMMUNE MODEL DEFINITIONS
%  ================================================================
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
% and caps the ratio to prevent overflow in the fractional power.
    C_safe = max(C, 1e-12);
    X_safe = max(X, 0);
    ratio = min(X_safe / C_safe, 1e6);
    r = ratio^p.k;
    D = p.deltaX * r / (p.a + r);
end

