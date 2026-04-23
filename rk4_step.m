function x_next = rk4_step(f, x, h)
% RK4_STEP  One step of classical fourth-order Runge-Kutta for dx/dt = f(x).
    k1 = f(x);
    k2 = f(x + 0.5*h*k1);
    k3 = f(x + 0.5*h*k2);
    k4 = f(x + h*k3);
    x_next = x + (h/6) * (k1 + 2*k2 + 2*k3 + k4);
end
