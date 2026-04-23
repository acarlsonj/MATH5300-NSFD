function x_next = euler_step(f, x, h)
% EULER_STEP  One step of Forward Euler for dx/dt = f(x).
    x_next = x + h * f(x);
end
