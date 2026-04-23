function x_next = nsfd_step(F, G, x, phi_h)
% NSFD_STEP  One step of the Mickens NSFD scheme for the class dx/dt = F(x) - G(x).*x
%
%   x_next = nsfd_step(F, G, x, phi_h) advances the state vector x by one
%   NSFD step of effective size phi_h. F and G are function handles returning
%   componentwise nonnegative production and per-capita loss rates.
%
%   Update rule: x_i^{n+1} = (x_i^n + phi_h * F_i(x^n)) / (1 + phi_h * G_i(x^n))

    Fx = F(x);
    Gx = G(x);
    x_next = (x + phi_h .* Fx) ./ (1 + phi_h .* Gx);
end
