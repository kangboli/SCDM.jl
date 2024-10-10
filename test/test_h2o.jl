using SCDM, LinearAlgebra, Random, WTP

src_dir = "./test/scdm_dataset/H2O_tiny"
f, orbital_set = load_problem(joinpath(src_dir, "pwscf.save"))

# Add this line to use Marzari Vanderbilt's objective function
# f = as_mv(f)

grad_f = make_grad_f(f)

# Use a random initial guess
@time u_opt = cg(SCDM.random_gauge(f), f, grad_f, qr_retract!)

# Use SCDM "initial guess"
u_scdm = scdm_gauge(orbital_set)
u_scdm_opt = cg(u_scdm, f, grad_f, qr_retract!)

@time cg(identity_gauge(f), f, grad_f, qr_retract!)

# The TDC center and spread

f(u_opt)
peek_centers(f)
peek_spreads(f)

# SCDM center and spread

f(u_scdm)
peek_centers(f)
peek_spreads(f)

# DC center and spread.
gauge_opt = Gauge(grid(orbital_set), f.n_e)
for k in grid(orbital_set)
    gauge_opt[k] = u_opt[:, :, linear_index(k)]
end
new_set = commit_gauge!(set_gauge(orbital_set, gauge_opt))

s = supercell(new_set)
r2, r̂2 = compute_r2(s)
rhos = reciprocal_densities(new_set)
center_spreads = map(rho->center_spread(rho,  r̂2), rhos)
