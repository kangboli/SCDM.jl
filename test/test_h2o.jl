using SCDM, LinearAlgebra, Random

src_dir = "./test/scdm_dataset/H2O_tiny"
f, orbital_set = load_problem(joinpath(src_dir, "pwscf.save"))
grad_f = make_grad_f(f)

# Three different initial guesses.
@time cg(random_gauge(f), f, grad_f, qr_retract!)

u_scdm = scdm_gauge(orbital_set)
@time cg(u_scdm, f, grad_f, qr_retract!)

@time cg(identity_gauge(f), f, grad_f, qr_retract!)
