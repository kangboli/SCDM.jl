using SCDM, LinearAlgebra, Random

src_dir = "./test/scdm_dataset/Si"
f, _ = load_problem(joinpath(src_dir, "aiida.save"))
grad_f = make_grad_f(f)
@time cg(random_gauge(f), f, grad_f, qr_retract!);
