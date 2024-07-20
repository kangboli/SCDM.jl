using SCDM, LinearAlgebra, Random, Profile

src_dir = "./test/scdm_dataset/BaTiO3/aiida.save"
f, _ = load_problem(src_dir)
grad_f = make_grad_f(f)
@time cg(random_gauge(f), f, grad_f, qr_retract!);

