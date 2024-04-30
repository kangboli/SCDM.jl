using SCDM
using LinearAlgebra
using ManifoldsBase
using Manopt
const ideg = 4

struct GaugeManifold <: AbstractManifold{ℂ}
    n_electron::Int64
    n_kpoints::Int64
    n_outer::Int64
    grad_work::Matrix{ComplexF64}
    U_work::Vector{ComplexF64}
end

function GaugeManifold(n_electron::Int64, n_kpoints::Int64, n_outer::Int64)
    size_u_work = 4 * Ne * Ne + ideg + 1
    return GaugeManifold(
        n_electron,
        n_kpoints,
        n_outer,
        zeros(ComplexF64, Ne, Ne),
        zeros(ComplexF64, size_u_work)
    )
end

get_n_kpoints(M::GaugeManifold) = M.n_kpoints
get_n_electron(M::GaugeManifold) = M.n_electron
get_n_outer(M::GaugeManifold) = M.n_outer

si_src_dir = "./test/scdm_dataset/Si"
MTensor, UTensor, w_list, kplusb, Nk, Nb, Ne = load_problem(si_src_dir)

grad_work = zeros(ComplexF64, Ne, Ne)

RTensor = zeros(ComplexF64, size(MTensor))

function ManifoldsBase.vector_transport_to(::GaugeManifold, p::Array{ComplexF64, 3}, X::Array{ComplexF64, 3}, q::Array{ComplexF64, 3}, m::ManifoldsBase.AbstractVectorTransportMethod
)
    return X
end

function ManifoldsBase.vector_transport_to!(::GaugeManifold, Y::Array{ComplexF64, 3}, p::Array{ComplexF64, 3}, X::Array{ComplexF64, 3}, q::Array{ComplexF64, 3}, m::ManifoldsBase.AbstractVectorTransportMethod
)
    Y[:, :, :] = X[:, :, :]
end

function ManifoldsBase.inner(::GaugeManifold, ::Array{ComplexF64, 3}, X::Array{ComplexF64, 3}, Y::Array{ComplexF64, 3})
    return dot(vec(X), vec(Y))
end
function ManifoldsBase.retract_project!(M::GaugeManifold, UMutate::Array{ComplexF64, 3}, UTensor::Array{ComplexF64,3},
    grad_omega::Array{ComplexF64,3}, step_size::Number)
    LinearAlgebra.BLAS.scal!(ComplexF64(step_size), grad_omega)
    SCDM.project!(UTensor, grad_omega, M.grad_work, get_n_kpoints(M), get_n_electron(M))
    SCDM.retract!(UMutate, grad_omega, M.U_work, get_n_kpoints(M), get_n_electron(M), ideg, length(M.U_work))
    return UMutate
end

function ManifoldsBase.zero_vector(M::GaugeManifold, ::Array{ComplexF64, 3})
    return zeros(ComplexF64, M.n_electron, M.n_electron, M.n_kpoints)
end



M = GaugeManifold(Ne, Nk, Ne)

UTensor = zeros(ComplexF64, Ne, Ne, Nk)
for k in 1:Nk
    UTensor[:, :, k] = diagm(ones(Ne))
end
grad_omega = zeros(ComplexF64, Ne, Ne, Nk)
UMutate = deepcopy(UTensor)
rho_hat = zeros(ComplexF64, Ne, Nb)

grad_res = 1e6
step_size = -0.05
n_iter = 0
Nj = Ne

struct GaugeOptProblem
    STensor::Array{ComplexF64, 4}
    R::Array{ComplexF64, 4}
    w_list::Vector{Float64}
    kplusb::Matrix{Int64}
    rho_hat::Matrix{ComplexF64}
    n_b::Int64
end

function make_f(gop::GaugeOptProblem) 
    function f(M::GaugeManifold, UTensor::Array{ComplexF64, 3})
        omega = zeros(Float64, 1)
        f_oracle!(gop.STensor, gop.R, UTensor, gop.w_list, gop.kplusb, gop.rho_hat, M.n_kpoints, gop.n_b, M.n_electron, M.n_outer, omega)
        return omega
    end
    return f
end

function make_grad_f(gop::GaugeOptProblem) 
    function grad_f!(M::GaugeManifold, grad_omega::Array{ComplexF64, 3}, UTensor::Array{ComplexF64, 3})
        omega = zeros(Float64, 1)
        f_oracle!(gop.STensor, gop.R, UTensor, gop.w_list, gop.kplusb, gop.rho_hat, M.n_kpoints, gop.n_b, M.n_electron, M.n_outer, omega)
        grad_f_oracle!(gop.R, gop.w_list, gop.rho_hat, M.n_kpoints, gop.n_b, M.n_electron, M.n_outer, grad_omega)
        return grad_omega
    end
    return grad_f!
end

problem = GaugeOptProblem(MTensor, RTensor, w_list, kplusb, rho_hat, Nb)
ManifoldsBase.retraction_method = ProjectionRetraction()

f = make_f(problem)
grad_f! = make_grad_f(problem)
U_opt = conjugate_gradient_descent(M, f, grad_f!, UTensor; 
    evaluation=InplaceEvaluation(),
    retraction_method = ProjectionRetraction(),
    stepsize = DecreasingStepsize(M; length=1.0), 
    stopping_criterion = StopAfterIteration(1000), 
)

f(M, U_opt)


while grad_res > Nk * 1e-6

    omega = f(M, UTensor)
    grad_f!(M, grad_omega, UTensor)
    #= f_oracle!(MTensor, RTensor, UTensor, w_list, kplusb, rho_hat, Nk, Nb, Ne, Nj, omega)
    grad_f_oracle!(RTensor, w_list, rho_hat, Nk, Nb, Ne, Nj, grad_omega) =#
    UTensor[:,:,:] = retract_project!(M, UMutate, UTensor, grad_omega, ComplexF64(step_size))
    grad_res = norm(abs.(grad_omega))^2
    println(grad_res)
    println("$(n_iter): $(first(omega))")
    n_iter += 1
end

