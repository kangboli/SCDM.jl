using SCDM
using Test


currrent_dir = pwd()
test_dir = endswith(currrent_dir, "SCDM.jl") ? joinpath(currrent_dir, "test") : currrent_dir
test_1_dir = "$(test_dir)/test_data/test_1"
test_2_dir = "$(test_dir)/test_data/test_2"
test_3_dir = "$(test_dir)/test_data/test_3"
test_4_dir = "$(test_dir)/test_data/test_4"
test_5_dir = "$(test_dir)/test_data/test_5"
test_6_dir = "$(test_dir)/test_data/test_6"

include("$(test_dir)/test_vanilla.jl")
include("$(test_dir)/test_gauge_transform.jl")
include("$(test_dir)/test_w90_center_spread.jl")
include("$(test_dir)/test_truncated_convolution.jl")
include("$(test_dir)/test_ila_gradient_descent.jl")
include("$(test_dir)/test_w90_gradient.jl")
