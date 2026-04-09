using Pkg
using ONNXRuntime


using CUDA
using cuDNN
using NIfTI

using Statistics
using ImageFiltering

using PythonCall



println("CUDA:", CUDA.functional())

if CUDA.functional()
    println("GPU", CUDA.name(CUDA.device()))
end

const MODEL_PATH = joinpath(@__DIR__, "..", "models", "swinunetr_brats.onnx")




const PROVIDERS = [
    ("CUDAExecutionProvider", (
        device_id = 0,
        cudnn_conv_algo_search = "HEURISTIC",
        gpu_mem_limit = 0,
        do_copy_in_default_stream = true,
        arena_extend_strategy = "kNextPowerOfTwo"
    )),
    "CPUExecutionProvider"
]

const SESSION = ONNXRunTime.load_inference(
    MODEL_PATH,
    execution_providers = PROVIDERS,
    intra_op_num_threads = 0,
    inter_op_num_threads = 0
)

println("ONNX Session loaded with CUDAExecutionProvider")

const INPUT_NAME = "input"
const OUTPUT_NAME = "output"

const PY_MODULE_NAME = "preprocessing_brats"
const PY_FUNCTION_NAME = "preprocessing_case_dir"

function py_path!(path::AbstractString = pwd())
    sys = pyimport("sys")
    paths = pyconvert(Vector{String}, sys.path)
    if !(path in paths)
        sys.path.append(path)
    end
    return nothing
end

function load_py_preprocessing()
    py_dir = normpath(joinpath(@__DIR__, "..", "python"))
    py_path!()
    pymod = pyimport(PY_MODULE_NAME)
    return getproperty(pymod, Symbol(PY_FUNCTION_NAME))
end

function preprocess_case(case_dir::AbstractString; pixdim=(1.0, 1.0, 1.0), roi_size=nothing)
    fn = load_py_preprocessing()

    x_py = if roi_size === nothing
        fn(case_dir; pixdim = pixdim)
    else
        fn(case_dir; pixdim = pixdim, roi_size = roi_size)
    end

    x = pyconvert(Array{Float32, 5}, x_py)

    @assert ndims(x) == 5 "Expected 5D tensor, got ndims=$(ndims(x))"
    @assert size(x, 1) == 1 "Expected batch size 1, got size(x,1)=$(size(x,1))"
    @assert size(x, 2) == 4 "Expected 4 modalities in channel dim, got size(x,2)=$(size(x,2))"
    @assert all(isfinite, x) "Preprocessed input contains NaN/Inf"

    return x

end


function pick_output(outputs::AbstractDict)
    if OUTPUT_NAME !== "output"
        @assert haskey(outputs, OUTPUT_NAME) "OUTPUT_NAME=$(OUTPUT_NAME) not found"
        return outputs[OUTPUT_NAME]
    end
    @assert length(outputs) == 1 "Multiple outputs found: $(collect(keys(outputs)))"
    return only(values(outputs))
end

function channel_argmax_mask(y::AbstractArray{T, S}) where {T<:Real}
    n, c, d, l, w = size(y)
    @assert n ==1
    @assert c >= 2

    mask = Array{UInt8}(undef, d , l, w)

    @inbounds for zz in 1:d, yy in 1:l, xx in 1:w
        best_c = 1
        best_val = y[1, 1, zz, yy, xx]
        for cc in 2:c
            val = y[cc, zz, yy, xx]
            if val > best_val
                best_val = val
                best_c =cc
            end
        end
        mask[zz, yy, xx] = UInt8(best_c - 1)
    end

    return mask
end

function infer_case(case_path::AbstractString; roi_size=nothing)
    x = preprocess_case(case_path; roi_size=roi_size)

    println("input shape = ", size(x))
    outputs = SESSION(Dict(INPUT_NAME => x))
    println("output shape = ", collect(keys(outputs)))

    y = pick_output(outputs)
    println("output size = ", size(y))
    @assert all(isinfinite, y) "Output contains NaN/Inf"

    mask = nothing
    if ndims(y) == S && size(y, 2) >= 2
        mask = channel_argmax_mask(y)
    end

    return x, y, mask
end


case_path = "../src/SplitFlowODESolver/data/dataset/"

@time begin
    x, y, mask = infer_case(case_path;roi_size=(1.0, 1.0, 1.0))
end

println("Inference successful. Prediction size", size(mask))

niwrite("prediction.nii.gz", NIVolume(mask))

