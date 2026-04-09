using ONNXRunTime
using PythonCall







function triage_score_from_output(y)
    if ndims(y) == 2 && size(y, 2) == 1
        logit = Float32(y[1, 1])
        prob = softmax(logit)
        return logit, prob
    elseif ndims(y) == 1 && length(y) == 1
        logit = Float32(y[1])
        prob = softmax(logit)
        return logit, prob
    else
        error("Expected triage output.shape == [1,1] || [1], but $(size(y))")
    end
end

function main()
    isempty(ARGS) && error("usage: julia --project=julia julia/inference.jl /path/to/BraTS_case_dir")
    case_dir = ARGS[1]

    x, y = infer_case(
        case_dir;
        pixdim=(1.0, 1.0, 1.0),
        roi_size=(128, 128, 128),
    )


    logit, prob = triage_score_from_output(y)
    println("triage logit=$(logit)")
    println("triage prob=$(prob)")
end

abspath(PROGRAM_FILE) == @__FILE__ && main()