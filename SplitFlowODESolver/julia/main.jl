include("inference.jl")
include("save_nifti.jl")




function main()
    isempty(ARGS) && error("Usage: julia --project julia/main.jl path/to")
    case_dir = ARGS[1]




    
    
    println("[main] inspect onnx started")
    inspet_onxx()

    println("[main] starting inference")
    
    @time begin
        x, y, mask = infer_case(case_dir; roi_size=(96, 96, 96))
    end    

    if mask === nothing
        println("no discrete mask generated automatically")
        return
    end

    ref_path = find_reference_path(case_dir)
    out_path = normpath(joinpath(@__DIR__, "..", "outputs", "pred_mask.nii.gz"))

    save_mask_reference(mask, ref_path, out_path)
    
    println("[main] saved >>", out_path)
    println("[main] Inference successful. Prediction size >> ", size(mask))
    niwrite("[main] prediction.nii.gz", NIVolume(mask))

end

main()