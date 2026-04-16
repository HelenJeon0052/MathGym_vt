using NIfTI




function find_reference_path(case_dir::AbstractString)
    files = readdir(case_dir; join=true)
    # Todo : loop all types ("t1c", "t1n", "t2f", "t2w")
    hits = sort(filter(f -> endswith(f, "-t1c.nii.gz"), files))
    legnth(hits) == 1 || error("expected '-t1c.nii.gz' file in $case_dir") 
    return only(hits)
end


function save_mask_reference(
    mask::AbstractArray{<:Real, 3}.
    reference_path::AbstractString,
    output_path::AbstractString,
)

    ref = niread(reference_path)

    @assert ndims(ref.raw) == 3 "referecne raw volume == 3, got ndims=$(ndims(ref.raw))"

    @assert size(mask) == size(ref.raw) "Mask size $(size(mask)) != reference size $(size(ref.raw))"

    out = deepcopy(ref)

    out.raw .= Float32.(mask)

    mkpath(dirname(output_path))
    niwrite(output_path, out)
    return output_path
    
end
