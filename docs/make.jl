using Documenter
using AshDieback

makedocs(;
         modules = [AshDieback],
         authors = "Simon Frost",
         sitename = "AshDieback.jl",
         checkdocs = :exports,
         format = Documenter.HTML(;
                                  prettyurls = get(ENV, "CI", "false") == "true",
                                  canonical = "https://ecorecipes.github.io/AshDieback.jl"),
         pages = ["Home" => "index.md",
                  "Tutorials" => ["The ash dieback model" => "tutorials/01_introduction.md",
                                  "Inference" => "tutorials/02_inference.md",
                                  "Validation and the likelihood surface" => "tutorials/03_validation.md",
                                  "Fast AMIS" => "tutorials/04_fast_amis.md",
                                  "Reproducing the paper" => "tutorials/05_reproducing_the_paper.md",
                                  "Interventions in Great Britain" => "tutorials/06_interventions_gb.md"],
                  "API Reference" => ["Data" => "api/data.md",
                                      "Model" => "api/model.md",
                                      "Inference" => "api/inference.md"]])

if get(ENV, "CI", "false") == "true"
    deploydocs(; repo = "github.com/ecorecipes/AshDieback.jl.git", push_preview = true)
end
