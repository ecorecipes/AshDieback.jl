"""
    AshDiebackError <: Exception

Root of the exceptions thrown by AshDieback.jl. Invalid arguments and keywords raise
`ArgumentError` instead.
"""
abstract type AshDiebackError <: Exception end

"""
    DataFormatError(file, message)

An input data file could not be parsed: a missing column, a value that is not a number,
or a coordinate outside the grid. `file` names the file and `message` says what was wrong.
"""
struct DataFormatError <: AshDiebackError
    file::String
    message::String
end

function Base.showerror(io::IO, e::DataFormatError)
    return print(io, "DataFormatError in ", e.file, ": ", e.message)
end
