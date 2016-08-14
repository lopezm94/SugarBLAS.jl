__precompile__(true)
module SugarBLAS

export @blas, @blas!

include("MathMatch.jl")
using .MathMatch

isempty(nl::Nullable) = nl.isnull

macro blas(expr::Expr)
    either = @match expr a*X
    isempty(either) || (d=either.value; return esc(:(scale($(d[:a]), $(d[:X])))))
    error("No match found")
end

#Must be ordered from most to least especific formulas
macro blas!(expr::Expr)
    either = @match expr a*X
    isempty(either) || (d=either.value; return esc(:(scale!($(d[:a]), $(d[:X])))))
    either = @match expr X = a*X
    isempty(either) || (d=either.value; return esc(:(scale!($(d[:a]), $(d[:X])))))
    either = @match expr X *= a
    isempty(either) || (d=either.value; return esc(:(scale!($(d[:a]), $(d[:X])))))
    either = @match expr Y = a*X + Y
    if !isempty(either)
        d=either.value
        return esc(:(Base.LinAlg.axpy!($(d[:a]), $(d[:X]), $(d[:Y]))))
    end
    either = @match expr Y += a*X
    if !isempty(either)
        d=either.value
        return esc(:(Base.LinAlg.axpy!($(d[:a]), $(d[:X]), $(d[:Y]))))
    end
    either = @match expr a*X + Y
    if !isempty(either)
        d=either.value
        return esc(:(Base.LinAlg.axpy!($(d[:a]), $(d[:X]), $(d[:Y]))))
    end
    either = @match expr X = Y
    isempty(either) || (d=either.value; return esc(:(copy!($(d[:X]), $(d[:Y])))))
    error("No match found")
end

end
