__precompile__(true)
module SugarBLAS

export @blas!

include("MathMatch.jl")
using .MathMatch

isempty(nl::Nullable) = nl.isnull

function expand(expr::Expr)
    @match(expr, A += B) && return :($A = $A + $B)
    expr
end

#Must be ordered from most to least especific formulas
macro blas!(expr::Expr)
    expr = expand(expr)
    @match(expr, X *= a) && return esc(:(scale!($a, $X)))
    @match(expr, X = a*X) && return esc(:(scale!($a, $X)))
    @match(expr, Y = a*X + Y) && return esc(:(Base.LinAlg.axpy!($a, $X, $Y)))
    @match(expr, Y = X + Y) && return esc(:(Base.LinAlg.axpy!(1.0, $X, $Y)))
    @match(expr, X = Y) && return esc(:(copy!($X, $Y)))
    error("No match found")
end

end
