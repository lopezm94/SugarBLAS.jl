__precompile__(true)
module SugarBLAS

export @blas, @blas!

include("MathMatch.jl")
using .MathMatch

ismsg{T<:AbstractString}(::Nullable{T}) = true
ismsg(::Nullable) = false

macro blas(expr::Expr)
    either = @match expr a*X
    ismsg(either) || (d=either.value; return esc(:(scale($(d[:a]), $(d[:X])))))
    error("No match found")
end

macro blas!(expr::Expr)
    println(1)
    either = @match expr a*X
    ismsg(either) || (d=either.value; return esc(:(scale!($(d[:a]), $(d[:X])))))
println(2)
    either = @match expr X = a*X
    ismsg(either) || (d=either.value; return esc(:(scale!($(d[:a]), $(d[:X])))))
println(3)
    either = @match expr X *= a
    ismsg(either) || (d=either.value; return esc(:(scale!($(d[:a]), $(d[:X])))))
println(4)
    either = @match expr Y = a*X + Y
    if !ismsg(either)
        d=either.value
        return esc(:(Base.LinAlg.axpy!($(d[:a]), $(d[:X]), $(d[:Y]))))
    end
println(5)
    either = @match expr Y += a*X
    if !ismsg(either)
        d=either.value
        return esc(:(Base.LinAlg.axpy!($(d[:a]), $(d[:X]), $(d[:Y]))))
    end
println(6)
    either = @match expr a*X + Y
    if !ismsg(either)
        d=either.value
        return esc(:(Base.LinAlg.axpy!($(d[:a]), $(d[:X]), $(d[:Y]))))
    end
println(7)
    either = @match expr X = Y
    ismsg(either) || (d=either.value; return esc(:(copy!($(d[:X]), $(d[:Y])))))
    error("No match found")
end

end
