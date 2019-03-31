__precompile__(true)

"""
Syntactic sugar for BLAS polynomials.
"""
module SugarBLAS

export ᵀ
export  @blas!

include("Match/Match.jl")
using .Match

struct ᵀ end

"""
Negate a number, symbol or expression
"""
neg(ast::Number) = -ast
function neg(ast::Union{Symbol, Expr})
    if @match(ast, -ast) | (ast == 0)
        ast
    else
        Expr(:call, :(-), ast)
    end
end

"""
Determine whether it is a substraction or not
"""
substracts(expr) = false
substracts(expr::Expr) = (expr.head == :call) & (expr.args[1] == :-)

"""
Make dictionary containing the kwargs contents.
"""
function kwargs_to_dict(kwargs::Tuple)
    dict = Dict()
    for kw in kwargs
        dict[kw.args[1]] = kw.args[2]
    end
    dict
end

"""
Wrap a expr with an expression.
"""
wrap(expr::Symbol) = QuoteNode(expr)
function wrap(expr::Expr)
    head = QuoteNode(expr.head)
    func = string(expr.args[1])
    :(Expr($head, Meta.parse($func), $(expr.args[2:end]...)))
end

"""
Expand mixed assignment operators.
"""
function expand(expr::Expr)
    @match(expr, A += B) && return :($A = $A + $B)
    @match(expr, A -= B) && return :($A = $A - $B)
    expr
end

"""
Execute escaped expr.
"""
macro call(expr::Expr)
    esc(:(esc($(wrap(expr)))))
end

#Changes made in Julia parser in v"0.6.0-dev.2613".
#JuliaLang Issue: https://github.com/JuliaLang/julia/pull/20327
"""
Transforms the custom case expression to a string representing the equivalent if-then-else block of code.
"""
construct_case_statement(lines::Vector) = construct_case_statement(lines, Val{VERSION>=v"0.6.0-dev.2613"})
function construct_case_statement(lines::Vector, ::Type{Val{true}})
  failproof(s) = s
  failproof(s::Char) = string("'",s,"'")
  line = lines[1]
  exec = "if $(line.args[2])\n$(failproof(line.args[3]))\n"
  for line in lines[2:end-1]
      (line.head == :line) && continue
      (line.head == :call && line.args[1] == :(=>)) || error("Each condition must be followed by `=>`")
      exec *= "elseif $(line.args[2])\n$(failproof(line.args[3]))\n"
  end
  line = lines[end]
  exec *= (line.args[2] == :otherwise) && ("else\n$(failproof(line.args[3]))\n")
  exec *= "end"
end
function construct_case_statement(lines::Vector, ::Type{Val{false}})
  failproof(s) = s
  failproof(s::Char) = string("'",s,"'")
  line = lines[1]
  exec = "if $(line.args[1])\n$(failproof(line.args[2]))\n"
  for line in lines[2:end-1]
      (line.head == :line) && continue
      line.head == :(=>) || error("Each condition must be followed by `=>`")
      exec *= "elseif $(line.args[1])\n$(failproof(line.args[2]))\n"
  end
  line = lines[end]
  exec *= (line.args[1] == :otherwise) && ("else\n$(failproof(line.args[2]))\n")
  exec *= "end"
end

"""
Sugar for if-then-else expression. Beautiful for one liners.
"""
macro case(expr::Expr)
    (expr.head == :block) || error("@case statement must be followed by `begin ... end`")
    lines = filter(line -> !is_metadata(line), expr.args)
    exec = construct_case_statement(lines)
    esc(Meta.parse(exec))
end

"""
Filter metadata lines that often appear in Julia expressions
"""
is_metadata(line::Expr) = line.head == :line #Needed for versions older than v0.7-DEV
is_metadata(::LineNumberNode) = true

###############
# BLAS macros #
###############

"""
    @blas!(expr)

Transform expr to most specific BLAS function. Resulting expression will be
Base.copy!, Base.scale! or Base.LinAlg.axpy!.

**Polynomials**

*copy!*:

- `X = Y`

*scale!*:

- `X *= a`
- `X = a*X`

*axpy!*:

- `Y ±= X`
- `Y ±= a*X`
"""
#Must be ordered from most to least especific formulas
macro blas!(expr::Expr)
    unkeyword!(expr)
    expr = expand(expr)
    @case begin
        @match(expr, X *= a)        => @call scale!(a,X)
        @match(expr, X = a*X)       => @call scale!(a,X)
        @match(expr, Y = Y - a*X)   => @call Base.LinAlg.axpy!(neg(a),X,Y)
        @match(expr, Y = Y - X)     => @call Base.LinAlg.axpy!(-1.0,X,Y)
        @match(expr, Y = a*X + Y)   => @call Base.LinAlg.axpy!(a,X,Y)
        @match(expr, Y = X + Y)     => @call Base.LinAlg.axpy!(1.0,X,Y)
        @match(expr, X = Y)         => @call copy!(X, Y)
        otherwise                   => error("No match found")
    end
end

"""
    @copy!(expr)

Copy all elements from collection `Y` to array `X`. Return `X`.

**Polynomials**

- `X = Y`
"""
macro copy!(expr::Expr)
    unkeyword!(expr)
    @case begin
        @match(expr, X = Y) => @call copy!(X,Y)
        otherwise           => error("No match found")
    end
end

"""
    @scale!(expr)

Scale an array `X` by a scalar `a` overwriting `X` in-place.

**Polynomials**

- `X *= a`
- `X = a*X`
"""
macro scale!(expr::Expr)
    unkeyword!(expr)
    @case begin
        @match(expr, X *= a)    => @call scale!(a,X)
        @match(expr, X = a*X)   => @call scale!(a,X)
        otherwise               => error("No match found")
    end
end

"""
    @axpy!(expr)

Overwrite `Y` with `a*X + Y`. Return `Y`.

**Polynomials**

- `Y ±= X`
- `Y ±= a*X`
"""
macro axpy!(expr::Expr)
    unkeyword!(expr)
    expr = expand(expr)
    @case begin
        @match(expr, Y = Y - a*X)   => @call(Base.LinAlg.axpy!(neg(a),X,Y))
        @match(expr, Y = Y - X)     => @call Base.LinAlg.axpy!(-1.0,X,Y)
        @match(expr, Y = a*X + Y)   => @call Base.LinAlg.axpy!(a,X,Y)
        @match(expr, Y = X + Y)     => @call Base.LinAlg.axpy!(1.0,X,Y)
        otherwise                   => error("No match found")
    end
end

"""
    @ger!(expr)

Rank-1 update of the matrix `A` with vectors `x` and `y` as `alpha*x*y' + A`.

**Polynomials**

- `A ±= alpha*x*y'`
"""
macro ger!(expr::Expr)
    unkeyword!(expr)
    expr = expand(expr)
    f = @case begin
        @match(expr, A = alpha*x*y' + A)    => identity
        @match(expr, A = A - alpha*x*y')    => neg
        otherwise                           => error("No match found")
    end
    @call Base.LinAlg.BLAS.ger!(f(alpha),x,y,A)
end

"""
    @syr!(expr)

Rank-1 update of the symmetric matrix `A` with vector `x` as `alpha*x*xᵀ + A`.
When left side has `A['U']` the upper triangle of `A` is updated (`'L'` for lower
triangle). Return `A`.

**Polynomials**

- `A[uplo] ±= alpha*x*xᵀ`
"""
macro syr!(expr::Expr)
    unkeyword!(expr)
    expr = expand(expr)
    @match(expr, A[uplo] = right) || error("No match found")
    f = @case begin
        @match(right, alpha*x*xᵀ + Y)  => identity
        @match(right, Y - alpha*x*xᵀ)  => neg
        otherwise                       => error("No match found")
    end
    (@match(Y, Y[uplo]) && (Y == A)) || (Y == A) || error("No match found")
    @call Base.LinAlg.BLAS.syr!(uplo,f(alpha),x,A)
end

"""
    @syrk!(expr)

Return either the upper triangle or the lower triangle, depending on
(`'U'` or `'L'`), of `alpha*A*(A)ᵀ` or `alpha*(A)ᵀ*A`.

**Polynomials**

- `alpha*A*(A)ᵀ uplo=ul`
- `alpha*(A)ᵀ*A uplo=ul`
"""
macro syrk(expr::Expr, kwargs...)
    kwargs = kwargs_to_dict(kwargs)
    uplo = kwargs[:uplo]
    f = @case begin
        @match(expr, alpha*X*Y) => identity
        otherwise               => error("No match found")
    end
    trans = @case begin
        @match(X, (A)ᵀ) && (Y == A)  => 'T'
        @match(Y, (A)ᵀ) && (X == A)  => 'N'
        otherwise                   => error("No match found")
    end
    @call Base.LinAlg.BLAS.syrk(uplo,trans,f(alpha),A)
end

"""
    @syrk!(expr)

Rank-k update of the symmetric matrix `C` as `alpha*A*(A)ᵀ + beta*C` or
`alpha*(A)ᵀ*A + beta*C`. When the left hand side is`C['U']` the upper triangle of `C`
is updated (`'L'` for lower triangle). Return `C`.

**Polynomials**

- `C[uplo] ±= alpha*A*(A)ᵀ`
- `C[uplo] = beta*C ± alpha*(A)ᵀ*A`
"""
macro syrk!(expr::Expr)
    unkeyword!(expr)
    expr = expand(expr)
    @match(expr, C[uplo] = right) || error("No match found")
    f = @case begin
        @match(right, alpha*X*Y + D)    => identity
        @match(right, D - alpha*X*Y)    => neg
        otherwise                       => error("No match found")
    end
    trans = @case begin
        @match(X, (A)ᵀ) && (Y == A)  => 'T'
        @match(Y, (A)ᵀ) && (X == A)  => 'N'
        otherwise                   => error("No match found")
    end
    @match(D, beta*D) || (beta = 1.0)
    (@match(D, D[uplo]) && (C == D)) || (C == D) || error("No match found")
    @call Base.LinAlg.BLAS.syrk!(uplo,trans,f(alpha),A,beta,C)
end

"""
    @her!(expr)

Methods for complex arrays only. Rank-1 update of the Hermitian matrix `A`
with vector `x` as `alpha*x*x' + A`. Whenthe left hand side is `A['U']`
the upper triangle of `A` is updated (`'L'` for lower triangle). Return `A`.

**Polynomials**

- `A[uplo] ±= alpha*x*x'`
"""
macro her!(expr::Expr)
    unkeyword!(expr)
    expr = expand(expr)
    @match(expr, A[uplo] = right) || error("No match found")
    f = @case begin
        @match(right, alpha*x*x' + Y)   => identity
        @match(right, Y - alpha*x*x')   => neg
        otherwise                       => error("No match found")
    end
    (@match(Y, Y[uplo]) && (Y == A)) || (Y == A) || error("No match found")
    @call Base.LinAlg.BLAS.her!(uplo,f(alpha),x,A)
end

"""
    @herk(expr)

Methods for complex arrays only. Returns either the upper triangle or the
lower triangle, according to uplo ('U' or 'L'), of alpha*A*A' or alpha*A'*A,
according to trans ('N' or 'T').

**Polynomials**

- `alpha*A*A' uplo=ul`
- `alpha*A'*A uplo=ul`
"""
macro herk(expr::Expr, kwargs...)
    kwargs = kwargs_to_dict(kwargs)
    uplo = kwargs[:uplo]
    @match(expr, alpha*X*Y) || error("No match found")
    trans = @case begin
        @match(X, A') && (Y == A)   =>  'T'
        @match(Y, A') && (X == A)   =>  'N'
        otherwise                   =>  error("No match found")
    end
    @call Base.LinAlg.BLAS.herk(uplo,trans,alpha,A)
end

"""
    @herk!(expr)

Methods for complex arrays only. Rank-k update of the Hermitian matrix `C` as
`alpha*A*A' + beta*C` or `alpha*A'*A + beta*C`. When the left hand side is `C['U']`
the upper triangle of `C` is updated (`'L'` for lower triangle). Return `C`.

**Polynomials**

- `C[uplo] ±= alpha*A*A'`
- `C[uplo] = beta*C ± alpha*A'*A`
"""
macro herk!(expr::Expr)
    unkeyword!(expr)
    expr = expand(expr)
    @match(expr, C[uplo] = right) || error("No match found")
    f = @case begin
        @match(right, alpha*X*Y + D)    => identity
        @match(right, D - alpha*X*Y)    => neg
        otherwise                       => error("No match found")
    end
    trans = @case begin
        @match(X, A') && (Y == A)   =>  'T'
        @match(Y, A') && (X == A)   =>  'N'
        otherwise                   =>  error("No match found")
    end
    @match(D, beta*D) || (beta = 1.0)
    (@match(D, D[crap]) && (C == D)) || (C == D) || error("No match found")
    @call Base.LinAlg.BLAS.herk!(uplo,trans,f(alpha),A,beta,C)
end

"""
    @gbmv(expr)

Return `alpha*A*x` or `alpha*A'*x`. The matrix `A` is a general band matrix
of dimension `m` by `size(A,2)` with `kl` sub-diagonals and `ku` super-diagonals.

**Polynomials**

- `alpha*A[kl:ku,h=m]*x`
- `alpha*A[h=m,kl:ku]'*x`
"""
macro gbmv(expr::Expr)
    @match(expr, alpha*Y*x) || error("No match found")
    trans = @match(Y, Y') ? 'T' : 'N'
    @match(Y, A[kl:ku,h=m])
    @call Base.LinAlg.BLAS.gbmv(trans,m,neg(kl),ku,alpha,A,x)
end

"""
    @gbmv!(expr)

Update vector `y` as `alpha*A*x + beta*y` or `alpha*A'*x + beta*y`.
The matrix `A` is a general band matrix of dimension `m` by `size(A,2)` with
`kl` sub-diagonals and `ku` super-diagonals. Return the updated `y`.

**Polynomials**

- `y ±= alpha*A[kl:ku,h=m]*x`
- `y = beta*y ± alpha*A[h=m,kl:ku]'*x`
"""
macro gbmv!(expr::Expr)
    unkeyword!(expr)
    expr = expand(expr)
    @match(expr, y = right) || error("No match found")
    f = @case begin
        @match(right, alpha*Y*x + w)    => identity
        @match(right, w - alpha*Y*x)    => neg
        otherwise                       => error("No match found")
    end
    trans = @match(Y, Y') ? 'T' : 'N'
    @match(Y, A[kl:ku,h=m])
    @match(w, beta*w) || (beta = 1.0)
    (y == w) || error("No match found")
    @call Base.LinAlg.BLAS.gbmv!(trans,m,neg(kl),ku,f(alpha),A,x,beta,y)
end

"""
    @sbmv(expr)

Return `alpha*A*x` where `A` is a symmetric band matrix of order `size(A,2)` with
`k` super-diagonals stored in the argument `A`.

**Polynomials**

- `A[0:k,uplo]*xv`
- `alpha*A[0:k,uplo]*x`
"""
macro sbmv(expr::Expr)
    @case begin
        @match(expr, alpha*A[0:k,uplo]*x)   => @call Base.LinAlg.BLAS.sbmv(uplo,k,alpha,A,x)
        @match(expr, A[0:k,uplo]*x)         => @call Base.LinAlg.BLAS.sbmv(uplo,k,A,x)
        otherwise                           => error("No match found")
    end
end

"""
    @sbmv!(expr)

Update vector `y` as `alpha*A*x + beta*y` where `A` is a a symmetric band matrix
of order `size(A,2)` with `k` super-diagonals stored in the argument `A`. If
`A[...,'U']` is used multiplication is done with `A`'s upper triangle, `L` is for the
lower triangle. Return updated `y`.

**Polynomials**

- `y ±= alpha*A[0:k,uplo]*x`
- `y = beta*y ± alpha*A[0:k,uplo]*x`
"""
macro sbmv!(expr::Expr)
    unkeyword!(expr)
    expr = expand(expr)
    @match(expr, y = right) || error("No match found")
    f = @case begin
        @match(right, alpha*A[0:k,uplo]*x + w)  => identity
        @match(right, w - alpha*A[0:k,uplo]*x)  => neg
        otherwise                               => error("No match found")
    end
    @match(w, beta*w) || (beta = 1.0)
    (@match(w, w[crap]) && (y == w)) || (y == w) || error("No match found")
    @call Base.LinAlg.BLAS.sbmv!(uplo,k,f(alpha),A,x,beta,y)
end

"""
    @gemm(expr)

Return `alpha*A*B`, `alpha*A'*B`, `alpha*A*B'` or `alpha*A'*B'`.

**Polynomials**

- `A*B`
- `A'*B`
- `A*B'`
- `A'*B'`
- `alpha*A*B`
- `alpha*A'*B`
- `alpha*A*B'`
- `alpha*A'*B'`
"""
macro gemm(expr::Expr)
    if @match(expr, alpha*A*B)
        tA = @match(A, A') ? 'T' : 'N'
        tB = @match(B, B') ? 'T' : 'N'
        @call Base.LinAlg.BLAS.gemm(tA,tB,alpha,A,B)
    elseif @match(expr, A*B)
        tA = @match(A, A') ? 'T' : 'N'
        tB = @match(B, B') ? 'T' : 'N'
        @call Base.LinAlg.BLAS.gemm(tA,tB,A,B)
    else
        error("No match found")
    end
end

"""
    @gemm!(expr)

Update `C` as `alpha*A*B + beta*C` or the other three variants according to the
combination of transposes of `A` and `B`. Return updated C.

**Polynomials**

- `C ±= alpha*A*B`
- `C ±= alpha*A'*B`
- `C ±= alpha*A*B'`
- `C ±= alpha*A'*B'`
- `C = beta*C ± alpha*A*B`
- `C = beta*C ± alpha*A'*B`
- `C = beta*C ± alpha*A*B'`
- `C = beta*C ± alpha*A'*B'`
"""
macro gemm!(expr::Expr)
    unkeyword!(expr)
    expr = expand(expr)
    @match(expr, C = right) || error("No match found")
    f = @case begin
        @match(right, alpha*A*B + D)    => identity
        @match(right, D - alpha*A*B)    => neg
        otherwise                       => error("No match found")
    end
    tA = @match(A, A') ? 'T' : 'N'
    tB = @match(B, B') ? 'T' : 'N'
    @match(D, beta*D) || (beta = 1.0)
    (C == D) || error("No match found")
    @call Base.LinAlg.BLAS.gemm!(tA,tB,f(alpha),A,B,beta,C)
end

"""
    @gemv(expr)

Return `alpha*A*x` or `alpha*A'*x`.

**Polynomials**

- `A*x`
- `A'*x`
- `alpha*A*x`
- `alpha*A'*x`
"""
macro gemv(expr::Expr)
    if @match(expr, alpha*A*x)
        tA = @match(A, A') ? 'T' : 'N'
        @call Base.LinAlg.BLAS.gemv(tA,alpha,A,x)
    elseif @match(expr, A*x)
        tA = @match(A, A') ? 'T' : 'N'
        @call Base.LinAlg.BLAS.gemv(tA,A,x)
    else
        error("No match found")
    end
end

"""
    @gemv!(expr)

Update the vector `y` as `alpha*A*x + beta*y` or `alpha*A'*x + beta*y`.
Return updated `y`.

**Polynomials**

- `y ±= alpha*A*x`
- `y ±= alpha*A'*x`
- `y = beta*y ± alpha*A*x`
- `y = beta*y ± alpha*A'*x`
"""
macro gemv!(expr::Expr)
    unkeyword!(expr)
    expr = expand(expr)
    @match(expr, y = right) || error("No match found")
    f = @case begin
        @match(right, alpha*A*x + w)    => identity
        @match(right, w - alpha*A*x)    => neg
        otherwise                       => error("No match found")
    end
    tA = @match(A, A') ? 'T' : 'N'
    @match(w, beta*w) || (beta = 1.0)
    (y == w) || error("No match found")
    @call Base.LinAlg.BLAS.gemv!(tA,f(alpha),A,x,beta,y)
end

"""
    @symm(expr)

Return `alpha*A*B` or `alpha*B*A` according to `"symm"`. `A` is assumed to be
symmetric. Only the `uplo` triangle of `A` is used (`'L'` for lower and `'U'` for upper).

**Polynomials**

- `A["symm", uplo]*B`
- `A*B["symm", uplo]`
- `alpha*A["symm", uplo]*B `
- `alpha*A*B["symm", uplo]`
"""
macro symm(expr::Expr)
    if @match(expr, alpha*A*B)
        side = @case begin
            @match(A, A["symm", uplo])  => 'L'
            @match(B, B["symm", uplo])  => 'R'
            otherwise                   => error("No match found")
        end
        @call Base.LinAlg.BLAS.symm(side,uplo,alpha,A,B)
    elseif @match(expr, A*B)
        side = @case begin
            @match(A, A["symm", uplo])  => 'L'
            @match(B, B["symm", uplo])  => 'R'
            otherwise                   => error("No match found")
        end
        @call Base.LinAlg.BLAS.symm(side,uplo,A,B)
    else
        error("No match found")
    end
end

"""
    @symm!(expr)

Update `C` as `alpha*A*B + beta*C` or `alpha*B*A + beta*C` according to `"symm"`.
`A` is assumed to be symmetric. Only the `uplo` triangle of `A` is used
(`'L'` for lower and `'U'` for upper). Return updated `C`.

**Polynomials**

- `C = alpha*A["symm",uplo]*B`
- `C = alpha*A*B["symm",uplo]`
- `C = beta*C ± alpha*A["symm",uplo]*B`
- `C = beta*C ± alpha*A*B["symm",uplo]`
"""
macro symm!(expr::Expr)
    unkeyword!(expr)
    expr = expand(expr)
    @match(expr, C = right) || error("No match found")
    f = @case begin
        @match(right, alpha*A*B + D)    => identity
        @match(right, D - alpha*A*B)    => neg
        otherwise                       => error("No match found")
    end
    side = @case begin
        @match(A, A["symm", uplo])   => 'L'
        @match(B, B["symm", uplo])   => 'R'
        otherwise                    => error("No match found")
    end
    @match(D, beta*D) || (beta = 1.0)
    (@match(D, D[crap]) && (C == D)) || (C == D) || error("No match found")
    @call Base.LinAlg.BLAS.symm!(side,uplo,f(alpha),A,B,beta,C)
end

"""
    @symv(expr)

Return `alpha*A*x`. `A` is assumed to be symmetric. Only the `uplo` triangle of `A`
is used (`'L'` for lower and `'U'` for upper).

**Polynomials**

- `A[uplo]*x`
- `alpha*A[uplo]*x`
"""
macro symv(expr::Expr)
    @case begin
        @match(expr, alpha*A[uplo]*x)   => @call Base.LinAlg.BLAS.symv(uplo,alpha,A,x)
        @match(expr, A[uplo]*x)         => @call Base.LinAlg.BLAS.symv(uplo,A,x)
        otherwise                       => error("No match found")
    end
end

"""
    @symv!(expr)

Update the vector `y` as `alpha*A*x + beta*y`. `A` is assumed to be symmetric.
Only the `uplo` triangle of `A` is used (`'L'` for lower and `'U'` for upper).
Return updated y.

**Polynomials**

- `y ±= alpha*A[uplo]*x`
- `y = beta*y ± alpha*A[uplo]*x`
"""
macro symv!(expr::Expr)
    unkeyword!(expr)
    expr = expand(expr)
    @match(expr, y = right) || error("No match found")
    f = @case begin
        @match(right, alpha*A[uplo]*x + w)  => identity
        @match(right, w - alpha*A[uplo]*x)  => neg
        otherwise                           => error("No match found")
    end
    @match(w, beta*w) || (beta = 1.0)
    (@match(w, w[crap]) && (y == w)) || (y == w) || error("No match found")
    @call Base.LinAlg.BLAS.symv!(uplo,f(alpha),A,x,beta,y)
end

end
