__precompile__(true)
module SugarBLAS

export  @blas!
export  @scale!, @axpy!, @copy!, @ger!, @syr!, @syrk!,
        @her!, @herk!, @gbmv!, @sbmv!, @gemm!, @gemv!

include("MathMatch.jl")
using .MathMatch

char(s::Symbol) = string(s)[1]
function char(expr::Expr)
    (expr.head == :quote) || error("char doesn't support $(expr.head)")
    char(expr.args[1])
end

isempty(nl::Nullable) = nl.isnull

function expand(expr::Expr)
    @match(expr, A += B) && return :($A = $A + $B)
    expr
end

macro place(expr::Expr)
    func = "$(expr.args[1])"
    expr = :(Expr(:call, parse($func), $(expr.args[2:end]...)))
    esc(:(return esc($expr)))
end

function absAST(ast)
    if @match(ast, -ast) | (ast == 0)
        ast
    else
        Expr(:call, :(-), ast)
    end
end

###########
# Mutable #
###########

#Must be ordered from most to least especific formulas
macro blas!(expr::Expr)
    expr = expand(expr)
    @match(expr, X *= a) && @place scale!(a,X)
    @match(expr, X = a*X) && @place scale!(a,X)
    @match(expr, Y = a*X + Y) && @place Base.LinAlg.axpy!(a,X,Y)
    @match(expr, Y = X + Y) && @place Base.LinAlg.axpy!(1.0,X,Y)
    @match(expr, X = Y) && @place copy!(X, Y)
    error("No match found")
end

macro copy!(expr::Expr)
    @match(expr, X = Y) && @place copy!(X,Y)
    error("No match found")
end

macro scale!(expr::Expr)
    @match(expr, X *= a) && @place scale!(a,X)
    @match(expr, X = a*X) && @place scale!(a,X)
    error("No match found")
end

macro axpy!(expr::Expr)
    expr = expand(expr)
    @match(expr, Y = a*X + Y) && @place Base.LinAlg.axpy!(a,X,Y)
    @match(expr, Y = X + Y) && @place Base.LinAlg.axpy!(1.0,X,Y)
    error("No match found")
end

macro ger!(expr::Expr)
    expr = expand(expr)
    if @match(expr, A = alpha*x*y' + A)
        @place Base.LinAlg.BLAS.ger!(alpha,x,y,A)
    end
    error("No match found")
end

macro syr!(expr::Expr)
    expr = expand(expr)
    if @match(expr, X = alpha*x*x.' + Y)
        if @match(X, A[uplo]) & (@match(Y, A) | @match(Y, A[uplo]))
            c = char(uplo)
            @place Base.LinAlg.BLAS.syr!(c,alpha,x,A)
        end
    end
    error("No match found")
end

macro syrk!(expr::Expr)
    if @match(expr, C[uplo] = alpha*X*Y + beta*C)
        c = char(uplo)
        trans = if @match(copy(X), A.') && (Y == A)
            'T'
        elseif @match(Y, A.') && (X == A)
            'N'
        end
        @place Base.LinAlg.BLAS.syrk!(c,trans,alpha,A,beta,C)
    end
    error("No match found")
end

macro her!(expr::Expr)
    expr = expand(expr)
    if @match(expr, X = alpha*x*x' + Y)
        if @match(X, A[uplo]) & (@match(Y, A) | @match(Y, A[uplo]))
            c = char(uplo)
            @place Base.LinAlg.BLAS.her!(c,alpha,x,A)
        end
    end
    error("No match found")
end

macro herk!(expr::Expr)
    if @match(expr, C[uplo] = alpha*X*Y + beta*C)
        c = char(uplo)
        trans = if @match(copy(X), A') && (Y == A)
            'T'
        elseif @match(Y, A') && (X == A)
            'N'
        end
        @place Base.LinAlg.BLAS.herk!(c,trans,alpha,A,beta,C)
    end
    error("No match found")
end

macro gbmv!(expr::Expr)
    if @match(expr, y = alpha*Y*x + beta*y)
        trans = @match(Y, Y') ? 'T' : 'N'
        @match(Y, A[kl:ku,h=m])
        kl = absAST(kl)
        @place Base.LinAlg.BLAS.gbmv!(trans,m,kl,ku,alpha,A,x,beta,y)
    end
    error("No match found")
end

macro sbmv!(expr::Expr)
    if @match(expr, y = alpha*A[0:k,uplo]*x + beta*y)
        c = char(uplo)
        @place Base.LinAlg.BLAS.sbmv!(c,k,alpha,A,x,beta,y)
    end
    error("No match found")
end

macro sbmv!(expr::Expr)
    if @match(expr, y = alpha*A[0:k,uplo]*x + beta*y)
        c = char(uplo)
        @place Base.LinAlg.BLAS.sbmv!(c,k,alpha,A,x,beta,y)
    end
    error("No match found")
end

macro gemm!(expr::Expr)
    if @match(expr, C = alpha*A*B + beta*C)
        tA = @match(A, A') ? 'T' : 'N'
        tB = @match(B, B') ? 'T' : 'N'
        @place Base.LinAlg.BLAS.gemm!(tA,tB,alpha,A,B,beta,C)
    end
    error("No match found")
end

macro gemv!(expr::Expr)
    if @match(expr, y = alpha*A*x + beta*y)
        tA = @match(A, A') ? 'T' : 'N'
        @place Base.LinAlg.BLAS.gemv!(tA,alpha,A,x,beta,y)
    end
    error("No match found")
end

end
