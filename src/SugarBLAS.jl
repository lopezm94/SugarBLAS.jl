__precompile__(true)
module SugarBLAS

import Base: copy

export  @blas!
export  @scale!, @axpy!, @copy!, @ger!, @syr!, @syrk!,
        @her!, @herk!, @gbmv!, @sbmv!, @gemm!, @gemv!

include("Match/Match.jl")
using .Match

char(s::Symbol) = string(s)[1]
function char(expr::Expr)
    (expr.head == :quote) || error("char doesn't support $(expr.head)")
    char(expr.args[1])
end

isempty(nl::Nullable) = nl.isnull

wrap(expr::Symbol) = QuoteNode(expr)
function wrap(expr::Expr)
    head = QuoteNode(expr.head)
    func = string(expr.args[1])
    :(Expr($head, parse($func), $(expr.args[2:end]...)))
end

function expand(expr::Expr)
    @match(expr, A += B) && return :($A = $A + $B)
    expr
end

macro call(expr::Expr)
    esc(:(esc($(wrap(expr)))))
end

function absAST(ast)
    if @match(ast, -ast) | (ast == 0)
        ast
    else
        Expr(:call, :(-), ast)
    end
end

copy(s::Symbol) = s

###########
# Mutable #
###########

#Must be ordered from most to least especific formulas
macro blas!(expr::Expr)
    expr = expand(expr)
    @match(expr, X *= a) && return @call scale!(a,X)
    @match(expr, X = a*X) && return @call scale!(a,X)
    @match(expr, Y = a*X + Y) && return @call Base.LinAlg.axpy!(a,X,Y)
    @match(expr, Y = X + Y) && return @call Base.LinAlg.axpy!(1.0,X,Y)
    @match(expr, X = Y) && return @call copy!(X, Y)
    error("No match found")
end

macro copy!(expr::Expr)
    @match(expr, X = Y) && return @call copy!(X,Y)
    error("No match found")
end

macro scale!(expr::Expr)
    @match(expr, X *= a) && return @call scale!(a,X)
    @match(expr, X = a*X) && return @call scale!(a,X)
    error("No match found")
end

macro axpy!(expr::Expr)
    expr = expand(expr)
    @match(expr, Y = a*X + Y) && return @call Base.LinAlg.axpy!(a,X,Y)
    @match(expr, Y = X + Y) && return @call Base.LinAlg.axpy!(1.0,X,Y)
    error("No match found")
end

macro ger!(expr::Expr)
    expr = expand(expr)
    if @match(expr, A = alpha*x*y' + A)
        return @call Base.LinAlg.BLAS.ger!(alpha,x,y,A)
    end
    error("No match found")
end

macro syr!(expr::Expr)
    expr = expand(expr)
    if @match(expr, X = alpha*x*x.' + Y)
        if @match(X, A[uplo]) & (@match(Y, A) | @match(Y, A[uplo]))
            c = char(uplo)
            return @call Base.LinAlg.BLAS.syr!(c,alpha,x,A)
        end
    end
    error("No match found")
end

macro syrk!(expr::Expr)
    if @match(expr, C[uplo] = alpha*X*Y + D)
        c = char(uplo)
        trans = if @match(copy(X), A.') && (Y == A)
            'T'
        elseif @match(Y, A.') && (X == A)
            'N'
        end
        if (D == C)
            return @call Base.LinAlg.BLAS.syrk!(c,trans,alpha,A,1.0,C)
        elseif @match(D, beta*E) && (E == C)
            return @call Base.LinAlg.BLAS.syrk!(c,trans,alpha,A,beta,C)
        else
            error("No match found")
        end
    end
    error("No match found")
end

macro her!(expr::Expr)
    expr = expand(expr)
    if @match(expr, X = alpha*x*x' + Y)
        if @match(X, A[uplo]) & (@match(Y, A) | @match(Y, A[uplo]))
            c = char(uplo)
            return @call Base.LinAlg.BLAS.her!(c,alpha,x,A)
        end
    end
    error("No match found")
end

macro herk!(expr::Expr)
    if @match(expr, C[uplo] = alpha*X*Y + D)
        c = char(uplo)
        trans = if @match(copy(X), A') && (Y == A)
            'T'
        elseif @match(Y, A') && (X == A)
            'N'
        end
        if (D == C)
            return @call Base.LinAlg.BLAS.herk!(c,trans,alpha,A,1.0,C)
        elseif @match(D, beta*E) && (E == C)
            return @call Base.LinAlg.BLAS.herk!(c,trans,alpha,A,beta,C)
        else
            error("No match found")
        end
    end
    error("No match found")
end

macro gbmv!(expr::Expr)
    if @match(expr, y = alpha*Y*x + w)
        trans = @match(Y, Y') ? 'T' : 'N'
        @match(Y, A[kl:ku,h=m])
        kl = absAST(kl)
        if (w == y)
            return @call Base.LinAlg.BLAS.gbmv!(trans,m,kl,ku,alpha,A,x,1.0,y)
        elseif @match(w, beta*u) && (u == y)
            return @call Base.LinAlg.BLAS.gbmv!(trans,m,kl,ku,alpha,A,x,beta,y)
        else
            error("No match found")
        end
    end
    error("No match found")
end

macro sbmv!(expr::Expr)
    if @match(expr, y = alpha*A[0:k,uplo]*x + w)
        c = char(uplo)
        if (w == y)
            return @call Base.LinAlg.BLAS.sbmv!(c,k,alpha,A,x,1.0,y)
        elseif @match(w, beta*u) && (u == y)
            return @call Base.LinAlg.BLAS.sbmv!(c,k,alpha,A,x,beta,y)
        else
            error("No match found")
        end
    end
    error("No match found")
end

macro gemm!(expr::Expr)
    if @match(expr, C = alpha*A*B + D)
        tA = @match(A, A') ? 'T' : 'N'
        tB = @match(B, B') ? 'T' : 'N'
        if (D == C)
            return @call Base.LinAlg.BLAS.gemm!(tA,tB,alpha,A,B,1.0,C)
        elseif @match(D, beta*E) && (E == C)
            return @call Base.LinAlg.BLAS.gemm!(tA,tB,alpha,A,B,beta,C)
        else
            error("No match found")
        end
    end
    error("No match found")
end

macro gemv!(expr::Expr)
    if @match(expr, y = alpha*A*x + w)
        tA = @match(A, A') ? 'T' : 'N'
        if (w == y)
            return @call Base.LinAlg.BLAS.gemv!(tA,alpha,A,x,1.0,y)
        elseif @match(w, beta*u) && (u == y)
            return @call Base.LinAlg.BLAS.gemv!(tA,alpha,A,x,beta,y)
        else
            error("No match found")
        end
    end
    error("No match found")
end

end
