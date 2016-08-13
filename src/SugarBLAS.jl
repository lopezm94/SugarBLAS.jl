precompile(true)
module SugarBLAS

export @blas!

_axpy!(out,ast) = Nullable{Expr}()
function _axpy!(out::Symbol, ast::Expr)
    try
        op = ast.args[1]
        args = ast.args[2:end]

        assert((ast.head == :call) & ((op == :+) | (op == :-)))
        assert(length(args) == 2)
        aux = find(args .== out)    #Finds out
        index = (aux[1] == 1) ? 2:1 #Get a*X index in AST
        assert((op == :+) | (index == 2))
        if isa(args[index], Symbol)
            a = (op == :-) ? -1 : 1
            X = args[index]
            Nullable{Expr}(esc(:(Base.LinAlg.BLAS.axpy!(eltype($X)($a),$X,$out))))
        elseif isa(args[index], Expr)
            assert((args[index].head == :call) & (args[index].args[1] == :*))
            a = args[index].args[2]
            a = (op == :-) ? :(-1*$a) : a
            X = args[index].args[3]
            Nullable{Expr}(esc(:(Base.LinAlg.BLAS.axpy!(eltype($X)($a),$X,$out))))
        else
            assert(false)
        end
    catch
        Nullable{Expr}()
    end
end

_scale(ast) = Nullable{Expr}()
function _scale(ast::Expr)
    try
        op = ast.args[1]
        args = ast.args[2:end]

        assert((ast.head == :call) & (op == :*))
        assert(length(args) == 2)
        a = args[1]
        X = args[2]
        Nullable{Expr}(esc(:(scale($a,$X))))
    catch
        Nullable{Expr}()
    end
end

_scale!(out,ast) = Nullable{Expr}()
function _scale!(out::Symbol, ast::Expr)
    try
        op = ast.args[1]
        args = ast.args[2:end]

        assert((ast.head == :call) & (op == :*))
        assert(length(args) == 2)
        a = args[1]
        X = args[2]
        assert(out == X)
        Nullable{Expr}(esc(:(scale!($a,$X))))
    catch e
        Nullable{Expr}()
    end
end

_copy!(X,Y) = Nullable{Expr}()
function _copy!(X::Symbol, Y::Symbol)
    Nullable{Expr}(esc(:(copy!($X,$Y))))
end

cat_able(::Symbol, ::Symbol) = false
function cat_able(op::Symbol, expr::Expr)
    (expr.head==:call) & (expr.args[1]==op) & (op in [:+,:*])
end

#Expand from "A (op)= B" to "A = A (op) B"
function expand(expr::Expr)
    op = expr.head
    op != :(=) || return expr
    op in [:(-=), :(+=), :(/=), :(\=), :(*=)] || return expr
    op = symbol(string(op)[1])  #Get the operator
    left = expr.args[1]
    right = expr.args[2]
    if cat_able(op, right)
        Expr(:(=), left, Expr(:call, op, left, right.args[2:end]...))
    else
        Expr(:(=), left, Expr(:call, op, left, right))
    end
end

macro blas!(expr::Expr)
    if expr.head == :call
        (aux = _scale(expr)).isnull || return aux.value
    else
        left, right = expand(expr).args
        (aux = _axpy!(left, right)).isnull || return aux.value
        (aux = _scale!(left, right)).isnull || return aux.value
        (aux = _copy!(left, right)).isnull || return aux.value
    end
    error("No match found")
end

end
