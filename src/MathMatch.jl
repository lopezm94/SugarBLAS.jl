module MathMatch

export @match

#TODO: Replace errors for booleans, if 2 same keys have different value it doesnt match.

iscall(expr::Expr) = expr.head == :call

iscommutative(op::Expr) = iscommutative(op.args[1])
iscommutative(op::Symbol) = _iscommutative(Val{op})

_iscommutative(::Type{Val{:(+)}}) = true
_iscommutative(::Type{Val{:(*)}}) = false
_iscommutative(::Type{Val{:(-)}}) = false
_iscommutative(::Type{Val{:(/)}}) = false
_iscommutative{T<:Val}(::Type{T}) = false

function partialmatch(expr::Expr, formula::Expr)
    samehead = (expr.head == formula.head)
    samelen = length(expr.args) == length(formula.args)
    (samehead & samelen) || return false
    _partialmatch(Val{iscall(expr)}, expr, formula)
end

_partialmatch(::Type{Val{false}}, expr::Expr, formula::Expr) = true
_partialmatch(::Type{Val{true}}, expr::Expr, formula::Expr) = expr.args[1]==formula.args[1]

function match_args(symbols::Dict, expr::Expr, formula::Expr)
    offset = iscall(expr) ? 1 : 0
    _match_args(Val{iscommutative(expr)}, offset, symbols, expr, formula)
end

function _match_args(::Type{Val{false}}, offset, symbols, expr, formula)
    eargs, margs = expr.args, formula.args
    for i in 1+offset:length(eargs)
        match(symbols, eargs[i], margs[i])
    end
end
function _match_args(::Type{Val{true}}, offset, symbols, expr, formula)
    eargs, margs = expr.args, formula.args
    n = length(eargs)-offset
    for perm in permutations(collect(1+offset:length(eargs)))
        try
            for i in 1:n
                match(symbols, eargs[i+offset], margs[perm[i]])
            end
            return
        catch
            continue
        end
    end
    error("Can't match $expr with $formula")
end

match(symbols::Dict, expr, s::Symbol) = symbols[s] = expr
function match(symbols::Dict, expr::Expr, formula::Expr)
    partialmatch(expr, formula) || error("can't match $expr with $formula")
    match_args(symbols, expr, formula)
end
match(::Dict, expr, formula::Expr) = error("can't match $expr with $formula")

macro match(expr, formula)
    symbols = Dict{Symbol, Any}()
    matchto = "$formula"
    esc(quote
        try
            $match($symbols, $expr, parse($matchto))
            Nullable($symbols)
        catch e
            Nullable("In " * $matchto * " " * e.msg)
        end
    end)
end

end
