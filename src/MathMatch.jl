module MathMatch

export @match

iscommutative(op::Expr) = iscommutative(op.args[1])
iscommutative(op::Symbol) = _iscommutative(Val{op})

_iscommutative(::Type{Val{:(+)}}) = true
_iscommutative(::Type{Val{:(*)}}) = false
_iscommutative(::Type{Val{:(-)}}) = false
_iscommutative(::Type{Val{:(/)}}) = false
_iscommutative{T<:Val}(::Type{T}) = false

iscall(expr::Expr) = expr.head == :call

function clear!(d::Dict)
    for key in keys(d)
        delete!(d, key)
    end
end

#Overwrite d1
#Returns false if successfull, true otherwise
function conflictmerge!(d1::Dict, d2::Dict)
    for key in keys(d2)
        haskey(d1, key) && (d1[key] != d2[key]) && return true
        d1[key] = d2[key]
    end
    false
end

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
        d = Dict()
        match(d, eargs[i], margs[i]) || return false
        (conflict = conflictmerge!(symbols, d)) && return false
    end
    true
end
function _match_args(::Type{Val{true}}, offset, symbols, expr, formula)
    eargs, margs = expr.args, formula.args
    n = length(eargs)-offset
    for perm in permutations(collect(1+offset:length(eargs)))
        success = true
        d1, d2 = Dict(), Dict()
        for i in 1:n
            match(d2, eargs[i+offset], margs[perm[i]]) || (success=false; break)
            (conflict=conflictmerge!(d1, d2)) && (success=false; break)
        end
        success && (merge!(symbols, d1); return true)
    end
    false
end

match(symbols::Dict, expr, s::Symbol) = (symbols[s] = expr; true)
function match(symbols::Dict, expr::Expr, formula::Expr)
    partialmatch(expr, formula) || return false
    match_args(symbols, expr, formula)
end
match(::Dict, expr, formula::Expr) = false

macro match(expr, formula)
    symbols = Dict{Symbol, Any}()
    matchto = "$formula"
    esc(quote
        $clear!($symbols)
        if $match($symbols, $expr, parse($matchto))
            Nullable($symbols)
        else
            Nullable()
        end
    end)
end

end
