module MathMatch

export @match

function iscommutative(op::Expr)
    iscall(op) && return iscommutative(op.args[1])
    isref(op) && return true
    false
end
iscommutative(op::Symbol) = _iscommutative(Val{op})

_iscommutative(::Type{Val{:(+)}}) = true
_iscommutative{T<:Val}(::Type{T}) = false

permutations(r::Range) = permutations(collect(r))
permutations(v::Vector) = @task permfactory(v)

function permfactory{T}(v::Vector{T})
    stack = Vector{Tuple{Vector{T}, Vector{T}}}()
    push!(stack, (Vector{T}(), v))
    while !isempty(stack)
        state = pop!(stack)
        taken = copy(state[1])
        left = copy(state[2])
        isempty(left) && (produce(taken); continue)
        for i in 1:length(left)
            new_state = (push!(copy(taken), left[i]), vcat(left[1:i-1],left[i+1:end]))
            push!(stack, new_state)
        end
    end
end

#Overwrite d
#Returns false if successfull, true otherwise
function conflictadd!(d::Dict, s::Symbol, v)
    haskey(d,s) && (d[s] != v) && return true
    d[s] = v
    false
end

function partialmatch(expr::Expr, formula::Expr)
    samehead = (expr.head == formula.head)
    samelen = length(expr.args) == length(formula.args)
    (samehead & samelen) || return false
    true
end

function match_args(symbols::Dict, expr::Expr, formula::Expr)
    offset = (iscall(expr) | isref(expr)) ? 1 : 0
    (!iscall(expr) | (expr.args[1] == formula.args[1])) || return false
    _match_args(Val{iscommutative(expr)}, offset, symbols, expr, formula)
end

function _match_args(::Type{Val{false}}, offset, symbols, expr, formula)
    eargs, margs = expr.args, formula.args
    (!isref(expr) | match(symbols, eargs[1], margs[1])) || return false
    for i in 1+offset:length(eargs)
        match(symbols, eargs[i], margs[i]) || return false
    end
    true
end
function _match_args(::Type{Val{true}}, offset, symbols, expr, formula)
    eargs, margs = expr.args, formula.args
    n = length(eargs)-offset
    (!isref(expr) | match(symbols, eargs[1], margs[1])) || return false
    for perm in permutations(1+offset:length(eargs))
        success = true
        d = copy(symbols)
        for i in 1:n
            match(d, eargs[i+offset], margs[perm[i]]) || (success=false; break)
        end
        success && (merge!(symbols, d); return true)
    end
    false
end

match(::Dict, expr, formula) = expr == formula
match(symbols::Dict, expr, s::Symbol) = !conflictadd!(symbols, s, expr)
function match(symbols::Dict, expr::Expr, formula::Expr)
    partialmatch(expr, formula) || return false
    match_args(symbols, expr, formula)
end
match(::Dict, expr, formula::Expr) = false

macro match(expr, formula)
    vars = getvars(formula)
    aux1 = "$formula"
    exec = quote end
    push!(exec.args, :(symbols = Dict{Symbol, Any}()))
    aux3 = esc(:($expr))
    push!(exec.args, :(success = $match(symbols, $aux3, parse($aux1))))
    for var in vars
        aux2 = "$var"
        aux3 = esc(:($var))
        push!(exec.args, :(success && ($aux3 = symbols[parse($aux2)])))
    end
    push!(exec.args, :(success))
    exec
end

_getvars(::Any) = Set{Symbol}()
_getvars(s::Symbol) = Set{Symbol}([s])
function _getvars(expr::Expr)
    start = (iscall(expr) | iskw(expr)) ? 2 : 1
    union(map(_getvars, expr.args[start:end])...)
end

getvars(expr::Symbol) = _getvars(expr)
getvars(expr::Expr) = _getvars(unkeyword!(expr))

function unkeyword!(expr::Expr)
    iskw(expr) && (expr.head = :(=))
    expr
end

iskw(expr::Expr) = expr.head == :kw
isref(expr::Expr) = expr.head == :ref
iscall(expr::Expr) = expr.head == :call

end
