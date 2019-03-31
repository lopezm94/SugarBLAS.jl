"""
Extract expression sub-trees.
"""
module Match

export @match
export unkeyword!

iskw(expr::Expr) = expr.head == :kw
isref(expr::Expr) = expr.head == :ref
iscall(expr::Expr) = expr.head == :call

"""
Determines whether expression has commutative property.
"""
function iscommutative(op::Expr)
    iscall(op) && return iscommutative(op.args[1])
    isref(op) && return true
    false
end
iscommutative(op::Symbol) = _iscommutative(Val{op})

_iscommutative(::Type{Val{:(+)}}) = true
_iscommutative(::Type{T}) where T<:Val = false

"""
Output true if dictionary 'd' has a key 's' with a different value than 'v'.
Otherwise add value to dictionary and output false.
"""
function conflictadd!(d::Dict, s::Symbol, v)
    haskey(d,s) && (d[s] != v) && return true
    d[s] = v
    false
end

"""
Determine whether both expressions have the same head and arguments length.
"""
function partialmatch(expr::Expr, formula::Expr)
    samehead = (expr.head == formula.head)
    samelen = length(expr.args) == length(formula.args)
    (samehead & samelen) || return false
    true
end

"""
Match set of arguments from formula with expr.
"""
function match_args(symbols::Dict, expr::Expr, formula::Expr)
    offset = (iscall(expr) | isref(expr)) ? 1 : 0
    (!iscall(expr) | (expr.args[1] == formula.args[1])) || return false
    match_args(Val{iscommutative(expr)}, offset, symbols, expr, formula)
end
match_args(::Type{Val{false}}, kwargs...) = static_match(kwargs...)
match_args(::Type{Val{true}}, kwargs...) = commutative_match(kwargs...)

"""
Match each argument of formula with each argument of expr in an orderly fashion.
"""
function static_match(offset, symbols, expr, formula)
    eargs, margs = expr.args, formula.args
    (!isref(expr) | match(symbols, eargs[1], margs[1])) || return false
    for i in 1+offset:length(eargs)
        match(symbols, eargs[i], margs[i]) || return false
    end
    true
end

"""
Match each argument of formula with each argument of expr in any order.
"""
function commutative_match(offset, symbols, expr, formula)
    eargs, margs = expr.args, formula.args
    n = length(eargs)-offset
    (!isref(expr) | match(symbols, eargs[1], margs[1])) || return false
    indexes = collect(1+offset:length(eargs))
    perms = permutations(indexes)
    for perm in perms
        isempty(perm) && break
        success = true
        d = copy(symbols)
        for i in 1:n
            match(d, eargs[i+offset], margs[perm[i]]) || (success=false; break)
        end
        success && (merge!(symbols, d); return true)
    end
    false
end

"""
Match formula with expr. Overwrite dictionary with matched values.
"""
match(::Dict, expr, formula) = expr == formula
match(symbols::Dict, expr, s::Symbol) = !conflictadd!(symbols, s, expr)
function match(symbols::Dict, expr::Expr, formula::Expr)
    partialmatch(expr, formula) || return false
    match_args(symbols, expr, formula)
end
match(::Dict, expr, formula::Expr) = false

"""
Match formula with expr. Overwrite matched values directly to formula declared symbols.
"""
macro match(expr, formula)
    vars = getvars(formula)
    aux1 = "$formula"
    exec = quote end
    push!(exec.args, :(symbols = Dict{Symbol, Any}()))
    aux3 = esc(:($expr))
    push!(exec.args, :(success = $match(symbols, $aux3, Meta.parse($aux1))))
    for var in vars
        aux2 = "$var"
        aux3 = esc(:($var))
        push!(exec.args, :(success && ($aux3 = symbols[Meta.parse($aux2)])))
    end
    push!(exec.args, :(success))
    exec
end

"""
Get leaf symbols from expression.
"""
getvars(expr::Symbol) = _getvars(expr)
getvars(expr::Expr) = _getvars(unkeyword!(expr))

_getvars(::Any) = Set{Symbol}()
_getvars(s::Symbol) = Set{Symbol}([s])
function _getvars(expr::Expr)
    start = (iscall(expr) | iskw(expr)) ? 2 : 1
    union(map(_getvars, expr.args[start:end])...)
end

"""
Transform keyword to assignment.
"""
function unkeyword!(expr::Expr)
    iskw(expr) && (expr.head = :(=))
    expr
end

struct Permutations{T}
    a::T
    t::Int
end

Base.eltype(::Type{Permutations{T}}) where {T} = Vector{eltype(T)}

Base.length(p::Permutations) = (0 <= p.t <= length(p.a)) ? factorial(length(p.a), length(p.a)-p.t) : 0

"""
    permutations(a)
Generate all permutations of an indexable object `a` in lexicographic order. Because the number of permutations
can be very large, this function returns an iterator object.
Use `collect(permutations(a))` to get an array of all permutations.
"""
permutations(a) = Permutations(a, length(a))

"""
    permutations(a, t)
Generate all size `t` permutations of an indexable object `a`.
"""
function permutations(a, t::Integer)
    if t < 0
        t = length(a) + 1
    end
    Permutations(a, t)
end

function Base.iterate(p::Permutations, s = collect(1:length(p.a)))
    (!isempty(s) && max(s[1], p.t) > length(p.a) || (isempty(s) && p.t > 0)) && return
    nextpermutation(p.a, p.t ,s)
end

function nextpermutation(m, t, state)
    perm = [m[state[i]] for i in 1:t]
    n = length(state)
    if t <= 0
        return(perm, [n+1])
    end
    s = copy(state)
    if t < n
        j = t + 1
        while j <= n &&  s[t] >= s[j]; j+=1; end
    end
    if t < n && j <= n
        s[t], s[j] = s[j], s[t]
    else
        if t < n
            reverse!(s, t+1)
        end
        i = t - 1
        while i>=1 && s[i] >= s[i+1]; i -= 1; end
        if i > 0
            j = n
            while j>i && s[i] >= s[j]; j -= 1; end
            s[i], s[j] = s[j], s[i]
            reverse!(s, i+1)
        else
            s[1] = n+1
        end
    end
    return (perm, s)
end

end
