# SugarBLAS

[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/JuliaLang/IterativeSolvers.jl/blob/master/LICENSE)
[![codecov](https://codecov.io/gh/lopezm94/SugarBLAS.jl/coverage.svg?branch=master)](https://codecov.io/gh/lopezm94/SugarBLAS.jl)
[![Build Status](https://travis-ci.org/lopezm94/SugarBLAS.jl.svg?branch=master)](https://travis-ci.org/lopezm94/SugarBLAS.jl?branch=master)

`BLAS` functions are unaesthetic and annoying without good knowledge of the positional
arguments. This package provides macros for `BLAS` functions representing polynomials.
The main macro of the package is `@blas!`.

The macros will output a function from `BASE` module, this allows defining
new behavior for custom types. Note that the output won't necessarily belong to the
julia `BLAS` API, e.g. `copy!` is used instead of `BASE.LinAlg.BLAS.blascopy!`.

For now the package supports few of the functions and not all of the parameters, the
macros only receive the polynomial elements.


## Installing

To install the package, use the following command inside Julia's REPL:
```julia
Pkg.add("SugarBLAS")
```

## Usage

`@blas!` matches the expression and decides which function to call. As long as
it is correctly parenthesized putting more variables won't be an issue.

```julia
julia> macroexpand(:(@blas! Y = (a*b +c)*(X*Z) + Y))
:(Base.LinAlg.axpy!(a * b + c,X * Z,Y))

julia> macroexpand(:(@blas! X = (a+c)*X))
:(scale!(a + c,X))
```

When doing this just imagine the BLAS expression.

```julia
Y = a*X + Y
->
a := (a*b +c); X := (X*Z)
->
Y = (a*b +c)*(X*Z) + Y
```

### Commutativity

`+` is assumed as the only commutative operator, it is important to note here
that `*` is not treated as commutative and therefore some expressions will lead
to errors.

```julia
julia> a = 2.3;

julia> X = rand(10,10);

julia> Y = rand(10,10);

julia> @blas! Y += X*a
ERROR: MethodError: `axpy!` has no method matching axpy!(::Array{Float64,2}, ::Float64, ::Array{Float64,2})
```

The package assumes types by its position in the multiplication, this doesn't happen
with addition and that's why it conserves its property.

```julia
julia> macroexpand(:(@blas! Y = X + Y)) == macroexpand(:(@blas! Y = Y + X))
true
```

## Functions

- [scale](#scale)
- [scale!](#scale!)
- [axpy!](#axpy!)
- [copy!](#copy!)


### *scale*

**Polynomials**

- `a*X`

**Example**

```julia
julia> macroexpand(:(@blas a*X))
:(scale(a,X))
```


### *scale!*

**Polynomials**

- `X *= a`
- `X = a*X`

**Example**

```julia
julia> macroexpand(:(@blas! X *= a))
:(scale!(a,X))
```


### *axpy!*

**Polynomials**

- `Y += X`
- `Y = X + Y`
- `Y += a*X`
- `Y = a*X + Y`

**Example**

```julia
julia> macroexpand(:(@blas! Y += X))
:(Base.LinAlg.axpy!(1.0,X,Y))

julia> macroexpand(:(@blas! Y += a*X))
:(Base.LinAlg.axpy!(a,X,Y))
```


### *copy!*

**Polynomials**

- `X = Y`

**Example**

```julia
julia> macroexpand(:(@blas! X = Y))
:(copy!(X,Y))
```
