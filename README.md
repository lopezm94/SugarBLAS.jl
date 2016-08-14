# SugarBLAS

[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/JuliaLang/IterativeSolvers.jl/blob/master/LICENSE)
[![Coverage Status](https://coveralls.io/repos/github/lopezm94/SugarBLAS.jl/badge.svg?branch=master)](https://coveralls.io/github/lopezm94/SugarBLAS.jl?branch=master)
[![codecov](https://codecov.io/gh/lopezm94/SugarBLAS.jl/coverage.svg?branch=master)](https://codecov.io/gh/lopezm94/SugarBLAS.jl)
[![Build Status](https://travis-ci.org/lopezm94/SugarBLAS.jl.svg?branch=master)](https://travis-ci.org/lopezm94/SugarBLAS.jl?branch=master)
[![IterativeSolvers](http://pkg.julialang.org/badges/IterativeSolvers_0.4.svg)](http://pkg.julialang.org/?pkg=IterativeSolvers&ver=0.4)
[![IterativeSolvers](http://pkg.julialang.org/badges/IterativeSolvers_0.5.svg)](http://pkg.julialang.org/?pkg=IterativeSolvers&ver=0.5)

`BLAS` functions are unaesthetic and annoying without good knowledge of the positional
arguments. This package provides macros for `BLAS` functions representing polynomials.

There are two main macros, `@blas` and `@blas!` depending whether and argument is
overwritten or not.

The macros will output a function from `BASE` module, this allows defining
new behavior for custom types. Note that the output won't necessarily belong to the
julia `BLAS` API, e.g. `copy!` is used instead of `BASE.LinAlg.BLAS.blascopy!`.

For now the package supports few of the functions and not all of the parameters, the
macros only receive the polynomial elements.

*Note:* Commutative operators are supported (Only `+` is such operator).

```julia
julia> macroexpand(:(@blas! Y = X + Y)) == macroexpand(:(@blas! Y = Y + X))
true
```

## Installing

To install the package, use the following command inside Julia's REPL:
```julia
Pkg.add("SugarBLAS")
```

## Functions

- [scale](#scale)
- [scale!](#scale!)
- [axpy!](#axpy!)
- [copy!](#copy!)

### scale

**Polynomials**

- `a*X`

**Example**

```julia
julia> macroexpand(:(@blas a*X))
:(scale(a,X))
```

### scale!

**Polynomials**

- `X *= a`
- `X = a*X`

**Example**

```julia
julia> macroexpand(:(@blas! X *= a))
:(scale!(a,X))
```

### axpy!

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

### copy!

**Polynomials**

- `X = Y`

**Example**

```julia
julia> macroexpand(:(@blas! X = Y))
:(copy!(X,Y))
```
