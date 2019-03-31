# SugarBLAS

[![codecov](https://codecov.io/gh/lopezm94/SugarBLAS.jl/coverage.svg?branch=master)](https://codecov.io/gh/lopezm94/SugarBLAS.jl)
[![Build Status](https://travis-ci.org/lopezm94/SugarBLAS.jl.svg?branch=master)](https://travis-ci.org/lopezm94/SugarBLAS.jl?branch=master)

`BLAS` functions are unaesthetic and annoying without good knowledge of the positional
arguments. This package provides macros for `BLAS` functions representing polynomials.
The main macro of the package is `@blas!` for most of the use cases: `copy!`, `scale!` and `axpy!`.
Non mutable versions of this operator are already very easy to write so they are not included.

The macros will output a function from `BASE` module, this allows defining
new behavior for custom types. Note that the output won't necessarily belong to the
julia `BLAS` API, e.g. `copy!` is used instead of `BASE.LinAlg.BLAS.blascopy!` for better performance.

For now the package supports the most common BLAS functions from the internal API. The access for these functions is private since the official API is private aswell and may change in the future.

This documentation offers great examples but is by no means super extensive, for more examples check the test folder of the repository.


## Installing

To install the package, use the following command inside Julia's REPL:
```julia
Pkg.add("SugarBLAS")
```

## Usage

`@blas!` matches the expression and decides which function to call. As long as
it is correctly parenthesized putting more variables won't be an issue.

```julia
julia> macroexpand(SugarBLAS, :(@blas! Y = (a*b +c)*(X*Z) + Y))
:(Base.LinAlg.axpy!(a * b + c,X * Z,Y))

julia> macroexpand(SugarBLAS, :(@blas! X = (a+c)*X))
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

### Updating operators

Both `*=` and `+=` are supported. `*=` can only be used for scaling given that is pretty unambigous.

```julia
julia> macroexpand(SugarBLAS, :(@blas! Y += X)) == macroexpand(SugarBLAS, :(@blas! Y = Y + X))
true
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
julia> macroexpand(SugarBLAS, :(@blas! Y = X + Y)) == macroexpand(SugarBLAS, :(@blas! Y = Y + X))
true
```


## Macros

### [*blas!*](#blas-1)
  - [scale!](#scale)
  - [axpy!](#axpy)
  - [copy!](#copy)

### [*Internal API*](#internal-api-1)
  - [scale!](#scale-1)
  - [axpy!](#axpy-1)
  - [copy!](#copy-1)
  - [ger!](#ger)
  - [syr!](#syr)
  - [syrk](#syrk)
  - [syrk!](#syrk-1)
  - [her!](#her)
  - [herk](#herk)
  - [herk!](#herk-1)
  - [gbmv](#gbmv)
  - [gbmv!](#gbmv-1)
  - [sbmv](#sbmv)
  - [sbmv!](#sbmv-1)
  - [gemm](#gemm)
  - [gemm!](#gemm-1)
  - [gemv](#gemv)
  - [gemv!](#gemv-1)
  - [symm](#symm)
  - [symm!](#symm-1)
  - [symv](#symv)
  - [symv!](#symv-1)


## *blas!*

Macro for most of the use cases: `copy!`, `scale!` and `axpy!`

### *scale!*

Scale an array `X` by a scalar `a` overwriting `X` in-place.

**Polynomials**

- `X *= a`
- `X = a*X`

**Example**

```julia
julia> macroexpand(SugarBLAS, :(@blas! X *= a))
:(scale!(a,X))
```


### *axpy!*

Overwrite `Y` with `a*X + Y`. Return `Y`.

**Polynomials**

- `Y += X`
- `Y += a*X`

**Example**

```julia
julia> macroexpand(SugarBLAS, :(@blas! Y += X))
:(Base.LinAlg.axpy!(1.0,X,Y))

julia> macroexpand(SugarBLAS, :(@blas! Y += a*X))
:(Base.LinAlg.axpy!(a,X,Y))
```


### *copy!*

Copy all elements from collection `Y` to array `X`. Return `X`.

**Polynomials**

- `X = Y`

**Example**

```julia
julia> macroexpand(SugarBLAS, :(@blas! X = Y))
:(copy!(X,Y))
```



## *Internal API*

Macro for most of the functions available in the JuliaLang internal BLAS API.

### *scale!*

Scale an array `X` by a scalar `a` overwriting `X` in-place.

**Polynomials**

- `X *= a`
- `X = a*X`

**Example**

```julia
julia> macroexpand(SugarBLAS, :(SugarBLAS.@scale! X *= a))
:(scale!(a,X))
```


### *axpy!*

Overwrite `Y` with `a*X + Y`. Return `Y`.

**Polynomials**

- `Y ±= X`
- `Y ±= a*X`

**Example**

```julia
julia> macroexpand(SugarBLAS, :(SugarBLAS.@axpy! Y += X))
:(Base.LinAlg.axpy!(1.0,X,Y))

julia> macroexpand(SugarBLAS, :(SugarBLAS.@axpy! Y += a*X))
:(Base.LinAlg.axpy!(a,X,Y))
```


### *copy!*

Copy all elements from collection `Y` to array `X`. Return `X`.

**Polynomials**

- `X = Y`

**Example**

```julia
julia> macroexpand(SugarBLAS, :(SugarBLAS.@copy! X = Y))
:(copy!(X,Y))
```


### *ger!*

Rank-1 update of the matrix `A` with vectors `x` and `y` as `alpha*x*y' + A`.

**Polynomials**

- `A ±= alpha*x*y'`

**Example**

```julia
julia> macroexpand(SugarBLAS, :(SugarBLAS.@ger! A -= alpha*x*y'))
:(Base.LinAlg.BLAS.ger!(-alpha,x,y,A))

julia> macroexpand(SugarBLAS, :(SugarBLAS.@ger! A += alpha*x*y'))
:(Base.LinAlg.BLAS.ger!(alpha,x,y,A))
```


### *syr!*

Rank-1 update of the symmetric matrix `A` with vector `x` as `alpha*x*(x)ᵀ + A`.
When left side has `A['U']` the upper triangle of `A` is updated (`'L'` for lower
triangle). Return `A`.

**Polynomials**

- `A[uplo] ±= alpha*x*(x)ᵀ`

**Example**

```julia
julia> macroexpand(SugarBLAS, :(SugarBLAS.@syr! A['U'] -= alpha*x*(x)ᵀ))
:(Base.LinAlg.BLAS.syr!('U',-alpha,x,A))

julia> macroexpand(SugarBLAS, :(SugarBLAS.@syr! A['L'] += alpha*x*(x)ᵀ))
:(Base.LinAlg.BLAS.syr!('L',alpha,x,A))
```


### *syrk*

Return either the upper triangle or the lower triangle, depending on
(`'U'` or `'L'`), of `alpha*A*(A)ᵀ` or `alpha*(A)ᵀ*A`.

**Polynomials**

- `alpha*A*(A)ᵀ uplo=ul`
- `alpha*(A)ᵀ*A uplo=ul`

**Example**

```julia
julia> macroexpand(SugarBLAS, :(SugarBLAS.@syrk alpha*A*(A)ᵀ uplo='U'))
:(Base.LinAlg.BLAS.syrk('U','N',alpha,A))

julia> macroexpand(SugarBLAS, :(SugarBLAS.@syrk alpha*(A)ᵀ*A uplo='L'))
:(Base.LinAlg.BLAS.syrk('L','T',alpha,A))
```


### *syrk!*

Rank-k update of the symmetric matrix `C` as `alpha*A*(A)ᵀ + beta*C` or
`alpha*(A)ᵀ*A + beta*C`. When the left hand side is`C['U']` the upper triangle of `C`
is updated (`'L'` for lower triangle). Return `C`.

**Polynomials**

- `C[uplo] ±= alpha*A*(A)ᵀ`
- `C[uplo] = beta*C ± alpha*(A)ᵀ*A`

**Example**

```julia
julia> macroexpand(SugarBLAS, :(SugarBLAS.@syrk! C['U'] -= alpha*A*(A)ᵀ))
:(Base.LinAlg.BLAS.syrk!('U','N',-alpha,A,1.0,C))

julia> macroexpand(SugarBLAS, :(SugarBLAS.@syrk! C['L'] = beta*C - alpha*(A)ᵀ*A))
:(Base.LinAlg.BLAS.syrk!('L','T',-alpha,A,beta,C))

julia> macroexpand(SugarBLAS, :(SugarBLAS.@syrk! C['U'] += alpha*A*(A)ᵀ))
:(Base.LinAlg.BLAS.syrk!('U','N',alpha,A,1.0,C))

julia> macroexpand(SugarBLAS, :(SugarBLAS.@syrk! C['L'] = alpha*(A)ᵀ*A + beta*C))
:(Base.LinAlg.BLAS.syrk!('L','T',alpha,A,beta,C))
```


### *her!*

Methods for complex arrays only. Rank-1 update of the Hermitian matrix `A`
with vector `x` as `alpha*x*x' + A`. Whenthe left hand side is `A['U']`
the upper triangle of `A` is updated (`'L'` for lower triangle). Return `A`.

**Polynomials**

- `A[uplo] ±= alpha*x*x'`

**Example**

```julia
julia> macroexpand(SugarBLAS, :(SugarBLAS.@her! A['U'] -= alpha*x*x'))
:(Base.LinAlg.BLAS.her!('U',-alpha,x,A))

julia> macroexpand(SugarBLAS, :(SugarBLAS.@her! A['L'] = A - alpha*x*x'))
:(Base.LinAlg.BLAS.her!('L',-alpha,x,A))

julia> macroexpand(SugarBLAS, :(SugarBLAS.@her! A['U'] += alpha*x*x'))
:(Base.LinAlg.BLAS.her!('U',alpha,x,A))
```


### *herk*

Methods for complex arrays only. Returns either the upper triangle or the
lower triangle, according to uplo ('U' or 'L'), of alpha*A*A' or alpha*A'*A,
according to trans ('N' or 'T').

**Polynomials**

- `alpha*A*A' uplo=ul`
- `alpha*A'*A uplo=ul`

**Example**

```julia
julia> macroexpand(SugarBLAS, :(SugarBLAS.@herk alpha*A*A' uplo='U'))
:(Base.LinAlg.BLAS.herk('U','N',alpha,A))

julia> macroexpand(SugarBLAS, :(SugarBLAS.@herk alpha*A'*A uplo='U'))
:(Base.LinAlg.BLAS.herk('U','T',alpha,A))

julia> macroexpand(SugarBLAS, :(SugarBLAS.@herk alpha*A*A' uplo='L'))
:(Base.LinAlg.BLAS.herk('L','N',alpha,A))

julia> macroexpand(SugarBLAS, :(SugarBLAS.@herk alpha*A'*A uplo='L'))
:(Base.LinAlg.BLAS.herk('L','T',alpha,A))
```


### *herk!*

Methods for complex arrays only. Rank-k update of the Hermitian matrix `C` as
`alpha*A*A' + beta*C` or `alpha*A'*A + beta*C`. When the left hand side is `C['U']`
the upper triangle of `C` is updated (`'L'` for lower triangle). Return `C`.

**Polynomials**

- `C[uplo] ±= alpha*A*A'`
- `C[uplo] = beta*C ± alpha*A'*A`

**Example**

```julia
julia> macroexpand(SugarBLAS, :(SugarBLAS.@herk! C['L'] -= alpha*A'*A))
:(Base.LinAlg.BLAS.herk!('L','T',-alpha,A,1.0,C))

julia> macroexpand(SugarBLAS, :(SugarBLAS.@herk! C['U'] = C - alpha*A*A'))
:(Base.LinAlg.BLAS.herk!('U','N',-alpha,A,1.0,C))

julia> macroexpand(SugarBLAS, :(SugarBLAS.@herk! C['L'] = beta*C - alpha*A'*A))
:(Base.LinAlg.BLAS.herk!('L','T',-alpha,A,beta,C))

julia> macroexpand(SugarBLAS, :(SugarBLAS.@herk! C['U'] += alpha*A*A'))
:(Base.LinAlg.BLAS.herk!('U','N',alpha,A,1.0,C))

julia> macroexpand(SugarBLAS, :(SugarBLAS.@herk! C['L'] = alpha*A'*A + beta*C))
:(Base.LinAlg.BLAS.herk!('L','T',alpha,A,beta,C))
```


### *gbmv*

Return `alpha*A*x` or `alpha*A'*x`. The matrix `A` is a general band matrix
of dimension `m` by `size(A,2)` with `kl` sub-diagonals and `ku` super-diagonals.

**Polynomials**

- `alpha*A[kl:ku,h=m]*x`
- `alpha*A[h=m,kl:ku]'*x`

**Example**

```julia
julia> macroexpand(SugarBLAS, :(SugarBLAS.@gbmv alpha*A[0:ku,h=2]*x))
:(Base.LinAlg.BLAS.gbmv('N',2,0,ku,alpha,A,x))

julia> macroexpand(SugarBLAS, :(SugarBLAS.@gbmv alpha*A[h=m,-kl:ku]*x))
:(Base.LinAlg.BLAS.gbmv('N',m,kl,ku,alpha,A,x))
```


### *gbmv!*

Update vector `y` as `alpha*A*x + beta*y` or `alpha*A'*x + beta*y`.
The matrix `A` is a general band matrix of dimension `m` by `size(A,2)` with
`kl` sub-diagonals and `ku` super-diagonals. Return the updated `y`.

**Polynomials**

- `y ±= alpha*A[kl:ku,h=m]*x`
- `y = beta*y ± alpha*A[h=m,kl:ku]'*x`

**Example**

```julia
@test macroexpand(SugarBLAS, :(SugarBLAS.@gbmv! y -= alpha*A[h=m,-kl:ku]*x))
:(Base.LinAlg.BLAS.gbmv!('N',m,kl,ku,-alpha,A,x,1.0,y))

@test macroexpand(SugarBLAS, :(SugarBLAS.@gbmv! y = beta*y - alpha*A[h=2, 0:ku]'*x))
:(Base.LinAlg.BLAS.gbmv!('T',2,0,ku,-alpha,A,x,beta,y))

@test macroexpand(SugarBLAS, :(SugarBLAS.@gbmv! y = alpha*A[0:ku,h=2]*x + y))
:(Base.LinAlg.BLAS.gbmv!('N',2,0,ku,alpha,A,x,1.0,y))

@test macroexpand(SugarBLAS, :(SugarBLAS.@gbmv! y += alpha*A[h=m,-kl:ku]*x))
:(Base.LinAlg.BLAS.gbmv!('N',m,kl,ku,alpha,A,x,1.0,y))

@test macroexpand(SugarBLAS, :(SugarBLAS.@gbmv! y = alpha*A[kl:ku, h=m]'*x + beta*y))
:(Base.LinAlg.BLAS.gbmv!('T',m,-kl,ku,alpha,A,x,beta,y))
```


### *sbmv*

Return `alpha*A*x` where `A` is a symmetric band matrix of order `size(A,2)` with
`k` super-diagonals stored in the argument `A`.


**Polynomials**

- `A[0:k,uplo]*xv`
- `alpha*A[0:k,uplo]*x`

**Example**

```julia
julia> macroexpand(SugarBLAS, :(SugarBLAS.@sbmv A['U',0:k]*x))
:(Base.LinAlg.BLAS.sbmv('U',k,A,x))

julia> macroexpand(SugarBLAS, :(SugarBLAS.@sbmv alpha*A[0:k,'L']*x))
:(Base.LinAlg.BLAS.sbmv('L',k,alpha,A,x))
```


### *sbmv!*

Update vector `y` as `alpha*A*x + beta*y` where `A` is a a symmetric band matrix
of order `size(A,2)` with `k` super-diagonals stored in the argument `A`. If
`A[...,'U']` is used multiplication is done with `A`'s upper triangle, `L` is for the
lower triangle. Return updated `y`.


**Polynomials**

- `y ±= alpha*A[0:k,uplo]*x`
- `y = beta*y ± alpha*A[0:k,uplo]*x`

**Example**

```julia
julia> macroexpand(SugarBLAS, :(SugarBLAS.@sbmv! y -= alpha*A['U',0:k]*x))
:(Base.LinAlg.BLAS.sbmv!('U',k,-alpha,A,x,1.0,y))

julia> macroexpand(SugarBLAS, :(SugarBLAS.@sbmv! y = beta*y - alpha*A[0:k,'U']*x))
:(Base.LinAlg.BLAS.sbmv!('U',k,-alpha,A,x,beta,y))

julia> macroexpand(SugarBLAS, :(SugarBLAS.@sbmv! y = beta*y - alpha*A[0:k,'L']*x))
:(Base.LinAlg.BLAS.sbmv!('L',k,-alpha,A,x,beta,y))

julia> macroexpand(SugarBLAS, :(SugarBLAS.@sbmv! y += alpha*A[0:k,'L']*x))
:(Base.LinAlg.BLAS.sbmv!('L',k,alpha,A,x,1.0,y))

julia> macroexpand(SugarBLAS, :(SugarBLAS.@sbmv! y = alpha*A['L',0:k]*x + beta*y))
:(Base.LinAlg.BLAS.sbmv!('L',k,alpha,A,x,beta,y))
```


### *gemm*

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

**Example**

```julia
@test macroexpand(SugarBLAS, :(SugarBLAS.@gemm alpha*A*B))
:(Base.LinAlg.BLAS.gemm('N','N',alpha,A,B))

@test macroexpand(SugarBLAS, :(SugarBLAS.@gemm A*B'))
:(Base.LinAlg.BLAS.gemm('N','T',A,B))
```


### *gemm!*

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

**Example**

```julia
julia> macroexpand(SugarBLAS, :(SugarBLAS.@gemm! C -= alpha*A*B))
:(Base.LinAlg.BLAS.gemm!('N','N',-alpha,A,B,1.0,C))

julia> macroexpand(SugarBLAS, :(SugarBLAS.@gemm! C = beta*C - alpha*A*B))
:(Base.LinAlg.BLAS.gemm!('N','N',-alpha,A,B,beta,C))

julia> macroexpand(SugarBLAS, :(SugarBLAS.@gemm! C += alpha*A*B))
:(Base.LinAlg.BLAS.gemm!('N','N',alpha,A,B,1.0,C))

julia> macroexpand(SugarBLAS, :(SugarBLAS.@gemm! C = 3.4*C - alpha*A'*B'))
:(Base.LinAlg.BLAS.gemm!('T','T',-alpha,A,B,3.4,C))

julia> macroexpand(SugarBLAS, :(SugarBLAS.@gemm! C = alpha*A'*B + beta*C))
:(Base.LinAlg.BLAS.gemm!('T','N',alpha,A,B,beta,C))
```


### *gemv*

Return `alpha*A*x` or `alpha*A'*x`.

**Polynomials**

- `A*x`
- `A'*x`
- `alpha*A*x`
- `alpha*A'*x`

**Example**

```julia
julia> macroexpand(SugarBLAS, :(SugarBLAS.@gemv A'*x))
:(Base.LinAlg.BLAS.gemv('T',A,x))

julia> macroexpand(SugarBLAS, :(SugarBLAS.@gemv alpha*A*x))
:(Base.LinAlg.BLAS.gemv('N',alpha,A,x))
```


### *gemv!*

Update the vector `y` as `alpha*A*x + beta*y` or `alpha*A'*x + beta*y`.
Return updated `y`.

**Polynomials**

- `y ±= alpha*A*x`
- `y ±= alpha*A'*x`
- `y = beta*y ± alpha*A*x`
- `y = beta*y ± alpha*A'*x`

**Example**

```julia
julia> macroexpand(SugarBLAS, :(SugarBLAS.@gemv! y -= alpha*A*x))
:(Base.LinAlg.BLAS.gemv!('N',-alpha,A,x,1.0,y))

julia> macroexpand(SugarBLAS, :(SugarBLAS.@gemv! y = beta*y - alpha*A*x))
:(Base.LinAlg.BLAS.gemv!('N',-alpha,A,x,beta,y))

julia> macroexpand(SugarBLAS, :(SugarBLAS.@gemv! y = beta*y - 1.5*A'*x))
:(Base.LinAlg.BLAS.gemv!('T',-1.5,A,x,beta,y))

julia> macroexpand(SugarBLAS, :(SugarBLAS.@gemv! y += alpha*A*x))
:(Base.LinAlg.BLAS.gemv!('N',alpha,A,x,1.0,y))

julia> macroexpand(SugarBLAS, :(SugarBLAS.@gemv! y = alpha*A*x + beta*y))
:(Base.LinAlg.BLAS.gemv!('N',alpha,A,x,beta,y))
```

### *symm*

Return `alpha*A*B` or `alpha*B*A` according to `"symm"`. `A` is assumed to be
symmetric. Only the `uplo` triangle of `A` is used (`'L'` for lower and `'U'` for upper).

**Polynomials**

- `A["symm", uplo]*B`
- `A*B["symm", uplo]`
- `alpha*A["symm", uplo]*B `
- `alpha*A*B["symm", uplo]`

**Example**

```julia
julia> macroexpand(SugarBLAS, :(SugarBLAS.@symm alpha*A["symm", 'L']*B))
:(Base.LinAlg.BLAS.symm('L','L',alpha,A,B))

julia> macroexpand(SugarBLAS, :(SugarBLAS.@symm A*B["symm", 'U']))
:(Base.LinAlg.BLAS.symm('R','U',A,B))
```

### *symm!*

Update `C` as `alpha*A*B + beta*C` or `alpha*B*A + beta*C` according to `"symm"`.
`A` is assumed to be symmetric. Only the `uplo` triangle of `A` is used
(`'L'` for lower and `'U'` for upper). Return updated `C`.

**Polynomials**

- `C = alpha*A["symm",uplo]*B`
- `C = alpha*A*B["symm",uplo]`
- `C = beta*C ± alpha*A["symm",uplo]*B`
- `C = beta*C ± alpha*A*B["symm",uplo]`

**Example**

```julia
julia> macroexpand(SugarBLAS, :(SugarBLAS.@symm! C -= alpha*A["symm", 'L']*B))
:(Base.LinAlg.BLAS.symm!('L','L',-alpha,A,B,1.0,C))

julia> macroexpand(SugarBLAS, :(SugarBLAS.@symm! C = C - alpha*A["symm", 'U']*B))
:(Base.LinAlg.BLAS.symm!('L','U',-alpha,A,B,1.0,C))

julia> macroexpand(SugarBLAS, :(SugarBLAS.@symm! C = beta*C - alpha*A["symm", 'L']*B))
:(Base.LinAlg.BLAS.symm!('L','L',-alpha,A,B,beta,C))
```

### *symv*

Return `alpha*A*x`. `A` is assumed to be symmetric. Only the `uplo` triangle of `A`
is used (`'L'` for lower and `'U'` for upper).

**Polynomials**

- `A[uplo]*x`
- `alpha*A[uplo]*x`

```julia
julia> macroexpand(SugarBLAS, :(SugarBLAS.@symv alpha*A['U']*x))
:(Base.LinAlg.BLAS.symv('U',alpha,A,x))

julia> macroexpand(SugarBLAS, :(SugarBLAS.@symv A['L']*x))
:(Base.LinAlg.BLAS.symv('L',A,x))
```

### *symv!*

Update the vector `y` as `alpha*A*x + beta*y`. `A` is assumed to be symmetric.
Only the `uplo` triangle of `A` is used (`'L'` for lower and `'U'` for upper).
Return updated y.

**Polynomials**

- `y ±= alpha*A[uplo]*x`
- `y = beta*y ± alpha*A[uplo]*x`

```julia
julia> macroexpand(SugarBLAS, :(SugarBLAS.@symv! y -= alpha*A['U']*x))
:(Base.LinAlg.BLAS.symv!('U',-alpha,A,x,1.0,y))

julia> macroexpand(SugarBLAS, :(SugarBLAS.@symv! y = y - alpha*A['L']*x))
(Base.LinAlg.BLAS.symv!('L',-alpha,A,x,1.0,y))

julia> macroexpand(SugarBLAS, :(SugarBLAS.@symv! y = beta*y + alpha*A['U']*x))
:(Base.LinAlg.BLAS.symv!('U',-alpha,A,x,beta,y))
```
