using SugarBLAS
using Base.Test

#scale
@test macroexpand(:(@blas a*X)) == :(scale(a, X))

#scale!
@test macroexpand(:(@blas! X *= a)) == :(scale!(a, X))
@test macroexpand(:(@blas! X = a*X)) == :(scale!(a, X))

#axpy!
@test macroexpand(:(@blas! Y += a*X)) == :(Base.LinAlg.axpy!(a, X, Y))
@test macroexpand(:(@blas! Y = a*X + Y)) == :(Base.LinAlg.axpy!(a, X, Y))
@test macroexpand(:(@blas! Y = Y + a*X)) == :(Base.LinAlg.axpy!(a, X, Y))
@test macroexpand(:(@blas! Y = X + Y)) == :(Base.LinAlg.axpy!(1.0, X, Y))
@test macroexpand(:(@blas! Y = Y + X)) == :(Base.LinAlg.axpy!(1.0, X, Y))
@test macroexpand(:(@blas! Y += X)) == :(Base.LinAlg.axpy!(1.0, X, Y))

#copy!
@test macroexpand(:(@blas! X = Y)) == :(copy!(X, Y))
