using SugarBLAS
using Base.Test

#scale!
@test macroexpand(:(SugarBLAS.@scale! X *= a)) == :(scale!(a, X))
@test macroexpand(:(SugarBLAS.@scale! X = a*X)) == :(scale!(a, X))
@test macroexpand(:(@blas! X *= a)) == :(scale!(a, X))
@test macroexpand(:(@blas! X = a*X)) == :(scale!(a, X))

#axpy!
@test macroexpand(:(SugarBLAS.@axpy! Y -= a*X)) == :(Base.LinAlg.axpy!(-a, X, Y))
@test macroexpand(:(SugarBLAS.@axpy! Y = Y - a*X)) == :(Base.LinAlg.axpy!(-a, X, Y))
@test macroexpand(:(SugarBLAS.@axpy! Y = Y - X)) == :(Base.LinAlg.axpy!(-1.0, X, Y))
@test macroexpand(:(SugarBLAS.@axpy! Y -= X)) == :(Base.LinAlg.axpy!(-1.0, X, Y))
@test macroexpand(:(SugarBLAS.@axpy! Y += a*X)) == :(Base.LinAlg.axpy!(a, X, Y))
@test macroexpand(:(SugarBLAS.@axpy! Y = a*X + Y)) == :(Base.LinAlg.axpy!(a, X, Y))
@test macroexpand(:(SugarBLAS.@axpy! Y = Y + a*X)) == :(Base.LinAlg.axpy!(a, X, Y))
@test macroexpand(:(SugarBLAS.@axpy! Y = X + Y)) == :(Base.LinAlg.axpy!(1.0, X, Y))
@test macroexpand(:(SugarBLAS.@axpy! Y = Y + X)) == :(Base.LinAlg.axpy!(1.0, X, Y))
@test macroexpand(:(SugarBLAS.@axpy! Y += X)) == :(Base.LinAlg.axpy!(1.0, X, Y))
@test macroexpand(:(@blas! Y -= a*X)) == :(Base.LinAlg.axpy!(-a, X, Y))
@test macroexpand(:(@blas! Y = Y - a*X)) == :(Base.LinAlg.axpy!(-a, X, Y))
@test macroexpand(:(@blas! Y = Y - X)) == :(Base.LinAlg.axpy!(-1.0, X, Y))
@test macroexpand(:(@blas! Y -= X)) == :(Base.LinAlg.axpy!(-1.0, X, Y))
@test macroexpand(:(@blas! Y += a*X)) == :(Base.LinAlg.axpy!(a, X, Y))
@test macroexpand(:(@blas! Y = a*X + Y)) == :(Base.LinAlg.axpy!(a, X, Y))
@test macroexpand(:(@blas! Y = Y + a*X)) == :(Base.LinAlg.axpy!(a, X, Y))
@test macroexpand(:(@blas! Y = X + Y)) == :(Base.LinAlg.axpy!(1.0, X, Y))
@test macroexpand(:(@blas! Y = Y + X)) == :(Base.LinAlg.axpy!(1.0, X, Y))
@test macroexpand(:(@blas! Y += X)) == :(Base.LinAlg.axpy!(1.0, X, Y))

#copy!
@test macroexpand(:(SugarBLAS.@copy! X = Y)) == :(copy!(X, Y))
@test macroexpand(:(@blas! X = Y)) == :(copy!(X, Y))

#ger!
@test macroexpand(:(SugarBLAS.@ger! A -= alpha*x*y')) == :(Base.LinAlg.BLAS.ger!(-alpha,x,y,A))
@test macroexpand(:(SugarBLAS.@ger! A = A - alpha*x*y')) == :(Base.LinAlg.BLAS.ger!(-alpha,x,y,A))
@test macroexpand(:(SugarBLAS.@ger! A += alpha*x*y')) == :(Base.LinAlg.BLAS.ger!(alpha,x,y,A))
@test macroexpand(:(SugarBLAS.@ger! A = alpha*x*y' + A)) == :(Base.LinAlg.BLAS.ger!(alpha,x,y,A))

#syr!
@test macroexpand(:(SugarBLAS.@syr! A['U'] -= alpha*x*x.')) == :(Base.LinAlg.BLAS.syr!('U',-alpha,x,A))
@test macroexpand(:(SugarBLAS.@syr! A['L'] -= alpha*x*x.')) == :(Base.LinAlg.BLAS.syr!('L',-alpha,x,A))
@test macroexpand(:(SugarBLAS.@syr! A['U'] = A - alpha*x*x.')) == :(Base.LinAlg.BLAS.syr!('U',-alpha,x,A))
@test macroexpand(:(SugarBLAS.@syr! A['L'] = A - alpha*x*x.')) == :(Base.LinAlg.BLAS.syr!('L',-alpha,x,A))
@test macroexpand(:(SugarBLAS.@syr! A['U'] += alpha*x*x.')) == :(Base.LinAlg.BLAS.syr!('U',alpha,x,A))
@test macroexpand(:(SugarBLAS.@syr! A['L'] += alpha*x*x.')) == :(Base.LinAlg.BLAS.syr!('L',alpha,x,A))
@test macroexpand(:(SugarBLAS.@syr! A['U'] = alpha*x*x.' + A)) == :(Base.LinAlg.BLAS.syr!('U',alpha,x,A))
@test macroexpand(:(SugarBLAS.@syr! A['L'] = alpha*x*x.' + A)) == :(Base.LinAlg.BLAS.syr!('L',alpha,x,A))

#syrk
@test macroexpand(:(SugarBLAS.@syrk alpha*A*A.' uplo='U')) == :(Base.LinAlg.BLAS.syrk('U','N',alpha,A))
@test macroexpand(:(SugarBLAS.@syrk alpha*A.'*A uplo='U')) == :(Base.LinAlg.BLAS.syrk('U','T',alpha,A))
@test macroexpand(:(SugarBLAS.@syrk alpha*A*A.' uplo='L')) == :(Base.LinAlg.BLAS.syrk('L','N',alpha,A))
@test macroexpand(:(SugarBLAS.@syrk alpha*A.'*A uplo='L')) == :(Base.LinAlg.BLAS.syrk('L','T',alpha,A))

#syrk!
@test macroexpand(:(SugarBLAS.@syrk! C['U'] -= alpha*A*A.')) == :(Base.LinAlg.BLAS.syrk!('U','N',-alpha,A,1.0,C))
@test macroexpand(:(SugarBLAS.@syrk! C['U'] -= alpha*A.'*A)) == :(Base.LinAlg.BLAS.syrk!('U','T',-alpha,A,1.0,C))
@test macroexpand(:(SugarBLAS.@syrk! C['U'] = C - alpha*A*A.')) == :(Base.LinAlg.BLAS.syrk!('U','N',-alpha,A,1.0,C))
@test macroexpand(:(SugarBLAS.@syrk! C['U'] = C - alpha*A.'*A)) == :(Base.LinAlg.BLAS.syrk!('U','T',-alpha,A,1.0,C))
@test macroexpand(:(SugarBLAS.@syrk! C['U'] = beta*C - alpha*A*A.')) == :(Base.LinAlg.BLAS.syrk!('U','N',-alpha,A,beta,C))
@test macroexpand(:(SugarBLAS.@syrk! C['L'] = beta*C - alpha*A.'*A)) == :(Base.LinAlg.BLAS.syrk!('L','T',-alpha,A,beta,C))
@test macroexpand(:(SugarBLAS.@syrk! C['U'] += alpha*A*A.')) == :(Base.LinAlg.BLAS.syrk!('U','N',alpha,A,1.0,C))
@test macroexpand(:(SugarBLAS.@syrk! C['U'] += alpha*A.'*A)) == :(Base.LinAlg.BLAS.syrk!('U','T',alpha,A,1.0,C))
@test macroexpand(:(SugarBLAS.@syrk! C['U'] = alpha*A*A.' + C)) == :(Base.LinAlg.BLAS.syrk!('U','N',alpha,A,1.0,C))
@test macroexpand(:(SugarBLAS.@syrk! C['U'] = C + alpha*A.'*A)) == :(Base.LinAlg.BLAS.syrk!('U','T',alpha,A,1.0,C))
@test macroexpand(:(SugarBLAS.@syrk! C['U'] = alpha*A*A.' + beta*C)) == :(Base.LinAlg.BLAS.syrk!('U','N',alpha,A,beta,C))
@test macroexpand(:(SugarBLAS.@syrk! C['L'] = alpha*A.'*A + beta*C)) == :(Base.LinAlg.BLAS.syrk!('L','T',alpha,A,beta,C))

#her!
@test macroexpand(:(SugarBLAS.@her! A['U'] -= alpha*x*x')) == :(Base.LinAlg.BLAS.her!('U',-alpha,x,A))
@test macroexpand(:(SugarBLAS.@her! A['L'] -= alpha*x*x')) == :(Base.LinAlg.BLAS.her!('L',-alpha,x,A))
@test macroexpand(:(SugarBLAS.@her! A['U'] = A - alpha*x*x')) == :(Base.LinAlg.BLAS.her!('U',-alpha,x,A))
@test macroexpand(:(SugarBLAS.@her! A['L'] = A - alpha*x*x')) == :(Base.LinAlg.BLAS.her!('L',-alpha,x,A))
@test macroexpand(:(SugarBLAS.@her! A['U'] = alpha*x*x' + A)) == :(Base.LinAlg.BLAS.her!('U',alpha,x,A))
@test macroexpand(:(SugarBLAS.@her! A['L'] = alpha*x*x' + A)) == :(Base.LinAlg.BLAS.her!('L',alpha,x,A))
@test macroexpand(:(SugarBLAS.@her! A['U'] += alpha*x*x')) == :(Base.LinAlg.BLAS.her!('U',alpha,x,A))
@test macroexpand(:(SugarBLAS.@her! A['L'] += alpha*x*x')) == :(Base.LinAlg.BLAS.her!('L',alpha,x,A))

#herk
@test macroexpand(:(SugarBLAS.@herk alpha*A*A' uplo='U')) == :(Base.LinAlg.BLAS.herk('U','N',alpha,A))
@test macroexpand(:(SugarBLAS.@herk alpha*A'*A uplo='U')) == :(Base.LinAlg.BLAS.herk('U','T',alpha,A))
@test macroexpand(:(SugarBLAS.@herk alpha*A*A' uplo='L')) == :(Base.LinAlg.BLAS.herk('L','N',alpha,A))
@test macroexpand(:(SugarBLAS.@herk alpha*A'*A uplo='L')) == :(Base.LinAlg.BLAS.herk('L','T',alpha,A))

#herk!
@test macroexpand(:(SugarBLAS.@herk! C['U'] -= alpha*A*A')) == :(Base.LinAlg.BLAS.herk!('U','N',-alpha,A,1.0,C))
@test macroexpand(:(SugarBLAS.@herk! C['L'] -= alpha*A'*A)) == :(Base.LinAlg.BLAS.herk!('L','T',-alpha,A,1.0,C))
@test macroexpand(:(SugarBLAS.@herk! C['U'] = C - alpha*A*A')) == :(Base.LinAlg.BLAS.herk!('U','N',-alpha,A,1.0,C))
@test macroexpand(:(SugarBLAS.@herk! C['L'] = C - alpha*A'*A)) == :(Base.LinAlg.BLAS.herk!('L','T',-alpha,A,1.0,C))
@test macroexpand(:(SugarBLAS.@herk! C['U'] = beta*C - alpha*A*A')) == :(Base.LinAlg.BLAS.herk!('U','N',-alpha,A,beta,C))
@test macroexpand(:(SugarBLAS.@herk! C['L'] = beta*C - alpha*A'*A)) == :(Base.LinAlg.BLAS.herk!('L','T',-alpha,A,beta,C))
@test macroexpand(:(SugarBLAS.@herk! C['U'] += alpha*A*A')) == :(Base.LinAlg.BLAS.herk!('U','N',alpha,A,1.0,C))
@test macroexpand(:(SugarBLAS.@herk! C['L'] += alpha*A'*A)) == :(Base.LinAlg.BLAS.herk!('L','T',alpha,A,1.0,C))
@test macroexpand(:(SugarBLAS.@herk! C['U'] = alpha*A*A' + C)) == :(Base.LinAlg.BLAS.herk!('U','N',alpha,A,1.0,C))
@test macroexpand(:(SugarBLAS.@herk! C['L'] = C + alpha*A'*A)) == :(Base.LinAlg.BLAS.herk!('L','T',alpha,A,1.0,C))
@test macroexpand(:(SugarBLAS.@herk! C['U'] = alpha*A*A' + beta*C)) == :(Base.LinAlg.BLAS.herk!('U','N',alpha,A,beta,C))
@test macroexpand(:(SugarBLAS.@herk! C['L'] = alpha*A'*A + beta*C)) == :(Base.LinAlg.BLAS.herk!('L','T',alpha,A,beta,C))

#gbmv
@test macroexpand(:(SugarBLAS.@gbmv alpha*A[0:ku,h=2]*x)) == :(Base.LinAlg.BLAS.gbmv('N',2,0,ku,alpha,A,x))
@test macroexpand(:(SugarBLAS.@gbmv alpha*A[h=m,-kl:ku]*x)) == :(Base.LinAlg.BLAS.gbmv('N',m,kl,ku,alpha,A,x))

#gbmv!
@test macroexpand(:(SugarBLAS.@gbmv! y = y - alpha*A[0:ku,h=2]*x)) == :(Base.LinAlg.BLAS.gbmv!('N',2,0,ku,-alpha,A,x,1.0,y))
@test macroexpand(:(SugarBLAS.@gbmv! y -= alpha*A[h=m,-kl:ku]*x)) == :(Base.LinAlg.BLAS.gbmv!('N',m,kl,ku,-alpha,A,x,1.0,y))
@test macroexpand(:(SugarBLAS.@gbmv! y = beta*y - alpha*A[0:ku,h=2]*x)) == :(Base.LinAlg.BLAS.gbmv!('N',2,0,ku,-alpha,A,x,beta,y))
@test macroexpand(:(SugarBLAS.@gbmv! y = beta*y - alpha*A[h=m,-kl:ku]*x)) == :(Base.LinAlg.BLAS.gbmv!('N',m,kl,ku,-alpha,A,x,beta,y))
@test macroexpand(:(SugarBLAS.@gbmv! y = beta*y - alpha*A[h=2, 0:ku]'*x)) == :(Base.LinAlg.BLAS.gbmv!('T',2,0,ku,-alpha,A,x,beta,y))
@test macroexpand(:(SugarBLAS.@gbmv! y = beta*y - alpha*A[kl:ku, h=m]'*x)) == :(Base.LinAlg.BLAS.gbmv!('T',m,-kl,ku,-alpha,A,x,beta,y))
@test macroexpand(:(SugarBLAS.@gbmv! y = alpha*A[0:ku,h=2]*x + y)) == :(Base.LinAlg.BLAS.gbmv!('N',2,0,ku,alpha,A,x,1.0,y))
@test macroexpand(:(SugarBLAS.@gbmv! y += alpha*A[h=m,-kl:ku]*x)) == :(Base.LinAlg.BLAS.gbmv!('N',m,kl,ku,alpha,A,x,1.0,y))
@test macroexpand(:(SugarBLAS.@gbmv! y = y + alpha*A[h=m,-kl:ku]*x)) == :(Base.LinAlg.BLAS.gbmv!('N',m,kl,ku,alpha,A,x,1.0,y))
@test macroexpand(:(SugarBLAS.@gbmv! y = alpha*A[0:ku,h=2]*x + beta*y)) == :(Base.LinAlg.BLAS.gbmv!('N',2,0,ku,alpha,A,x,beta,y))
@test macroexpand(:(SugarBLAS.@gbmv! y = alpha*A[h=m,-kl:ku]*x + beta*y)) == :(Base.LinAlg.BLAS.gbmv!('N',m,kl,ku,alpha,A,x,beta,y))
@test macroexpand(:(SugarBLAS.@gbmv! y = alpha*A[h=2, 0:ku]'*x + beta*y)) == :(Base.LinAlg.BLAS.gbmv!('T',2,0,ku,alpha,A,x,beta,y))
@test macroexpand(:(SugarBLAS.@gbmv! y = alpha*A[kl:ku, h=m]'*x + beta*y)) == :(Base.LinAlg.BLAS.gbmv!('T',m,-kl,ku,alpha,A,x,beta,y))

#sbmv
@test macroexpand(:(SugarBLAS.@sbmv A['U',0:k]*x)) == :(Base.LinAlg.BLAS.sbmv('U',k,A,x))
@test macroexpand(:(SugarBLAS.@sbmv alpha*A[0:k,'L']*x)) == :(Base.LinAlg.BLAS.sbmv('L',k,alpha,A,x))

#sbmv!
@test macroexpand(:(SugarBLAS.@sbmv! y -= alpha*A['U',0:k]*x)) == :(Base.LinAlg.BLAS.sbmv!('U',k,-alpha,A,x,1.0,y))
@test macroexpand(:(SugarBLAS.@sbmv! y -= alpha*A[0:k,'L']*x)) == :(Base.LinAlg.BLAS.sbmv!('L',k,-alpha,A,x,1.0,y))
@test macroexpand(:(SugarBLAS.@sbmv! y = y - alpha*A['U',0:k]*x)) == :(Base.LinAlg.BLAS.sbmv!('U',k,-alpha,A,x,1.0,y))
@test macroexpand(:(SugarBLAS.@sbmv! y = beta*y - alpha*A['U',0:k]*x)) == :(Base.LinAlg.BLAS.sbmv!('U',k,-alpha,A,x,beta,y))
@test macroexpand(:(SugarBLAS.@sbmv! y = beta*y - alpha*A[0:k,'U']*x)) == :(Base.LinAlg.BLAS.sbmv!('U',k,-alpha,A,x,beta,y))
@test macroexpand(:(SugarBLAS.@sbmv! y = beta*y - alpha*A['L',0:k]*x)) == :(Base.LinAlg.BLAS.sbmv!('L',k,-alpha,A,x,beta,y))
@test macroexpand(:(SugarBLAS.@sbmv! y = beta*y - alpha*A[0:k,'L']*x)) == :(Base.LinAlg.BLAS.sbmv!('L',k,-alpha,A,x,beta,y))
@test macroexpand(:(SugarBLAS.@sbmv! y += alpha*A['U',0:k]*x)) == :(Base.LinAlg.BLAS.sbmv!('U',k,alpha,A,x,1.0,y))
@test macroexpand(:(SugarBLAS.@sbmv! y += alpha*A[0:k,'L']*x)) == :(Base.LinAlg.BLAS.sbmv!('L',k,alpha,A,x,1.0,y))
@test macroexpand(:(SugarBLAS.@sbmv! y = alpha*A['U',0:k]*x + y)) == :(Base.LinAlg.BLAS.sbmv!('U',k,alpha,A,x,1.0,y))
@test macroexpand(:(SugarBLAS.@sbmv! y = y + alpha*A[0:k,'L']*x)) == :(Base.LinAlg.BLAS.sbmv!('L',k,alpha,A,x,1.0,y))
@test macroexpand(:(SugarBLAS.@sbmv! y = alpha*A['U',0:k]*x + beta*y)) == :(Base.LinAlg.BLAS.sbmv!('U',k,alpha,A,x,beta,y))
@test macroexpand(:(SugarBLAS.@sbmv! y = beta*y + alpha*A[0:k,'U']*x)) == :(Base.LinAlg.BLAS.sbmv!('U',k,alpha,A,x,beta,y))
@test macroexpand(:(SugarBLAS.@sbmv! y = alpha*A['L',0:k]*x + beta*y)) == :(Base.LinAlg.BLAS.sbmv!('L',k,alpha,A,x,beta,y))
@test macroexpand(:(SugarBLAS.@sbmv! y = beta*y + alpha*A[0:k,'L']*x)) == :(Base.LinAlg.BLAS.sbmv!('L',k,alpha,A,x,beta,y))

#gemm
@test macroexpand(:(SugarBLAS.@gemm alpha*A*B)) == :(Base.LinAlg.BLAS.gemm('N','N',alpha,A,B))
@test macroexpand(:(SugarBLAS.@gemm A*B')) == :(Base.LinAlg.BLAS.gemm('N','T',A,B))

#gemm!
@test macroexpand(:(SugarBLAS.@gemm! C -= alpha*A*B)) == :(Base.LinAlg.BLAS.gemm!('N','N',-alpha,A,B,1.0,C))
@test macroexpand(:(SugarBLAS.@gemm! C -= 1.5*A*B')) == :(Base.LinAlg.BLAS.gemm!('N','T',-1.5,A,B,1.0,C))
@test macroexpand(:(SugarBLAS.@gemm! C = C - alpha*A*B)) == :(Base.LinAlg.BLAS.gemm!('N','N',-alpha,A,B,1.0,C))
@test macroexpand(:(SugarBLAS.@gemm! C = C - 1.5*A*B')) == :(Base.LinAlg.BLAS.gemm!('N','T',-1.5,A,B,1.0,C))
@test macroexpand(:(SugarBLAS.@gemm! C = beta*C - alpha*A*B)) == :(Base.LinAlg.BLAS.gemm!('N','N',-alpha,A,B,beta,C))
@test macroexpand(:(SugarBLAS.@gemm! C = beta*C - 1.5*A*B')) == :(Base.LinAlg.BLAS.gemm!('N','T',-1.5,A,B,beta,C))
@test macroexpand(:(SugarBLAS.@gemm! C = beta*C - alpha*A'*B)) == :(Base.LinAlg.BLAS.gemm!('T','N',-alpha,A,B,beta,C))
@test macroexpand(:(SugarBLAS.@gemm! C = 3.4*C - alpha*A'*B')) == :(Base.LinAlg.BLAS.gemm!('T','T',-alpha,A,B,3.4,C))
@test macroexpand(:(SugarBLAS.@gemm! C += alpha*A*B)) == :(Base.LinAlg.BLAS.gemm!('N','N',alpha,A,B,1.0,C))
@test macroexpand(:(SugarBLAS.@gemm! C += 1.5*A*B')) == :(Base.LinAlg.BLAS.gemm!('N','T',1.5,A,B,1.0,C))
@test macroexpand(:(SugarBLAS.@gemm! C = alpha*A*B + C)) == :(Base.LinAlg.BLAS.gemm!('N','N',alpha,A,B,1.0,C))
@test macroexpand(:(SugarBLAS.@gemm! C = C + 1.5*A*B')) == :(Base.LinAlg.BLAS.gemm!('N','T',1.5,A,B,1.0,C))
@test macroexpand(:(SugarBLAS.@gemm! C = alpha*A*B + beta*C)) == :(Base.LinAlg.BLAS.gemm!('N','N',alpha,A,B,beta,C))
@test macroexpand(:(SugarBLAS.@gemm! C = beta*C + 1.5*A*B')) == :(Base.LinAlg.BLAS.gemm!('N','T',1.5,A,B,beta,C))
@test macroexpand(:(SugarBLAS.@gemm! C = alpha*A'*B + beta*C)) == :(Base.LinAlg.BLAS.gemm!('T','N',alpha,A,B,beta,C))
@test macroexpand(:(SugarBLAS.@gemm! C = 3.4*C + alpha*A'*B')) == :(Base.LinAlg.BLAS.gemm!('T','T',alpha,A,B,3.4,C))

#gemv
@test macroexpand(:(SugarBLAS.@gemv alpha*A*x)) == :(Base.LinAlg.BLAS.gemv('N',alpha,A,x))
@test macroexpand(:(SugarBLAS.@gemv A'*x)) == :(Base.LinAlg.BLAS.gemv('T',A,x))

#gemv!
@test macroexpand(:(SugarBLAS.@gemv! y -= alpha*A*x)) == :(Base.LinAlg.BLAS.gemv!('N',-alpha,A,x,1.0,y))
@test macroexpand(:(SugarBLAS.@gemv! y -= 1.5*A'*x)) == :(Base.LinAlg.BLAS.gemv!('T',-1.5,A,x,1.0,y))
@test macroexpand(:(SugarBLAS.@gemv! y = y - alpha*A*x)) == :(Base.LinAlg.BLAS.gemv!('N',-alpha,A,x,1.0,y))
@test macroexpand(:(SugarBLAS.@gemv! y = y - 1.5*A'*x)) == :(Base.LinAlg.BLAS.gemv!('T',-1.5,A,x,1.0,y))
@test macroexpand(:(SugarBLAS.@gemv! y = beta*y - alpha*A*x)) == :(Base.LinAlg.BLAS.gemv!('N',-alpha,A,x,beta,y))
@test macroexpand(:(SugarBLAS.@gemv! y = beta*y - 1.5*A'*x)) == :(Base.LinAlg.BLAS.gemv!('T',-1.5,A,x,beta,y))
@test macroexpand(:(SugarBLAS.@gemv! y += alpha*A*x)) == :(Base.LinAlg.BLAS.gemv!('N',alpha,A,x,1.0,y))
@test macroexpand(:(SugarBLAS.@gemv! y += 1.5*A'*x)) == :(Base.LinAlg.BLAS.gemv!('T',1.5,A,x,1.0,y))
@test macroexpand(:(SugarBLAS.@gemv! y = alpha*A*x + y)) == :(Base.LinAlg.BLAS.gemv!('N',alpha,A,x,1.0,y))
@test macroexpand(:(SugarBLAS.@gemv! y = y + 1.5*A'*x)) == :(Base.LinAlg.BLAS.gemv!('T',1.5,A,x,1.0,y))
@test macroexpand(:(SugarBLAS.@gemv! y = alpha*A*x + beta*y)) == :(Base.LinAlg.BLAS.gemv!('N',alpha,A,x,beta,y))
@test macroexpand(:(SugarBLAS.@gemv! y = beta*y + 1.5*A'*x)) == :(Base.LinAlg.BLAS.gemv!('T',1.5,A,x,beta,y))

#symm
@test macroexpand(:(SugarBLAS.@symm alpha*A[:symm]*B uplo='L')) == :(Base.LinAlg.BLAS.symm('L','L',alpha,A,B))
@test macroexpand(:(SugarBLAS.@symm A*B[:symm] uplo='U')) == :(Base.LinAlg.BLAS.symm('R','U',A,B))

#symm!
@test macroexpand(:(SugarBLAS.@symm! C['L'] -= alpha*A[:symm]*B)) == :(Base.LinAlg.BLAS.symm!('L','L',-alpha,A,B,1.0,C))
@test macroexpand(:(SugarBLAS.@symm! C['U'] -= 1.5*A*B[:symm])) == :(Base.LinAlg.BLAS.symm!('R','U',-1.5,A,B,1.0,C))
@test macroexpand(:(SugarBLAS.@symm! C['L'] = C - alpha*A[:symm]*B)) == :(Base.LinAlg.BLAS.symm!('L','L',-alpha,A,B,1.0,C))
@test macroexpand(:(SugarBLAS.@symm! C['U'] = C - 1.5*A*B[:symm])) == :(Base.LinAlg.BLAS.symm!('R','U',-1.5,A,B,1.0,C))
@test macroexpand(:(SugarBLAS.@symm! C['L'] = beta*C - alpha*A[:symm]*B)) == :(Base.LinAlg.BLAS.symm!('L','L',-alpha,A,B,beta,C))
@test macroexpand(:(SugarBLAS.@symm! C['L'] += alpha*A[:symm]*B)) == :(Base.LinAlg.BLAS.symm!('L','L',alpha,A,B,1.0,C))
@test macroexpand(:(SugarBLAS.@symm! C['U'] += 1.5*A*B[:symm])) == :(Base.LinAlg.BLAS.symm!('R','U',1.5,A,B,1.0,C))
@test macroexpand(:(SugarBLAS.@symm! C['L'] = C + alpha*A[:symm]*B)) == :(Base.LinAlg.BLAS.symm!('L','L',alpha,A,B,1.0,C))
@test macroexpand(:(SugarBLAS.@symm! C['U'] = 1.5*A*B[:symm] + C)) == :(Base.LinAlg.BLAS.symm!('R','U',1.5,A,B,1.0,C))
@test macroexpand(:(SugarBLAS.@symm! C['L'] = beta*C + alpha*A[:symm]*B)) == :(Base.LinAlg.BLAS.symm!('L','L',alpha,A,B,beta,C))

#symv
@test macroexpand(:(SugarBLAS.@symv alpha*A['U']*x uplo='U')) == :(Base.LinAlg.BLAS.symv('U',alpha,A,x))
@test macroexpand(:(SugarBLAS.@symv A['L']*x uplo='L')) == :(Base.LinAlg.BLAS.symv('L',A,x))

#symv!
@test macroexpand(:(SugarBLAS.@symv! y -= alpha*A['U']*x)) == :(Base.LinAlg.BLAS.symv!('U',-alpha,A,x,1.0,y))
@test macroexpand(:(SugarBLAS.@symv! y -= 1.5*A['L']*x)) == :(Base.LinAlg.BLAS.symv!('L',-1.5,A,x,1.0,y))
@test macroexpand(:(SugarBLAS.@symv! y = y - alpha*A['U']*x)) == :(Base.LinAlg.BLAS.symv!('U',-alpha,A,x,1.0,y))
@test macroexpand(:(SugarBLAS.@symv! y = y - 1.5*A['L']*x)) == :(Base.LinAlg.BLAS.symv!('L',-1.5,A,x,1.0,y))
@test macroexpand(:(SugarBLAS.@symv! y = beta*y - alpha*A['U']*x)) == :(Base.LinAlg.BLAS.symv!('U',-alpha,A,x,beta,y))
@test macroexpand(:(SugarBLAS.@symv! y = beta*y - 1.5*A['L']*x)) == :(Base.LinAlg.BLAS.symv!('L',-1.5,A,x,beta,y))
@test macroexpand(:(SugarBLAS.@symv! y += alpha*A['U']*x)) == :(Base.LinAlg.BLAS.symv!('U',alpha,A,x,1.0,y))
@test macroexpand(:(SugarBLAS.@symv! y += 1.5*A['L']*x)) == :(Base.LinAlg.BLAS.symv!('L',1.5,A,x,1.0,y))
@test macroexpand(:(SugarBLAS.@symv! y = alpha*A['U']*x + y)) == :(Base.LinAlg.BLAS.symv!('U',alpha,A,x,1.0,y))
@test macroexpand(:(SugarBLAS.@symv! y = y + 1.5*A['L']*x)) == :(Base.LinAlg.BLAS.symv!('L',1.5,A,x,1.0,y))
@test macroexpand(:(SugarBLAS.@symv! y = alpha*A['U']*x + beta*y)) == :(Base.LinAlg.BLAS.symv!('U',alpha,A,x,beta,y))
@test macroexpand(:(SugarBLAS.@symv! y = beta*y + 1.5*A['L']*x)) == :(Base.LinAlg.BLAS.symv!('L',1.5,A,x,beta,y))
