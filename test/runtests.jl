using SugarBLAS
using Test

#scale!
@test macroexpand(SugarBLAS, :(SugarBLAS.@scale! X *= a)) == :(scale!(a, X))
@test macroexpand(SugarBLAS, :(SugarBLAS.@scale! X = a*X)) == :(scale!(a, X))
@test macroexpand(SugarBLAS, :(@blas! X *= a)) == :(scale!(a, X))
@test macroexpand(SugarBLAS, :(@blas! X = a*X)) == :(scale!(a, X))

#axpy!
@test macroexpand(SugarBLAS, :(SugarBLAS.@axpy! Y -= a*X)) == :(Base.LinAlg.axpy!(-a, X, Y))
@test macroexpand(SugarBLAS, :(SugarBLAS.@axpy! Y = Y - a*X)) == :(Base.LinAlg.axpy!(-a, X, Y))
@test macroexpand(SugarBLAS, :(SugarBLAS.@axpy! Y = Y - X)) == :(Base.LinAlg.axpy!(-1.0, X, Y))
@test macroexpand(SugarBLAS, :(SugarBLAS.@axpy! Y -= X)) == :(Base.LinAlg.axpy!(-1.0, X, Y))
@test macroexpand(SugarBLAS, :(SugarBLAS.@axpy! Y += a*X)) == :(Base.LinAlg.axpy!(a, X, Y))
@test macroexpand(SugarBLAS, :(SugarBLAS.@axpy! Y = a*X + Y)) == :(Base.LinAlg.axpy!(a, X, Y))
@test macroexpand(SugarBLAS, :(SugarBLAS.@axpy! Y = Y + a*X)) == :(Base.LinAlg.axpy!(a, X, Y))
@test macroexpand(SugarBLAS, :(SugarBLAS.@axpy! Y = X + Y)) == :(Base.LinAlg.axpy!(1.0, X, Y))
@test macroexpand(SugarBLAS, :(SugarBLAS.@axpy! Y = Y + X)) == :(Base.LinAlg.axpy!(1.0, X, Y))
@test macroexpand(SugarBLAS, :(SugarBLAS.@axpy! Y += X)) == :(Base.LinAlg.axpy!(1.0, X, Y))
@test macroexpand(SugarBLAS, :(@blas! Y -= a*X)) == :(Base.LinAlg.axpy!(-a, X, Y))
@test macroexpand(SugarBLAS, :(@blas! Y = Y - a*X)) == :(Base.LinAlg.axpy!(-a, X, Y))
@test macroexpand(SugarBLAS, :(@blas! Y = Y - X)) == :(Base.LinAlg.axpy!(-1.0, X, Y))
@test macroexpand(SugarBLAS, :(@blas! Y -= X)) == :(Base.LinAlg.axpy!(-1.0, X, Y))
@test macroexpand(SugarBLAS, :(@blas! Y += a*X)) == :(Base.LinAlg.axpy!(a, X, Y))
@test macroexpand(SugarBLAS, :(@blas! Y = a*X + Y)) == :(Base.LinAlg.axpy!(a, X, Y))
@test macroexpand(SugarBLAS, :(@blas! Y = Y + a*X)) == :(Base.LinAlg.axpy!(a, X, Y))
@test macroexpand(SugarBLAS, :(@blas! Y = X + Y)) == :(Base.LinAlg.axpy!(1.0, X, Y))
@test macroexpand(SugarBLAS, :(@blas! Y = Y + X)) == :(Base.LinAlg.axpy!(1.0, X, Y))
@test macroexpand(SugarBLAS, :(@blas! Y += X)) == :(Base.LinAlg.axpy!(1.0, X, Y))

#copy!
@test macroexpand(SugarBLAS, :(SugarBLAS.@copy! X = Y)) == :(copy!(X, Y))
@test macroexpand(SugarBLAS, :(@blas! X = Y)) == :(copy!(X, Y))

#ger!
@test macroexpand(SugarBLAS, :(SugarBLAS.@ger! A -= alpha*x*y')) == :(Base.LinAlg.BLAS.ger!(-alpha,x,y,A))
@test macroexpand(SugarBLAS, :(SugarBLAS.@ger! A = A - alpha*x*y')) == :(Base.LinAlg.BLAS.ger!(-alpha,x,y,A))
@test macroexpand(SugarBLAS, :(SugarBLAS.@ger! A += alpha*x*y')) == :(Base.LinAlg.BLAS.ger!(alpha,x,y,A))
@test macroexpand(SugarBLAS, :(SugarBLAS.@ger! A = alpha*x*y' + A)) == :(Base.LinAlg.BLAS.ger!(alpha,x,y,A))

#syr!
@test macroexpand(SugarBLAS, :(SugarBLAS.@syr! A['U'] -= alpha*x*x.')) == :(Base.LinAlg.BLAS.syr!('U',-alpha,x,A))
@test macroexpand(SugarBLAS, :(SugarBLAS.@syr! A['L'] -= alpha*x*x.')) == :(Base.LinAlg.BLAS.syr!('L',-alpha,x,A))
@test macroexpand(SugarBLAS, :(SugarBLAS.@syr! A['U'] = A - alpha*x*x.')) == :(Base.LinAlg.BLAS.syr!('U',-alpha,x,A))
@test macroexpand(SugarBLAS, :(SugarBLAS.@syr! A['L'] = A - alpha*x*x.')) == :(Base.LinAlg.BLAS.syr!('L',-alpha,x,A))
@test macroexpand(SugarBLAS, :(SugarBLAS.@syr! A['U'] += alpha*x*x.')) == :(Base.LinAlg.BLAS.syr!('U',alpha,x,A))
@test macroexpand(SugarBLAS, :(SugarBLAS.@syr! A['L'] += alpha*x*x.')) == :(Base.LinAlg.BLAS.syr!('L',alpha,x,A))
@test macroexpand(SugarBLAS, :(SugarBLAS.@syr! A['U'] = alpha*x*x.' + A)) == :(Base.LinAlg.BLAS.syr!('U',alpha,x,A))
@test macroexpand(SugarBLAS, :(SugarBLAS.@syr! A['L'] = alpha*x*x.' + A)) == :(Base.LinAlg.BLAS.syr!('L',alpha,x,A))

#syrk
@test macroexpand(SugarBLAS, :(SugarBLAS.@syrk alpha*A*A.' uplo='U')) == :(Base.LinAlg.BLAS.syrk('U','N',alpha,A))
@test macroexpand(SugarBLAS, :(SugarBLAS.@syrk alpha*A.'*A uplo='U')) == :(Base.LinAlg.BLAS.syrk('U','T',alpha,A))
@test macroexpand(SugarBLAS, :(SugarBLAS.@syrk alpha*A*A.' uplo='L')) == :(Base.LinAlg.BLAS.syrk('L','N',alpha,A))
@test macroexpand(SugarBLAS, :(SugarBLAS.@syrk alpha*A.'*A uplo='L')) == :(Base.LinAlg.BLAS.syrk('L','T',alpha,A))

#syrk!
@test macroexpand(SugarBLAS, :(SugarBLAS.@syrk! C['U'] -= alpha*A*A.')) == :(Base.LinAlg.BLAS.syrk!('U','N',-alpha,A,1.0,C))
@test macroexpand(SugarBLAS, :(SugarBLAS.@syrk! C['U'] -= alpha*A.'*A)) == :(Base.LinAlg.BLAS.syrk!('U','T',-alpha,A,1.0,C))
@test macroexpand(SugarBLAS, :(SugarBLAS.@syrk! C['U'] = C - alpha*A*A.')) == :(Base.LinAlg.BLAS.syrk!('U','N',-alpha,A,1.0,C))
@test macroexpand(SugarBLAS, :(SugarBLAS.@syrk! C['U'] = C - alpha*A.'*A)) == :(Base.LinAlg.BLAS.syrk!('U','T',-alpha,A,1.0,C))
@test macroexpand(SugarBLAS, :(SugarBLAS.@syrk! C['U'] = beta*C - alpha*A*A.')) == :(Base.LinAlg.BLAS.syrk!('U','N',-alpha,A,beta,C))
@test macroexpand(SugarBLAS, :(SugarBLAS.@syrk! C['L'] = beta*C - alpha*A.'*A)) == :(Base.LinAlg.BLAS.syrk!('L','T',-alpha,A,beta,C))
@test macroexpand(SugarBLAS, :(SugarBLAS.@syrk! C['U'] += alpha*A*A.')) == :(Base.LinAlg.BLAS.syrk!('U','N',alpha,A,1.0,C))
@test macroexpand(SugarBLAS, :(SugarBLAS.@syrk! C['U'] += alpha*A.'*A)) == :(Base.LinAlg.BLAS.syrk!('U','T',alpha,A,1.0,C))
@test macroexpand(SugarBLAS, :(SugarBLAS.@syrk! C['U'] = alpha*A*A.' + C)) == :(Base.LinAlg.BLAS.syrk!('U','N',alpha,A,1.0,C))
@test macroexpand(SugarBLAS, :(SugarBLAS.@syrk! C['U'] = C + alpha*A.'*A)) == :(Base.LinAlg.BLAS.syrk!('U','T',alpha,A,1.0,C))
@test macroexpand(SugarBLAS, :(SugarBLAS.@syrk! C['U'] = alpha*A*A.' + beta*C)) == :(Base.LinAlg.BLAS.syrk!('U','N',alpha,A,beta,C))
@test macroexpand(SugarBLAS, :(SugarBLAS.@syrk! C['L'] = alpha*A.'*A + beta*C)) == :(Base.LinAlg.BLAS.syrk!('L','T',alpha,A,beta,C))

#her!
@test macroexpand(SugarBLAS, :(SugarBLAS.@her! A['U'] -= alpha*x*x')) == :(Base.LinAlg.BLAS.her!('U',-alpha,x,A))
@test macroexpand(SugarBLAS, :(SugarBLAS.@her! A['L'] -= alpha*x*x')) == :(Base.LinAlg.BLAS.her!('L',-alpha,x,A))
@test macroexpand(SugarBLAS, :(SugarBLAS.@her! A['U'] = A - alpha*x*x')) == :(Base.LinAlg.BLAS.her!('U',-alpha,x,A))
@test macroexpand(SugarBLAS, :(SugarBLAS.@her! A['L'] = A - alpha*x*x')) == :(Base.LinAlg.BLAS.her!('L',-alpha,x,A))
@test macroexpand(SugarBLAS, :(SugarBLAS.@her! A['U'] = alpha*x*x' + A)) == :(Base.LinAlg.BLAS.her!('U',alpha,x,A))
@test macroexpand(SugarBLAS, :(SugarBLAS.@her! A['L'] = alpha*x*x' + A)) == :(Base.LinAlg.BLAS.her!('L',alpha,x,A))
@test macroexpand(SugarBLAS, :(SugarBLAS.@her! A['U'] += alpha*x*x')) == :(Base.LinAlg.BLAS.her!('U',alpha,x,A))
@test macroexpand(SugarBLAS, :(SugarBLAS.@her! A['L'] += alpha*x*x')) == :(Base.LinAlg.BLAS.her!('L',alpha,x,A))

#herk
@test macroexpand(SugarBLAS, :(SugarBLAS.@herk alpha*A*A' uplo='U')) == :(Base.LinAlg.BLAS.herk('U','N',alpha,A))
@test macroexpand(SugarBLAS, :(SugarBLAS.@herk alpha*A'*A uplo='U')) == :(Base.LinAlg.BLAS.herk('U','T',alpha,A))
@test macroexpand(SugarBLAS, :(SugarBLAS.@herk alpha*A*A' uplo='L')) == :(Base.LinAlg.BLAS.herk('L','N',alpha,A))
@test macroexpand(SugarBLAS, :(SugarBLAS.@herk alpha*A'*A uplo='L')) == :(Base.LinAlg.BLAS.herk('L','T',alpha,A))

#herk!
@test macroexpand(SugarBLAS, :(SugarBLAS.@herk! C['U'] -= alpha*A*A')) == :(Base.LinAlg.BLAS.herk!('U','N',-alpha,A,1.0,C))
@test macroexpand(SugarBLAS, :(SugarBLAS.@herk! C['L'] -= alpha*A'*A)) == :(Base.LinAlg.BLAS.herk!('L','T',-alpha,A,1.0,C))
@test macroexpand(SugarBLAS, :(SugarBLAS.@herk! C['U'] = C - alpha*A*A')) == :(Base.LinAlg.BLAS.herk!('U','N',-alpha,A,1.0,C))
@test macroexpand(SugarBLAS, :(SugarBLAS.@herk! C['L'] = C - alpha*A'*A)) == :(Base.LinAlg.BLAS.herk!('L','T',-alpha,A,1.0,C))
@test macroexpand(SugarBLAS, :(SugarBLAS.@herk! C['U'] = beta*C - alpha*A*A')) == :(Base.LinAlg.BLAS.herk!('U','N',-alpha,A,beta,C))
@test macroexpand(SugarBLAS, :(SugarBLAS.@herk! C['L'] = beta*C - alpha*A'*A)) == :(Base.LinAlg.BLAS.herk!('L','T',-alpha,A,beta,C))
@test macroexpand(SugarBLAS, :(SugarBLAS.@herk! C['U'] += alpha*A*A')) == :(Base.LinAlg.BLAS.herk!('U','N',alpha,A,1.0,C))
@test macroexpand(SugarBLAS, :(SugarBLAS.@herk! C['L'] += alpha*A'*A)) == :(Base.LinAlg.BLAS.herk!('L','T',alpha,A,1.0,C))
@test macroexpand(SugarBLAS, :(SugarBLAS.@herk! C['U'] = alpha*A*A' + C)) == :(Base.LinAlg.BLAS.herk!('U','N',alpha,A,1.0,C))
@test macroexpand(SugarBLAS, :(SugarBLAS.@herk! C['L'] = C + alpha*A'*A)) == :(Base.LinAlg.BLAS.herk!('L','T',alpha,A,1.0,C))
@test macroexpand(SugarBLAS, :(SugarBLAS.@herk! C['U'] = alpha*A*A' + beta*C)) == :(Base.LinAlg.BLAS.herk!('U','N',alpha,A,beta,C))
@test macroexpand(SugarBLAS, :(SugarBLAS.@herk! C['L'] = alpha*A'*A + beta*C)) == :(Base.LinAlg.BLAS.herk!('L','T',alpha,A,beta,C))

#gbmv
@test macroexpand(SugarBLAS, :(SugarBLAS.@gbmv alpha*A[0:ku,h=2]*x)) == :(Base.LinAlg.BLAS.gbmv('N',2,0,ku,alpha,A,x))
@test macroexpand(SugarBLAS, :(SugarBLAS.@gbmv alpha*A[h=m,-kl:ku]*x)) == :(Base.LinAlg.BLAS.gbmv('N',m,kl,ku,alpha,A,x))

#gbmv!
@test macroexpand(SugarBLAS, :(SugarBLAS.@gbmv! y = y - alpha*A[0:ku,h=2]*x)) == :(Base.LinAlg.BLAS.gbmv!('N',2,0,ku,-alpha,A,x,1.0,y))
@test macroexpand(SugarBLAS, :(SugarBLAS.@gbmv! y -= alpha*A[h=m,-kl:ku]*x)) == :(Base.LinAlg.BLAS.gbmv!('N',m,kl,ku,-alpha,A,x,1.0,y))
@test macroexpand(SugarBLAS, :(SugarBLAS.@gbmv! y = beta*y - alpha*A[0:ku,h=2]*x)) == :(Base.LinAlg.BLAS.gbmv!('N',2,0,ku,-alpha,A,x,beta,y))
@test macroexpand(SugarBLAS, :(SugarBLAS.@gbmv! y = beta*y - alpha*A[h=m,-kl:ku]*x)) == :(Base.LinAlg.BLAS.gbmv!('N',m,kl,ku,-alpha,A,x,beta,y))
@test macroexpand(SugarBLAS, :(SugarBLAS.@gbmv! y = beta*y - alpha*A[h=2, 0:ku]'*x)) == :(Base.LinAlg.BLAS.gbmv!('T',2,0,ku,-alpha,A,x,beta,y))
@test macroexpand(SugarBLAS, :(SugarBLAS.@gbmv! y = beta*y - alpha*A[kl:ku, h=m]'*x)) == :(Base.LinAlg.BLAS.gbmv!('T',m,-kl,ku,-alpha,A,x,beta,y))
@test macroexpand(SugarBLAS, :(SugarBLAS.@gbmv! y = alpha*A[0:ku,h=2]*x + y)) == :(Base.LinAlg.BLAS.gbmv!('N',2,0,ku,alpha,A,x,1.0,y))
@test macroexpand(SugarBLAS, :(SugarBLAS.@gbmv! y += alpha*A[h=m,-kl:ku]*x)) == :(Base.LinAlg.BLAS.gbmv!('N',m,kl,ku,alpha,A,x,1.0,y))
@test macroexpand(SugarBLAS, :(SugarBLAS.@gbmv! y = y + alpha*A[h=m,-kl:ku]*x)) == :(Base.LinAlg.BLAS.gbmv!('N',m,kl,ku,alpha,A,x,1.0,y))
@test macroexpand(SugarBLAS, :(SugarBLAS.@gbmv! y = alpha*A[0:ku,h=2]*x + beta*y)) == :(Base.LinAlg.BLAS.gbmv!('N',2,0,ku,alpha,A,x,beta,y))
@test macroexpand(SugarBLAS, :(SugarBLAS.@gbmv! y = alpha*A[h=m,-kl:ku]*x + beta*y)) == :(Base.LinAlg.BLAS.gbmv!('N',m,kl,ku,alpha,A,x,beta,y))
@test macroexpand(SugarBLAS, :(SugarBLAS.@gbmv! y = alpha*A[h=2, 0:ku]'*x + beta*y)) == :(Base.LinAlg.BLAS.gbmv!('T',2,0,ku,alpha,A,x,beta,y))
@test macroexpand(SugarBLAS, :(SugarBLAS.@gbmv! y = alpha*A[kl:ku, h=m]'*x + beta*y)) == :(Base.LinAlg.BLAS.gbmv!('T',m,-kl,ku,alpha,A,x,beta,y))

#sbmv
@test macroexpand(SugarBLAS, :(SugarBLAS.@sbmv A['U',0:k]*x)) == :(Base.LinAlg.BLAS.sbmv('U',k,A,x))
@test macroexpand(SugarBLAS, :(SugarBLAS.@sbmv alpha*A[0:k,'L']*x)) == :(Base.LinAlg.BLAS.sbmv('L',k,alpha,A,x))

#sbmv!
@test macroexpand(SugarBLAS, :(SugarBLAS.@sbmv! y -= alpha*A['U',0:k]*x)) == :(Base.LinAlg.BLAS.sbmv!('U',k,-alpha,A,x,1.0,y))
@test macroexpand(SugarBLAS, :(SugarBLAS.@sbmv! y -= alpha*A[0:k,'L']*x)) == :(Base.LinAlg.BLAS.sbmv!('L',k,-alpha,A,x,1.0,y))
@test macroexpand(SugarBLAS, :(SugarBLAS.@sbmv! y = y - alpha*A['U',0:k]*x)) == :(Base.LinAlg.BLAS.sbmv!('U',k,-alpha,A,x,1.0,y))
@test macroexpand(SugarBLAS, :(SugarBLAS.@sbmv! y = beta*y - alpha*A['U',0:k]*x)) == :(Base.LinAlg.BLAS.sbmv!('U',k,-alpha,A,x,beta,y))
@test macroexpand(SugarBLAS, :(SugarBLAS.@sbmv! y = beta*y - alpha*A[0:k,'U']*x)) == :(Base.LinAlg.BLAS.sbmv!('U',k,-alpha,A,x,beta,y))
@test macroexpand(SugarBLAS, :(SugarBLAS.@sbmv! y = beta*y - alpha*A['L',0:k]*x)) == :(Base.LinAlg.BLAS.sbmv!('L',k,-alpha,A,x,beta,y))
@test macroexpand(SugarBLAS, :(SugarBLAS.@sbmv! y = beta*y - alpha*A[0:k,'L']*x)) == :(Base.LinAlg.BLAS.sbmv!('L',k,-alpha,A,x,beta,y))
@test macroexpand(SugarBLAS, :(SugarBLAS.@sbmv! y += alpha*A['U',0:k]*x)) == :(Base.LinAlg.BLAS.sbmv!('U',k,alpha,A,x,1.0,y))
@test macroexpand(SugarBLAS, :(SugarBLAS.@sbmv! y += alpha*A[0:k,'L']*x)) == :(Base.LinAlg.BLAS.sbmv!('L',k,alpha,A,x,1.0,y))
@test macroexpand(SugarBLAS, :(SugarBLAS.@sbmv! y = alpha*A['U',0:k]*x + y)) == :(Base.LinAlg.BLAS.sbmv!('U',k,alpha,A,x,1.0,y))
@test macroexpand(SugarBLAS, :(SugarBLAS.@sbmv! y = y + alpha*A[0:k,'L']*x)) == :(Base.LinAlg.BLAS.sbmv!('L',k,alpha,A,x,1.0,y))
@test macroexpand(SugarBLAS, :(SugarBLAS.@sbmv! y = alpha*A['U',0:k]*x + beta*y)) == :(Base.LinAlg.BLAS.sbmv!('U',k,alpha,A,x,beta,y))
@test macroexpand(SugarBLAS, :(SugarBLAS.@sbmv! y = beta*y + alpha*A[0:k,'U']*x)) == :(Base.LinAlg.BLAS.sbmv!('U',k,alpha,A,x,beta,y))
@test macroexpand(SugarBLAS, :(SugarBLAS.@sbmv! y = alpha*A['L',0:k]*x + beta*y)) == :(Base.LinAlg.BLAS.sbmv!('L',k,alpha,A,x,beta,y))
@test macroexpand(SugarBLAS, :(SugarBLAS.@sbmv! y = beta*y + alpha*A[0:k,'L']*x)) == :(Base.LinAlg.BLAS.sbmv!('L',k,alpha,A,x,beta,y))

#gemm
@test macroexpand(SugarBLAS, :(SugarBLAS.@gemm alpha*A*B)) == :(Base.LinAlg.BLAS.gemm('N','N',alpha,A,B))
@test macroexpand(SugarBLAS, :(SugarBLAS.@gemm A*B')) == :(Base.LinAlg.BLAS.gemm('N','T',A,B))

#gemm!
@test macroexpand(SugarBLAS, :(SugarBLAS.@gemm! C -= alpha*A*B)) == :(Base.LinAlg.BLAS.gemm!('N','N',-alpha,A,B,1.0,C))
@test macroexpand(SugarBLAS, :(SugarBLAS.@gemm! C -= 1.5*A*B')) == :(Base.LinAlg.BLAS.gemm!('N','T',-1.5,A,B,1.0,C))
@test macroexpand(SugarBLAS, :(SugarBLAS.@gemm! C = C - alpha*A*B)) == :(Base.LinAlg.BLAS.gemm!('N','N',-alpha,A,B,1.0,C))
@test macroexpand(SugarBLAS, :(SugarBLAS.@gemm! C = C - 1.5*A*B')) == :(Base.LinAlg.BLAS.gemm!('N','T',-1.5,A,B,1.0,C))
@test macroexpand(SugarBLAS, :(SugarBLAS.@gemm! C = beta*C - alpha*A*B)) == :(Base.LinAlg.BLAS.gemm!('N','N',-alpha,A,B,beta,C))
@test macroexpand(SugarBLAS, :(SugarBLAS.@gemm! C = beta*C - 1.5*A*B')) == :(Base.LinAlg.BLAS.gemm!('N','T',-1.5,A,B,beta,C))
@test macroexpand(SugarBLAS, :(SugarBLAS.@gemm! C = beta*C - alpha*A'*B)) == :(Base.LinAlg.BLAS.gemm!('T','N',-alpha,A,B,beta,C))
@test macroexpand(SugarBLAS, :(SugarBLAS.@gemm! C = 3.4*C - alpha*A'*B')) == :(Base.LinAlg.BLAS.gemm!('T','T',-alpha,A,B,3.4,C))
@test macroexpand(SugarBLAS, :(SugarBLAS.@gemm! C += alpha*A*B)) == :(Base.LinAlg.BLAS.gemm!('N','N',alpha,A,B,1.0,C))
@test macroexpand(SugarBLAS, :(SugarBLAS.@gemm! C += 1.5*A*B')) == :(Base.LinAlg.BLAS.gemm!('N','T',1.5,A,B,1.0,C))
@test macroexpand(SugarBLAS, :(SugarBLAS.@gemm! C = alpha*A*B + C)) == :(Base.LinAlg.BLAS.gemm!('N','N',alpha,A,B,1.0,C))
@test macroexpand(SugarBLAS, :(SugarBLAS.@gemm! C = C + 1.5*A*B')) == :(Base.LinAlg.BLAS.gemm!('N','T',1.5,A,B,1.0,C))
@test macroexpand(SugarBLAS, :(SugarBLAS.@gemm! C = alpha*A*B + beta*C)) == :(Base.LinAlg.BLAS.gemm!('N','N',alpha,A,B,beta,C))
@test macroexpand(SugarBLAS, :(SugarBLAS.@gemm! C = beta*C + 1.5*A*B')) == :(Base.LinAlg.BLAS.gemm!('N','T',1.5,A,B,beta,C))
@test macroexpand(SugarBLAS, :(SugarBLAS.@gemm! C = alpha*A'*B + beta*C)) == :(Base.LinAlg.BLAS.gemm!('T','N',alpha,A,B,beta,C))
@test macroexpand(SugarBLAS, :(SugarBLAS.@gemm! C = 3.4*C + alpha*A'*B')) == :(Base.LinAlg.BLAS.gemm!('T','T',alpha,A,B,3.4,C))

#gemv
@test macroexpand(SugarBLAS, :(SugarBLAS.@gemv alpha*A*x)) == :(Base.LinAlg.BLAS.gemv('N',alpha,A,x))
@test macroexpand(SugarBLAS, :(SugarBLAS.@gemv A'*x)) == :(Base.LinAlg.BLAS.gemv('T',A,x))

#gemv!
@test macroexpand(SugarBLAS, :(SugarBLAS.@gemv! y -= alpha*A*x)) == :(Base.LinAlg.BLAS.gemv!('N',-alpha,A,x,1.0,y))
@test macroexpand(SugarBLAS, :(SugarBLAS.@gemv! y -= 1.5*A'*x)) == :(Base.LinAlg.BLAS.gemv!('T',-1.5,A,x,1.0,y))
@test macroexpand(SugarBLAS, :(SugarBLAS.@gemv! y = y - alpha*A*x)) == :(Base.LinAlg.BLAS.gemv!('N',-alpha,A,x,1.0,y))
@test macroexpand(SugarBLAS, :(SugarBLAS.@gemv! y = y - 1.5*A'*x)) == :(Base.LinAlg.BLAS.gemv!('T',-1.5,A,x,1.0,y))
@test macroexpand(SugarBLAS, :(SugarBLAS.@gemv! y = beta*y - alpha*A*x)) == :(Base.LinAlg.BLAS.gemv!('N',-alpha,A,x,beta,y))
@test macroexpand(SugarBLAS, :(SugarBLAS.@gemv! y = beta*y - 1.5*A'*x)) == :(Base.LinAlg.BLAS.gemv!('T',-1.5,A,x,beta,y))
@test macroexpand(SugarBLAS, :(SugarBLAS.@gemv! y += alpha*A*x)) == :(Base.LinAlg.BLAS.gemv!('N',alpha,A,x,1.0,y))
@test macroexpand(SugarBLAS, :(SugarBLAS.@gemv! y += 1.5*A'*x)) == :(Base.LinAlg.BLAS.gemv!('T',1.5,A,x,1.0,y))
@test macroexpand(SugarBLAS, :(SugarBLAS.@gemv! y = alpha*A*x + y)) == :(Base.LinAlg.BLAS.gemv!('N',alpha,A,x,1.0,y))
@test macroexpand(SugarBLAS, :(SugarBLAS.@gemv! y = y + 1.5*A'*x)) == :(Base.LinAlg.BLAS.gemv!('T',1.5,A,x,1.0,y))
@test macroexpand(SugarBLAS, :(SugarBLAS.@gemv! y = alpha*A*x + beta*y)) == :(Base.LinAlg.BLAS.gemv!('N',alpha,A,x,beta,y))
@test macroexpand(SugarBLAS, :(SugarBLAS.@gemv! y = beta*y + 1.5*A'*x)) == :(Base.LinAlg.BLAS.gemv!('T',1.5,A,x,beta,y))

#symm
@test macroexpand(SugarBLAS, :(SugarBLAS.@symm alpha*A["symm", 'L']*B)) == :(Base.LinAlg.BLAS.symm('L','L',alpha,A,B))
@test macroexpand(SugarBLAS, :(SugarBLAS.@symm A*B["symm", 'U'])) == :(Base.LinAlg.BLAS.symm('R','U',A,B))

#symm!
@test macroexpand(SugarBLAS, :(SugarBLAS.@symm! C -= alpha*A["symm", 'L']*B)) == :(Base.LinAlg.BLAS.symm!('L','L',-alpha,A,B,1.0,C))
@test macroexpand(SugarBLAS, :(SugarBLAS.@symm! C -= 1.5*A*B["symm", 'U'])) == :(Base.LinAlg.BLAS.symm!('R','U',-1.5,A,B,1.0,C))
@test macroexpand(SugarBLAS, :(SugarBLAS.@symm! C = C - alpha*A["symm", 'L']*B)) == :(Base.LinAlg.BLAS.symm!('L','L',-alpha,A,B,1.0,C))
@test macroexpand(SugarBLAS, :(SugarBLAS.@symm! C = C - 1.5*A*B["symm", 'U'])) == :(Base.LinAlg.BLAS.symm!('R','U',-1.5,A,B,1.0,C))
@test macroexpand(SugarBLAS, :(SugarBLAS.@symm! C = beta*C - alpha*A["symm", 'L']*B)) == :(Base.LinAlg.BLAS.symm!('L','L',-alpha,A,B,beta,C))
@test macroexpand(SugarBLAS, :(SugarBLAS.@symm! C += alpha*A["symm", 'L']*B)) == :(Base.LinAlg.BLAS.symm!('L','L',alpha,A,B,1.0,C))
@test macroexpand(SugarBLAS, :(SugarBLAS.@symm! C += 1.5*A*B["symm", 'U'])) == :(Base.LinAlg.BLAS.symm!('R','U',1.5,A,B,1.0,C))
@test macroexpand(SugarBLAS, :(SugarBLAS.@symm! C = C + alpha*A["symm", 'L']*B)) == :(Base.LinAlg.BLAS.symm!('L','L',alpha,A,B,1.0,C))
@test macroexpand(SugarBLAS, :(SugarBLAS.@symm! C = 1.5*A*B["symm", 'U'] + C)) == :(Base.LinAlg.BLAS.symm!('R','U',1.5,A,B,1.0,C))
@test macroexpand(SugarBLAS, :(SugarBLAS.@symm! C = beta*C + alpha*A["symm", 'L']*B)) == :(Base.LinAlg.BLAS.symm!('L','L',alpha,A,B,beta,C))

#symv
@test macroexpand(SugarBLAS, :(SugarBLAS.@symv alpha*A['U']*x)) == :(Base.LinAlg.BLAS.symv('U',alpha,A,x))
@test macroexpand(SugarBLAS, :(SugarBLAS.@symv A['L']*x)) == :(Base.LinAlg.BLAS.symv('L',A,x))

#symv!
@test macroexpand(SugarBLAS, :(SugarBLAS.@symv! y -= alpha*A['U']*x)) == :(Base.LinAlg.BLAS.symv!('U',-alpha,A,x,1.0,y))
@test macroexpand(SugarBLAS, :(SugarBLAS.@symv! y -= 1.5*A['L']*x)) == :(Base.LinAlg.BLAS.symv!('L',-1.5,A,x,1.0,y))
@test macroexpand(SugarBLAS, :(SugarBLAS.@symv! y = y - alpha*A['U']*x)) == :(Base.LinAlg.BLAS.symv!('U',-alpha,A,x,1.0,y))
@test macroexpand(SugarBLAS, :(SugarBLAS.@symv! y = y - 1.5*A['L']*x)) == :(Base.LinAlg.BLAS.symv!('L',-1.5,A,x,1.0,y))
@test macroexpand(SugarBLAS, :(SugarBLAS.@symv! y = beta*y - alpha*A['U']*x)) == :(Base.LinAlg.BLAS.symv!('U',-alpha,A,x,beta,y))
@test macroexpand(SugarBLAS, :(SugarBLAS.@symv! y = beta*y - 1.5*A['L']*x)) == :(Base.LinAlg.BLAS.symv!('L',-1.5,A,x,beta,y))
@test macroexpand(SugarBLAS, :(SugarBLAS.@symv! y += alpha*A['U']*x)) == :(Base.LinAlg.BLAS.symv!('U',alpha,A,x,1.0,y))
@test macroexpand(SugarBLAS, :(SugarBLAS.@symv! y += 1.5*A['L']*x)) == :(Base.LinAlg.BLAS.symv!('L',1.5,A,x,1.0,y))
@test macroexpand(SugarBLAS, :(SugarBLAS.@symv! y = alpha*A['U']*x + y)) == :(Base.LinAlg.BLAS.symv!('U',alpha,A,x,1.0,y))
@test macroexpand(SugarBLAS, :(SugarBLAS.@symv! y = y + 1.5*A['L']*x)) == :(Base.LinAlg.BLAS.symv!('L',1.5,A,x,1.0,y))
@test macroexpand(SugarBLAS, :(SugarBLAS.@symv! y = alpha*A['U']*x + beta*y)) == :(Base.LinAlg.BLAS.symv!('U',alpha,A,x,beta,y))
@test macroexpand(SugarBLAS, :(SugarBLAS.@symv! y = beta*y + 1.5*A['L']*x)) == :(Base.LinAlg.BLAS.symv!('L',1.5,A,x,beta,y))
