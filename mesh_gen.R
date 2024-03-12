#install.packages("devtools")
#devtools::install_github("davidcsterratt/RTriangle", subdir="pkg")

neck = 0.3
a = asin(neck)
a = seq(0,pi-a,len=5)

k = 0
points = matrix(c(
    sin(a),neck,1,3,3.2,1,1,2,0.5,0,
    cos(a),-1.4,-1.4,-1,-1.8,-2.3,-5,-10,-10,-6
),ncol=2)
n = nrow(points)
points = rbind(points,points[(n-1):2,] %*% diag(c(-1,1)))
n = nrow(points)
segments = matrix(c(1:n,2:n,1),ncol=2)
k = k + n



plot(points,asp=1)
segments(points[segments[,1],1],points[segments[,1],2],points[segments[,2],1],points[segments[,2],2])

p = RTriangle::pslg(points,S = segments)

ret = RTriangle::triangulate(p,a=0.3,q=30)

plot(ret$P,asp=1)
segments(ret$P[ret$T[,1],1],ret$P[ret$T[,1],2],ret$P[ret$T[,2],1],ret$P[ret$T[,2],2])
segments(ret$P[ret$T[,2],1],ret$P[ret$T[,2],2],ret$P[ret$T[,3],1],ret$P[ret$T[,3],2])
segments(ret$P[ret$T[,3],1],ret$P[ret$T[,3],2],ret$P[ret$T[,1],1],ret$P[ret$T[,1],2])

write.table(ret$P, "man_points.txt", row.names=FALSE,col.names=FALSE)
write.table(ret$T-1, "man_triangles.txt", row.names=FALSE,col.names=FALSE)


tab = ret$P
tab[] = sprintf("%.2f",tab)
tab = paste0("[",tab[,1],",",tab[,1],"]")
n = length(tab)
tab = matrix(tab,nrow=3)
tab[-(1:n)] = NA
apply(tab,2,paste0,collapse=",")
