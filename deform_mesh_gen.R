#install.packages("devtools")
#devtools::install_github("davidcsterratt/RTriangle", subdir="pkg")

k = 0
points = matrix(c(
    0, 0, 3, 3,
    0, 1, 1, 0),ncol=2)
n = nrow(points)
segments = matrix(c(1:n,2:n,1),ncol=2)
k = k + n

n = 20
a = seq(0,2*pi,len=n+1)[-1]
points = rbind(points, matrix(c(cos(a)*0.25+1.5,sin(a)*0.25+0.5),ncol=2))
segments = rbind(segments, matrix(c(1:n,2:n,1),ncol=2)+k)
k = k + n

meshes = list(
    size1=matrix(c(1,0,0,1),2,2),
    size2=matrix(c(2,0,0,2),2,2),
    size3=matrix(c(3,0,0,3),2,2),
    size4=matrix(c(4,0,0,4),2,2),
    size5=matrix(c(5,0,0,5),2,2),
    size6=matrix(c(6,0,0,6),2,2),
    asp12=matrix(c(1,0,0,2),2,2),
    asp13=matrix(c(1,0,0,3),2,2),
    asp21=matrix(c(2,0,0,1),2,2),
    asp31=matrix(c(3,0,0,1),2,2),
    shear1=matrix(c(1,1,0,1),2,2),
    shear2=matrix(c(1,2,0,1),2,2),
    shear3=matrix(c(1,3,0,1),2,2)
)

for (nm in names(meshes)) {
    M = meshes[[nm]]
    p = RTriangle::pslg(points %*% M,S = segments, H=matrix(c(1.5,0.5),ncol=2) %*% M)
    ret = RTriangle::triangulate(p,a=0.01,q=30)
    ret$P = ret$P %*% solve(M)
    plot(ret$P,asp=1)
    segments(ret$P[ret$T[,1],1],ret$P[ret$T[,1],2],ret$P[ret$T[,2],1],ret$P[ret$T[,2],2])
    segments(ret$P[ret$T[,2],1],ret$P[ret$T[,2],2],ret$P[ret$T[,3],1],ret$P[ret$T[,3],2])
    segments(ret$P[ret$T[,3],1],ret$P[ret$T[,3],2],ret$P[ret$T[,1],1],ret$P[ret$T[,1],2])
    write.table(ret$P, paste0("mesh/",nm,"_points.txt"), row.names=FALSE,col.names=FALSE)
    write.table(ret$T-1, paste0("mesh/",nm,"_triangles.txt"), row.names=FALSE,col.names=FALSE)
}
