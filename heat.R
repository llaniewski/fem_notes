rm(list=ls())

library(rgl) # for the nice color gadient triangle elements

Lx = 2
Ly = 1
H = 0.001 # 1mm
nx = 21
ny = 10
alpha = 111e-6 # 111mm^/s

points = expand.grid(x=seq(0,Lx,len=nx),y=seq(0,Ly,len=ny))
m = nrow(points)
i0 = which(points$x != Lx & points$y != Ly) - 1 # 0-index
elements = data.frame(
    i0 = c(i0,i0),
    i1 = c(i0+1,i0+nx+1),
    i2 = c(i0+nx+1,i0+nx),
    h = H
)

elements$center_x = (points$x[elements$i0+1]+points$x[elements$i1+1]+points$x[elements$i2+1])/3
elements$center_y = (points$y[elements$i0+1]+points$y[elements$i1+1]+points$y[elements$i2+1])/3

sel = elements$center_x > Lx*0.2 & elements$center_x < Lx*0.4 & elements$center_y > Ly*0.3333
sel = sel | (elements$center_x > Lx*0.6 & elements$center_x < Lx*0.8 & elements$center_y < Ly*0.6666)
elements$h[sel] = H/10

plot_mesh = function(px,py,i0,i1,i2,col=0,alpha=1) {
    plot(px,py,asp=1)
    plot3d(px, py,0, asp="iso")
    if (max(alpha) > 0) {
        alpha = alpha / max(alpha)
    }
    if (max(col) > min(col)) {
        col = (col-min(col))/(max(col)-min(col))
        col = colorRamp(c("black","red","yellow","white"))(col)
        col = rgb(col,max=255)
    } else {
        col[] = 3
    }
    i = as.vector(rbind(i0,i1,i2))
    if (length(alpha) == length(i0)) alpha=rep(alpha,each=3)
    if (length(alpha) == length(px)) alpha=alpha[i+1]
    if (length(col) == length(i0)) col=rep(col,each=3)
    if (length(col) == length(px)) col=col[i+1]
    triangles3d(
        px[i+1],
        py[i+1],
        0,
        col=col,
        alpha=alpha
    )
}

plot_mesh(points$x, points$y, elements$i0, elements$i1, elements$i2, alpha=elements$h)

dofs = nrow(points)
RHS = rep(0,dofs)
S = matrix(0, dofs, dofs)
M = matrix(0, dofs, dofs)

for (i in seq_len(nrow(elements))) {
    i0 = elements$i0[i]
    i1 = elements$i1[i]
    i2 = elements$i2[i]
    h = elements$h[i]
    J = matrix(c(
        points$x[i1+1] - points$x[i0+1], points$y[i1+1] - points$y[i0+1],
        points$x[i2+1] - points$x[i0+1], points$y[i2+1] - points$y[i0+1]
        ),2,2)
    local_dof = c(i0,i1,i2)
    tmp1 = det(J) * solve(t(J) %*% J)
    tmp2 = matrix(c(-1,-1,1,0,0,1),2,3)
    local_S = t(tmp2) %*% tmp1 %*% tmp2 * h * alpha
    S[local_dof+1,local_dof+1] = S[local_dof+1,local_dof+1] + local_S
    local_M = det(S) * matrix(c(2,1,1,1,2,1,1,1,2),3,3)/24
    M[local_dof+1,local_dof+1] = M[local_dof+1,local_dof+1] + local_M
}

to_fix = which(points$x == 0)-1
S[to_fix+1, ] = diag(dofs)[to_fix+1, ]
RHS[to_fix+1] = 100

to_fix = which(points$x == Lx)-1
S[to_fix+1, ] = diag(dofs)[to_fix+1, ]
RHS[to_fix+1] = 0

x = solve(S,RHS)

plot_mesh(points$x, points$y, elements$i0, elements$i1, elements$i2, col=x, alpha=elements$h)
