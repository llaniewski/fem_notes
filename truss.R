E = 210e9 # [Pa]
g = 9.81 # m/s^2
rho = 7850 # kg/m^3
A1 = 18e-3 # m^2


points <- data.frame(
    id= c(0, 1, 2, 3, 4,   5,   6,   7,   8),
    x = c(0, 1, 2, 3, 4, 0.5, 1.5, 2.5, 3.5), # [m]
    y = c(0, 0, 0, 0, 0,   1,   1,   1,   1)  # [m]
)

elements <- data.frame(
    i1 = c(0,1,2,3,5,6,7,0,5,1,6,2,7,3,8), # 0-indexing
    i2 = c(1,2,3,4,6,7,8,5,1,6,2,7,3,8,4), # 0-indexing
    A =  A1 # [m^2]
)

plot_beams = function(px,py,i1,i2) {
    plot(px, py, asp=1)
    segments(px[i1+1],py[i1+1],px[i2+1],py[i2+1])
}

plot_beams(points$x, points$y, elements$i1, elements$i2)

dofs = nrow(points)*2
S = matrix(0,dofs,dofs)
RHS = rep(0,dofs)

for (i in seq_len(nrow(elements))) {
    i1 = elements$i1[i]
    i2 = elements$i2[i]
    A = elements$A[i]
    local_dof = c(2*i1,2*i1+1,2*i2,2*i2+1)
    n_x = points$x[i2+1] - points$x[i1+1]
    n_y = points$y[i2+1] - points$y[i1+1]
    L = sqrt(n_x^2+n_y^2)
    n_x = n_x / L
    n_y = n_y / L
    N = as.matrix(c(-n_x,-n_y,n_x,n_y))
    local_S = N %*% (A*E/L) %*% t(N)
    S[local_dof+1,local_dof+1] = S[local_dof+1,local_dof+1] + local_S
    local_load = c(0,-0.5,0,-0.5)*g*A*L*rho
    RHS[local_dof+1] = RHS[local_dof+1] + local_load
}
total_weigth = sum(RHS)

to_load = c(2*2+1)
load = -1e3*g # 1 tonne
RHS[to_load+1] = load

to_fix = c(2*0+0,2*0+1,2*4+0,2*4+1)
S[to_fix+1, ] = diag(dofs)[to_fix+1, ]
RHS[to_fix+1] = 0

x = solve(S, RHS)
displacement = matrix(x,ncol=2,byrow = TRUE)

scale = 10000
plot_beams(points$x + scale*displacement[,1], points$y + scale*displacement[,2], elements$i1, elements$i2)
