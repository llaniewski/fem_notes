# Importing numpy for matrices and linear equation solving
import numpy as np
# Importing matplotlib for making plots
import matplotlib.pyplot as plt

# Material properties
H = 0.001  # 1mm
alpha = 111e-6  # 111mm^2/s
source = 1e-5 # K/s

# Reading vertices and triangles from files
points = np.loadtxt("mesh/mesh1_points.txt")
elements = np.loadtxt("mesh/mesh1_triangles.txt",dtype=np.int64)

# n=number of points, m=number of elements
n = points.shape[0]
m = elements.shape[0]

# Thickness of all elements is set to H
element_h = np.repeat(H,m)

# Plotting mesh
plt.axes().set_aspect('equal')
plt.triplot(points[:,0],points[:,1],elements)
plt.show()

# Initialising RHS and matrices with zeros
dofs = n*2
RHS = np.zeros(dofs)
S = np.zeros((dofs, dofs))
M = np.zeros((dofs, dofs))

E = 200e9 # [Pa]
pois = 0.25
mu = E/(2*(1+pois))
lam = E*pois/((1+pois)*(1-2*pois))

# Iterating over elements and their thicknesses
for el, h in zip(elements, element_h):
    # Constructing the Jacobian matrix
    J = np.array([points[el[1],:]-points[el[0],:],points[el[2],:]-points[el[0],:]]).T
    # Selecting the element's DOFs
    local_dof = np.concatenate([el,el+n])
    # Calculating element stiffness matrix
    tmp1 = np.linalg.inv(J.T)
    tmp2 = np.array([[-1, 1, 0], [-1, 0, 1]])
    tmp3 = np.dot(tmp1,tmp2)
    grad = np.kron(np.eye(2),tmp3)
    eps_xx = grad[0]
    eps_xy = (grad[1]+grad[2])/2
    eps_yy = grad[3]
    tr_eps = eps_xx + eps_yy
    sigma_xx = 2*mu*eps_xx + lam*tr_eps
    sigma_xy = 2*mu*eps_xy
    sigma_yy = 2*mu*eps_yy + lam*tr_eps
    sigma = np.array([sigma_xx,sigma_xy,sigma_xy,sigma_yy])
    local_S = np.linalg.det(J) * np.dot(grad.T,sigma) * h
    # Adding the element matrix to the global matrix
    S[np.ix_(local_dof, local_dof)] += local_S
    # Calculating element mass matrix
    tmp1 = np.array([[2, 1, 1], [1, 2, 1], [1, 1, 2]]) / 24
    local_M = np.linalg.det(J) * np.kron(np.eye(2),tmp1)
    # Adding the element matrix to the global matrix
    M[np.ix_(local_dof, local_dof)] += local_M

# # Plotting the stiffness matrix
# plt.imshow(S)
# plt.show()
# # Plotting the mass matrix
# plt.imshow(M)
# plt.show()


from scipy.linalg import eigh

eigvals, eigvecs = eigh(S,M)
plt.plot(eigvals)
plt.show()

v = eigvecs[:,5]
u = v.reshape([2,n]).T
plt.axes().set_aspect('equal')
p = points + u*1
plt.triplot(p[:,0],p[:,1],elements)
plt.show()



import matplotlib.animation as animation

fig, ax = plt.subplots()
artists = []
ax.set_aspect('equal')
for i in range(20):
    p = points + u*1*np.cos(2*np.pi*i/20)
    act = ax.triplot(p[:,0],p[:,1],elements,"-",c="black")
    artists.append(act)
ani = animation.ArtistAnimation(fig=fig, artists=artists, interval=100)
ani.save("test.gif")
plt.show()


for i in range(20):
    data += rng.integers(low=0, high=10, size=data.shape)
    container = ax.barh(x, data, color=colors)
    artists.append(container)
ani = animation.FuncAnimation(fig=fig, func=update, frames=40, interval=30)
#plt.show()
ani.save("out.gif")




to_fix = np.where(points[:, 0] == 0)[0]
to_fix = np.concatenate([to_fix,to_fix+n])
S[to_fix, :] = np.eye(dofs)[to_fix, :]
S[:, to_fix] = np.eye(dofs)[:,to_fix]
RHS[to_fix] = 0

# Fixing right edge at 15'C
Lx = points[:,0].max()
to_fix = np.where(points[:, 0] == Lx)[0]
to_fix = to_fix+n
RHS[to_fix] = 1

# Solving the matrix equation
x = np.linalg.solve(S, RHS)

u = x.reshape([2,n]).T

plt.axes().set_aspect('equal')
plt.tripcolor(points[:,0],points[:,1],elements,u[:,0],shading='gouraud')
plt.show()

plt.axes().set_aspect('equal')
plt.tripcolor(points[:,0],points[:,1],elements,u[:,1],shading='gouraud')
plt.show()

# Plotting mesh
plt.axes().set_aspect('equal')
p = points + u*1e5
plt.triplot(p[:,0],p[:,1],elements)
plt.show()


