# Importing numpy for matrices and linear equation solving
import numpy as np
# Importing matplotlib for making plots
import matplotlib.pyplot as plt

# Material properties
H = 0.001  # 1mm
alpha = 111e-6  # 111mm^2/s
source = 1e-5 # K/s

E = 200e9 # [Pa]
pois = 0.25
mu = E/(2*(1+pois))
lam = E*pois/((1+pois)*(1-2*pois))

def solve(name):
    # Reading vertices and triangles from files
    points = np.loadtxt(f"mesh/{name}_points.txt")
    elements = np.loadtxt(f"mesh/{name}_triangles.txt",dtype=np.int64)
    # n=number of points, m=number of elements
    n = points.shape[0]
    m = elements.shape[0]
    # Thickness of all elements is set to H
    element_h = np.repeat(H,m)
    # Initialising RHS and matrices with zeros
    dofs = n*2
    RHS = np.zeros(dofs)
    S = np.zeros((dofs, dofs))
    M = np.zeros((dofs, dofs))

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

    RHS = np.dot(M, np.concatenate([np.zeros([n]),np.ones([n])]))

    to_fix = np.where(points[:, 0] == 0)[0]
    to_fix = np.concatenate([to_fix,to_fix+n])
    S[to_fix, :] = np.eye(dofs)[to_fix, :]
    S[:, to_fix] = np.eye(dofs)[:,to_fix]
    RHS[to_fix] = 0

    # Solving the matrix equation
    x = np.linalg.solve(S, RHS)

    u = x.reshape([2,n]).T

    return [points, elements, u]

meshes = ["size1","size2","size3","size4","size5","size6","asp12","asp13","asp21","asp31","shear1","shear2","shear3"]
res = []
for nm in meshes:
    [points, elements, u] = solve(nm)
    corner = np.where(np.bitwise_and(points[:,0] == 3,points[:,1] == 0))[0][0]
    res.append(np.concatenate([[points.shape[0],elements.shape[0]],u[corner,:]]))

res = np.vstack(res)

plt.plot(res)
plt.show()

x = res[:,1]
y = np.abs(res[:,3] - res[5,3])
plt.loglog(x,y,"o")
for [x0,y0,nm] in zip(x,y,meshes):
    plt.text(x0,y0,nm)
plt.show()
