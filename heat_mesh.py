# Importing numpy for matrices and linear equation solving
import numpy as np
# Importing matplotlib for making plots
import matplotlib.pyplot as plt

# Material properties
H = 0.001  # 1mm
alpha = 111e-6  # 111mm^2/s
source = 1e-5 # K/s

# Reading vertices and triangles from files
points = np.loadtxt("mesh1_points.txt")
elements = np.loadtxt("mesh1_triangles.txt",dtype=np.int64)

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
dofs = n
RHS = np.zeros(dofs)
S = np.zeros((dofs, dofs))
M = np.zeros((dofs, dofs))

# Iterating over elements and their thicknesses
for el, h in zip(elements, element_h):
    # Constructing the Jacobian matrix
    J = np.array([points[el[1],:]-points[el[0],:],points[el[2],:]-points[el[0],:]]).T
    # Selecting the element's DOFs
    local_dof = el
    # Calculating element stiffness matrix
    tmp1 = np.linalg.inv(J.T)
    tmp2 = np.array([[-1, 1, 0], [-1, 0, 1]])
    tmp3 = np.dot(tmp1,tmp2)
    local_S = np.linalg.det(J) * np.dot(tmp3.T, tmp3) * h * alpha
    # Adding the element matrix to the global matrix
    S[np.ix_(local_dof, local_dof)] += local_S
    # Calculating element mass matrix
    local_M = np.linalg.det(J) * np.array([[2, 1, 1], [1, 2, 1], [1, 1, 2]]) / 24
    # Adding the element matrix to the global matrix
    M[np.ix_(local_dof, local_dof)] += local_M

# Plotting the stiffness matrix
plt.imshow(S)
plt.show()
# Plotting the mass matrix
plt.imshow(M)
plt.show()

# Adding uniform heat source (uncomment for source)
# RHS += np.dot(M, np.ones(n)*source)

# Fixing left edge at 60'C
to_fix = np.where(points[:, 0] == 0)[0]
S[to_fix, :] = np.eye(dofs)[to_fix, :]
RHS[to_fix] = 60

# Fixing right edge at 15'C
Lx = points[:,0].max()
to_fix = np.where(points[:, 0] == Lx)[0]
S[to_fix, :] = np.eye(dofs)[to_fix, :]
RHS[to_fix] = 0

# Solving the matrix equation
x = np.linalg.solve(S, RHS)

# Plotting the temperature field
plt.axes().set_aspect('equal')
plt.tripcolor(points[:,0],points[:,1],elements,x,shading='gouraud')
plt.show()

# Plotting the temperature in 3D
ax = plt.figure().add_subplot(projection='3d')
ax.plot_trisurf(points[:,0],points[:,1],elements,x, linewidth=0.2, antialiased=True)
plt.show()


# --- plotting flux ---
# Calculating centers of elements
el_centers = (points[elements[:,0],] + points[elements[:,1],] + points[elements[:,2],])/3
# Initialising flux with 0
Q = np.zeros(el_centers.shape)

# Calculating fluxed for each element
i = 0
for el, h in zip(elements, element_h):
    J = np.array([points[el[1],:]-points[el[0],:],points[el[2],:]-points[el[0],:]]).T
    local_dof = el
    tmp1 = np.linalg.inv(J.T)
    tmp2 = np.array([[-1, 1, 0], [-1, 0, 1]])
    tmp3 = np.dot(tmp1,tmp2)
    Q[i,:] = -np.dot(tmp3, x[el]) * h * alpha
    i = i + 1

# Plotting the fluxes
scale = 1e4
plt.axes().set_aspect('equal')
for p,v in zip(el_centers,Q):
        plt.plot(
            [p[0],p[0]+v[0]*scale],
            [p[1],p[1]+v[1]*scale],
        'k-')
plt.tripcolor(points[:,0],points[:,1],elements,x,shading='gouraud')
plt.show()


