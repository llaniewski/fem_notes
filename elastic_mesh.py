# Importing numpy for matrices and linear equation solving
import numpy as np
# Importing matplotlib for making plots
import matplotlib.pyplot as plt

# Material properties
H = 0.001  # 1mm
alpha = 111e-6  # 111mm^2/s
source = 1e-5 # K/s

# Reading vertices and triangles from files
points = np.loadtxt("man_points.txt")
elements = np.loadtxt("man_triangles.txt",dtype=np.int64)

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

# print(np.array_str(grad, precision=2, suppress_small=True))
prefix = "points = np.array("
print(prefix,np.array2string(points.T, precision=2, suppress_small=True, separator=",",prefix=prefix))
prefix = "elements = np.array("
print(prefix,np.array2string(elements.T, precision=2, suppress_small=True, separator=",",prefix=prefix))

print(np.array2string(points.T, precision=2, suppress_small=True, separator=",",prefix="  "))
print(np.array2string(elements.T, precision=2, suppress_small=True, separator=",",prefix="  "))

points = np.array(
  [[  0.  ,  0.65,  0.99,  0.85,  0.3 ,  0.3 ,  1.  ,  3.  ,  3.2 ,  1.  ,
      1.  ,  2.  ,  0.5 ,  0.  , -0.5 , -2.  , -1.  , -1.  , -3.2 , -3.  ,
     -1.  , -0.3 , -0.3 , -0.85, -0.99, -0.65, -0.  , -0.65, -0.03,  0.  ,
      0.65,  0.44,  1.  ,  1.  , -0.52,  0.33,  0.47, -0.19,  0.36, -2.02,
     -2.1 ,  2.1 ,  2.  , -1.  ,  0.24, -2.6 ,  2.59, -0.36,  0.49, -0.09,
      1.  , -1.  , -1.  , -0.41, -0.23, -0.38, -1.5 ,  1.5 , -0.54, -0.05,
      0.49, -1.25,  1.25, -0.69,  1.12, -0.25,  0.12,  0.25,  0.55,  0.65,
      0.06,  0.67,  0.75, -0.12,  1.38,  1.75, -0.52, -1.38,  0.38, -0.79,
      1.62, -0.87, -1.75, -0.19,  0.31,  0.82,  1.25,  0.72,  0.96,  1.27,
      1.88, -0.99, -1.62, -0.38, -0.6 , -1.18, -1.25, -1.48,  1.48, -0.44,
     -0.91, -1.88, -0.95],
   [  1.  ,  0.76,  0.15, -0.53, -0.95, -1.4 , -1.4 , -1.  , -1.8 , -2.3 ,
     -5.  ,-10.  ,-10.  , -6.  ,-10.  ,-10.  , -5.  , -2.3 , -1.8 , -1.  ,
     -1.4 , -1.4 , -0.95, -0.53,  0.15,  0.76, -0.  , -1.85, -2.7 , -1.99,
     -1.85, -2.38, -3.65, -2.98, -0.11, -0.42,  0.26,  0.5 , -3.31, -1.2 ,
     -2.05, -2.05, -1.2 , -3.65, -4.33, -1.53, -1.53, -3.33, -3.84, -3.85,
     -4.33, -4.33, -2.98, -2.31, -5.23, -4.4 , -7.5 , -7.5 , -4.85, -4.76,
     -4.8 , -6.25, -6.25, -5.79, -5.62, -8.  , -7.  , -8.  , -5.42, -6.05,
     -6.5 , -6.68, -7.43, -7.  , -6.88, -8.75, -6.44, -6.88, -9.  , -7.32,
     -8.12, -6.81, -8.75, -7.5 , -8.5 , -8.69,-10.  , -8.2 , -9.37, -8.95,
     -9.38, -8.03, -8.12, -9.  , -8.46, -8.59,-10.  , -1.85, -1.85, -9.5 ,
     -9.05, -9.38, -9.57]]).T
elements = np.array(
  [[ 38, 97, 79, 31, 38, 63, 32, 21, 19, 39, 53, 19, 61, 38,  5, 83, 34,
     37,  4,  3, 97,  2, 64, 34, 98,  4, 54, 29,  5, 22, 36, 30, 47,  7,
      1, 34, 37, 37,  9, 98, 35, 29,  9, 21, 36, 35, 37, 67, 48, 22,  7,
     30, 49, 52, 44, 18, 40, 42, 41, 28, 48, 49, 44, 64, 59, 43, 44, 58,
     28, 43, 17, 52, 43, 43, 16, 73, 10, 59, 51, 54, 44, 44, 60, 16, 54,
     64, 13, 72, 60, 56, 68, 71, 71, 71, 69, 72, 74, 89, 74, 75, 11, 72,
     87, 63, 56, 73, 61, 85, 86, 72, 61, 77,100, 94, 12, 79, 84, 80, 89,
     88, 72, 85, 88, 82, 90, 93, 95, 91, 95, 93, 92,100,101,102, 39, 40,
      9, 42, 82, 99,102, 14,101],
   [ 31, 17, 91, 29, 48, 76, 33, 29, 45, 40, 29, 18, 63, 33, 22, 73, 22,
     24, 35, 35, 27, 35, 68, 26, 30, 22, 13, 21, 29, 34, 37,  6, 38, 42,
     37, 24,  0,  1, 31, 42,  2, 31, 30, 20,  2, 36, 25, 84, 50, 35, 46,
     98, 48, 47, 49, 40, 39, 41,  8, 53, 49, 38, 60, 69, 55, 51, 50, 51,
     29, 47, 53, 53, 55, 49, 54, 81, 50, 58, 58, 68, 55, 59, 59, 63, 63,
     10, 76, 71, 68, 91, 13, 69, 70, 62, 13, 66, 62, 80, 72, 80, 90, 74,
     80, 61, 79, 76, 77, 84, 88, 87, 81, 79, 99, 65, 88, 65, 85, 87, 90,
     85, 67, 88, 86, 95, 89, 94, 94, 56,100, 65, 95, 82,102, 14, 97, 17,
     41, 98,100,100,101,102, 15],
   [ 28, 27, 65, 28, 32, 13, 38,  5, 39, 97, 27, 45, 16, 31, 21, 79, 26,
     34, 22,  4, 20,  3, 69, 37,  9,  5, 68, 27, 30, 23, 26,  5, 28, 46,
     36, 23, 25,  0, 33,  6, 36, 30, 31, 27,  1, 26, 24, 87, 32, 26,  8,
      6, 38, 28, 55, 45, 45, 46, 46, 52, 44, 47, 50, 62, 58, 55, 48, 16,
     53, 52, 27, 17, 49, 47, 58, 79, 60, 54, 55, 60, 59, 60, 54, 54, 13,
     68, 73, 66, 10, 79, 69, 70, 66, 69, 70, 67, 71, 85, 57, 89, 86, 71,
     57, 76, 77, 81, 81, 78, 12, 57, 76, 81, 93, 91, 78, 83, 87, 85, 75,
     78, 87, 89, 90, 92, 88,100, 91, 92, 94, 94, 91,101,100, 99, 20, 97,
     98, 41, 95,102, 96, 96, 96]]).T

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




# Adding uniform heat source (uncomment for source)
# RHS += np.dot(M, np.ones(n)*source)

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


