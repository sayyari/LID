import numpy as np

# Computes the LGL nodes - Modified
# From https://www.mathworks.com/matlabcentral/fileexchange/4775-legende-gauss-lobatto-nodes-and-weights
# Written by Greg von Winckel - 04/17/2004
# Contact: gregvw@chtm.unm.edu
# Input:
# - p is the order of accuracy
def lglnodes(p):

    # Truncation + 1
    N = p+1
    # Use the Chebyshev-Gauss-Lobatto nodes as the first guess
    x = np.cos(np.pi*np.array([i for i in range(N)])/p)
    # The Legendre Vandermonde Matrix
    P = np.zeros((N,N))
    # Compute P_(N) using the recursion relation
    # Compute its first and second derivatives and 
    # update x using the Newton-Raphson method.
    xold = 2*np.ones(N)
    eps = 10e-16
    while max(abs(x-xold)) > eps:
        xold = x

        P.T[0] = np.ones(N)
        P.T[1] = x

        for k in range(1,p):
            P.T[k+1] = ((2*(k+1)-1)*x*P.T[k]-(k)*P.T[k-1])/(k+1)

        x = xold - (x*P.T[N-1]-P.T[p-1])/(N*P.T[N-1])

    return x

# Lagrange polynomial
# Input:
# - x are the set of points definig the roots of the Lagrange polynomial
def Lagrange(x):
    # x are the collocation points
    
    N = len(x)
    
    def LagrangePolynomial(y):
        L = np.ones(N)
        
        for j in range(N):
            for k in range(N):
                if k != j:
                    L[j] *= (y-x[k])/(x[j]-x[k])
        
        return L
    
    return LagrangePolynomial

# Interpolating function
# Input:
# - p is the original order of accuracy
# - q is the new order of accuracy
# - u is a global solution on the LGL points
def interpolate(u,p,q):
    
    N = p + 1
    M = q + 1
    
    # Define the lglnodes for the original accuracy
    x = lglnodes(p)

    # Define the lglnodes for the new accuracy
    y = lglnodes(q)
    
    # Assemble the interpolation operator using LGL points
    L = Lagrange(x)
    Li = []
    for yi in y: Li += [L(yi)]
    Li = np.array(Li)
    
    # Interpolate the solution on the new set of points for each element
    uy = Li.dot(u[:N])
        
    return y,uy

# Interpolating function
# Input:
# - p is the original order of accuracy
# - n is the number of new equidistant points
# - u is a global solution on the LGL points
def visualize(u,p,n=100):
    
    N = p + 1
    
    # Define the lglnodes for the original accuracy
    x = lglnodes(p)

    # Define the lglnodes for the new accuracy
    y = np.linspace(-1,1,n)
    
    # Assemble the interpolation operator using LGL points
    L = Lagrange(x)
    Li = []
    for yi in y: Li += [L(yi)]
    Li = np.array(Li)
    
    # Interpolate the solution on the new set of points for each element
    uy = Li.dot(u[:N])
        
    return y,uy

import sys,os
sys.path.insert(0,os.environ['SSDC_DIR']+"/lib/ssdc/python")
from ssdc.io import *

# Load data from ssdc data files
def loadDat(solutionData, gridData, bodyData=""):
    
    # Load grid data
    io = IO().open(gridData)
    grid = io.load()
    deg, nodes = grid
    
    # Load solution data
    io = IO().open(solutionData)
    state = io.load()
    step, time, soln = state
    soln.shape = (nodes.shape[0], -1)
    
    elem = []
    sln = []
    for i in range(len(deg)):
        elem += [nodes[i*(deg[i]+1)**2:(i+1)*(deg[i]+1)**2]]
        sln += [soln[i*(deg[i]+1)**2:(i+1)*(deg[i]+1)**2]]
        
    # Load body data
    if bodyData != "":
        body = np.loadtxt(bodyData, dtype=int)
        marker = np.zeros(nodes.shape[0], dtype=int)
        marker[body] = 1
        return elem, sln, deg, step, time, marker
        
    return elem, sln, deg, step, time

# Load a CSV file
def loadCSV(file):
    
    file = open(file,"r")
    lines = file.readlines()

    csvsolution = []
    for line in lines[5:]:

        data = line.split(',')

        results = []
        for word in data:
            results += [float(word)]

        csvsolution += [np.array(results)]

    csvsolution = np.array(csvsolution)

    nodes = csvsolution.T[:2]
    body = csvsolution.T[2]
    boundary = csvsolution.T[3]
    soln = csvsolution.T[4:8]
    deg = csvsolution.T[-1]
    
    tmp = []
    for i in range(len(deg)):
        tmp += [int(deg[i])]
    deg = np.array(tmp)
    
    nodes = nodes.T
    soln = soln.T
    
    elem = []
    sln = []
    for i in range(len(deg)):
        elem += [nodes[i*(deg[i]+1)**2:(i+1)*(deg[i]+1)**2]]
        sln += [soln[i*(deg[i]+1)**2:(i+1)*(deg[i]+1)**2]]
    
    return elem, sln, deg

# Helper function to determine the boundary layer nodes
# Input:
# - x, y are the sets of coordinates.
# - i is the node index.
# - body is the set of flags for body nodes.
def boundaryLayer(x,y,i,body):
        
    for b in body:
        error = np.linalg.norm(np.array([x[i]-x[b],y[i]-y[b]]))
        if error < 0.26:
            return 1
    return 0

# Return marked body edges
# Input:
# - NN is the number of nodes in each element
# - nel is the number of elements
# - side is the side ("n","e","s","w") or anything else for no body edge
def newBody(NN,nel,side):
    
    N = int(np.sqrt(NN))
    marker = []
    for e in range(nel):
        tmp = np.zeros(NN)
    
        # North
        if side[e] == "n":
            for i in range(N):
                tmp[i] = 1

        # East
        elif side[e] == "e":
            for i in range(NN):
                if (i+1)%N == 0:
                    tmp[i] = 1

        # South
        elif side[e] == "s":
            for i in range(N):
                tmp[-(i+1)] = 1

        # West
        elif side[e] == "w":
            for i in range(NN):
                if i%N == 0:
                    tmp[i] = 1
                    
        marker += list(tmp)

    return np.array(marker, dtype = int)

# Return the side to be marked
# Input:
# - marker is the old marked indices
# - nel is the number of elements
def sideMarker(marker,nel):
    oldN = int(np.sqrt(len(marker)/nel))

    mark = np.zeros((oldN)**2)
    side = []
    for i in range(nel):
        mark = marker[i*(oldN)**2:(i+1)*(oldN)**2]
        smark = "f"
        if mark[0] == 1 and mark[1] == 1:
            smark = "n"
        if mark[oldN-1] == 1 and mark[oldN*2-1] == 1:
            smark = "e"
        if mark[-1] == 1 and mark[-2] == 1:
            smark = "s"
        if mark[-(oldN)] == 1 and mark[-(2*oldN)] == 1:
            smark = "w"

        side += [smark]

    return side

def writeCSV(file,elem,soln,deg,body=[],marker=[]):
    
    # must be marked for the new grid by ssdc
    # with a one step run on the mesh
    if len(marker) > 0 and len(body) > 0:
        nodes = elem2data(elem)
        x, y = nodes.T
        
        boundary = []
        for i in range(len(nodes)):
            boundary += [boundaryLayer(x,y,i,body)]
            
    else:
        mark  = np.zeros((deg[0]+1)**2, dtype = int)
        layer = np.zeros((deg[0]+1)**2, dtype = int)

    # Write the CSV file
    file = open(file,"a")
    header = "Elements, Points\n"
    file.write(header)
    file.write("{}, {}\n".format(len(elem), len(elem)*(deg[0]+1)**2))
    header = "X, Y, BODY, BLAYER, RHO, U, V, TEMP, ELEM, Deg\n"
    file.write(header)
    for e in range(len(elem)):

        NN = len(elem[e])
        x, y = elem[e].T
        if len(marker) > 0 and len(body) > 0:
            mark = marker[e*NN:(e+1)*NN]
            layer = boundary[e*NN:(e+1)*NN]
        density, Xvel, Yvel, temp = soln[e].T
        for i in range(NN):
            line = "{}, {}, {}, {}, {}, {}, {}, {}, {}, {}\n".format(x[i],y[i],mark[i], layer[i],density[i],Xvel[i],Yvel[i],temp[i],e,deg[i])
            file.write(line)
    file.close()

#################################################
################ Interpolation ##################
#################################################
# Interpolation functionalities should not be used 
# directly by the user

# Convert elementwise node pairing to one vector
def elem2data(elem):
    dataElem = []
    for e in range(len(elem)):
        dataElem += list(elem[e])

    return np.array(dataElem)

# Chop an element into edges
# in direction A (rows)
def quad2edgesA(element):

    N = int(np.sqrt(len(element)))
    dofs = element.T
    
    elemA = []
    for i in range(N):
        tmp = []
        for j in range(len(dofs)):
            tmp += [dofs[j][i*N:(i+1)*N]]
        elemA += [np.array(tmp)]
    
    return elemA

# Chop an element into sets of edges
# in direction B (columns)
# This is achieved by means of transposition
# Thus, run edges to quad and then run it 
# again to realign
def quad2edgesB(element):
    
    N = int(np.sqrt(len(element)))
    M = int(len(element)/N)
    dofs = element.T

    # create the set of points in direction B
    dofsT = [[] for _ in range(len(dofs))]
    for i in range(M):
        for j in range(N):
            for k in range(len(dofs)):
                dofsT[k] += [dofs[k][j*(M)+i]]
                
    # construct elements
    elemB = []
    for i in range(int(len(element)/N)):
        tmp = []
        for j in range(len(dofsT)):
            tmp += [dofsT[j][i*N:(i+1)*N]]
        elemB += [np.array(tmp)]
        
    return elemB

# Convert sets of edges into an element
def edges2quad(A):
    element = []
    for i in range(len(A)):
        for j in range(len(A[i].T)):
            element += [A[i].T[j]]

    return np.array(element)

########################################################
######################## Main ##########################
########################################################
# The main function for interpolating the whole domain
# input:
# - obj is the solution degrees of freedom or set of coordinates 
#       They can be in any dimension, thus, you may pass x, y and 
#       solution as one object.
# - deg is the set of degrees of accuracy for each element 
#       (must correspond to obj).
# - q is the desired degree of accuracy.
def mapP2Q(obj,deg,q):
    
    # Initialize the new object and set of degrees of accuracy
    newObj = []
    newDeg = []
    for e in range(len(deg)):

        # Interpolate in direction A
        edgesA = quad2edgesA(obj[e])
        newA = []
        for i in range(len(edgesA)):
            tmp = []
            for j in range(len(edgesA[i])):
                tmp += [interpolate(edgesA[i][j],deg[e],q)[1]]
            newA += [np.array(tmp)]
        newElem = edges2quad(newA) # Revert back to element

        # Interpolate in direction B
        edgesB = quad2edgesB(newElem)
        newB = []
        for i in range(len(edgesB)):
            tmp = []
            for j in range(len(edgesB[i])):
                tmp += [interpolate(edgesB[i][j],deg[e],q)[1]]
            newB += [np.array(tmp)]
        tempElem = edges2quad(newB) # Revert back to element

        # Transpose the element back to original orientation
        tempA = quad2edgesB(tempElem)
        newElem = edges2quad(tempA)

        # Populate the sets of elements and degrees
        newObj += [newElem]
        newDeg += [q]

    return newObj, newDeg

# The main function for visualizing the whole domain
# input:
# - obj is the solution degrees of freedom or set of coordinates 
#       They can be in any dimension, thus, you may pass x, y and 
#       solution as one object.
# - deg is the set of degrees of accuracy for each element 
#       (must correspond to obj).
# - n is the desired number of visualization points
def mapP2Visualize(obj,deg,n=100):
    
    # Initialize the new object and set of degrees of accuracy
    newObj = []
    newDeg = []
    for e in range(len(deg)):

        # Interpolate in direction A
        edgesA = quad2edgesA(obj[e])
        newA = []
        for i in range(len(edgesA)):
            tmp = []
            for j in range(len(edgesA[i])):
                tmp += [visualize(edgesA[i][j],deg[e],n)[1]]
            newA += [np.array(tmp)]
        newElem = edges2quad(newA) # Revert back to element

        # Interpolate in direction B
        edgesB = quad2edgesB(newElem)
        newB = []
        for i in range(len(edgesB)):
            tmp = []
            for j in range(len(edgesB[i])):
                tmp += [visualize(edgesB[i][j],deg[e],n)[1]]
            newB += [np.array(tmp)]
        tempElem = edges2quad(newB) # Revert back to element

        # Transpose the element back to original orientation
        tempA = quad2edgesB(tempElem)
        newElem = edges2quad(tempA)

        # Populate the sets of elements and degrees
        newObj += [newElem]
        newDeg += [n-1]

    return newObj, newDeg

