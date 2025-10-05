import numpy
import math
from numba import jit

@jit(nopython=True)
def _ij2symmetric(i,j,size):
    return (size * (size + 1))/2 - (size-i)*((size - i + 1))/2 + j - i

@jit(nopython=True)
def _jitted_ADMM_z(x, u, lamb, rho, numBlocks, sizeBlocks, length):
    a = x + u
    probSize = numBlocks*sizeBlocks
    z_update = numpy.zeros(length)

    for i in range(numBlocks):
        elems = numBlocks if i==0 else (2*numBlocks - 2*i)/2 # i=0 is diagonal
        for j in range(sizeBlocks):
            startPoint = j if i==0 else 0
            for k in range(startPoint, sizeBlocks):
                # locList = [((l+i)*sizeBlocks + j, l*sizeBlocks+k) for l in range(int(elems))]
                locList_0 = numpy.array([(l+i)*sizeBlocks + j for l in range(int(elems))])
                locList_1 = numpy.array([l*sizeBlocks+k for l in range(int(elems))])
                if i == 0:
                    lamSum = 0
                    for idx in range(len(locList_0)):
                        lamSum += lamb[locList_0[idx], locList_1[idx]]

                    indices = numpy.zeros(len(locList_0))
                    for idx in range(len(locList_0)):
                        indices[idx] = _ij2symmetric(locList_0[idx], locList_1[idx], probSize)
                else:
                    lamSum = 0
                    for idx in range(len(locList_0)):
                        lamSum += lamb[locList_1[idx], locList_0[idx]]

                    indices = numpy.zeros(len(locList_0))
                    for idx in range(len(locList_0)):
                        indices[idx] = _ij2symmetric(locList_1[idx], locList_0[idx], probSize)

                pointSum = 0
                for index in indices:
                    pointSum += a[int(index)]
                rhoPointSum = rho * pointSum

                #Calculate soft threshold
                ans = 0
                #If answer is positive
                if rhoPointSum > lamSum:
                    ans = max((rhoPointSum - lamSum)/(rho*elems),0)
                elif rhoPointSum < -1*lamSum:
                    ans = min((rhoPointSum + lamSum)/(rho*elems),0)

                for index in indices:
                    z_update[int(index)] = ans
    return z_update

class ADMMSolver:
    def __init__(self, lamb, num_stacked, size_blocks, rho, S, rho_update_func=None):
        self.lamb = lamb
        self.numBlocks = num_stacked
        self.sizeBlocks = size_blocks
        probSize = num_stacked*size_blocks
        self.length = int(probSize*(probSize+1)/2)
        self.x = numpy.zeros(self.length)
        self.z = numpy.zeros(self.length)
        self.u = numpy.zeros(self.length)
        self.rho = float(rho)
        self.S = S
        self.status = 'initialized'
        self.rho_update_func = rho_update_func

    def upper2Full(self, a):
        n = int((-1  + numpy.sqrt(1+ 8*a.shape[0]))/2)
        A = numpy.zeros([n,n])
        A[numpy.triu_indices(n)] = a
        temp = A.diagonal()
        A = (A + A.T) - numpy.diag(temp)
        return A

    def Prox_logdet(self, S, A, eta):
        d, q = numpy.linalg.eigh(eta*A-S)
        q = numpy.matrix(q)
        X_var = ( 1/(2*float(eta)) )*q*( numpy.diag(d + numpy.sqrt(numpy.square(d) + (4*eta)*numpy.ones(d.shape))) )*q.T
        x_var = X_var[numpy.triu_indices(S.shape[1])] # extract upper triangular part as update variable
        return numpy.matrix(x_var).T

    def ADMM_x(self):
        a = self.z-self.u
        A = self.upper2Full(a)
        eta = self.rho
        x_update = self.Prox_logdet(self.S, A, eta)
        self.x = numpy.array(x_update).T.reshape(-1)

    def ADMM_z(self, index_penalty = 1):
        self.z = _jitted_ADMM_z(self.x, self.u, self.lamb, self.rho, self.numBlocks, self.sizeBlocks, self.length)

    def ADMM_u(self):
        u_update = self.u + self.x - self.z
        self.u = u_update

    # Returns True if convergence criteria have been satisfied
    # eps_abs = eps_rel = 0.01
    # r = x - z
    # s = rho * (z - z_old)
    # e_pri = sqrt(length) * e_abs + e_rel * max(||x||, ||z||)
    # e_dual = sqrt(length) * e_abs + e_rel * ||rho * u||
    # Should stop if (||r|| <= e_pri) and (||s|| <= e_dual)
    # Returns (boolean shouldStop, primal residual value, primal threshold,
    #          dual residual value, dual threshold)
    def CheckConvergence(self, z_old, e_abs, e_rel, verbose):
        norm = numpy.linalg.norm
        r = self.x - self.z
        s = self.rho * (self.z - z_old)
        # Primal and dual thresholds. Add .0001 to prevent the case of 0.
        e_pri = math.sqrt(self.length) * e_abs + e_rel * max(norm(self.x), norm(self.z)) + .0001
        e_dual = math.sqrt(self.length) * e_abs + e_rel * norm(self.rho * self.u) + .0001
        # Primal and dual residuals
        res_pri = norm(r)
        res_dual = norm(s)
        if verbose:
            # Debugging information to print(convergence criteria values)
            print('  r:', res_pri)
            print('  e_pri:', e_pri)
            print('  s:', res_dual)
            print('  e_dual:', e_dual)
        stop = (res_pri <= e_pri) and (res_dual <= e_dual)
        return (stop, res_pri, e_pri, res_dual, e_dual)

    #solve
    def __call__(self, maxIters, eps_abs, eps_rel, verbose):
        num_iterations = 0
        self.status = 'Incomplete: max iterations reached'
        for i in range(maxIters):
            z_old = numpy.copy(self.z)
            self.ADMM_x()
            self.ADMM_z()
            self.ADMM_u()
            if i != 0:
                stop, res_pri, e_pri, res_dual, e_dual = self.CheckConvergence(z_old, eps_abs, eps_rel, verbose)
                if stop:
                    self.status = 'Optimal'
                    break
                new_rho = self.rho
                if self.rho_update_func:
                    new_rho = rho_update_func(self.rho, res_pri, e_pri, res_dual, e_dual)
                scale = self.rho / new_rho
                rho = new_rho
                self.u = scale*self.u
            if verbose:
                # Debugging information prints current iteration #
                print('Iteration %d' % i)
        return self.x
