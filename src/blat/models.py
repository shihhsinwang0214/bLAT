import jax.numpy as jnp
import jax.scipy as jsp
import jax
from functools import partial
from smlib.helpers import get_complex, get_real

#from jax.config import config
jax.config.update("jax_enable_x64", True)

"""
Notes:
Layer:
contains specific implementation for matrix_forward
init_params is generalized based on self.num_param_vec

Composition:
contains multiple layers inside of self.layers list
generalized init_params and matrix_forward

Layers:
 - unitary matrix
 - diagonal matrix
   - scale elements so max |d_i| <= max_scale
   - enforce real(d_i) <= 0
 - upper triangular
 - activation
 - 

"""

@partial(jax.jit, static_argnums=(1,2))
def _access_params(params, start, end):
    return params[start:end]

@jax.jit
def get_wv(wv_params, k):
    """
    input:
    wv_params: array of complex numbers
    k:         scaling parameter for wv

    output:
    wv: complex array that satisfies |wv| <= 1
    """

    wv_params_norm = jnp.linalg.norm(wv_params)
    scale = jnp.tanh(k * wv_params_norm)
    wv = scale * wv_params / wv_params_norm

    return wv

@partial(jax.jit, static_argnums=(2))
def psi_p(wv, y, L):
    """
    multiply by psi matrix in Polcari parameterization
    """
    # L = jnp.size(wv)
    u1 = y[0:L]
    u2 = y[L][None, :]

    last = jnp.sqrt(jnp.maximum(1 - wv.T.conj() @ wv, 0))
    dot = wv.T.conj() @ u1

    y = y.at[0:L].set(u1 - (wv @ dot) / (last + 1) - wv @ u2)
    y = y.at[L].set(dot[0] + last[0] * u2[0])

    return y

@partial(jax.jit, static_argnums=(2))
def psi_str_p(wv, y, L):
    """
    multiply by conjugate transpose of psi matrix in Polcari parameterization
    """
    # L = jnp.size(wv)
    u1 = y[0:L]
    u2 = y[L][None, :]

    last = jnp.sqrt(jnp.maximum(1 - wv.T.conj() @ wv, 0.0))
    dot = wv.T.conj() @ u1

    y = y.at[0:L].set(u1 - (wv @ dot) / (last + 1) + wv @ u2)
    y = y.at[L].set(- dot[0] + last[0] * u2[0])

    return y

# @partial(jax.jit, static_argnums=(3,4,5))
# def psis_p(wv_params_vec, y, k, M, dist, end):
def psis_p(wv_params_vec, y, k, starts):
    """
    multiple by a string of psi matrices from polcari parameterization

    k:      scaling parameter for wv
    N:      size of square unitary matrix
    starts: indices into wv_params_vec
            wv_params{i} = wv_params_vec[starts[i]:starts[i+1]]
    """
    M = len(starts)
    dist = starts[-1]-starts[-2]
    end = starts[-1]
    
    for i in range(M-2, -1, -1):
        start = end - dist

        wv_params = _access_params(wv_params_vec, start, end)[:, None]
        wv = get_wv(wv_params, k)
        y = psi_p(wv, y, dist)

        dist -= 1
        end = start

    return y


def psi_strs_p(wv_params_vec, y, k, starts):
    """
    multiple by a string of conjugate transpose of psi matrices from polcari parameterization

    k:      scaling parameter for wv
    starts: indices into wv_params_vec
            wv_params{i} = wv_params_vec[starts[i]:starts[i+1]]
    """
    M = len(starts)
    for i in range(M-1):
        wv_params = _access_params(wv_params_vec, starts[i], starts[i+1])[:, None]
        wv = get_wv(wv_params, k)
        L = starts[i+1] - starts[i]
        y = psi_str_p(wv, y, L)
    return y

class Layer():
    """
    Abstract class for a single layer in a network architecture

    Derived classes implement the function
    matrix_forward(params, x)

    And supply
    self.num_params_vec
    """
    def __init__(self):
        self.num_params_vec = [0]
        self.is_linear = True
        self.param_starts = [0]
        self.use_type = complex

    def init_params(self, count=0):
        params = [] 
        self.num_params = 0
        prev = 0
        for num in self.num_params_vec: 
            params.append(jax.random.normal(shape=[num], key=jax.random.PRNGKey(count)))
            self.param_starts.append(prev+num)
            prev += num
            count += 1
            self.num_params += num

        return jnp.concatenate(params, axis=0)

    def init_model_params(self, models, count=0):
        params = []
        total = 0
        for model in models:
            new_params = model.init_params(count)
            params.append(new_params)
            self.param_starts.append(total+len(new_params))
            total += len(new_params)
            count += 1

        self.num_params = total
        return jnp.concatenate(params, axis=0)
        

    def form_operator(self, params):

        x = jnp.eye((self.N), dtype=self.use_type)
        operator = self.matrix_forward(params, x)
        return operator

    def form_hermitian_operator(self, params):

        x = jnp.eye((self.M), dtype=self.use_type)
        operator = self.matrix_hermitian_forward(params, x)
        return operator

class UpperTriangular(Layer):
    """
    purely upper triangular, no diagonal
    """
    def __init__(self, M, use_complex=False, diagonal=False):
        super().__init__()
        self.M = M
        self.N = M
        self.use_complex = use_complex
        if use_complex:
            self.num_params_vec = [M ** 2,
                                   M ** 2]
        else:
            self.num_params_vec = [M ** 2]
            self.use_type = float

        if diagonal:
            self.mask = jnp.tri(M, k=0, dtype=self.use_type).T
        else:
            self.mask = jnp.tri(M, k=-1, dtype=self.use_type).T

    def matrix_forward(self, params, x):
        """
        Inputs:
        x: (N x Nd) tensor where N is the dimension of the matrix and Nd is the number of data points

        Outputs:
        y: (M x Nd) tensor, y = D x, where D is diagonal
        """

        if self.use_complex:
            a = + params[self.param_starts[0]:self.param_starts[1]] \
                + 1j * params[self.param_starts[1]:self.param_starts[2]]
        else:
            a = + params[self.param_starts[0]:self.param_starts[1]]

        if len(a.shape) == 2:
            a = a.reshape((self.M, self.M, -1))
            return _UpperTriangular_matrix_forward_multi(x, a, self.mask)
        else:
            a = a.reshape((self.M, self.M))
            return _UpperTriangular_matrix_forward(x, a, self.mask)

@jax.jit
def _UpperTriangular_matrix_forward(x, a, mask):
    return (mask * a) @ x

@jax.jit
def _UpperTriangular_matrix_forward_multi(x, a, mask):
    out = jnp.tensordot(mask[:, :, None] * a, x[:, :], axes=(1, 0))
    out = out.swapaxes(1, 2)
    return out


class DiagonalScaling(Layer):
    """
    Apply the (vector) scaling "exp(i theta) mu"

    theta, mu \in R^N
    |mu| < max_scale
    """
    def __init__(self, M, max_scale=1.0, option='scale', gap=None, dim=None, use_complex=True):
        super().__init__()
        self.M = M
        self.N = M
        self.max_scale = max_scale
        self.k = 0.25
        self.use_complex = use_complex
        if use_complex:
            self.num_params_vec = [M, M]
            self.use_type = complex
        else:
            self.num_params_vec = [M]
            self.use_type = float
        if option == 'scale':
            self.matrix_forward = self.matrix_forward_scale
        elif option == 'minmax':
            if max_scale < 1.0:
                print("max_scale should be greater than 1, recieved %f" % (max_scale))
                assert(1==0)
            self.matrix_forward = self.matrix_forward_minmax
        elif option == 'real':
            self.matrix_forward = self.matrix_forward_real
        elif option == 'gap':
            self.matrix_forward = self.matrix_forward_gap
            self.gap = gap
            self.dim = dim
        elif option == 'sparse':
            self.matrix_forward = self.matrix_forward_sparse
            self.dim = dim
            self.gap = gap
            self.num_params_vec = [M - dim, M - dim]
        else:
            print("invalid option")

            assert(1 == 0)

    def matrix_forward_scale(self, params, x):
        """
        Inputs:
        x: (N x Nd) tensor where N is the dimension of the matrix and Nd is the number of data points

        Outputs:
        y: (M x Nd) tensor, y = D x, where D is diagonal
        """

        M, Nd = jnp.shape(x)
        if self.use_complex:
            mu = params[self.param_starts[0]:self.param_starts[1]]
            theta = params[self.param_starts[1]:self.param_starts[2]]

            abs_mu = jnp.abs(mu)
            scale = jnp.exp(1j * theta) * jnp.tanh(mu) * self.max_scale
        else:
            mu = params[self.param_starts[0]:self.param_starts[1]]
            scale = jnp.tanh(mu) * self.max_scale
        y = scale[:, None] * x

        return y

    def matrix_forward_minmax(self, params, x):
        """
        Inputs:
        x: (N x Nd) tensor where N is the dimension of the matrix and Nd is the number of data points

        Outputs:
        y: (M x Nd) tensor, y = D x, where D is diagonal
        """

        M, Nd = jnp.shape(x)

        mu = params[self.param_starts[0]:self.param_starts[1]]
        C = jnp.log(self.max_scale)
        scale = jnp.exp(C * jnp.tanh(mu))
        y = scale[:, None] * x

        return y    

    def matrix_forward_real(self, params, x):
        M, Nd = jnp.shape(x)
        a = params[self.param_starts[0]:self.param_starts[1]]
        b = params[self.param_starts[1]:self.param_starts[2]]

        scale = - jnp.abs(a) + 1j * b
        y = scale[:, None] * x

        return y

    def matrix_forward_gap(self, params, x):
        M, Nd = jnp.shape(x)
        a = params[self.param_starts[0]:self.param_starts[1]]
        b = params[self.param_starts[1]:self.param_starts[2]]

        lambd_r = - jnp.cumsum(jnp.abs(a))
        lambd_r = lambd_r.at[self.dim:].add(-self.gap)
        scale = lambd_r + 1j * b
        y = scale[:, None] * x

        return y

    def matrix_forward_sparse(self, params, x):
        M, Nd = jnp.shape(x)
        a = params[self.param_starts[0]:self.param_starts[1]]
        b = params[self.param_starts[1]:self.param_starts[2]]

        y = x
        y = y.at[:self.dim].mul(0)
        scale = - jnp.abs(a) + 1j * b
        y = y.at[self.dim:].mul(scale[:, None])
        y = y.at[self.dim:].add(-self.gap)
        return y


class Householder(Layer):
    """
    Q = Q_1 ... Q_n, where
    Q \in R^{MxN}

    Q_k = I - 2 (v_k v_k^T) / (v_k^T v_k),

    v_k = [0;
           tilde{v}_k]
    v_k \in R^M
    tilde{v}_k \in R^k
    """
    def __init__(self, M, N=None):

        if N is None:
            N = M
        self.M = M
        self.N = N
        self.p = max(M, N)

        self.use_complex = False
        self.use_type = float

        self.mask = jnp.tri(M, N, k=0, dtype=self.use_type)
        
        self.param_starts = [0]
        # self.num_params_vec = [M * N]
        self.num_params_vec = [M * N]
        
    def matrix_forward(self, params, x):

        if len(params.shape) > 1:
            params = params.reshape((self.M, self.N, -1))

            if self.M >= self.N:
                return _Householder_matrix_forward_skinny_multi(params, self.mask, x, self.M, self.N)
            else:
                return _Householder_matrix_forward_fat_multi(params, self.mask, x, self.M, self.N)
        else:
            params = params.reshape((self.M, self.N))
            if self.M >= self.N:
                return _Householder_matrix_forward_skinny(params, self.mask, x, self.M, self.N)
            else:
                return _Householder_matrix_forward_fat(params, self.mask, x, self.M, self.N)


    def matrix_transpose_forward(self, params, x):
        if len(params.shape) > 1:
            params = params.reshape((self.M, self.N, -1))

            if self.M >= self.N:
                return _Householder_matrix_transpose_forward_skinny_multi(params, self.mask, x, self.M, self.N)
            else:
                return _Householder_matrix_transpose_forward_fat_multi(params, self.mask, x, self.M, self.N)
        else:
            params = params.reshape((self.M, self.N))

            if self.M >= self.N:
                return _Householder_matrix_transpose_forward_skinny(params, self.mask, x, self.M, self.N)
            else:
                return _Householder_matrix_transpose_forward_fat(params, self.mask, x, self.M, self.N)


@jax.jit
def linear_0(V, y):
    out = jnp.tensordot(V, y, axes=(0, 0))
    return out

linear_map = jax.jit(jax.vmap(linear_0, (1, 1), (0)))
linear_map_0 = jax.jit(jax.vmap(linear_0, (None, 1), (0)))

# todo: functions that don't depend on M, N...
# reshape before functions

# @jax.jit
@partial(jax.jit, static_argnums=(3,4))
def _Householder_matrix_forward_skinny(params, mask, x, M, N):

    # M, N = params.shape
    V = mask[:, :] * params[:, :]

    x = jnp.eye(M, N) @ x

    for k in range(N):
        x = x - 2 * V[:, k, None] * linear_map_0(V[:, k], x) / jnp.dot(V[:, k], V[:, k])

    return x

# @jax.jit
@partial(jax.jit, static_argnums=(3,4))
def _Householder_matrix_forward_skinny_multi(params, mask, x, M, N):

    # M, N, Nd = params.shape
    V = mask[:, :, None] * params[:, :]

    x = jnp.eye(M, N) @ x

    for k in range(N):
        x = x - 2 * V[:, k, :] * linear_map(V[:, k, :], x) / linear_map(V[:, k, :], V[:, k, :])

    return x

# @jax.jit
@partial(jax.jit, static_argnums=(3,4))
def _Householder_matrix_transpose_forward_skinny(params, mask, x, M, N):

    # M, N = params.shape
    V = mask[:, :] * params[:, :]

    for k in range(N - 1, -1, -1):
        x = x - 2 * V[:, k, None] * linear_map_0(V[:, k], x) / jnp.dot(V[:, k], V[:, k])

    x = jnp.eye(N, M) @ x
    return x

# @jax.jit
@partial(jax.jit, static_argnums=(3,4))
def _Householder_matrix_transpose_forward_skinny_multi(params, mask, x, M, N):

    # M, N, Nd = params.shape
    V = mask[:, :, None] * params[:, :]

    for k in range(N - 1, -1, -1):
        x = x - 2 * V[:, k, :] * linear_map(V[:, k, :], x) / linear_map(V[:, k, :], V[:, k, :])

    x = jnp.eye(N, M) @ x
    return x


# @jax.jit
@partial(jax.jit, static_argnums=(3,4))
def _Householder_matrix_forward_fat_multi(params, mask, x, M, N):
    
    # M, N, Nd = params.shape
    V = mask[:, :, None] * params[:, :]

    for k in range(N):
        x = x - 2 * V[k, :, :] * linear_map(V[k, :, :], x) / linear_map(V[k, :, :], V[k, :, :])

    x = x[:M]
    return x


# @jax.jit
@partial(jax.jit, static_argnums=(3,4))
def _Householder_matrix_forward_fat(params, mask, x, M, N):
    
    # M, N = params.shape
    V = mask[:, :] * params[:, :]

    # x: (N, Nd)

    for k in range(N):
        x = x - 2 * V[k, :, None] * linear_map_0(V[k, :], x) / jnp.dot(V[k, :], V[k, :])

    x = x[:M]
    return x

# @jax.jit
@partial(jax.jit, static_argnums=(3,4))
def _Householder_matrix_transpose_forward_fat_multi(params, mask, x, M, N):
    
    # M, N, Nd = params.shape
    V = mask[:, :, None] * params[:, :]

    x = jnp.eye(N, M) @ x
    for k in range(N - 1, -1, -1):
        x = x - 2 * V[k, :, :] * linear_map(V[k, :, :], x) / linear_map(V[k, :, :], V[k, :, :])

    return x


# @jax.jit
@partial(jax.jit, static_argnums=(3,4))
def _Householder_matrix_transpose_forward_fat(params, mask, x, M, N):
    
    # M, N = params.shape
    V = mask[:, :] * params[:, :]

    x = jnp.eye(N, M) @ x
    # x: (N, Nd)

    for k in range(N - 1, -1, -1):
        x = x - 2 * V[k, :, None] * linear_map_0(V[k, :], x) / jnp.dot(V[k, :], V[k, :])

    return x






    
class Orthogonal(Layer):

    """
    Q = exp(A), where A = - A^T (skew symmetric)
    """
    def __init__(self, M, N=None):

        if N is None:
            N = M
        self.M = M
        self.N = N
        self.p = max(M, N)
        self.U = UpperTriangular(self.p, use_complex=False)
        assert(M == N)

        self.use_complex = False
        self.use_type = float

        self.param_starts = [0]

    def init_params(self, count=0):

        models = [self.U]
        return self.init_model_params(models, count)

    def matrix_forward(self, params, x):
        """
        Inputs:
        x: (N x Nd) tensor where N is the dimension of the matrix and Nd is the number of data points

        Outputs:
        y: (M x Nd) tensor, y = Q x, where Q is orthogonal
        """

        params_U = params[self.param_starts[0]:self.param_starts[1]]
        U = self.U.form_operator(params_U)
        if len(jnp.shape(U)) == 3:
            return _Orthogonal_matrix_forward_multi(U, x)
        else:
            return _Orthogonal_matrix_forward(U, x)

    # def matrix_free_forward(self, params, x):
    #     params_U = params[self.param_starts[0]:self.param_starts[1]]
    #     breakpoint()
    #     jsp.sparse.linalg.expm_multiply(params_U, x)
    
    def inverse(self, params, x):
        params_U = params[self.param_starts[0]:self.param_starts[1]]
        U = self.U.form_operator(params_U)
        return _Orthogonal_inverse(U, x)


@jax.jit
def _Orthogonal_matrix_forward(U, x):
        Q = jsp.linalg.expm(U - U.T)
        return Q @ x

@jax.jit
def _Orthogonal_matrix_forward_multi(U, x):

    U = U.swapaxes(2, 1)
    U = U.swapaxes(1, 0)
    UT = U.swapaxes(1, 2)

    breakpoint()
    Q = jsp.linalg.expm(U - UT)

    breakpoint()
    jnp.tensordot(Q, x, axes=(1, 0))
    return 
    
@jax.jit
def _Orthogonal_inverse(U, x):
        Q = jsp.linalg.expm(U - U.T)
        return Q.T @ x            
    
class UnitaryModel(Layer):
    """
    Unitary matrix, C^{M x N}
    Polcari decomp
    """
    def __init__(self, M, N=None, apply_signs=True, use_complex=True):
        super().__init__()

        self.M = M
        if N is None:
            N = M
        self.N = N

        self.dim_max = max(M, N)
        self.dim_min = min(M, N)

        self.apply_signs = apply_signs
        self.use_complex = use_complex
        
        self.k = 0.25

        # mapping between flat coordinates and list coordinates
        self.starts = []

        # count number of parameters for flattened vector
        total = 0
        for n in range(self.dim_min):
            total += self.dim_max - 1 - n
        self.total = total

        self.starts = []
        start = total
        n = 0
        while start > 0:
            self.starts.append(start)
            start -= self.dim_max - 1 - n
            n += 1
        self.starts.append(start)
        self.starts = self.starts[-1::-1]

        if self.use_complex:
            self.num_params_vec = [self.total, self.total]
        else:
            self.num_params_vec = [self.total]
            self.use_type = float
        if self.apply_signs:
            assert(self.use_complex)
            self.num_params_vec.append(self.dim_min)


    def matrix_forward(self, params, x):
        """
        Inputs:
        x: (N x Nd) tensor where N is the dimension of the matrix and Nd is the number of data points

        Outputs:
        y: (M x Nd) tensor, y = U x, where U is unitary
        """

        N, Nd = jnp.shape(x)

        if self.use_complex:
            wv_params_vec = \
                params[self.param_starts[0]:self.param_starts[1]] \
                + 1j * params[self.param_starts[1]:self.param_starts[2]]
        else:
            wv_params_vec = \
                _access_params(params, self.param_starts[0], self.param_starts[1])
        if self.apply_signs:
            angles = params[self.param_starts[2]:self.param_starts[3]]

        y = x
        if self.N == self.dim_min:
            if self.apply_signs:
                y = jnp.exp(1j * angles)[:, None] * y
            if self.N != self.M:
                new_y = jnp.zeros((self.M, Nd), dtype=self.use_type)
                y = new_y.at[self.M-self.N:].set(y)

            M = len(self.starts)
            dist = self.starts[-1]-self.starts[-2]
            end = self.starts[-1]
            
            y = psis_p(wv_params_vec, y, self.k, self.starts)
            # y = psis_p(wv_params_vec, y, self.k, M, dist, end)
        if self.N != self.dim_min:
            y = psi_strs_p(wv_params_vec, y, self.k, self.starts)
            y = y[self.N-self.M:, :]
            if self.apply_signs:
                y = jnp.exp(- 1j * angles)[:, None] * y
                
        return y
        
    def matrix_hermitian_forward(self, params, x):
        """
        y = Q^* x

        Inputs:
        x: (M x Nd) tensor where N is the dimension of the matrix and Nd is the number of data points

        Outputs:
        y: (N x Nd) tensor, y = U x, where U is unitary
        """

        M, Nd = jnp.shape(x)

        if self.use_complex:
            wv_params_vec = \
                params[self.param_starts[0]:self.param_starts[1]] \
                + 1j * params[self.param_starts[1]:self.param_starts[2]]
        else:
            wv_params_vec = \
                params[self.param_starts[0]:self.param_starts[1]]
        if self.apply_signs:
            angles = params[self.param_starts[2]:self.param_starts[3]]

        y = jnp.zeros((self.M, Nd), dtype=self.use_type)
        y += x

        if self.N == self.dim_min:

            y = psi_strs_p(wv_params_vec, y, self.k, self.starts)

            if self.N != self.M:
                new_y = jnp.zeros((self.M, Nd), dtype=self.use_type)
                y = y[self.M-self.N:, :]
                
            if self.apply_signs:
                y = jnp.exp(- 1j * angles)[:, None] * y

        if self.N != self.dim_min:
            # new_y = jnp.zeros((self.M, Nd), dtype=complex)
            if self.apply_signs:
                y = jnp.exp(1j * angles)[:, None] * y

            new_y = jnp.zeros((self.N, Nd), dtype=self.use_type)
            y = new_y.at[self.N-self.M:].set(y)
            y = psis_p(wv_params_vec, y, self.k, self.starts)

        return y


class SimpleMatrix(Layer):
    def __init__(self, M, N):
        super().__init__()

        self.M = M
        self.N = N
        self.num_params_vec = [M * N]
        self.use_type = float
        self.use_complex = False
        

    def matrix_forward(self, params, x):
        """
        Inputs:
        x: (N x Nd) tensor where N is the dimension of the matrix and Nd is the number of data points

        Outputs:
        y: (M x Nd) tensor,
        """

        return _SimpleMatrix_matrix_forward(params, x, self.M, self.N)

    def inverse(self, params, x):
        """
        Inputs:
        x: (N x Nd) tensor where N is the dimension of the matrix and Nd is the number of data points

        Outputs:
        y: (M x Nd) tensor,
        """

        M, Nd = jnp.shape(x)
        A = params[self.param_starts[0]:self.param_starts[1]].reshape((self.M, self.N))

        y = jnp.linalg.inv(A) @ x
        
        return y

@partial(jax.jit, static_argnums=(2,3))
def _SimpleMatrix_matrix_forward(params, x, M, N):
    A = params.reshape((M, N))
    y = A @ x
    return y
    

class Bias(Layer):
    def __init__(self, M):
        super().__init__()

        self.M = M
        self.num_params_vec = [M]
        self.use_type = float
        self.use_complex = False

    def matrix_forward(self, params, x):
        """
        Inputs:
        x: (N x Nd) tensor where N is the dimension of the matrix and Nd is the number of data points

        Outputs:
        y: (N x Nd) tensor,
        """

        return _Bias_matrix_forward(params, x)


class Activation(Layer):
    def __init__(self, func):
        super().__init__()
        self.k = 1.0
        self.func = jax.jit(func)
        self.is_linear = False

    def matrix_forward(self, params, x):
        # y = + self.func(self.k * jnp.real(x)) \
        #     + self.func(self.k * jnp.imag(x))
        # y = + jax.jit(self.func)(self.k * x)
        return _Activation_matrix_forward(self.func, self.k, x)


@partial(jax.jit, static_argnums=(0))
def _Activation_matrix_forward(func, k, x):
    return func(k * x)
    

class MatrixFromSVD(Layer):
    def __init__(self, M, N, max_scale=1.0, use_complex=True, with_bias=False, scaling="scale"):
        # method:
        super().__init__()
        self.M = M
        self.N = N
        p = min(M, N)
        self.with_bias = with_bias

        self.use_complex = use_complex

        # unitary
        if use_complex:
            self.U = UnitaryModel(M, p, apply_signs=False, use_complex=use_complex)
            self.Vstr = UnitaryModel(p, N, apply_signs=False, use_complex=use_complex)
        elif M != N:
            self.U = UnitaryModel(M, p, apply_signs=False, use_complex=use_complex)
            self.Vstr = UnitaryModel(p, N, apply_signs=False, use_complex=use_complex)
        else:
            self.U = Orthogonal(M, p)
            self.Vstr = Orthogonal(p, N)
        self.D = DiagonalScaling(p, option=scaling, max_scale=max_scale, use_complex=use_complex)
        self.use_complex = use_complex
        if use_complex:
            self.use_type = complex            
        else:
            self.use_type = float
        if with_bias:
            self.bias = Bias(M)

        self.param_starts = [0]


    def init_params(self, count=0):

        models = [self.U, self.D, self.Vstr]
        if self.with_bias:
            models += [self.bias]
        return self.init_model_params(models, count)

    def matrix_forward(self, params, x):
        """
        y = U D Vstr x
        """
        params_U = params[self.param_starts[0]:self.param_starts[1]]
        params_D = params[self.param_starts[1]:self.param_starts[2]]
        params_Vstr = params[self.param_starts[2]:self.param_starts[3]]

        # Vstr = self.Vstr.form_operator(params_Vstr)
        # D = self.D.form_operator(params_D)
        # U = self.U.form_operator(params_U)

        # # breakpoint()
        # x = Vstr @ x
        # x = D @ x
        # x = U @ x

        x = self.Vstr.matrix_forward(params_Vstr, x)
        x = self.D.matrix_forward(params_D, x)
        x = self.U.matrix_forward(params_U, x)

        if self.with_bias:
            params_b = params[self.param_starts[3]:self.param_starts[4]]
            x += params_b[:, None]

        return x

    def form_operators(self, params):
        
        params_U = params[self.param_starts[0]:self.param_starts[1]]
        params_D = params[self.param_starts[1]:self.param_starts[2]]
        params_Vstr = params[self.param_starts[2]:self.param_starts[3]]

        Vstr = self.Vstr.form_operator(params_Vstr)
        D = self.D.form_operator(params_D)
        U = self.U.form_operator(params_U)
        return U, D, Vstr

    def form_operator(self, params):
        U, D, Vstr = self.form_operators(params)
        return U @ D @ Vstr

    def inverse(self, params, x):
        if self.with_bias:
            params_b = params[self.param_starts[3]:self.param_starts[4]]
            x = x - params_b[:, None]

        params_U = params[self.param_starts[0]:self.param_starts[1]]
        params_D = params[self.param_starts[1]:self.param_starts[2]]
        params_Vstr = params[self.param_starts[2]:self.param_starts[3]]

        # Vstr = self.Vstr.form_operator(params_Vstr)
        D = self.D.form_operator(params_D)
        # U = self.U.form_operator(params_U)
        
        # x = (U.T @ x)
        # x = x / jnp.diag(D)[:, None]
        # x = Vstr.T @ x

        x = self.U.inverse(params_U, x)
        x = x / jnp.diag(D)[:, None]
        x = self.Vstr.inverse(params_Vstr, x)
        
        return x


class SingleLayer(Layer):
    def __init__(self, M, N, k, max_scale=1.0):
        """
        M: output dimension
        N: input dimension
        k: hidden dimension
        """
        
        # method:
        super().__init__()
        self.M = M
        self.N = M

        self.use_complex = False
        self.use_type = float

        # unitary
        self.A = MatrixFromSVD(k, N, max_scale=max_scale, use_complex=False)
        self.C = MatrixFromSVD(M, k, max_scale=max_scale, use_complex=False)
        self.additional_params = k
        self.param_starts = [0]


    def init_params(self, count=0):

        models = [self.A, self.C]
        params = []
        total = 0
        for model in models:
            new_params = model.init_params(count)
            params.append(new_params)
            self.param_starts.append(total+len(new_params))
            total += len(new_params)
            count += 1

        # bias term
        params.append(jax.random.normal(shape=[self.additional_params], key=jax.random.PRNGKey(count)))
        self.param_starts.append(total+self.additional_params)

        return jnp.concatenate(params, axis=0)

    def matrix_forward(self, params, x):
        """
        """
        params_A = params[self.param_starts[0]:self.param_starts[1]]
        params_C = params[self.param_starts[1]:self.param_starts[2]]
        params_b = params[self.param_starts[2]:self.param_starts[3]]

        h = self.A.matrix_forward(params_A, x)
        h = params_b[:, None] + h
        h = jnp.tanh(h)
        y = self.C.matrix_forward(params_C, h)

        return y

    def matrix_forward_operator(self, params, x):
        """
        """
        params_A = params[self.param_starts[0]:self.param_starts[1]]
        params_C = params[self.param_starts[1]:self.param_starts[2]]
        params_b = params[self.param_starts[2]:self.param_starts[3]]

        A = self.A.form_operator(params_A)
        C = self.C.form_operator(params_C)

        # h = self.A.matrix_forward(params_A, x)
        h = A @ x
        h = params_b[:, None] + h
        h = jnp.tanh(h)
        # y = self.C.matrix_forward(params_C, h)
        y = C @ h

        return y

    
    def grad_matrix_forward_dot_y(self, params, x, y):
        """
        """
        params_A = params[self.param_starts[0]:self.param_starts[1]]
        params_C = params[self.param_starts[1]:self.param_starts[2]]
        params_b = params[self.param_starts[2]:self.param_starts[3]]

        A = self.A.form_operator(params_A)
        C = self.C.form_operator(params_C)
        
        z = self.A.matrix_forward(params_A, x)
        z = params_b[:, None] + z
        ThetaPrime = (1 / jnp.cosh(z)) ** 2

        y = C.T @ y
        y = ThetaPrime * y
        y = A.T @ y

        return y

        
class Composition():
    """
    Abstract class for a composition of layers
    """
    def __init__(self):
        pass

    def init_params(self, count=0):
        start = 0
        params = []
        self.param_starts = []
        for layer in self.layers:
            self.param_starts.append(start)
            params.append(layer.init_params(count=start+count))
            start += layer.num_params
        self.param_starts.append(start)

        return jnp.concatenate(params, axis=0)

    def matrix_forward(self, params, y):
        """
        """
        for i, layer in enumerate(self.layers):
            param = params[self.param_starts[i]:self.param_starts[i+1]]
            y = layer.matrix_forward(param, y)

        return y


class MatrixModel(Composition):
    """
    Matrix A \in C^{M x N}, |A| <= max_scale 
    """
    def __init__(self, M, N, max_scale=1.0):
        super().__init__()

        self.M = M
        self.N = N
        p = min(M, N)
        self.p = p

        # M x N matrix
        self.layers = MatrixLayers(M, N, max_scale)


def MatrixLayers(M, N, max_scale=1.0, use_complex=False):
    p = min(M, N)
    if use_complex:
        return [UnitaryModel(p, N, False, use_complex=use_complex),
                DiagonalScaling(p, max_scale=max_scale, use_complex=use_complex),
                UnitaryModel(M, p, False, use_complex=use_complex)]
    else:
        return [Orthogonal(p, N),
                DiagonalScaling(p, max_scale=max_scale, use_complex=use_complex),
                Orthogonal(M, p)]
        

def MatrixBiasLayers(M, N, max_scale=1.0, use_complex=False):
    p = min(M, N)
    # if use_complex or M != N:
    if M != N:
        # return [UnitaryModel(p, N, False, use_complex=use_complex),
        #         DiagonalScaling(p, max_scale=max_scale, use_complex=use_complex),
        #         UnitaryModel(M, p, False, use_complex=use_complex),
        #         Bias(M)]
        return [Householder(p, N),
                DiagonalScaling(p, max_scale=max_scale, use_complex=use_complex),
                Householder(M, p),
                Bias(M)]
    else:
        return [Orthogonal(p, N),
                DiagonalScaling(p, max_scale=max_scale, use_complex=use_complex),
                Orthogonal(M, p),
                Bias(M)]


class FeedForward(Composition):
    """
    input dimension: N
    inner dimension: d
    output dimension: M
    layers: p
    """
    def __init__(self, M, N, d, p, scale=True, max_scale=1.0, sigma=jnp.tanh):
        super().__init__()

        self.M = M
        self.N = N

        # self.sigma_prime = jax.grad(sigma)
        # self.sigma_prime = lambda x: 1 - jnp.tanh(x) ** 2
        
        if scale:
            # d x N matrix
            self.layers = MatrixBiasLayers(d, N, max_scale=max_scale) + [Activation(sigma)]
            for k in range(p - 1):
                # d x d matrix
                self.layers += MatrixBiasLayers(d, d, max_scale=max_scale) + [Activation(sigma)]
            # M x d matrix
            self.layers += MatrixBiasLayers(M, d, max_scale=max_scale)
        else:
            self.layers = [SimpleMatrix(d, N),
                           Bias(d),
                           Activation(sigma)]
            for k in range(p):
                # d x d matrix
                self.layers += [SimpleMatrix(d, d),
                                Bias(d),
                                Activation(sigma)]
            # M x d matrix
            self.layers += [SimpleMatrix(M, d),
                            Bias(M)]

    def form_operator(self, params):
        func = lambda x: self.matrix_forward(params, x)
        return func

    def forward_and_jacobian(self, params, x, z):
        # compute:
        #   f(x)
        #   D f(x) z

        Q = z
        i = 0
        N_passes = len(self.layers)//3
        for j in range(N_passes + 1):
            # matrix
            param_mat = params[self.param_starts[i+0]:self.param_starts[i+1]]
            layer_mat = self.layers[i]
            # bias
            param_bias = params[self.param_starts[i+1]:self.param_starts[i+2]]
            layer_bias = self.layers[i+1]

            Ax = layer_mat.matrix_forward(param_mat, x)
            Aq = layer_mat.matrix_forward(param_mat, Q)
            Ax_b = layer_bias.matrix_forward(param_bias, Ax)

            # activation
            if j < N_passes:
                param_activation = params[self.param_starts[i+2]:self.param_starts[i+3]]
                layer_activation = self.layers[i+2]
                
                theta_Ax_b = layer_activation.matrix_forward(param_activation, Ax_b)
                theta_p_Ax_b = 1.0 - theta_Ax_b ** 2

                x = theta_Ax_b
                Q = theta_p_Ax_b * Aq
            else:
                x = Ax_b
                Q = Aq

            i += 3

        return x, Q
        

