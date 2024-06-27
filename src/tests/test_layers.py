import jax
import jax.numpy as jnp
import jax.scipy as jsp

from blat.models import DiagonalScaling, UnitaryModel, \
    MatrixFromSVD, Orthogonal, FeedForward, Householder
import matplotlib.pyplot as plt

from jax.config import config
config.update("jax_enable_x64", True)

plt.rcParams.update({'font.size': 18})





def test_householder():
    TOL = 1e-14
    for M, N in [[3, 3], [4, 4], [9, 3], [3, 9]]:
        func = Householder(M, N)
        
        Nd = 10

        ### 1D params test
        # y = Q x
        params = func.init_params()
        x = jax.random.normal(shape=[N, Nd], key=jax.random.PRNGKey(0))
        y = func.matrix_forward(params, x)

        if M >= N:
            assert(jnp.abs(jnp.linalg.norm(y) - jnp.linalg.norm(x)) < TOL)

        # Get Q and QT
        Q = func.form_operator(params)
        Q_check = func.matrix_forward(params, jnp.eye(N, N))

        QT = func.matrix_transpose_forward(params, jnp.eye(M, M))
        assert(jnp.linalg.norm(Q - Q_check) < TOL)
        assert(jnp.linalg.norm(Q.T - QT) < TOL)
        if M >= N:
            assert(jnp.linalg.norm(Q.T @ Q - jnp.eye(N, N)) < TOL)
        if M <= N:
            assert(jnp.linalg.norm(Q @ Q.T - jnp.eye(M, M)) < TOL)

        ### 2D params test
        Np = len(params)
        # y = Q x
        params = jax.random.normal(shape=[Np, Nd], key=jax.random.PRNGKey(0))
        x = jax.random.normal(shape=[N, Nd], key=jax.random.PRNGKey(0))
        y = func.matrix_forward(params, x)

        if M >= N:
            assert(jnp.abs(jnp.linalg.norm(y) - jnp.linalg.norm(x)) < TOL)

        for n in range(Nd):
            y_check = func.matrix_forward(params[:, n], x[:, n][:, None])
            assert(jnp.linalg.norm(y_check[:, 0] - y[:, n]) < TOL)

        x = jax.random.normal(shape=[M, Nd], key=jax.random.PRNGKey(0))
        y = func.matrix_transpose_forward(params, x)
        
        for n in range(Nd):
            y_check = func.matrix_transpose_forward(params[:, n], x[:, n][:, None])
            assert(jnp.linalg.norm(y_check[:, 0] - y[:, n]) < TOL)


            
        
def test_orthogonal():
    # note: only use when square!
    M = 5
    N = 3

    for M, N in [[3, 3], [4, 4], [5, 5]]:
        func = Orthogonal(M, N)
        params = func.init_params()

        Nd = 10
        x = jax.random.normal(shape=[N, Nd], key=jax.random.PRNGKey(0))

        y = func.matrix_forward(params, x)
        Q = func.form_operator(params)

        TOL = 1e-14
        if M >= N:
            assert(jnp.linalg.norm(Q.T @ Q - jnp.eye(N, N)) < TOL)
        if M <= N:
            assert(jnp.linalg.norm(Q @ Q.T - jnp.eye(M, M)) < TOL)


                
def test_diagonal_scale():
    M = 100
    Nd = 80
    scale = 2.0
    func = DiagonalScaling(M, max_scale=scale, option='scale')
    params = func.init_params()
    x = jnp.ones((M, Nd))
    y = func.matrix_forward(params, x)
    assert(jnp.max(jnp.abs(y)) <= scale)

    matrix = func.form_operator(params)
    assert(jnp.linalg.norm(jnp.diag(jnp.diag(matrix)) - matrix) == 0)


def test_real_diagonal_scale():
    M = 100
    Nd = 80
    scale = 0.5
    func = DiagonalScaling(M, max_scale=scale, option='scale', use_complex=False)
    params = func.init_params()
    x = jnp.ones((M, Nd))
    y = func.matrix_forward(params, x)
    assert(jnp.max(jnp.abs(y)) <= scale)

    matrix = func.form_operator(params)
    assert(jnp.linalg.norm(jnp.diag(jnp.diag(matrix)) - matrix) == 0)
    

    
    
def test_diagonal_real():
    M = 100
    Nd = 80
    func = DiagonalScaling(M, option='real')
    params = func.init_params()
    x = jnp.ones((M, Nd))
    y = func.matrix_forward(params, x)
    assert(jnp.max(jnp.real(y)) <= 0)

    matrix = func.form_operator(params)
    assert(jnp.linalg.norm(jnp.diag(jnp.diag(matrix)) - matrix) == 0)

def test_square_unitary():
    M = 5
    N = 5

    func = UnitaryModel(M, N=N, apply_signs=True)
    params = func.init_params()
    matrix = func.form_operator(params)

    TOL = 1e-12
    assert(jnp.linalg.norm(matrix.T.conj() @ matrix - jnp.eye(M, N)) <= TOL)
    assert(jnp.linalg.norm(matrix @ matrix.T.conj() - jnp.eye(M, N)) <= TOL)

def test_fat_unitary():
    M = 3
    N = 5

    func = UnitaryModel(M, N=N, apply_signs=True)
    params = func.init_params()
    matrix = func.form_operator(params)

    TOL = 1e-12
    assert(jnp.linalg.norm(matrix @ matrix.T.conj() - jnp.eye(M, M)) <= TOL)

def test_slender_unitary():
    M = 5
    N = 3

    func = UnitaryModel(M, N=N, apply_signs=True)
    params = func.init_params()
    matrix = func.form_operator(params)

    TOL = 1e-12
    assert(jnp.linalg.norm(matrix.T.conj() @ matrix - jnp.eye(N, N)) <= TOL)


def test_square_orthogonal():
    M = 5
    N = 5

    func = UnitaryModel(M, N=N, apply_signs=False, use_complex=False)
    params = func.init_params()
    matrix = func.form_operator(params)

    TOL = 1e-12
    assert(jnp.linalg.norm(matrix.T @ matrix - jnp.eye(M, N)) <= TOL)
    assert(jnp.linalg.norm(matrix @ matrix.T - jnp.eye(M, N)) <= TOL)

def test_fat_orthogonal():
    M = 3
    N = 5

    func = UnitaryModel(M, N=N, apply_signs=False, use_complex=False)
    params = func.init_params()
    matrix = func.form_operator(params)

    TOL = 1e-12
    assert(jnp.linalg.norm(matrix @ matrix.T - jnp.eye(M, M)) <= TOL)

def test_slender_orthogonal():
    M = 5
    N = 3

    func = UnitaryModel(M, N=N, apply_signs=False, use_complex=False)
    params = func.init_params()
    matrix = func.form_operator(params)

    TOL = 1e-12
    assert(jnp.linalg.norm(matrix.T @ matrix - jnp.eye(N, N)) <= TOL)

    

def test_hermitian_square_unitary():
    M = 5
    N = 5

    func = UnitaryModel(M, N=N, apply_signs=True)
    params = func.init_params()
    matrix = func.form_operator(params)
    matrix_herm = func.form_hermitian_operator(params)

    TOL = 1e-12
    assert(jnp.linalg.norm(matrix - matrix_herm.T.conj()) <= TOL)

def test_hermitian_slender_unitary():
    M = 5
    N = 3

    func = UnitaryModel(M, N=N, apply_signs=True)
    params = func.init_params()
    matrix = func.form_operator(params)
    matrix_herm = func.form_hermitian_operator(params)

    TOL = 1e-12
    assert(jnp.linalg.norm(matrix - matrix_herm.T.conj()) <= TOL)
    
def test_hermitian_fat_unitary():
    M = 3
    N = 5

    func = UnitaryModel(M, N=N, apply_signs=True)
    params = func.init_params()
    matrix = func.form_operator(params)
    matrix_herm = func.form_hermitian_operator(params)

    TOL = 1e-12
    assert(jnp.linalg.norm(matrix - matrix_herm.T.conj()) <= TOL)


    
    
