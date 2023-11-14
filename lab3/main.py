import numpy as np

def lab():
    y_func = np.loadtxt('y2.txt', dtype=np.float64)

    delta = np.float64(0.2)
    beta = np.array([10., 18., 15.], dtype=np.float64)  # [m1, m2, m3]
    eps = 1e-7
    abs_eps = 1e-9
    _, n = y_func.shape

    current = 1.0
    previous = 0
    iteration_num = 0

    while current > eps:
        a_mat = a_matrix(beta)
        iteration_num += 1

        a_deriv = a_matrix_derivatives(a_mat, beta)

        # Runge-Kutta method
        left_int_part = 0.
        right_int_part = 0.
        new_identification_beta = 0.

        U = np.zeros((6, 3))
        y_vec = np.copy(y_func[:, 0].reshape(-1, 1))

        for i in range(1, n):

            delta_U = get_new_delta(a_deriv, y_vec)

            U += runge_kutta_step(lambda x: a_mat @ x + delta_U, U, delta)

            # Calculate new y
            y_vec += runge_kutta_step(lambda x: a_mat @ x, y_vec, delta)

            left_int_part += U.T @ U
            right_int_part += U.T @ (y_func[:, i].reshape(-1, 1) - y_vec)

            new_identification_beta += (y_func[:, i].reshape(-1, 1) - y_vec).T @ (y_func[:, i] - y_vec.reshape(-1))

        d_beta = np.linalg.inv(left_int_part * delta) @ (right_int_part * delta)
        beta += d_beta.reshape(-1)

        previous = current
        current = new_identification_beta * delta

        print('Iteration {0} : {1:.15f}'.format(iteration_num, current[0]))

    print('m1 = ', beta[0])
    print('m2 = ', beta[1])
    print('m3 = ', beta[2])


def runge_kutta_step(f, x, delta):
    k1 = delta * f(x)
    k2 = delta * f(x + k1 / 2.)
    k3 = delta * f(x + k2 / 2.)
    k4 = delta * f(x + k3)
    return (k1 + 2. * k2 + 2. * k3 + k4) / 6.


def get_new_delta(dA, y_vec):
    return np.column_stack(((dA[0] @ y_vec).reshape(-1),
                            (dA[1] @ y_vec).reshape(-1),
                            (dA[2] @ y_vec).reshape(-1)))


def a_matrix(beta):
    cs = np.array([0.14, 0.3, 0.2, 0.12], dtype=np.float64)

    weights = np.array([beta[0], beta[1], beta[2]], dtype=np.float64)

    a_matrix = np.zeros((6, 6), dtype=np.float64)

    a_matrix[0, 1] = 1.
    a_matrix[1, 0] = -(cs[0] + cs[1]) / weights[0]
    a_matrix[1, 2] = cs[1] / weights[0]
    a_matrix[2, 3] = 1.
    a_matrix[3, 0] = cs[1] / weights[1]
    a_matrix[3, 2] = -(cs[1] + cs[2]) / weights[1]
    a_matrix[3, 4] = cs[2] / weights[1]
    a_matrix[4, 5] = 1.
    a_matrix[5, 2] = cs[2] / weights[2]
    a_matrix[5, 4] = -(cs[3] + cs[2]) / weights[2]
    return a_matrix


def a_matrix_derivatives(A, beta):
    # [c1, c2, c3, c4]
    cs = np.array([0.14, 0.3, 0.2, 0.12], dtype=np.float64)

    # [m1, m2, m3]
    weight = np.array([beta[0], beta[1], beta[2]], dtype=np.float64)

    derivatives = [
          np.zeros_like(A),
          np.zeros_like(A),
          np.zeros_like(A)
    ]

    # [m1, m2, m3]
    derivatives[0][1, 0] = (cs[1] + cs[0]) / (weight[0] * weight[0])
    derivatives[0][1, 2] = -(cs[1]) / (weight[0] * weight[0])

    derivatives[1][3, 0] = -(cs[1]) / (weight[1] * weight[1])
    derivatives[1][3, 2] = (cs[1] + cs[2]) / (weight[1] * weight[1])
    derivatives[1][3, 5] = -(cs[2]) / (weight[1] * weight[1])

    derivatives[2][5, 2] = -(cs[2]) / (weight[2] * weight[2])
    derivatives[2][5, 4] = (cs[3] + cs[2]) / (weight[2] * weight[2])

    return derivatives


lab()