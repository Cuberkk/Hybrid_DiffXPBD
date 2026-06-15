import numpy as np

class AdamOptimizer:
    def __init__(
        self,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ):
        """
        Adam optimizer.

        Parameters
        ----------
        lr : float
            Learning rate.
        beta1 : float
            Exponential decay rate for the first moment estimate.
        beta2 : float
            Exponential decay rate for the second moment estimate.
        eps : float
            Small constant for numerical stability.
        """
        if lr <= 0:
            raise ValueError("lr must be positive.")
        if not (0.0 < beta1 < 1.0):
            raise ValueError("beta1 must be in (0, 1).")
        if not (0.0 < beta2 < 1.0):
            raise ValueError("beta2 must be in (0, 1).")
        if eps <= 0:
            raise ValueError("eps must be positive.")

        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.t = 0
        self.m = None
        self.v = None

    def reset(self):
        """Reset optimizer state."""
        self.t = 0
        self.m = None
        self.v = None

    def step(self, param, grad):
        """
        Update parameter using Adam.

        Parameters
        ----------
        param : float or np.ndarray
            Current parameter value.
        grad : float or np.ndarray
            Gradient of the parameter.

        Returns
        -------
        updated_param : same type as param
            Updated parameter.
        """
        param_arr = np.asarray(param, dtype=np.float64)
        grad_arr = np.asarray(grad, dtype=np.float64)

        if param_arr.shape != grad_arr.shape:
            raise ValueError(
                f"Shape mismatch: param.shape={param_arr.shape}, grad.shape={grad_arr.shape}"
            )

        if self.m is None:
            self.m = np.zeros_like(param_arr, dtype=np.float64)
            self.v = np.zeros_like(param_arr, dtype=np.float64)

        self.t += 1

        self.m = self.beta1 * self.m + (1.0 - self.beta1) * grad_arr
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * (grad_arr ** 2)

        m_hat = self.m / (1.0 - self.beta1 ** self.t)
        v_hat = self.v / (1.0 - self.beta2 ** self.t)

        updated = param_arr - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

        if np.isscalar(param):
            return float(updated)
        return updated.astype(np.asarray(param).dtype, copy=False)