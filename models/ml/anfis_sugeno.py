"""
ANFIS de type Sugeno (Takagi-Sugeno-Kang, ordre 1).

Architecture :
  - Fonctions d'appartenance gaussiennes (une par règle par feature)
  - Conséquences linéaires : f_i = p_i · x + q_i
  - Apprentissage hybride : LSE pour les conséquences, gradient pour les prémisses

Interface compatible scikit-learn (fit / predict).
"""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


class ANFISSugeno(BaseEstimator, RegressorMixin):

    def __init__(self, n_rules: int = 5, n_epochs: int = 200, lr: float = 0.01,
                 random_state: int = 42):
        self.n_rules = n_rules
        self.n_epochs = n_epochs
        self.lr = lr
        self.random_state = random_state

    def _mf(self, X):
        """Retourne mu[n, r, f] = exp(-((x_f - c_rf) / sigma_rf)^2)."""
        diff = X[:, None, :] - self.centers_[None, :, :]  # (n, r, f)
        return np.exp(-(diff / self.sigmas_[None, :, :]) ** 2)

    def _firing(self, X):
        """Force de déclenchement w[n, r] = mean_f mu[n, r, f].
        Moyenne au lieu du produit pour éviter l'effondrement en haute dimension.
        """
        mu = self._mf(X)
        return np.mean(mu, axis=2)  # (n, r)

    def _norm_firing(self, W):
        """Normalisation : wbar[n, r] = w[n, r] / sum_r w[n, r]."""
        denom = W.sum(axis=1, keepdims=True) + 1e-12
        return W / denom  # (n, r)

    def _consequent_matrix(self, X, Wbar):
        """
        Construit la matrice Phi pour LSE : y ≈ Phi @ theta.
        Chaque règle i contribue : wbar_i * (p_i · x + q_i)
        Phi shape : (n, r*(f+1))
        """
        n, f = X.shape
        r = self.n_rules
        Phi = np.zeros((n, r * (f + 1)))
        for i in range(r):
            col = i * (f + 1)
            Phi[:, col:col + f] = Wbar[:, i:i + 1] * X
            Phi[:, col + f] = Wbar[:, i]
        return Phi

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state)
        n, f = X.shape
        r = self.n_rules

        # Initialisation des prémisses : centres sur des points aléatoires
        idx = rng.choice(n, r, replace=False)
        self.centers_ = X[idx].copy()  # (r, f)
        self.sigmas_ = np.ones((r, f)) * X.std(axis=0)
        self.sigmas_ = np.clip(self.sigmas_, 1e-3, None)

        y = np.asarray(y, dtype=float)

        # Normalise y pour stabiliser les gradients (dénormalise en predict)
        self.y_mean_ = y.mean()
        self.y_std_ = y.std() + 1e-12
        y_n = (y - self.y_mean_) / self.y_std_

        for epoch in range(self.n_epochs):
            # --- Passe avant ---
            W = self._firing(X)
            Wbar = self._norm_firing(W)
            Phi = self._consequent_matrix(X, Wbar)

            # LSE pour les conséquences (sur y normalisé)
            theta, _, _, _ = np.linalg.lstsq(Phi, y_n, rcond=None)
            self.theta_ = theta  # (r*(f+1),)

            y_pred = Phi @ theta
            residual = y_pred - y_n  # (n,)

            # dy/dWbar_i = sum_j x_j * theta[i*(f+1)+j] + theta[i*(f+1)+f]
            dL_dWbar = np.zeros((n, r))
            for i in range(r):
                col = i * (f + 1)
                fi = X @ theta[col:col + f] + theta[col + f]
                dL_dWbar[:, i] = residual * fi  # (n,)

            # dWbar_i/dW_j = (delta_ij * sum_W - W_i) / sum_W^2
            sumW = W.sum(axis=1, keepdims=True) + 1e-12
            dL_dW = np.zeros((n, r))
            for i in range(r):
                for j in range(r):
                    if i == j:
                        dWbar_dW = (sumW[:, 0] - W[:, i]) / sumW[:, 0] ** 2
                    else:
                        dWbar_dW = -W[:, i] / sumW[:, 0] ** 2
                    dL_dW[:, j] += dL_dWbar[:, i] * dWbar_dW

            mu = self._mf(X)  # (n, r, f)
            # dW_r/dmu_r_f = 1/f  (dérivée de la moyenne)
            for ri in range(r):
                for fi_ in range(f):
                    dW_dmu = np.full(n, 1.0 / f)
                    dL_dmu = dL_dW[:, ri] * dW_dmu

                    diff = X[:, fi_] - self.centers_[ri, fi_]
                    sig = self.sigmas_[ri, fi_]

                    dmu_dc = mu[:, ri, fi_] * 2 * diff / sig ** 2
                    dmu_dsig = mu[:, ri, fi_] * 2 * diff ** 2 / sig ** 3

                    grad_c = (dL_dmu * dmu_dc).mean()
                    grad_sig = (dL_dmu * dmu_dsig).mean()

                    self.centers_[ri, fi_] -= self.lr * grad_c
                    self.sigmas_[ri, fi_] -= self.lr * grad_sig
                    self.sigmas_[ri, fi_] = max(self.sigmas_[ri, fi_], 1e-3)

        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------
    def predict(self, X):
        X = np.asarray(X, dtype=float)
        W = self._firing(X)
        Wbar = self._norm_firing(W)
        Phi = self._consequent_matrix(X, Wbar)
        return (Phi @ self.theta_) * self.y_std_ + self.y_mean_
