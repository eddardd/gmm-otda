import torch
import numpy as np
from src.prob_utils import diag_gmm_log_probs


class GMMOTDA:
    """Gaussian Mixture Model-based Optimal Transport for Domain Adaptation"""

    def __init__(self,
                 ot_solver,
                 clustering_source,
                 clustering_target,
                 min_var=0.01):
        self.ot_solver = ot_solver
        self.clustering_source = clustering_source
        self.clustering_target = clustering_target
        self.min_var = min_var

        self.stds_src = None
        self.weights_src = None
        self.labels_src = None
        self.centroids_src = None

        self.stds_tgt = None
        self.labels_tgt = None
        self.weights_tgt = None
        self.centroids_tgt = None
        self.estimated_labels_tgt = None

        self.ot_plan = None
        self.n_dim = None
        self.n_classes = None

        self.fitted_gmm = False
        self.fitted_ot = False

    def fit_gmms(self, Xs, Ys, Xt, Yt=None):
        """Fit GMMs to source and target domain data"""
        self.n_dim = Xs.shape[1]
        self.n_classes = Ys.shape[1]

        w_s, m_s, v_s, y_s = self.clustering_source(Xs, Ys)
        v_s[v_s < self.min_var] = self.min_var
        self.weights_src = w_s
        self.stds_src = v_s ** 0.5
        self.labels_src = y_s
        self.centroids_src = m_s

        w_t, m_t, v_t, y_t = self.clustering_target(Xt, Yt)
        v_t[v_t < self.min_var] = self.min_var
        self.labels_tgt = y_t
        self.stds_tgt = v_t ** 0.5
        self.weights_tgt = w_t
        self.centroids_tgt = m_t

        self.fitted_gmm = True

        return self

    def fit_ot(self):
        """Solves the GMM-OT problem"""
        if not self.fitted_gmm:
            raise ValueError("Expected 'fit_gmm' to be previously called.")

        C = torch.cdist(self.centroids_src, self.centroids_tgt, p=2) ** 2 + \
            torch.cdist(self.stds_src, self.stds_tgt, p=2) ** 2
        self.ot_plan = self.ot_solver(
            self.weights_src,
            self.weights_tgt,
            C
        )

        self.estimated_labels_tgt = torch.mm(
            (self.ot_plan / self.ot_plan.sum(dim=0)[None, :]).T,
            self.labels_src)

        self.fitted_ot = True
        return self

    def fit(self, Xs, Ys, Xt, Yt=None):
        """Fit pipeline. First, fits GMMs, then, fits the GMM-OT problem"""
        return self.fit_gmms(Xs, Ys, Xt, Yt).fit_ot()

    def predict_target_components(self, X, return_proba=False):
        """Performs k = argmax P_T(K|X) (return_proba = False)
        or computes the probabiliities P_T(K|X) (return_proba = True)."""
        if not self.fitted_gmm:
            raise ValueError("Expected 'fit_gmm' to be called previously.")
        if not self.fitted_ot:
            raise ValueError("Expected 'fit_ot' to be called previously.")

        log_probs = diag_gmm_log_probs(
            X=X,
            weights=self.weights_tgt,
            means=self.centroids_tgt,
            stds=self.stds_tgt
        )
        log_proba_components = (
            log_probs - log_probs.logsumexp(dim=0)[None, :])
        if return_proba:
            return log_proba_components.exp()
        return log_proba_components.argmax(dim=0)

    def predict_source_components(self, X, return_proba=False):
        """Performs k = argmax P_S(K|X) (return_proba = False)
        or computes the probabiliities P_S(K|X) (return_proba = True)."""
        if not self.fitted_gmm:
            raise ValueError("Expected 'fit_gmm' to be called previously.")

        log_probs = diag_gmm_log_probs(
            X=X,
            weights=self.weights_src,
            means=self.centroids_src,
            stds=self.stds_src
        )
        log_proba_components = (
            log_probs - log_probs.logsumexp(dim=0)[None, :])
        if return_proba:
            return log_proba_components.exp()
        return log_proba_components.argmax(dim=0)

    def sample_from_source(self, n):
        """Samples from the source domain GMM"""
        if not self.fitted_gmm:
            raise ValueError("Expected 'fit_gmm' to be called previously.")
        Xsyn = []
        Ysyn = []

        for _ in range(n):
            k = np.random.choice(
                np.arange(len(self.weights_src)),
                p=(self.weights_src / self.weights_src.sum()).numpy())

            _x = self.stds_src[k, :] * np.random.randn(self.n_dim) + \
                self.centroids_src[k, :]
            _y = torch.nn.functional.one_hot(
                self.labels_src[k, :].argmax(),
                num_classes=self.n_classes).float()

            Xsyn.append(_x)
            Ysyn.append(_y)
        Xsyn = torch.stack(Xsyn).float()
        Ysyn = torch.stack(Ysyn).float()
        return Xsyn, Ysyn

    def sample_from_target(self, n, use_estimated_labels=True):
        """Samples from the target domain GMM"""
        if not self.fitted_gmm:
            raise ValueError("Expected 'fit_gmm' to be called previously.")
        if use_estimated_labels and not self.fitted_ot:
            raise ValueError("Expected 'fit_ot' to be called previously.")
        if not use_estimated_labels and not self.labels_tgt:
            raise ValueError(
                "If not using estimated labels, expects target GMM to"
                " be fitted using labels."
            )
        Xsyn = []
        Ysyn = []

        for _ in range(n):
            k = np.random.choice(
                np.arange(len(self.weights_tgt)),
                p=(self.weights_tgt / self.weights_tgt.sum()).numpy())

            _x = self.stds_tgt[k, :] * np.random.randn(self.n_dim) + \
                self.centroids_tgt[k, :]
            if use_estimated_labels:
                _y = torch.nn.functional.one_hot(
                    self.estimated_labels_tgt[k, :].argmax(),
                    num_classes=self.n_classes).float()
            else:
                _y = torch.nn.functional.one_hot(
                    self.labels_tgt[k, :].argmax(),
                    num_classes=self.n_classes).float()

            Xsyn.append(_x)
            Ysyn.append(_y)
        Xsyn = torch.stack(Xsyn).float()
        Ysyn = torch.stack(Ysyn).float()
        return Xsyn, Ysyn

    def predict_source_labels(self, X):
        """Predicts class labels using the source GMM"""
        if not self.fitted_gmm:
            raise ValueError("Expected 'fit_gmm' to be called previously.")
        proba_components = self.predict_source_components(X, return_proba=True)
        cluster_labels = torch.mm(
            self.labels_src.T, proba_components).T
        return cluster_labels

    def predict_target_labels(self, X, use_estimated_labels=True):
        """Predicts class labels using the target GMM"""
        if not self.fitted_gmm:
            raise ValueError("Expected 'fit_gmm' to be called previously.")
        if not self.fitted_ot:
            raise ValueError("Expected 'fit_ot' to be called previously.")
        proba_components = self.predict_target_components(X, return_proba=True)
        if use_estimated_labels:
            cluster_labels = torch.mm(
                self.estimated_labels_tgt.T, proba_components).T
        else:
            cluster_labels = torch.mm(
                self.labels_tgt.T, proba_components).T
        return cluster_labels

    def compute_source_nll(self, X):
        """Computes the Negative Log-Likelihood (NLL) using the
        source domain GMM."""
        if not self.fitted_gmm:
            raise ValueError("Expected 'fit_gmm' to be called previously.")

        log_probs = diag_gmm_log_probs(
            X=X,
            weights=self.weights_src,
            means=self.centroids_src,
            stds=self.stds_src
        )
        return - log_probs.logsumexp(dim=0).mean()

    def compute_target_nll(self, X):
        """Computes the Negative Log-Likelihood (NLL) using the
        target domain GMM."""
        if not self.fitted_gmm:
            raise ValueError("Expected 'fit_gmm' to be called previously.")

        log_probs = diag_gmm_log_probs(
            X=X,
            weights=self.weights_tgt,
            means=self.centroids_tgt,
            stds=self.stds_tgt
        )
        return - log_probs.logsumexp(dim=0).mean()

    def transport_samples(self, X, Y, numel=None):
        """Computes the weighted map from the source to the target domain."""
        if numel is None:
            numel = self.ot_plan.shape[0] + self.ot_plan.shape[1] - 1
        q = np.quantile(self.ot_plan.flatten(),
                        1 - numel / self.ot_plan.numel())
        ind_s, ind_t = np.where(self.ot_plan > q)

        transp_w = []
        transp_X = []
        transp_y = []

        components_s = self.predict_source_components(X)
        for i_s, i_t in zip(ind_s, ind_t):
            idx = np.where(components_s == i_s)[0]
            x = X[idx]
            y = Y[idx]

            w = self.ot_plan[i_s, i_t]
            A = self.stds_tgt[i_t] / (self.stds_src[i_s] + 1e-9)
            b = self.centroids_tgt[i_t] - self.centroids_src[i_s] * A

            transp_w.append(torch.Tensor([w] * len(x)))
            transp_X.append(x * A + b)
            transp_y.append(y)
        transp_w = torch.cat(transp_w, dim=0)
        transp_X = torch.cat(transp_X, dim=0)
        transp_y = torch.cat(transp_y, dim=0)

        return transp_w, transp_X, transp_y

    def rand_transport(self, X, Y, numel=None):
        """Computes the rand transport of (Delon and Desolneux, 2020) between
        source and target domains."""
        proba_components_src = self.predict_source_components(
            X, return_proba=True).T
        sampling_probs = torch.zeros([
            len(X), len(self.weights_src), len(self.weights_tgt)])
        for k1 in range(len(self.weights_src)):
            for k2 in range(len(self.weights_tgt)):
                sampling_probs[:, k1, k2] = (
                    (self.ot_plan[k1, k2] / self.weights_src[k1]) *
                    proba_components_src[:, k1])
        sampling_probs = sampling_probs.numpy()

        indices = np.arange(len(self.ot_plan.flatten()))
        indices_PQ = np.array([
            (k1, k2)
            for k1 in range(self.ot_plan.shape[0])
            for k2 in range(self.ot_plan.shape[1])])
        mapped_x = []

        for pi, xi in zip(sampling_probs, X):
            idx = np.random.choice(indices, p=pi.flatten())
            k1, k2 = indices_PQ[idx]

            A = self.stds_tgt[k2] / (self.stds_src[k1] + 1e-9)
            b = self.centroids_tgt[k2] - self.centroids_src[k1] * A

            mapped_x.append(xi * A + b)
        mapped_x = torch.stack(mapped_x)
        return mapped_x, Y
