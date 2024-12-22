import torch
from sklearn.mixture import GaussianMixture


def em_gmm(X, Y, n_clusters, random_state=None, dtype=torch.float32):
    """Fits a GMM using the expectation-maximization algorithm."""
    clustering = GaussianMixture(
        n_components=n_clusters,
        covariance_type='diag',
        random_state=random_state).fit(X)
    w = torch.from_numpy(clustering.weights_).to(dtype)
    m = torch.from_numpy(clustering.means_).to(dtype)
    v = torch.from_numpy(clustering.covariances_).to(dtype)

    return w, m, v, None


def conditional_em_gmm(X, Y, n_clusters, random_state=None,
                       dtype=torch.float32):
    """Fits GMMs on the conditoinals of P(X|Y)
    using expectation maximization"""
    n_classes = Y.shape[1]
    w_s, m_s, v_s, y_s = [], [], [], []
    for c in Y.argmax(dim=1).unique():
        ind = torch.where(Y.argmax(dim=1) == c)[0]
        wc, mc, vc, _ = em_gmm(
            X=X[ind],
            Y=None,
            n_clusters=n_clusters,
            random_state=random_state,
            dtype=dtype)
        w_s.append(wc)
        m_s.append(mc)
        v_s.append(vc)
        y_s.append(
            torch.nn.functional.one_hot(
                torch.Tensor([c] * len(mc)).long(),
                num_classes=n_classes
            ).to(dtype)
        )
    w_s = torch.cat(w_s).to(dtype)
    w_s /= w_s.sum()
    m_s = torch.cat(m_s, dim=0).to(dtype)
    v_s = torch.cat(v_s, dim=0).to(dtype)
    y_s = torch.cat(y_s, dim=0).to(dtype)

    return w_s, m_s, v_s, y_s
