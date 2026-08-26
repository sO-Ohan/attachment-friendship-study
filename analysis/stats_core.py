#!/usr/bin/env python3
"""
Statistical engine, part 1 of 2.
Attachment style, fear of rejection and friendship formation among BRACU undergraduates.

Reads the refined bias-free analytic sample (n = 103) and produces every number the
paper reports. Nothing is rounded until output; everything is written to out/results.json
so the LaTeX build and the figure script read identical values.

Implemented by hand (no statsmodels/pingouin dependency): Cronbach alpha with bootstrap
CI, McDonald's omega, EFA with varimax, Fisher-z correlation CIs, Benjamini-Hochberg FDR,
Welch ANOVA, Brown-Forsythe, Games-Howell, omega-squared, Pillai MANOVA, OLS with VIF and
semipartials, bootstrap mediation, k-means with adjusted Rand, Cochran's Q, logistic
regression by Newton-Raphson, and achieved/sensitivity power for one-way F.
"""
import json, itertools, warnings
import numpy as np, pandas as pd
from scipy import stats, optimize

warnings.filterwarnings("ignore")
RNG = np.random.default_rng(20260827)
BOOT = 10000

SRC = ("/mnt/storage/Accademics/2-2/psy101/final_project_paper/servay/"
       "REFINED_BIAS_FREE_DATASET/data/05_analytic_sample.csv")
OUT = "/mnt/storage/Accademics/2-2/psy101/final_project_paper/PAPER/analysis/out"

R = {}   # everything the paper cites

d = pd.read_csv(SRC)
N = len(d)
ORDER = ["Secure", "Preoccupied", "Dismissing", "Fearful"]
d["attachment_style"] = pd.Categorical(d["attachment_style"], categories=ORDER, ordered=False)

SCALES = {
    "anxiety":         ("Attachment anxiety",              4, (1, 7)),
    "avoidance":       ("Attachment avoidance",            4, (1, 7)),
    "scenario_anx":    ("Anticipated anxiety (bids)",      6, (1, 6)),
    "scenario_expect": ("Expected acceptance (bids)",      6, (1, 6)),
    "fne":             ("Fear of negative evaluation",     6, (1, 5)),
    "loneliness":      ("Loneliness (UCLA-3)",             3, (1, 4)),
}
CORE = ["anxiety", "avoidance", "fne", "scenario_anx", "scenario_expect",
        "loneliness", "friends_have", "friend_gap", "approach_inhibition", "convo_started"]
LABEL = {"anxiety": "Attachment anxiety", "avoidance": "Attachment avoidance",
         "fne": "Fear of negative evaluation", "scenario_anx": "Anticipated anxiety",
         "scenario_expect": "Expected acceptance", "loneliness": "Loneliness",
         "friends_have": "Close friends held", "friend_gap": "Friendship deficit",
         "approach_inhibition": "Approach inhibition", "convo_started": "Conversations initiated"}


# ---------------------------------------------------------------- helpers
def items(key):
    return d[[f"{key}_{j+1}" for j in range(SCALES[key][1])]]


def cronbach(X):
    X = np.asarray(X, float)
    X = X[~np.isnan(X).any(1)]
    k = X.shape[1]
    if k < 2 or len(X) < 3:
        return np.nan
    return (k / (k - 1)) * (1 - X.var(0, ddof=1).sum() / X.sum(1).var(ddof=1))


def boot_ci(fn, X, B=BOOT, lo=2.5, hi=97.5):
    X = np.asarray(X, float)
    n = len(X)
    vals = np.empty(B)
    for b in range(B):
        vals[b] = fn(X[RNG.integers(0, n, n)])
    vals = vals[np.isfinite(vals)]
    return float(np.percentile(vals, lo)), float(np.percentile(vals, hi))


def varimax(F, tol=1e-7, it=200):
    p, k = F.shape
    Rm = np.eye(k)
    dsum = 0
    for _ in range(it):
        old = dsum
        L = F @ Rm
        u, s, vt = np.linalg.svd(F.T @ (L**3 - L @ np.diag(np.sum(L**2, 0)) / p))
        Rm = u @ vt
        dsum = s.sum()
        if old != 0 and dsum / old < 1 + tol:
            break
    return F @ Rm


def efa(X, nf=2):
    """Principal-axis factoring with squared-multiple-correlation communalities."""
    X = np.asarray(X, float)
    X = X[~np.isnan(X).any(1)]
    Rc = np.corrcoef(X.T)
    h2 = 1 - 1 / np.diag(np.linalg.inv(Rc))
    Rr = Rc.copy()
    for _ in range(60):
        np.fill_diagonal(Rr, h2)
        w, v = np.linalg.eigh(Rr)
        idx = np.argsort(w)[::-1][:nf]
        L = v[:, idx] * np.sqrt(np.maximum(w[idx], 0))
        newh = (L**2).sum(1)
        if np.max(np.abs(newh - h2)) < 1e-6:
            h2 = newh
            break
        h2 = newh
    ev = np.sort(np.linalg.eigvalsh(Rc))[::-1]
    return varimax(L), h2, ev, Rc


def omega_total(X):
    """McDonald's omega-total from a one-factor principal-axis solution."""
    X = np.asarray(X, float)
    X = X[~np.isnan(X).any(1)]
    L, h2, _, Rc = efa(X, nf=1)
    l = L[:, 0]
    num = l.sum() ** 2
    return float(num / (num + (1 - l**2).sum()))


def fisher_ci(r, n):
    if not np.isfinite(r) or n < 4:
        return (np.nan, np.nan)
    z = np.arctanh(np.clip(r, -0.999999, 0.999999))
    se = 1 / np.sqrt(n - 3)
    return float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se))


def bh_fdr(p):
    p = np.asarray(p, float)
    m = len(p)
    o = np.argsort(p)
    q = np.empty(m)
    prev = 1.0
    for rank, i in enumerate(o[::-1]):
        k = m - rank
        prev = min(prev, p[i] * m / k)
        q[i] = prev
    return q


def welch_anova(groups):
    k = len(groups)
    n = np.array([len(g) for g in groups], float)
    m = np.array([g.mean() for g in groups])
    v = np.array([g.var(ddof=1) for g in groups])
    w = n / v
    mw = (w * m).sum() / w.sum()
    A = ((w * (m - mw) ** 2).sum()) / (k - 1)
    lam = (((1 - w / w.sum()) ** 2) / (n - 1)).sum()
    B = 1 + 2 * (k - 2) / (k**2 - 1) * lam
    F = A / B
    df2 = (k**2 - 1) / (3 * lam)
    return float(F), float(k - 1), float(df2), float(stats.f.sf(F, k - 1, df2))


def brown_forsythe_var(groups):
    """Levene / Brown-Forsythe test of variance homogeneity (median-centred)."""
    z = [np.abs(g - np.median(g)) for g in groups]
    F, p = stats.f_oneway(*z)
    return float(F), float(p)


def games_howell(groups, names):
    out = []
    for (i, j) in itertools.combinations(range(len(groups)), 2):
        a, b = groups[i], groups[j]
        na, nb = len(a), len(b)
        va, vb = a.var(ddof=1), b.var(ddof=1)
        se = np.sqrt(va / na + vb / nb)
        diff = a.mean() - b.mean()
        t = diff / se
        df = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
        p = stats.studentized_range.sf(np.abs(t) * np.sqrt(2), len(groups), df)
        crit = stats.studentized_range.ppf(0.95, len(groups), df) / np.sqrt(2)
        sp = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
        out.append(dict(a=names[i], b=names[j], diff=float(diff), se=float(se),
                        t=float(t), df=float(df), p=float(p),
                        lo=float(diff - crit * se), hi=float(diff + crit * se),
                        d=float(diff / sp) if sp > 0 else np.nan))
    return out


def hedges_g(a, b):
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    dd = (a.mean() - b.mean()) / sp
    J = 1 - 3 / (4 * (na + nb) - 9)
    g = dd * J
    se = np.sqrt((na + nb) / (na * nb) + g**2 / (2 * (na + nb - 2)))
    return float(g), float(g - 1.96 * se), float(g + 1.96 * se)


def ols(y, X, names):
    """X without intercept. Returns full regression table with sr2 and VIF."""
    m = np.isfinite(y) & np.isfinite(X).all(1)
    y, X = y[m], X[m]
    n = len(y)
    Xd = np.column_stack([np.ones(n), X])
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    resid = y - Xd @ beta
    p = X.shape[1]
    dfe = n - p - 1
    mse = resid @ resid / dfe
    XtXi = np.linalg.inv(Xd.T @ Xd)
    se = np.sqrt(np.diag(XtXi) * mse)
    t = beta / se
    pv = 2 * stats.t.sf(np.abs(t), dfe)
    sst = ((y - y.mean()) ** 2).sum()
    sse = resid @ resid
    r2 = 1 - sse / sst
    adj = 1 - (1 - r2) * (n - 1) / dfe
    F = (r2 / p) / ((1 - r2) / dfe)
    # standardised betas, semipartial r2, VIF
    sy = y.std(ddof=1)
    std_b, sr2, vif = [], [], []
    for j in range(p):
        sx = X[:, j].std(ddof=1)
        std_b.append(beta[j + 1] * sx / sy)
        keep = [c for c in range(p) if c != j]
        if keep:
            Xr = np.column_stack([np.ones(n), X[:, keep]])
            br, *_ = np.linalg.lstsq(Xr, y, rcond=None)
            sser = ((y - Xr @ br) ** 2).sum()
            sr2.append(float((sser - sse) / sst))
            Xj = np.column_stack([np.ones(n), X[:, keep]])
            bj, *_ = np.linalg.lstsq(Xj, X[:, j], rcond=None)
            rj2 = 1 - ((X[:, j] - Xj @ bj) ** 2).sum() / ((X[:, j] - X[:, j].mean()) ** 2).sum()
            vif.append(float(1 / (1 - rj2)) if rj2 < 1 else np.inf)
        else:
            sr2.append(float(r2)); vif.append(1.0)
    dw = float(((np.diff(resid)) ** 2).sum() / sse)
    return dict(n=int(n), r2=float(r2), adj_r2=float(adj), F=float(F), df1=p, df2=int(dfe),
                p=float(stats.f.sf(F, p, dfe)), rmse=float(np.sqrt(mse)),
                durbin_watson=dw,
                shapiro_resid_p=float(stats.shapiro(resid)[1]) if n <= 5000 else np.nan,
                terms=[dict(name=names[j], b=float(beta[j + 1]), se=float(se[j + 1]),
                            beta=float(std_b[j]), t=float(t[j + 1]), p=float(pv[j + 1]),
                            sr2=sr2[j], vif=vif[j],
                            lo=float(beta[j + 1] - stats.t.ppf(.975, dfe) * se[j + 1]),
                            hi=float(beta[j + 1] + stats.t.ppf(.975, dfe) * se[j + 1]))
                       for j in range(p)],
                intercept=float(beta[0]), sse=float(sse), sst=float(sst))


def r2_only(y, X):
    m = np.isfinite(y) & np.isfinite(X).all(1)
    y, X = y[m], X[m]
    Xd = np.column_stack([np.ones(len(y)), X])
    b, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    return 1 - ((y - Xd @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum(), len(y)


def hierarchical(y, blocks, names):
    """blocks: list of (label, ndarray). Returns step-wise R2 change table."""
    steps, X = [], None
    for lab, B in blocks:
        X = B if X is None else np.column_stack([X, B])
        r2, n = r2_only(y, X)
        p = X.shape[1]
        prev = steps[-1]["r2"] if steps else 0.0
        pprev = steps[-1]["k"] if steps else 0
        dR = r2 - prev
        dfn = p - pprev
        dfd = n - p - 1
        Fc = (dR / dfn) / ((1 - r2) / dfd) if dfn > 0 and r2 < 1 else np.nan
        steps.append(dict(block=lab, k=p, n=int(n), r2=float(r2),
                          adj_r2=float(1 - (1 - r2) * (n - 1) / dfd),
                          dR2=float(dR), F_change=float(Fc), df1=int(dfn), df2=int(dfd),
                          p_change=float(stats.f.sf(Fc, dfn, dfd)) if np.isfinite(Fc) else np.nan))
    return steps


def mediate(Xv, Mv, Yv, covs=None, B=BOOT):
    """Percentile bootstrap of the indirect effect a*b."""
    cols = [Xv, Mv, Yv] + (covs or [])
    m = d[cols].dropna()
    X, M, Y = m[Xv].values, m[Mv].values, m[Yv].values
    C = m[covs].values if covs else np.empty((len(m), 0))
    n = len(m)

    def fit(idx):
        x, mm, y, c = X[idx], M[idx], Y[idx], C[idx]
        A = np.column_stack([np.ones(n), x, c])
        a = np.linalg.lstsq(A, mm, rcond=None)[0][1]
        Bm = np.column_stack([np.ones(n), x, mm, c])
        bb = np.linalg.lstsq(Bm, y, rcond=None)[0]
        cprime = bb[1]; b = bb[2]
        ctot = np.linalg.lstsq(A, y, rcond=None)[0][1]
        return a, b, cprime, ctot

    a, b, cp, ct = fit(np.arange(n))
    ind = np.empty(B)
    for i in range(B):
        idx = RNG.integers(0, n, n)
        aa, bb2, _, _ = fit(idx)
        ind[i] = aa * bb2
    lo, hi = np.percentile(ind, [2.5, 97.5])
    return dict(n=int(n), a=float(a), b=float(b), c_total=float(ct), c_prime=float(cp),
                indirect=float(a * b), lo=float(lo), hi=float(hi),
                prop_mediated=float(a * b / ct) if ct != 0 else np.nan,
                boot=int(B))


def logistic(y, X, names):
    m = np.isfinite(y) & np.isfinite(X).all(1)
    y, X = y[m].astype(float), X[m]
    n = len(y)
    Xd = np.column_stack([np.ones(n), X])
    beta = np.zeros(Xd.shape[1])
    for _ in range(200):
        eta = Xd @ beta
        p = 1 / (1 + np.exp(-eta))
        W = p * (1 - p)
        g = Xd.T @ (y - p)
        H = Xd.T @ (Xd * W[:, None])
        try:
            step = np.linalg.solve(H + 1e-9 * np.eye(len(beta)), g)
        except np.linalg.LinAlgError:
            break
        beta += step
        if np.max(np.abs(step)) < 1e-9:
            break
    eta = Xd @ beta
    p = np.clip(1 / (1 + np.exp(-eta)), 1e-12, 1 - 1e-12)
    ll = float((y * np.log(p) + (1 - y) * np.log(1 - p)).sum())
    p0 = y.mean()
    ll0 = float((y * np.log(p0) + (1 - y) * np.log(1 - p0)).sum())
    cov = np.linalg.inv(Xd.T @ (Xd * (p * (1 - p))[:, None]) + 1e-9 * np.eye(len(beta)))
    se = np.sqrt(np.diag(cov))
    z = beta / se
    # concordance (AUC)
    pos, neg = p[y == 1], p[y == 0]
    auc = float((pos[:, None] > neg[None, :]).mean() + .5 * (pos[:, None] == neg[None, :]).mean())
    return dict(n=int(n), ll=ll, ll_null=ll0, chi2=float(2 * (ll - ll0)),
                df=int(Xd.shape[1] - 1),
                p=float(stats.chi2.sf(2 * (ll - ll0), Xd.shape[1] - 1)),
                mcfadden=float(1 - ll / ll0),
                nagelkerke=float((1 - np.exp(-2 * (ll - ll0) / n)) / (1 - np.exp(2 * ll0 / n))),
                auc=auc, accuracy=float(((p > .5) == (y == 1)).mean()),
                intercept=float(beta[0]),
                terms=[dict(name=names[j], b=float(beta[j + 1]), se=float(se[j + 1]),
                            z=float(z[j + 1]), p=float(2 * stats.norm.sf(abs(z[j + 1]))),
                            OR=float(np.exp(beta[j + 1])),
                            lo=float(np.exp(beta[j + 1] - 1.96 * se[j + 1])),
                            hi=float(np.exp(beta[j + 1] + 1.96 * se[j + 1])))
                       for j in range(X.shape[1])])


def pillai_manova(Y, g):
    """One-way MANOVA, Pillai's trace, with F approximation."""
    m = np.isfinite(Y).all(1)
    Y, g = Y[m], np.asarray(g)[m]
    lv = pd.unique(g)
    n, p = Y.shape
    k = len(lv)
    gm = Y.mean(0)
    H = np.zeros((p, p)); E = np.zeros((p, p))
    for l in lv:
        Yi = Y[g == l]
        dm = (Yi.mean(0) - gm)[:, None]
        H += len(Yi) * dm @ dm.T
        Ci = Yi - Yi.mean(0)
        E += Ci.T @ Ci
    V = float(np.trace(H @ np.linalg.inv(H + E)))
    s = min(k - 1, p)
    mm = (abs(k - 1 - p) - 1) / 2
    nn = (n - k - p - 1) / 2
    df1 = s * (2 * mm + s + 1)
    df2 = s * (2 * nn + s + 1)
    F = (2 * nn + s + 1) / (2 * mm + s + 1) * V / (s - V)
    return dict(pillai=V, F=float(F), df1=float(df1), df2=float(df2),
                p=float(stats.f.sf(F, df1, df2)), eta2_mult=float(V / s), n=int(n), k=int(k))


def kmeans(X, k, iters=500, restarts=100):
    X = np.asarray(X, float)
    best, bl, bi = np.inf, None, None
    for _ in range(restarts):
        C = X[RNG.choice(len(X), k, replace=False)]
        lab = None
        for _ in range(iters):
            dist = ((X[:, None, :] - C[None]) ** 2).sum(2)
            nl = dist.argmin(1)
            if lab is not None and (nl == lab).all():
                break
            lab = nl
            for j in range(k):
                if (lab == j).any():
                    C[j] = X[lab == j].mean(0)
        w = ((X - C[lab]) ** 2).sum()
        if w < best:
            best, bl, bi = w, lab.copy(), C.copy()
    return bl, bi, float(best)


def adj_rand(a, b):
    ct = pd.crosstab(pd.Series(a), pd.Series(b)).values
    n = ct.sum()
    def c2(x): return x * (x - 1) / 2
    sij = c2(ct).sum(); sa = c2(ct.sum(1)).sum(); sb = c2(ct.sum(0)).sum()
    exp = sa * sb / c2(n)
    mx = (sa + sb) / 2
    return float((sij - exp) / (mx - exp))


def silhouette(X, lab):
    X = np.asarray(X, float)
    D = np.sqrt(((X[:, None, :] - X[None]) ** 2).sum(2))
    s = np.empty(len(X))
    for i in range(len(X)):
        same = lab == lab[i]
        same[i] = False
        a = D[i, same].mean() if same.any() else 0
        bb = min(D[i, lab == j].mean() for j in np.unique(lab) if j != lab[i])
        s[i] = (bb - a) / max(a, bb) if max(a, bb) > 0 else 0
    return float(s.mean())


def cochran_q(M):
    M = np.asarray(M, int)
    k = M.shape[1]
    Cj = M.sum(0); Ri = M.sum(1); Nn = M.sum()
    Q = k * (k - 1) * ((Cj - Nn / k) ** 2).sum() / (k * Ri.sum() - (Ri**2).sum())
    return float(Q), int(k - 1), float(stats.chi2.sf(Q, k - 1))


def power_oneway(f2_eta, n, k, alpha=.05):
    """Achieved power for one-way F given eta-squared."""
    f2 = f2_eta / (1 - f2_eta)
    lam = f2 * n
    crit = stats.f.ppf(1 - alpha, k - 1, n - k)
    return float(stats.ncf.sf(crit, k - 1, n - k, lam))


def sensitivity_oneway(n, k, alpha=.05, power=.80):
    def g(f):
        return stats.ncf.sf(stats.f.ppf(1 - alpha, k - 1, n - k), k - 1, n - k, f**2 * n) - power
    return float(optimize.brentq(g, 1e-4, 3))


def power_r(r, n, alpha=.05):
    z = np.arctanh(abs(r)); se = 1 / np.sqrt(n - 3)
    zc = stats.norm.ppf(1 - alpha / 2)
    return float(stats.norm.sf(zc - z / se) + stats.norm.cdf(-zc - z / se))


def sensitivity_r(n, alpha=.05, power=.80):
    def g(r):
        return power_r(r, n, alpha) - power
    return float(optimize.brentq(g, 1e-4, .95))


def desc(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return dict(n=int(len(x)), mean=float(x.mean()), sd=float(x.std(ddof=1)),
                median=float(np.median(x)), min=float(x.min()), max=float(x.max()),
                skew=float(stats.skew(x, bias=False)), kurt=float(stats.kurtosis(x, bias=False)),
                se=float(x.std(ddof=1) / np.sqrt(len(x))),
                ci_lo=float(x.mean() - stats.t.ppf(.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))),
                ci_hi=float(x.mean() + stats.t.ppf(.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))),
                shapiro_p=float(stats.shapiro(x)[1]))


# ================================================================ 1. SAMPLE
R["meta"] = dict(n_raw=104, n_analytic=N, excluded=1, seed=20260827, bootstrap=BOOT,
                 source="REFINED_BIAS_FREE_DATASET/data/05_analytic_sample.csv")
R["sample"] = dict(
    age=desc(d.age), semester=desc(d.semester),
    gender={k: int(v) for k, v in d.gender.value_counts().items()},
    department={k: int(v) for k, v in d.department.value_counts().items()},
    caregiver={k: int(v) for k, v in d.caregiver.value_counts().items()},
    family_moves={str(k): int(v) for k, v in d.family_moves.value_counts().sort_index().items()},
    style={k: int(v) for k, v in d.attachment_style.value_counts().reindex(ORDER).items()},
    style_pct={k: round(100 * v / N, 1) for k, v in d.attachment_style.value_counts().reindex(ORDER).items()},
    willing={k: int(v) for k, v in d.willing_followup.value_counts().items()},
    graduated=int(d.graduated_flag.sum()),
    straightlining_flagged=int(d.straightlining_flag.sum()),
    missing={c: int(d[c].isna().sum()) for c in CORE if d[c].isna().sum() > 0},
    pct_complete_core=float(100 * d[CORE].notna().all(1).mean()),
)
# style distribution vs uniform
obs = d.attachment_style.value_counts().reindex(ORDER).values
chi, pchi = stats.chisquare(obs)
R["sample"]["style_chi2"] = dict(chi2=float(chi), df=3, p=float(pchi),
                                 cramers_v=float(np.sqrt(chi / (N * 3))))
# insecure vs secure binomial
ins = int(N - obs[0])
R["sample"]["insecure_binomial"] = dict(
    insecure=ins, n=N, prop=float(ins / N),
    ci=[float(x) for x in stats.binomtest(ins, N).proportion_ci(0.95)],
    p_vs_half=float(stats.binomtest(ins, N, 0.5).pvalue))
# gender x style independence
ct = pd.crosstab(d.gender, d.attachment_style)
c2, pg, dfg, _ = stats.chi2_contingency(ct)
R["sample"]["gender_style_chi2"] = dict(chi2=float(c2), df=int(dfg), p=float(pg),
                                        cramers_v=float(np.sqrt(c2 / (ct.values.sum() * min(ct.shape) - ct.values.sum()))) if min(ct.shape) > 1 else np.nan)

# ================================================================ 2. RELIABILITY
rel = {}
for k, (lab, ni, rng) in SCALES.items():
    X = items(k)
    a = cronbach(X)
    lo, hi = boot_ci(cronbach, X.dropna().values)
    Xc = X.dropna().values
    Rc = np.corrcoef(Xc.T)
    aic = float((Rc.sum() - ni) / (ni * (ni - 1)))
    tot = Xc.sum(1)
    itc = [float(np.corrcoef(Xc[:, j], tot - Xc[:, j])[0, 1]) for j in range(ni)]
    drop = []
    for j in range(ni):
        keep = [c for c in range(ni) if c != j]
        drop.append(float(cronbach(Xc[:, keep])))
    rel[k] = dict(label=lab, k=ni, range=list(rng), alpha=float(a), alpha_lo=lo, alpha_hi=hi,
                  omega=omega_total(X), avg_inter_item=aic,
                  item_total=itc, alpha_if_dropped=drop,
                  **desc(d[k]))
R["reliability"] = rel

# ================================================================ 3. FACTOR STRUCTURE
ecr = pd.concat([items("anxiety"), items("avoidance")], axis=1)
L, h2, ev, Rc = efa(ecr, nf=2)
kmo_num = (Rc[np.triu_indices_from(Rc, 1)] ** 2).sum()
Pc = np.linalg.inv(Rc); Ac = -Pc / np.sqrt(np.outer(np.diag(Pc), np.diag(Pc)))
kmo_den = kmo_num + (Ac[np.triu_indices_from(Ac, 1)] ** 2).sum()
nb = len(ecr.dropna())
chi_b = -(nb - 1 - (2 * Rc.shape[0] + 5) / 6) * np.log(np.linalg.det(Rc))
df_b = Rc.shape[0] * (Rc.shape[0] - 1) / 2
R["efa_ecr"] = dict(
    n=int(nb), loadings=[[float(x) for x in row] for row in L],
    items=[f"ANX{j+1}" for j in range(4)] + [f"AVO{j+1}" for j in range(4)],
    communalities=[float(x) for x in h2], eigenvalues=[float(x) for x in ev],
    var_explained=[float(x) for x in (L**2).sum(0) / L.shape[0]],
    kmo=float(kmo_num / kmo_den),
    bartlett=dict(chi2=float(chi_b), df=int(df_b), p=float(stats.chi2.sf(chi_b, df_b))))

# ================================================================ 4. CORRELATIONS
cm = np.full((len(CORE), len(CORE)), np.nan)
pm = np.ones_like(cm); nm = np.zeros_like(cm)
cil = np.full_like(cm, np.nan); cih = np.full_like(cm, np.nan)
sp = np.full_like(cm, np.nan)
flat = []
for i, a in enumerate(CORE):
    for j, b in enumerate(CORE):
        mm = d[[a, b]].dropna()
        if i == j:
            cm[i, j] = 1.0; nm[i, j] = len(mm); continue
        r, p = stats.pearsonr(mm[a], mm[b])
        rs, _ = stats.spearmanr(mm[a], mm[b])
        cm[i, j] = r; pm[i, j] = p; nm[i, j] = len(mm); sp[i, j] = rs
        cil[i, j], cih[i, j] = fisher_ci(r, len(mm))
        if i < j:
            flat.append(dict(a=a, b=b, r=float(r), p=float(p), n=int(len(mm)),
                             lo=cil[i, j], hi=cih[i, j], rho=float(rs)))
q = bh_fdr([f["p"] for f in flat])
for f, qq in zip(flat, q):
    f["q"] = float(qq)
    f["sig_fdr"] = bool(qq < .05)
R["correlations"] = dict(vars=CORE, labels=[LABEL[c] for c in CORE],
                         r=[[None if not np.isfinite(x) else float(x) for x in row] for row in cm],
                         p=[[float(x) for x in row] for row in pm],
                         n=[[int(x) for x in row] for row in nm],
                         ci_lo=[[None if not np.isfinite(x) else float(x) for x in row] for row in cil],
                         ci_hi=[[None if not np.isfinite(x) else float(x) for x in row] for row in cih],
                         spearman=[[None if not np.isfinite(x) else float(x) for x in row] for row in sp],
                         pairs=flat, n_tests=len(flat),
                         n_sig_raw=int(sum(f["p"] < .05 for f in flat)),
                         n_sig_fdr=int(sum(f["sig_fdr"] for f in flat)))

json.dump(R, open(f"{OUT}/results_part1.json", "w"), indent=1)
print("part1 written; keys:", list(R.keys()))
print("alpha:", {k: round(v["alpha"], 3) for k, v in rel.items()})
print("omega:", {k: round(v["omega"], 3) for k, v in rel.items()})
print("KMO", round(R["efa_ecr"]["kmo"], 3), "Bartlett p", R["efa_ecr"]["bartlett"]["p"])
print("corr sig raw/fdr:", R["correlations"]["n_sig_raw"], "/", R["correlations"]["n_sig_fdr"], "of", len(flat))
