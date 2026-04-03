"""
generation_module.py
=====================
Génération de profils de consommation conditionnelle (RS / RP).
Deux approches disponibles :
  1. CVAE  — Conditional Variational AutoEncoder (PyTorch)
  2. GMM   — Gaussian Mixture Model par classe (scikit-learn), baseline statistique

Depuis les vecteurs de features générés, un modèle paramétrique
reconstitue des courbes de charge journalières au pas 30 min (48 points/jour).

Métriques d'évaluation :
  - MMD  (Maximum Mean Discrepancy)
  - Distance de Wasserstein par feature
  - Test KS  (Kolmogorov-Smirnov) par feature
  - Comparaison des moments statistiques (µ, σ)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from scipy import stats
from scipy.stats import wasserstein_distance

# ──────────────────────────────────────────────────────────────
# CONSTANTES
# ──────────────────────────────────────────────────────────────
FEATURE_NAMES = [
    "conso_moyenne",      # Consommation moyenne (Wh / 30 min)
    "conso_std",          # Écart-type de consommation
    "ratio_we_semaine",   # Rapport weekend / semaine
    "ratio_hiver_ete",    # Rapport hiver / été  (lié au chauffage électrique)
    "prop_conso_nulle",   # Proportion d'horodates à consommation nulle (≈ absence)
]
N_STEPS = 48          # Résolution 30 min → 48 pas/jour
N_INPUT = len(FEATURE_NAMES)


# ══════════════════════════════════════════════════════════════
#  ARCHITECTURE  —  CVAE
# ══════════════════════════════════════════════════════════════

class _Encoder(nn.Module):
    def __init__(self, input_dim: int, cond_dim: int, latent_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + cond_dim, hidden),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.LeakyReLU(0.2),
        )
        self.fc_mu     = nn.Linear(hidden // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden // 2, latent_dim)

    def forward(self, x, c):
        h = self.net(torch.cat([x, c], dim=1))
        return self.fc_mu(h), self.fc_logvar(h)


class _Decoder(nn.Module):
    def __init__(self, latent_dim: int, cond_dim: int, output_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, hidden // 2),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(hidden // 2),
            nn.Linear(hidden // 2, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, z, c):
        return self.net(torch.cat([z, c], dim=1))


class CVAE(nn.Module):
    """
    Autoencodeur Variationnel Conditionnel.
    Conditionnement : c = 0 (RP) ou c = 1 (RS).
    """

    def __init__(
        self,
        input_dim: int  = N_INPUT,
        cond_dim: int   = 1,
        latent_dim: int = 8,
        hidden: int     = 64,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder    = _Encoder(input_dim, cond_dim, latent_dim, hidden)
        self.decoder    = _Decoder(latent_dim, cond_dim, input_dim, hidden)

    def reparameterize(self, mu, logvar):
        if self.training:
            return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return mu

    def forward(self, x, c):
        mu, logvar = self.encoder(x, c)
        z          = self.reparameterize(mu, logvar)
        recon      = self.decoder(z, c)
        return recon, mu, logvar

    @torch.no_grad()
    def generate(self, n: int, label: int, device: str = "cpu") -> np.ndarray:
        """Génère n vecteurs de features normalisés pour le type `label`."""
        self.eval()
        z = torch.randn(n, self.latent_dim, device=device)
        c = torch.full((n, 1), float(label), device=device)
        return self.decoder(z, c).cpu().numpy()


def _cvae_loss(recon, x, mu, logvar, beta: float = 5e-4):
    """Fonction de perte ELBO (MSE reconstruction + KL divergence)."""
    mse = nn.MSELoss()(recon, x)
    kl  = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return mse + beta * kl, float(mse.detach()), float(kl.detach())


# ══════════════════════════════════════════════════════════════
#  ENTRAÎNEMENT  —  CVAE
# ══════════════════════════════════════════════════════════════

def train_cvae(
    X_scaled: np.ndarray,
    labels: np.ndarray,
    n_epochs: int   = 300,
    latent_dim: int = 8,
    hidden: int     = 64,
    batch_size: int = 32,
    lr: float       = 1e-3,
    beta: float     = 5e-4,
    device: str     = "cpu",
    progress_callback=None,
) -> tuple:
    """
    Entraîne le CVAE.

    Paramètres
    ----------
    X_scaled : features normalisées  (n_samples × N_INPUT)
    labels   : array d'entiers 0/1   (n_samples,)
    progress_callback : fonction(epoch, loss) appelée à chaque époque
                        (utile pour la barre Streamlit)

    Returns
    -------
    model  : CVAE entraîné
    losses : liste de (total_loss, recon_loss, kl_loss) par époque
    scaler : StandardScaler déjà fitté (à sauvegarder pour l'inversion)
    """
    X_t = torch.FloatTensor(X_scaled).to(device)
    c_t = torch.FloatTensor(labels.reshape(-1, 1)).to(device)

    loader = DataLoader(TensorDataset(X_t, c_t), batch_size=batch_size, shuffle=True)
    model  = CVAE(latent_dim=latent_dim, hidden=hidden).to(device)
    opt    = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched  = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

    losses = []
    model.train()

    for ep in range(n_epochs):
        ep_total, ep_recon, ep_kl = 0.0, 0.0, 0.0
        for xb, cb in loader:
            opt.zero_grad()
            recon, mu, logvar = model(xb, cb)
            loss, r, k = _cvae_loss(recon, xb, mu, logvar, beta)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_total += float(loss) * len(xb)
            ep_recon += r * len(xb)
            ep_kl    += k * len(xb)

        sched.step()
        n = len(X_scaled)
        losses.append((ep_total / n, ep_recon / n, ep_kl / n))

        if progress_callback is not None:
            progress_callback(ep + 1, ep_total / n)

    return model, losses


# ══════════════════════════════════════════════════════════════
#  BASELINE  —  GMM par classe
# ══════════════════════════════════════════════════════════════

def fit_gmm(X_scaled: np.ndarray, labels: np.ndarray, n_components: int = 4) -> dict:
    """
    Entraîne un GMM indépendant pour chaque classe (RP / RS).

    Returns
    -------
    dict  {0: GaussianMixture (RP), 1: GaussianMixture (RS)}
    """
    gmms = {}
    for cls in [0, 1]:
        X_cls = X_scaled[labels == cls]
        n_comp = min(n_components, len(X_cls) // 5)
        g = GaussianMixture(n_components=n_comp, covariance_type="full",
                            random_state=42, max_iter=300)
        g.fit(X_cls)
        gmms[cls] = g
    return gmms


def generate_gmm(gmms: dict, label: int, n: int) -> np.ndarray:
    """Génère n vecteurs normalisés depuis le GMM de la classe `label`."""
    samples, _ = gmms[label].sample(n)
    return samples


# ══════════════════════════════════════════════════════════════
#  RECONSTRUCTION  —  profils journaliers (48 pas)
# ══════════════════════════════════════════════════════════════

_T = np.linspace(0, 24, N_STEPS, endpoint=False)  # axe temps [0, 24[


def _daily_template(label: int) -> np.ndarray:
    """
    Gabarit de forme du profil journalier (normalisé à somme=1).

    RP (0) : bimodal  — pic matin 7h30 + pic soir 20h  (chauffage électrique)
    RS (1) : unimodal — pic soirée 19h, base plus plate (usage secondaire)
    """
    t = _T
    if label == 0:
        night   = 0.15 * np.ones(N_STEPS)
        morning = 1.00 * np.exp(-0.5 * ((t - 7.5) / 1.2) ** 2)
        evening = 1.45 * np.exp(-0.5 * ((t - 20.0) / 2.0) ** 2)
        profile = night + morning + evening
    else:
        base    = 0.20 * np.ones(N_STEPS)
        evening = 0.80 * np.exp(-0.5 * ((t - 19.0) / 3.0) ** 2)
        profile = base + evening

    return profile / (profile.sum() + 1e-9)


def _daily_template_we(label: int) -> np.ndarray:
    """Gabarits spécifiques pour le weekend."""
    t = _T
    if label == 0:
        night     = 0.12 * np.ones(N_STEPS)
        morning   = 0.80 * np.exp(-0.5 * ((t - 10.0) / 2.0) ** 2)
        afternoon = 0.40 * np.exp(-0.5 * ((t - 15.0) / 2.5) ** 2)
        evening   = 1.20 * np.exp(-0.5 * ((t - 20.5) / 2.0) ** 2)
        profile   = night + morning + afternoon + evening
    else:
        base    = 0.35 * np.ones(N_STEPS)   # RS : plus présent le weekend
        morning = 0.40 * np.exp(-0.5 * ((t - 11.0) / 2.5) ** 2)
        evening = 1.00 * np.exp(-0.5 * ((t - 19.5) / 2.5) ** 2)
        profile = base + morning + evening

    return profile / (profile.sum() + 1e-9)


def reconstruct_profiles(
    features_orig: np.ndarray,
    label: int,
    n_days: int     = 14,
    noise_std: float = 0.12,
    seed: int       = 42,
) -> np.ndarray:
    """
    Reconstruit des courbes de charge journalières (Wh / 30 min)
    à partir de vecteurs de features en espace ORIGINAL (non normalisé).

    Paramètres
    ----------
    features_orig : (n_samples, N_INPUT)  — vecteurs de features denormalisés
    label         : 0 = RP, 1 = RS
    n_days        : nombre de jours consécutifs à générer par client
    noise_std     : niveau de bruit relatif (0.12 → 12 % de σ)

    Returns
    -------
    profils : (n_samples, n_days, N_STEPS)   — Wh par pas 30 min
    """
    rng      = np.random.RandomState(seed)
    tmpl_wd  = _daily_template(label)
    tmpl_we  = _daily_template_we(label)

    all_profiles = []

    for feat in features_orig:
        conso_mean   = max(float(feat[0]), 10.0)   # Wh / 30 min  ≥ 10
        conso_std    = max(float(feat[1]),  1.0)
        ratio_we     = max(float(feat[2]),  0.3)
        ratio_hiver  = max(float(feat[3]),  0.2)
        prop_nulle   = float(np.clip(feat[4], 0.0, 0.99))

        # énergie journalière moyenne (Wh/jour)
        # conso_moyenne est en Wh/30min donc × 48 = Wh/jour
        daily_energy_base = conso_mean * 48.0

        day_profiles = []
        for d in range(n_days):
            is_we = (d % 7) in [5, 6]

            # Simulation des absences
            if rng.rand() < prop_nulle:
                day_profiles.append(np.zeros(N_STEPS))
                continue

            tmpl = tmpl_we if is_we else tmpl_wd
            we_scale = ratio_we if is_we else 1.0

            # Application de la modulation saisonnière et journalière
            season = rng.uniform(0.7 * ratio_hiver, min(ratio_hiver * 1.3, ratio_hiver + 2))
            daily_energy = daily_energy_base * we_scale * rng.uniform(0.8, 1.2)
            profile = tmpl * daily_energy

            # Ajout du bruit gaussien
            noise   = rng.normal(0.0, noise_std * profile.mean(), N_STEPS)
            profile = np.maximum(0.0, profile + noise)

            day_profiles.append(profile)

        all_profiles.append(np.array(day_profiles))   # (n_days, 48)

    return np.array(all_profiles)   # (n_samples, n_days, 48)


# ══════════════════════════════════════════════════════════════
#  DÉTECTION  —  logement habité / non habité
# ══════════════════════════════════════════════════════════════

def compute_occupancy_score(features_orig: np.ndarray) -> pd.DataFrame:
    """
    Calcule un score d'occupation du logement à partir des features.

    Un score élevé → logement occupé en permanence (RP typique).
    Un score faible → logement souvent vide (RS ou logement vacant).

    Indicateurs utilisés :
      • prop_conso_nulle  : proportion de relevés nuls — normalisée relativement
                            au max du dataset (pas en valeur absolue, car même les
                            RS n'ont que ~10-40% de relevés nuls sur l'année entière)
      • conso_std / conso_moyenne : coefficient de variation — irrégularité de conso
      • ratio_we_semaine  : si ratio >> 1 → présence surtout le weekend (RS typique)
      • ratio_hiver_ete   : fort ratio = chauffage actif en hiver = RP avec chauffage élec

    Note sur la normalisation relative :
        Les features sont des agrégats annuels. Une RS absente 2 mois/an = seulement
        ~17% de relevés nuls. Avec une normalisation absolue (÷1), tout le monde
        serait proche de 1. On normalise donc par le max du groupe pour étaler
        la distribution et rendre les seuils 0.4 / 0.7 effectivement atteints.
    """
    df = pd.DataFrame(features_orig, columns=FEATURE_NAMES)

    # ── Indicateur 1 : absences (normalisé par le max du dataset) ──────────
    # Sans cette normalisation relative, prop_conso_nulle max ≈ 0.39
    # → absence_score minimum = 1 - 0.39 = 0.61 → seuil 0.4 jamais atteint
    max_nulle = df["prop_conso_nulle"].max()
    if max_nulle > 0:
        absence_norm = df["prop_conso_nulle"] / max_nulle          # 0 = jamais absent, 1 = le plus absent du dataset
    else:
        absence_norm = df["prop_conso_nulle"]
    absence_score = 1.0 - absence_norm                             # 1 = toujours occupé, 0 = très souvent absent

    # ── Indicateur 2 : régularité de la consommation ────────────────────────
    # Coefficient de variation : CV élevé = consommation très variable = présence irrégulière
    cv = (df["conso_std"] / (df["conso_moyenne"] + 1e-9))
    cv_norm = (cv - cv.min()) / (cv.max() - cv.min() + 1e-9)       # Normalisation min-max
    stability_score = 1.0 - cv_norm                                # 1 = très régulier = occupé en permanence

    # ── Indicateur 3 : équilibre semaine / weekend ──────────────────────────
    # ratio_we ≈ 1 → présence uniforme → RP
    # ratio_we >> 1 → surtout le weekend → RS (résidence de vacances)
    we_deviation = (df["ratio_we_semaine"] - 1.0).abs()
    we_dev_norm  = (we_deviation / (we_deviation.max() + 1e-9))
    we_score     = 1.0 - we_dev_norm                               # 1 = présence uniforme

    # ── Indicateur 4 : ratio hiver/été (spécifique RES2 — chauffage élec) ──
    # Un fort ratio hiver/été sur RES2 = chauffage électrique actif en hiver
    # = logement habité en hiver = plutôt RP
    # Un ratio ≈ 1 ou faible = pas de chauffage hiver = logement peu utilisé en hiver = RS
    hiver_norm   = (df["ratio_hiver_ete"] - df["ratio_hiver_ete"].min()) / \
                   (df["ratio_hiver_ete"].max() - df["ratio_hiver_ete"].min() + 1e-9)
    hiver_score  = hiver_norm.clip(0, 1)                           # 1 = fort usage hiver = RP

    # ── Score final (somme pondérée) ────────────────────────────────────────
    occupancy = (
        0.40 * absence_score     # Absences directes : indicateur le plus fort
      + 0.25 * stability_score   # Régularité de la conso
      + 0.20 * we_score          # Équilibre semaine/weekend
      + 0.15 * hiver_score       # Usage hivernal (spécifique RES2)
    ).clip(0, 1)

    df["score_occupation"] = occupancy.round(3)
    df["statut_occupation"] = occupancy.apply(
        lambda s: "Occupé en permanence" if s > 0.7
        else ("Occupation partielle" if s > 0.4
              else "Probablement vide")
    )
    return df[["score_occupation", "statut_occupation"]]


# ══════════════════════════════════════════════════════════════
#  MÉTRIQUES  —  évaluation de la génération
# ══════════════════════════════════════════════════════════════

def _rbf_kernel(X: np.ndarray, Y: np.ndarray, sigma: float) -> np.ndarray:
    dists = np.sum((X[:, None, :] - Y[None, :, :]) ** 2, axis=2)
    return np.exp(-dists / (2.0 * sigma ** 2))


def compute_mmd(X_real: np.ndarray, X_fake: np.ndarray, sigma: float = 1.0) -> float:
    """Calcule la Maximum Mean Discrepancy (noyau RBF). Une valeur proche de 0 indique des distributions similaires."""
    n, m = len(X_real), len(X_fake)
    Kxx  = _rbf_kernel(X_real, X_real, sigma)
    Kyy  = _rbf_kernel(X_fake, X_fake, sigma)
    Kxy  = _rbf_kernel(X_real, X_fake, sigma)
    return float(Kxx.sum() / (n * n) - 2 * Kxy.sum() / (n * m) + Kyy.sum() / (m * m))


def evaluate_generation(
    X_real: np.ndarray,
    X_gen: np.ndarray,
    scaler: StandardScaler,
    label: int,
) -> dict:
    """
    Compare les distributions réelle et générée.

    Parameters
    ----------
    X_real, X_gen : vecteurs en espace ORIGINAL (non normalisé)
    scaler        : pour calculer le MMD dans l'espace normalisé
    label         : 0=RP, 1=RS (pour l'affichage)

    Returns
    -------
    dict : métriques structurées
    """
    label_name = "RP (Résidence Principale)" if label == 0 else "RS (Résidence Secondaire)"
    results = {"classe": label_name}

    # MMD (espace normalisé)
    Xr_sc = scaler.transform(X_real)
    Xg_sc = scaler.transform(X_gen)
    results["MMD"] = round(compute_mmd(Xr_sc, Xg_sc), 6)

    # Wasserstein + KS par feature
    wd, ks_stat, ks_pval = {}, {}, {}
    for j, name in enumerate(FEATURE_NAMES):
        wd[name]      = round(wasserstein_distance(X_real[:, j], X_gen[:, j]), 3)
        stat, pval    = stats.ks_2samp(X_real[:, j], X_gen[:, j])
        ks_stat[name] = round(float(stat), 3)
        ks_pval[name] = round(float(pval), 3)

    results["wasserstein"] = wd
    results["ks_statistic"] = ks_stat
    results["ks_pvalue"]    = ks_pval

    # Moments statistiques 
    results["real_mean"] = dict(zip(FEATURE_NAMES, X_real.mean(axis=0).round(3)))
    results["gen_mean"]  = dict(zip(FEATURE_NAMES, X_gen.mean(axis=0).round(3)))
    results["real_std"]  = dict(zip(FEATURE_NAMES, X_real.std(axis=0).round(3)))
    results["gen_std"]   = dict(zip(FEATURE_NAMES, X_gen.std(axis=0).round(3)))

    return results


# ══════════════════════════════════════════════════════════════
#  PIPELINE COMPLET  (appelé depuis Streamlit)
# ══════════════════════════════════════════════════════════════

def prepare_data(df_features: pd.DataFrame):
    """
    Sépare les features et les labels, normalise X.

    Returns
    -------
    X_scaled  : np.ndarray  (n_samples, N_INPUT)
    labels    : np.ndarray  (n_samples,)
    X_orig    : np.ndarray  (n_samples, N_INPUT)  — espace original
    scaler    : StandardScaler fitté
    """
    X_orig  = df_features[FEATURE_NAMES].values.astype(np.float32)
    labels  = df_features["label"].values.astype(np.float32)
    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X_orig).astype(np.float32)
    return X_scaled, labels, X_orig, scaler
