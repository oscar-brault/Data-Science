"""
generation_page.py
===================
Page Streamlit « 5. Génération » à intégrer dans app.py.

Comment l'utiliser dans app.py :
    from generation_page import render_generation_page
    ...
    elif menu == "5. Génération":
        render_generation_page(features_base)
"""

import time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import StandardScaler

from generation_module import (
    FEATURE_NAMES,
    N_STEPS,
    CVAE,
    train_cvae,
    fit_gmm,
    generate_gmm,
    reconstruct_profiles,
    compute_occupancy_score,
    evaluate_generation,
    prepare_data,
)

# ─────────────────────────────────────────────────────────────
# Palette de couleurs
# ─────────────────────────────────────────────────────────────
COLOR_RP = "#2196F3"   # Bleu  — Résidence Principale
COLOR_RS = "#FF5722"   # Orange — Résidence Secondaire
COLOR_GEN = "#4CAF50"  # Vert  — Données générées

# Axe temporel pour une journée : 48 pas de 30 minutes (0.0 à 23.5 heures)
LABEL_NAMES = {0: "RP (Résidence Principale)", 1: "RS (Résidence Secondaire)"}
T_AXIS = np.arange(N_STEPS) * 0.5 


# ══════════════════════════════════════════════════════════════
#  HELPERS (Fonctions utilitaires avec mise en cache)
# ══════════════════════════════════════════════════════════════

# Sauvegarde les données en mémoire. Si le dataframe d'entrée ne
# change pas, Streamlit ne refera pas les calculs de préparation.
@st.cache_data
def _prepare(df: pd.DataFrame):
    return prepare_data(df)

# Est utilisé pour les objets non-sérialisables comme les modèles ML.
# Cela évite de ré-entraîner le réseau de neurones à chaque fois que
# l'utilisateur clique sur un bouton !
@st.cache_resource
def _train_cvae_cached(X_scaled, labels_arr, n_epochs, latent_dim, hidden, lr, beta):
    """Entraîne et met en cache le CVAE (évite de ré-entraîner à chaque interaction)."""
    model, losses = train_cvae(
        X_scaled, labels_arr,
        n_epochs=n_epochs,
        latent_dim=latent_dim,
        hidden=hidden,
        lr=lr,
        beta=beta,
    )
    return model, losses


@st.cache_resource
def _fit_gmm_cached(X_scaled, labels_arr, n_comp):
    return fit_gmm(X_scaled, labels_arr, n_components=n_comp)


# ─────────────────────────────────────────────────────────────
# Figure : courbes journalières générées
# ─────────────────────────────────────────────────────────────

def _plot_daily_profiles(profiles_rp, profiles_rs, n_show=5, model_name=""):
    """
    Compare les profils journaliers RP et RS générés.
    `profiles_*` : (n_samples, n_days, 48)
    """
    label_suffix = f" [{model_name}]" if model_name else ""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=True)

    for ax, profiles, color, label in [
        (axes[0], profiles_rp, COLOR_RP, "RP"),
        (axes[1], profiles_rs, COLOR_RS, "RS"),
    ]:
        # Quelques courbes individuelles aléatoires pour montrer la variance
        for i in range(min(n_show, len(profiles))):
            day_idx = np.random.randint(0, profiles.shape[1]) # Choix de jour au hazard
            ax.plot(T_AXIS, profiles[i, day_idx], alpha=0.25, color=color, lw=0.8)

        # Calcul et tracé du profil moyen global
        flat = profiles.reshape(-1, N_STEPS)
        mean = flat.mean(axis=0)
        std  = flat.std(axis=0)
        ax.plot(T_AXIS, mean, color=color, lw=2.0, label="Profil moyen")
        # Remplissage de l'écart-type (+/- 1 sigma)
        ax.fill_between(T_AXIS, mean - std, mean + std, alpha=0.15, color=color, label="±1σ")

        ax.set_title(f"Profils générés — {label}{label_suffix}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Heure de la journée (h)")
        ax.set_ylabel("Consommation (Wh / 30 min)")
        ax.set_xticks(range(0, 25, 3))
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────
# Figure : comparaison distributions features
# ─────────────────────────────────────────────────────────────

def _plot_feature_distributions(X_real_rp, X_real_rs, X_gen_rp, X_gen_rs):
    """
    Affiche la distribution des variables pour vérifier si les données 
    générées se superposent bien aux données réelles.
    """
    fig, axes = plt.subplots(1, len(FEATURE_NAMES), figsize=(18, 3))

    short_names = ["Conso moy.", "Conso σ", "Ratio WE", "Ratio H/E", "Prop. nulle"]
    for j, (ax, name) in enumerate(zip(axes, short_names)):
        # Affichage des densités Réelles (KDE)
        for X_real, color, lbl in [(X_real_rp, COLOR_RP, "RP réel"),
                                    (X_real_rs, COLOR_RS, "RS réel")]:
            vals = X_real[:, j]
            ax.hist(vals, bins=18, density=True, alpha=0.35, color=color, label=lbl)

        # Affichage des densités générées
        for X_gen, color, lbl in [(X_gen_rp, COLOR_RP, "RP gén."),
                                   (X_gen_rs, COLOR_RS, "RS gén.")]:
            vals = X_gen[:, j]
            ax.hist(vals, bins=18, density=True, alpha=0.0, color=color)
            # contour seulement
            counts, bins_ = np.histogram(vals, bins=18, density=True)
            centers = (bins_[:-1] + bins_[1:]) / 2
            ax.plot(centers, counts, color=color, lw=1.8, linestyle="--", label=lbl)

        ax.set_title(name, fontsize=9)
        ax.set_xlabel("")
        ax.grid(True, alpha=0.2)
        if j == 0:
            ax.legend(fontsize=7, loc="upper right")

    fig.suptitle("Distributions réelles (histogrammes) vs générées (pointillés)", fontsize=11)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────
# Figure : heatmap de consommation (type calendrier)
# ─────────────────────────────────────────────────────────────

def _plot_heatmap(profiles, label_name, n_days=14):
    """Affiche une heatmap (jours × heures) pour le premier client généré."""
    data = profiles[0, :n_days, :]   # On sélectionne uniquement le premier client (indice 0)
    fig, ax = plt.subplots(figsize=(14, 4))
    im = ax.imshow(data, aspect="auto", cmap="YlOrRd",
                   extent=[0, 24, n_days, 0])
    ax.set_xlabel("Heure")
    ax.set_ylabel("Jour")
    ax.set_title(f"Heatmap des profils générés — {label_name}", fontweight="bold")
    ax.set_xticks(range(0, 25, 3))
    ax.set_yticks(range(n_days))
    ax.set_yticklabels([f"J{i+1}" for i in range(n_days)], fontsize=8)
    fig.colorbar(im, ax=ax, label="Wh / 30 min")
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────
# Figure : métriques sous forme de tableau coloré
# ─────────────────────────────────────────────────────────────

def _render_metrics_table(eval_rp, eval_rs):
    """Génère un tableau Pandas stylisé pour les tests statistiques (Kolmogorov-Smirnov, etc.)."""
    rows = []
    for ev, cls_name in [(eval_rp, "RP"), (eval_rs, "RS")]:
        for feat in FEATURE_NAMES:
            rows.append({
                "Classe": cls_name,
                "Feature": feat,
                "Wasserstein ↓": ev["wasserstein"][feat],
                "KS statistic ↓": ev["ks_statistic"][feat],
                "KS p-value ↑": ev["ks_pvalue"][feat],
                "Moyenne réelle": ev["real_mean"][feat],
                "Moyenne générée": ev["gen_mean"][feat],
            })
    df = pd.DataFrame(rows)

    def _color_ks(val):
        """Vert si p > 0.05 (distributions semblables), rouge sinon."""
        color = "#c8f7c5" if val > 0.05 else "#f7c5c5"
        return f"background-color: {color}"

    styled = (
        df.style
        .format({
            "Wasserstein ↓": "{:.3f}",
            "KS statistic ↓": "{:.3f}",
            "KS p-value ↑": "{:.3f}",
            "Moyenne réelle": "{:.2f}",
            "Moyenne générée": "{:.2f}",
        })
        .applymap(_color_ks, subset=["KS p-value ↑"])
        .set_table_styles([
            {"selector": "th", "props": [("font-weight", "bold"), ("background-color", "#f0f2f6")]},
        ])
    )
    return styled


# ══════════════════════════════════════════════════════════════
#  PAGE PRINCIPALE
# ══════════════════════════════════════════════════════════════

def render_generation_page(df_features: pd.DataFrame):
    """Point d'entrée appelé depuis app.py."""

    st.title("Génération de Profils de Consommation")
    st.markdown(
        """
        Cette page génère des **courbes de charge synthétiques** conditionnées
        sur le type de client (**RS** ou **RP**) à l'aide d'un modèle d'IA entraîné
        sur les données réelles Enedis.

        | Approche | Description |
        |---|---|
        | **CVAE** | *Conditional Variational AutoEncoder* — génère des vecteurs de features plausibles dans l'espace latent |
        | **GMM**  | *Gaussian Mixture Model* par classe — baseline statistique classique |
        """
    )

    # ── préparation des données ──────────────────────────────
    # Les réseaux de neurones (CVAE) et les GMM nécessitent des données standardisées (moyenne 0, variance 1)
    X_scaled, labels_f, X_orig, scaler = _prepare(df_features)
    labels_int = labels_f.astype(int)

    X_real_rp = X_orig[labels_int == 0]
    X_real_rs = X_orig[labels_int == 1]

    # ── sidebar / contrôles ──────────────────────────────────
    with st.sidebar:
        st.markdown("### Paramètres de génération")
        model_choice = st.radio(
            "Modèle génératif",
            ["CVAE (Variational AutoEncoder)", "GMM (Baseline statistique)"],
            help="Le CVAE apprend une représentation latente. Le GMM modélise directement la distribution des features.",
        )
        n_gen = st.slider("Nombre de profils à générer", 20, 300, 80, step=10)
        n_days = st.slider("Jours par profil", 7, 28, 14)
        noise_level = st.slider("Bruit sur les profils (%)", 5, 30, 12) / 100.0

        st.markdown("---")
        st.markdown("### Hyperparamètres CVAE")
        n_epochs  = st.slider("Époques", 50, 500, 200, step=50)
        latent_d  = st.selectbox("Dim. latente", [4, 8, 16], index=1)
        hidden_d  = st.selectbox("Neurones cachés", [32, 64, 128], index=1)
        lr_val    = st.select_slider("Learning rate", [1e-4, 5e-4, 1e-3, 5e-3], value=1e-3)
        beta_val  = st.select_slider("β (KL weight)", [1e-4, 5e-4, 1e-3, 5e-3], value=5e-4)

    # ── 4 onglets ─────────────────────────────────────────────
    tab_train, tab_gen, tab_eval, tab_occ = st.tabs([
        "Entraînement",
        "Profils générés",
        "Évaluation",
        "Détection d'occupation",
    ])

    # ════════════════════════════════════════════════════════
    #  TAB 1 — Entraînement
    # ════════════════════════════════════════════════════════
    with tab_train:
        st.subheader("Entraînement du modèle génératif")

        col_info_rp, col_info_rs = st.columns(2)
        col_info_rp.metric("Clients RP (label 0)", int((labels_int == 0).sum()))
        col_info_rs.metric("Clients RS (label 1)", int((labels_int == 1).sum()))

        use_cvae_train = "CVAE" in model_choice

        # ── Cas A : Modèle Deep Learning (CVAE) ──────────────────────────────────────
        if use_cvae_train:
            st.info(
                "**Architecture CVAE** : l'encodeur compresse les features en un espace latent "
                f"de dimension {latent_d}. Le décodeur reconstruit les features en étant conditionné "
                "sur le type de client (RS=1 / RP=0)."
            )

            st.markdown(
                """
                ```
                Features (dim {len(FEATURE_NAMES)})  +  label (0 ou 1)
                             │
                        ┌────▼────┐
                        │ Encodeur│  → μ, σ  (dim {latent_d})
                        └────┬────┘
                             │  reparametrisation : z = μ + ε·σ
                        ┌────▼────┐
                        │ Décodeur│  + label conditionnel
                        └────┬────┘
                             │
                      Features reconstruites (dim {len(FEATURE_NAMES)})
                ```
                Perte = **MSE** (reconstruction) + β·**KL divergence**
                """
            )

            if st.button("Lancer l'entraînement CVAE", use_container_width=True, type="primary"):
                with st.spinner("Entraînement du CVAE en cours…"):
                    t0 = time.time()
                    # On lance l'entraînement
                    model, losses = train_cvae(
                        X_scaled, labels_f,
                        n_epochs=n_epochs,
                        latent_dim=latent_d,
                        hidden=hidden_d,
                        lr=lr_val,
                        beta=beta_val,
                    )
                    elapsed = time.time() - t0

                st.session_state["cvae_model"]  = model
                st.session_state["cvae_losses"] = losses
                st.session_state["cvae_scaler"] = scaler
                st.success(f"Entraînement CVAE terminé en {elapsed:.1f}s")

            # Courbe de perte CVAE
            if "cvae_losses" in st.session_state:
                losses_cv = st.session_state["cvae_losses"]
                total  = [l[0] for l in losses_cv]
                recon  = [l[1] for l in losses_cv]
                kl     = [l[2] for l in losses_cv]

                fig_loss, ax_l = plt.subplots(figsize=(10, 3))
                ax_l.plot(total, label="Perte totale", color="#1a237e", lw=1.5)
                ax_l.plot(recon, label="Reconstruction (MSE)", color="#0288d1", lw=1.2, linestyle="--")
                ax_l.plot(kl,    label="KL divergence", color="#e53935", lw=1.2, linestyle=":")
                ax_l.set_xlabel("Époque")
                ax_l.set_ylabel("Perte")
                ax_l.set_title("Courbe d'apprentissage du CVAE")
                ax_l.legend()
                ax_l.grid(True, alpha=0.3)
                st.pyplot(fig_loss)

                col1, col2, col3 = st.columns(3)
                col1.metric("Perte finale totale", f"{total[-1]:.4f}")
                col2.metric("MSE finale", f"{recon[-1]:.4f}")
                col3.metric("KL finale", f"{kl[-1]:.4f}")
            else:
                st.info("Cliquez sur **Lancer l'entraînement CVAE** pour démarrer.")

        # ── Cas B : Modèle Statistique (GMM) ───────────────────────────────────────
        else:
            st.info(
                "**GMM (Gaussian Mixture Model)** : modèle probabiliste classique. "
                "Un GMM indépendant est ajusté sur chaque classe (RP et RS). "
                "**Aucun entraînement manuel n'est nécessaire** — le modèle se calibre "
                "automatiquement à la génération."
            )

            st.markdown(
                """
                ```
                Pour chaque classe c ∈ {RP, RS} :

                  Données réelles (classe c)
                          │
                  ┌───────▼────────┐
                  │ GaussianMixture│  n_components = 4
                  │ covariance     │  "full"
                  └───────┬────────┘
                          │  sample(n)
                  Vecteurs synthétiques (dim 5)
                ```
                Génération = tirage direct dans la distribution apprise.
                """
            )

            with st.spinner("Calibration du GMM en cours…"):
                gmms_preview = _fit_gmm_cached(X_scaled, labels_f, n_comp=4)

            st.success("GMM calibré et prêt à générer — rendez-vous dans l'onglet **Profils générés**.")

            st.markdown("#### Paramètres des mélanges gaussiens")
            col_rp, col_rs = st.columns(2)
            for col, cls, label_name in [(col_rp, 0, "RP"), (col_rs, 1, "RS")]:
                g = gmms_preview[cls]
                col.markdown(f"**{label_name}** — {g.n_components} composantes")
                weights_df = pd.DataFrame({
                    "Composante": range(g.n_components),
                    "Poids": g.weights_.round(3),
                })
                col.dataframe(weights_df, use_container_width=True, hide_index=True)

    # ════════════════════════════════════════════════════════
    #  TAB 2 — Profils générés
    # ════════════════════════════════════════════════════════
    with tab_gen:
        use_cvae = "CVAE" in model_choice
        model_name = "CVAE" if use_cvae else "GMM"

        st.subheader(f"Visualisation des profils générés — {model_name}")

        if use_cvae and "cvae_model" not in st.session_state:
            st.warning("Veuillez d'abord entraîner le CVAE dans l'onglet **Entraînement**.")
            st.stop() # Bloque l'exécution de l'onglet si le modèle n'existe pas

        # ── génération des features ──────────────────────────
        with st.spinner(f"Génération via {model_name} en cours…"):
            # Génération des données dans l'espace "Standardisé" (valeurs entre -3 et 3)
            if use_cvae:
                model  = st.session_state["cvae_model"]
                scaler_gen = st.session_state["cvae_scaler"]
                gen_rp_sc  = model.generate(n_gen, label=0)
                gen_rs_sc  = model.generate(n_gen, label=1)
                # Retour aux unités réelles
                gen_rp_orig = scaler_gen.inverse_transform(gen_rp_sc)
                gen_rs_orig = scaler_gen.inverse_transform(gen_rs_sc)
            else:
                # GMM
                gmms       = _fit_gmm_cached(X_scaled, labels_f, n_comp=4)
                gen_rp_sc  = generate_gmm(gmms, label=0, n=n_gen)
                gen_rs_sc  = generate_gmm(gmms, label=1, n=n_gen)
                gen_rp_orig = scaler.inverse_transform(gen_rp_sc)
                gen_rs_orig = scaler.inverse_transform(gen_rs_sc)

            # Nettoyage physique (pas de consommation négative)
            gen_rp_orig = np.clip(gen_rp_orig, 0, None)
            gen_rs_orig = np.clip(gen_rs_orig, 0, None)

            # ── reconstruction des profils journaliers ──────
            profiles_rp = reconstruct_profiles(gen_rp_orig, label=0, n_days=n_days, noise_std=noise_level)
            profiles_rs = reconstruct_profiles(gen_rs_orig, label=1, n_days=n_days, noise_std=noise_level)

        # Sauvegarder les résultats ET le nom du modèle utilisé
        st.session_state["gen_rp_orig"]   = gen_rp_orig
        st.session_state["gen_rs_orig"]   = gen_rs_orig
        st.session_state["profiles_rp"]   = profiles_rp
        st.session_state["profiles_rs"]   = profiles_rs
        st.session_state["gen_scaler"]    = scaler
        st.session_state["gen_model_name"] = model_name

        # ── KPIs ────────────────────────────────────────────
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Modèle utilisé", model_name)
        col2.metric("Profils générés (RP + RS)", n_gen * 2)
        col3.metric("Conso moy. RP (Wh/30min)", f"{gen_rp_orig[:, 0].mean():.0f}")
        col4.metric("Conso moy. RS (Wh/30min)", f"{gen_rs_orig[:, 0].mean():.0f}")

        # ── courbes journalières ─────────────────────────────
        st.markdown(f"#### Courbes de charge journalières — {model_name} (48 pas × 30 min)")
        fig_daily = _plot_daily_profiles(profiles_rp, profiles_rs, model_name=model_name)
        st.pyplot(fig_daily)

        # ── heatmaps ─────────────────────────────────────────
        st.markdown("#### Heatmaps de consommation (jours × heures)")
        col_h1, col_h2 = st.columns(2)
        with col_h1:
            fig_hm_rp = _plot_heatmap(profiles_rp, f"RP — Résidence Principale [{model_name}]", n_days=n_days)
            st.pyplot(fig_hm_rp)
        with col_h2:
            fig_hm_rs = _plot_heatmap(profiles_rs, f"RS — Résidence Secondaire [{model_name}]", n_days=n_days)
            st.pyplot(fig_hm_rs)

        # ── features générées ────────────────────────────────
        st.markdown("#### Vecteurs de features générés (aperçu)")
        df_gen = pd.DataFrame(
            np.vstack([
                np.hstack([gen_rp_orig[:5], np.zeros((5, 1))]),
                np.hstack([gen_rs_orig[:5], np.ones((5, 1))]),
            ]),
            columns=FEATURE_NAMES + ["label"]
        )
        df_gen["label"] = df_gen["label"].map({0.0: "RP", 1.0: "RS"})
        st.dataframe(df_gen.style.format({c: "{:.3f}" for c in FEATURE_NAMES}))

    # ════════════════════════════════════════════════════════
    #  TAB 3 — Évaluation
    # ════════════════════════════════════════════════════════
    with tab_eval:
        st.subheader("Évaluation de la qualité de génération")

        if "gen_rp_orig" not in st.session_state:
            st.warning("Générez d'abord des profils dans l'onglet **Profils générés**.")
            st.stop()

        gen_rp = st.session_state["gen_rp_orig"]
        gen_rs = st.session_state["gen_rs_orig"]
        sc_ev  = st.session_state["gen_scaler"]
        eval_model_name = st.session_state.get("gen_model_name", "?")

        st.caption(f"Résultats calculés pour le modèle : **{eval_model_name}**")
        
        # Calcul des métriques statistiques complexes (MMD, KS test) pour prouver
        # que l'IA a bien reproduit la réalité et ne fait pas n'importe quoi.
        eval_rp = evaluate_generation(X_real_rp, gen_rp, sc_ev, label=0)
        eval_rs = evaluate_generation(X_real_rs, gen_rs, sc_ev, label=1)

        # ── KPIs MMD ─────────────────────────────────────────
        st.markdown("#### Maximum Mean Discrepancy (MMD)")
        st.caption(
            "Le MMD mesure la distance entre deux distributions dans l'espace des features normalisées. "
            "**Valeur proche de 0 → distribution générée fidèle à la réalité.**"
        )
        col_mmd1, col_mmd2 = st.columns(2)
        col_mmd1.metric(
            "MMD — RP",
            f"{eval_rp['MMD']:.5f}",
            delta=f"{'Bon' if eval_rp['MMD'] < 0.05 else 'À améliorer'}",
            delta_color="off",
        )
        col_mmd2.metric(
            "MMD — RS",
            f"{eval_rs['MMD']:.5f}",
            delta=f"{'Bon' if eval_rs['MMD'] < 0.05 else 'À améliorer'}",
            delta_color="off",
        )

        # ── Tableau métriques détaillées ─────────────────────
        st.markdown("#### Métriques détaillées par feature")
        st.caption(
            "**KS p-value** : si p > 0.05, on ne peut pas rejeter H₀ "
            "(distributions similaires). Si p < 0.05, les distributions diffèrent significativement."
        )
        styled_table = _render_metrics_table(eval_rp, eval_rs)
        st.dataframe(styled_table, use_container_width=True)

        # ── Comparaison des distributions ────────────────────
        st.markdown("#### Distributions réelles vs générées")
        fig_dist = _plot_feature_distributions(X_real_rp, X_real_rs, gen_rp, gen_rs)
        st.pyplot(fig_dist)

        # ── Cohérence des profils ─────────────────────────────
        st.markdown("#### Cohérence physique des profils")
        profiles_rp = st.session_state["profiles_rp"]
        profiles_rs = st.session_state["profiles_rs"]

        flat_rp = profiles_rp.reshape(-1, N_STEPS)
        flat_rs = profiles_rs.reshape(-1, N_STEPS)

        stats_rows = []
        for profiles, cls in [(flat_rp, "RP"), (flat_rs, "RS")]:
            daily_totals = profiles.sum(axis=1) / 2.0   # Wh → kWh  (÷ 2 pour 30min)
            stats_rows.append({
                "Classe": cls,
                "Énergie journalière moy. (kWh)": round(daily_totals.mean(), 2),
                "Min (kWh)": round(daily_totals.min(), 2),
                "Max (kWh)": round(daily_totals.max(), 2),
                "Std (kWh)": round(daily_totals.std(), 2),
                "% jours à conso nulle": round((daily_totals == 0).mean() * 100, 1),
            })

        st.dataframe(pd.DataFrame(stats_rows), use_container_width=True)

        st.markdown(
            """
            **Interprétation des métriques :**
            - **MMD < 0.05** : distributions générées très proches des réelles
            - **KS p-value > 0.05** : les features générées ne diffèrent pas significativement des réelles
            - **Énergie journalière** : cohérente avec une résidence française (3-15 kWh/jour en hiver)
            """
        )

    # ════════════════════════════════════════════════════════
    #  TAB 4 — Détection d'occupation
    # ════════════════════════════════════════════════════════
    with tab_occ:
        st.subheader("Détection de l'occupation du logement")
        st.markdown(
            """
            Au-delà de la classification RS/RP, on peut estimer si un logement
            est **occupé en permanence**, **partiellement occupé** ou **probablement vide**,
            à partir de trois indicateurs :

            | Indicateur | Poids | Interprétation |
            |---|---|---|
            | `prop_conso_nulle` | 50 % | Proportion de relevés à 0 → absence directe |
            | Coefficient de variation (σ/µ) | 30 % | Profil irrégulier → présence intermittente |
            | `ratio_we_semaine` | 20 % | Ratio ≫ 1 → présent surtout le weekend (RS) |
            """
        )

        # Le calcul du score d'occupation
        occ_df = compute_occupancy_score(X_orig)
        occ_df["label"] = labels_int
        occ_df["type"]  = occ_df["label"].map({0: "RP", 1: "RS"})

        # Affichage de la distribution des scores
        col1, col2, col3 = st.columns(3)
        for cls, col in [(0, col1), (1, col2)]:
            sub = occ_df[occ_df["label"] == cls]
            mean_sc = sub["score_occupation"].mean()
            col.metric(
                f"Score moyen — {'RP' if cls==0 else 'RS'}",
                f"{mean_sc:.2f} / 1.00",
            )
        col3.metric(
            "% logements probablement vides",
            f"{(occ_df['statut_occupation']=='Probablement vide').mean()*100:.1f} %"
        )

        # Histogramme des scores
        fig_occ, ax_occ = plt.subplots(figsize=(10, 4))
        for cls, color, label in [(0, COLOR_RP, "RP"), (1, COLOR_RS, "RS")]:
            sub = occ_df[occ_df["label"] == cls]["score_occupation"]
            ax_occ.hist(sub, bins=20, alpha=0.55, color=color, label=f"{label} (n={len(sub)})",
                        density=True, edgecolor="white", lw=0.5)
        
        # Ajout des seuils métiers sur le graphique
        ax_occ.axvline(0.4, color="orange", lw=1.5, linestyle="--", label="Seuil : occupation partielle")
        ax_occ.axvline(0.7, color="green",  lw=1.5, linestyle="--", label="Seuil : occupé en permanence")
        ax_occ.set_xlabel("Score d'occupation")
        ax_occ.set_ylabel("Densité")
        ax_occ.set_title("Distribution du score d'occupation par type de résidence")
        ax_occ.legend()
        ax_occ.grid(True, alpha=0.3)
        st.pyplot(fig_occ)

        # Tableau de distribution des statuts
        st.markdown("#### Répartition des statuts d'occupation")
        pivot = (
            occ_df.groupby(["type", "statut_occupation"])
            .size()
            .unstack(fill_value=0)
        )
        st.dataframe(pivot)

        # Simulation interactive : Voir les stats d'un client spécifique
        st.markdown("---")
        st.markdown("#### Explorer un client spécifique")
        client_idx = st.slider(
            "Index du client (données réelles)", 0, len(X_orig) - 1, 0
        )
        row_feat  = X_orig[client_idx]
        row_occ   = occ_df.iloc[client_idx]
        row_label = labels_int[client_idx]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Type réel", "RP" if row_label == 0 else "RS")
        c2.metric("Score d'occupation", f"{row_occ['score_occupation']:.2f}")
        c3.metric("Statut", row_occ["statut_occupation"])
        c4.metric("Prop. conso nulle", f"{row_feat[4]*100:.1f} %")

        # On reconstruit visuellement le profil de ce client précis pour comprendre son score
        prof_client = reconstruct_profiles(
            row_feat[None, :], label=row_label, n_days=7, seed=client_idx
        )
        fig_cli, ax_cli = plt.subplots(figsize=(10, 3))
        color_cli = COLOR_RP if row_label == 0 else COLOR_RS
        for d in range(7):
            ax_cli.plot(T_AXIS, prof_client[0, d], alpha=0.4, color=color_cli, lw=1)
        ax_cli.plot(T_AXIS, prof_client[0].mean(axis=0), color=color_cli, lw=2.5,
                    label="Profil moyen (7 jours)")
        ax_cli.set_xlabel("Heure")
        ax_cli.set_ylabel("Conso (Wh / 30 min)")
        ax_cli.set_title(
            f"Profil reconstruit — Client #{client_idx} "
            f"({'RP' if row_label==0 else 'RS'}) — {row_occ['statut_occupation']}"
        )
        ax_cli.legend()
        ax_cli.grid(True, alpha=0.3)
        st.pyplot(fig_cli)
