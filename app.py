"""
app.py — Dashboard principal de détection des Résidences Secondaires / Principales
======================================================================================
Ce fichier constitue le point d'entrée de l'application Streamlit. Il orchestre
l'ensemble du pipeline Data Science :
    1. Exploration des données & réduction de dimension (ACP)
    2. Clustering non-supervisé (K-Means) pour générer les labels
    3. Classification supervisée (Random Forest, Régression Logistique, SVM, MLP)
    4. Prévision de la consommation (séries temporelles)
    5. Génération de profils synthétiques (module externe)

Les données sources proviennent des relevés de consommation électrique Enedis.
"""

# =============================================================================
# IMPORTS
# =============================================================================

import streamlit as st          
import pandas as pd            
import numpy as np              
import matplotlib.pyplot as plt 
import seaborn as sns           

# --- Prétraitement & réduction de dimension ---
from sklearn.preprocessing import StandardScaler  # Normalisation des features (moyenne=0, écart-type=1)
from sklearn.decomposition import PCA             # Analyse en Composantes Principales

# --- Clustering ---
from sklearn.cluster import KMeans 

# --- Métriques & évaluation ---
from sklearn.metrics import classification_report, confusion_matrix  
from sklearn.model_selection import train_test_split                 

# --- Modèles de classification ---
from sklearn.ensemble import RandomForestClassifier   
from sklearn.linear_model import LogisticRegression   
from sklearn.svm import SVC                          
from sklearn.neural_network import MLPClassifier

# --- Modèles de régression (prévision temporelle) ---
from sklearn.neural_network import MLPRegressor       
from sklearn.ensemble import RandomForestRegressor 
from sklearn.metrics import mean_absolute_error 


# =============================================================================
# 1. CONFIGURATION DE LA PAGE STREAMLIT
# =============================================================================

# Définit le titre de l'onglet navigateur
st.set_page_config(page_title="Dashboard Enedis", layout="wide")

# Applique un thème visuel global
sns.set_theme(style="whitegrid")


# =============================================================================
# 2. FONCTIONS DE CHARGEMENT DES DONNÉES
# =============================================================================

@st.cache_data
def load_features():
    """
    Charge les features pré-calculées depuis le fichier CSV de features engineering.

    Le décorateur @st.cache_data mémorise le résultat en mémoire : la fonction
    n'est exécutée qu'une seule fois, puis Streamlit retourne directement le
    DataFrame mis en cache lors des interactions suivantes (gain de performance).

    Les colonnes 'label' et 'cluster' sont supprimées si elles existent,
    afin de conserver uniquement les variables explicatives (features brutes),
    sans biais lié aux sorties précédentes.

    Retourne :
        pd.DataFrame : DataFrame indexé par 'id', contenant uniquement les features.
    """
    df = pd.read_csv('features_pour_classification.csv', index_col='id')
    if 'label' in df.columns:   df = df.drop(columns=['label'])    # Supprime le label si présent
    if 'cluster' in df.columns: df = df.drop(columns=['cluster'])  # Supprime le cluster si présent
    return df


@st.cache_data
def load_features_with_labels():
    """
    Charge les features avec les colonnes 'label' et 'cluster' conservées.

    Cette version complète est réservée à la page '5. Génération', qui a besoin
    des étiquettes pour conditionner la génération de profils synthétiques.

    Retourne :
        pd.DataFrame : DataFrame indexé par 'id', avec toutes les colonnes du CSV.
    """
    return pd.read_csv('features_pour_classification.csv', index_col='id')


@st.cache_data
def charger_donnees_brutes(chemin_fichier):
    """
    Charge et prépare les relevés de consommation bruts (page Prévision).

    La colonne 'horodate' (timestamps) est convertie en objet datetime avec
    gestion du fuseau horaire UTC. Cela permet d'utiliser les méthodes de
    ré-échantillonnage temporel de Pandas (ex : resample('D') pour agréger
    par journée).

    Paramètres :
        chemin_fichier : str ou fichier uploadé via st.file_uploader

    Retourne :
        pd.DataFrame : DataFrame avec la colonne 'horodate' au format datetime UTC.
    """
    df = pd.read_csv(chemin_fichier)
    df['horodate'] = pd.to_datetime(df['horodate'], utc=True)  # Conversion en datetime avec timezone UTC
    return df


# Chargement initial des données au démarrage de l'application.
features_base        = load_features()             # Features seules (sans labels)
features_avec_labels = load_features_with_labels() # Features + labels (pour la page Génération)


# =============================================================================
# 3. BARRE LATÉRALE — NAVIGATION
# =============================================================================

# Titre affiché en haut de la barre de navigation latérale
st.sidebar.title("Pipeline Data Science")

# Menu de navigation principal : chaque option correspond à une étape du pipeline.
menu = st.sidebar.radio(
    "Étapes de l'analyse :",
    [
        "1. Exploration & ACP",
        "2. Clustering (K-Means)",
        "3. Classification (Modèles)",
        "4. Prévision",
        "5. Génération",
    ]
)

# Séparateur visuel horizontal dans la sidebar
st.sidebar.markdown("---")

# Message d'information
st.sidebar.info(
    "Dashboard interactif pour la détection des Résidences Secondaires (RS) "
    "et Principales (RP) à partir de données Enedis."
)

# =============================================================================
# PAGE 1 : EXPLORATION ET ACP (Analyse en Composantes Principales)
# =============================================================================
if menu == "1. Exploration & ACP":
    st.title("Exploration et Réduction de dimension")

    # Aperçu des 5 premières lignes du DataFrame de features
    st.write("Aperçu des caractéristiques extraites (Feature Engineering) :")
    st.dataframe(features_base.head())

    st.subheader("Analyse en Composantes Principales (ACP)")
    st.write("L'ACP permet de réduire nos variables en 2 dimensions pour pouvoir les visualiser sur un graphique.")

    # --- Étape 1 : Normalisation des données ---
    # StandardScaler centre les données (moyenne = 0) et les réduit (écart-type = 1).
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_base)

    # --- Étape 2 : Calcul de l'ACP à 2 composantes ---
    # PCA(n_components=2) projette les données dans un espace de dimension 2
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(features_scaled)  # Tableau (n_clients, 2)

    # --- Étape 3 : Visualisation du nuage de points ACP ---
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.5, c='#3498db')

    # Les labels des axes indiquent le pourcentage de variance capturé par chaque composante.
    # Plus ce pourcentage est élevé, plus la composante résume bien l'information d'origine.
    ax.set_xlabel(f"Composante 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"Composante 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title("Projection des profils clients (ACP 2D)")
    st.pyplot(fig)


# =============================================================================
# PAGE 2 : CLUSTERING INTERACTIF (K-Means)
# =============================================================================
elif menu == "2. Clustering (K-Means)":
    st.title("Clustering et Création des Labels")

    # --- Section : Méthode du Coude ---
    st.subheader("Méthode du Coude (Choix du k)")
    st.write("Le graphique ci-dessous calcule l'inertie pour différents nombres de clusters.")

    @st.cache_data
    def calculer_inerties(data):
        """
        Calcule l'inertie intra-cluster pour k allant de 1 à 14.

        L'inertie mesure la dispersion des points autour du centre de leur cluster.
        La 'méthode du coude' consiste à choisir le k à partir duquel l'inertie
        cesse de décroître fortement (le 'coude' de la courbe).

        Paramètres :
            data : np.ndarray — données normalisées

        Retourne :
            K_vals : range — valeurs de k testées (1 à 14)
            inerts : list  — inertie correspondante pour chaque k
        """
        inerts = []
        K_vals = range(1, 15)
        for k in K_vals:
            # n_init='auto' laisse scikit-learn choisir automatiquement le nombre d'initialisations aléatoires des centroïdes
            km = KMeans(n_clusters=k, random_state=42, n_init='auto')
            km.fit(data)
            inerts.append(km.inertia_)  # Somme des distances au carré au centroïde le plus proche
        return K_vals, inerts

    # Normalisation des données pour la méthode du coude
    scaler_coude      = StandardScaler()
    data_scaled_coude = scaler_coude.fit_transform(features_base)
    K_vals, inerts    = calculer_inerties(data_scaled_coude)

    # Tracé de la courbe inertie f(k) on cherche visuellement le "coude"
    fig_coude, ax_coude = plt.subplots(figsize=(8, 3))
    ax_coude.plot(K_vals, inerts, marker='o', linestyle='--', color='#9b59b6')
    ax_coude.set_title("Inertie en fonction de k")
    ax_coude.set_xlabel("Nombre de clusters (k)")
    ax_coude.set_ylabel("Inertie")
    st.pyplot(fig_coude)

    st.markdown("---")

    # Disposition en deux colonnes : paramètres à gauche, résultats à droite
    col1, col2 = st.columns([1, 2])

    with col1:
        #Paramètres interactifs du clustering
        st.subheader("Paramètres de l'algorithme")

        #Choisir le nombre de clusters k
        k_clusters = st.slider("Nombre de clusters (k)", min_value=2, max_value=15, value=10)

        # Seuil décisionnel : un cluster est classé RS (Résidence Secondaire) si sa proportion moyenne de consommation nulle dépasse ce seuil.
        seuil_rs   = st.slider("Seuil proportion conso nulle", min_value=0.0, max_value=0.20,
                                value=0.03, step=0.01)
        st.info("Les clusters ayant une proportion de consommation nulle supérieure à ce seuil "
                "seront classés en Résidences Secondaires (1).")

    with col2:
        #Application du K-Means avec les paramètres choisis

        # Re-normalisation des features avant le clustering
        scaler          = StandardScaler()
        features_scaled = scaler.fit_transform(features_base)

        # Entraînement du K-Means : chaque client est assigné à un cluster
        kmeans   = KMeans(n_clusters=k_clusters, random_state=42, n_init='auto')
        clusters = kmeans.fit_predict(features_scaled)  # Tableau d'entiers (0 à k-1) par client

        # Ajout des clusters au DataFrame de features
        features_clustered            = features_base.copy()
        features_clustered['cluster'] = clusters

        # Calcul du profil moyen de chaque cluster (moyenne de chaque feature par groupe)
        cluster_profiles = features_clustered.groupby('cluster').mean()

        # Identification des clusters RS : ceux dont la proportion de conso nulle > seuil
        clusters_rs = cluster_profiles[cluster_profiles['prop_conso_nulle'] > seuil_rs].index.tolist()

        # Création du label binaire : 1 = RS (Résidence Secondaire), 0 = RP (Résidence Principale)
        features_clustered['label'] = features_clustered['cluster'].apply(
            lambda c: 1 if c in clusters_rs else 0
        )

        # Sauvegarde du DataFrame labélisé dans le session_state pour le partager entre les pages sans recalculer
        st.session_state['data_labeled'] = features_clustered

        #Affichage du bilan de classification
        st.subheader(f"Résultat : {len(clusters_rs)} cluster(s) identifié(s) comme RS")
        nb_rs = features_clustered['label'].sum()  # Nombre total de RS (label = 1)
        nb_rp = len(features_clustered) - nb_rs    # Nombre total de RP (label = 0)
        st.write(f"**Résidences Principales (0) :** {nb_rp} clients")
        st.write(f"**Résidences Secondaires (1) :** {nb_rs} clients")

        # --- Graphique : proportion de conso nulle par cluster ---
        # Les barres rouges dépassent le seuil = RS
        # Les barres vertes sont en dessous du seuil = RP
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.barplot(
            x=cluster_profiles.index,
            y='prop_conso_nulle',
            data=cluster_profiles,
            palette=['#e74c3c' if c in clusters_rs else '#2ecc71' for c in cluster_profiles.index],
            ax=ax,
        )
        ax.axhline(seuil_rs, color='black', linestyle='--', label='Seuil (RS)')  # Ligne du seuil
        ax.set_title("Proportion moyenne de conso nulle par cluster")
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Prop. conso nulle")
        ax.legend()
        st.pyplot(fig)


# =============================================================================
# PAGE 3 : CLASSIFICATION SUPERVISÉE
# =============================================================================
elif menu == "3. Classification (Modèles)":
    st.title("Entraînement des modèles de Classification")

    # Vérification : les labels doivent avoir été générés depuis la page Clustering
    if 'data_labeled' not in st.session_state:
        st.warning("Veuillez d'abord aller sur la page '2. Clustering (K-Means)' pour générer les labels !")
    else:
        # Récupération du DataFrame labélisé depuis la mémoire de session
        df_labeled = st.session_state['data_labeled']

        # Sélection de l'algorithme de classification par l'utilisateur
        modele_choisi = st.selectbox(
            "Choisissez l'algorithme à entraîner :",
            ["Random Forest", "Régression Logistique", "SVM", "MLPClassifier (Réseau de Neurones)"]
        )

        # Séparation features (X) / cible (y)
        X = df_labeled.drop(columns=['label', 'cluster'])
        y = df_labeled['label']

        # On vérifie qu'il y a au moins 2 classes 
        if len(y.unique()) < 2:
            st.error("Erreur : Vos paramètres de clustering n'ont généré qu'une seule classe. "
                     "Ajustez le seuil dans la page Clustering.")
        else:
            #Découpage Train / Test (70% / 30%)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, stratify=y, random_state=42
            )

            #Normalisation
            scaler_clf     = StandardScaler()
            X_train_scaled = scaler_clf.fit_transform(X_train)
            X_test_scaled  = scaler_clf.transform(X_test)

            #modèle sélectionné
            if modele_choisi == "Random Forest":
                clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

            elif modele_choisi == "Régression Logistique":
                clf = LogisticRegression(random_state=42, class_weight='balanced')

            elif modele_choisi == "SVM":
                clf = SVC(kernel='rbf', class_weight='balanced', random_state=42)

            else:
                clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)

            #Entraînement et prédiction
            with st.spinner("Entraînement du modèle en cours..."):
                clf.fit(X_train_scaled, y_train)      
                y_pred = clf.predict(X_test_scaled)

            #Affichage des résultats en deux colonnes
            col1, col2 = st.columns(2)

            with col1:
                # Matrice de confusion : montre les vrais positifs, vrais négatifs, faux positifs et faux négatifs sous forme de heatmap
                st.subheader("Matrice de Confusion")
                cm  = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
                ax.set_xlabel("Prédiction")
                ax.set_ylabel("Vrai Label")
                st.pyplot(fig)

            with col2:
                # Rapport de classification : précision, rappel, F1-score par classe
                st.subheader("Rapport de Classification")
                report    = classification_report(y_test, y_pred, output_dict=True)
                df_report = pd.DataFrame(report).transpose()
                st.dataframe(df_report.style.format("{:.2f}"))

            #Importance des variables (selon le modèle)
            if modele_choisi == "Random Forest":
                # feature_importances_ : mesure la réduction moyenne d'impureté apportée par chaque variable à travers tous les arbres de la forêt
                st.subheader("Importance des variables (Random Forest)")
                importances = clf.feature_importances_
                df_imp = pd.DataFrame({'Variable': X.columns, 'Importance': importances}).sort_values(
                    by='Importance', ascending=False
                )
                fig, ax = plt.subplots(figsize=(10, 4))
                sns.barplot(x='Importance', y='Variable', data=df_imp, palette='viridis', ax=ax)
                st.pyplot(fig)

            elif modele_choisi == "Régression Logistique":
                # Les coefficients  indiquent l'influence de chaque featuresur la décision. On prend la valeur absolue car le signe indique le sens (positif = favorise RS, négatif = favorise RP), mais pas l'importance.
                st.subheader("Importance des variables (Régression Logistique)")
                importances = np.abs(clf.coef_[0])
                df_imp = pd.DataFrame({'Variable': X.columns, 'Importance': importances}).sort_values(
                    by='Importance', ascending=False
                )
                fig, ax = plt.subplots(figsize=(10, 4))
                sns.barplot(x='Importance', y='Variable', data=df_imp, palette='magma', ax=ax)
                st.pyplot(fig)

            elif modele_choisi in ["SVM", "MLPClassifier (Réseau de Neurones)"]:
                # SVM et MLP sont des modèles "boîte noire": leurs paramètres internes ne s'interprètent pas directement comme des importances de variables
                st.info(f"Le modèle {modele_choisi} est considéré comme une 'boîte noire' "
                        "mathématique. Il n'est pas possible d'afficher simplement l'importance "
                        "individuelle de chaque variable.")


# =============================================================================
# PAGE 4 : PRÉVISION DE LA CONSOMMATION (SÉRIES TEMPORELLES)
# =============================================================================
elif menu == "4. Prévision":
    st.title("Prévision de la consommation (Séries Temporelles)")
    st.write("Prédisez la consommation future d'un client spécifique grâce au Machine Learning.")

    # Widget d'upload de fichier CSV
    fichier_brut = st.file_uploader("Chargez le fichier brut des relevés (CSV)", type=["csv"])

    if fichier_brut is not None:
        with st.spinner('Chargement et préparation des données en cours...'):
            df_brut = charger_donnees_brutes(fichier_brut)

        st.success("Données chargées avec succès !")

        #Sélection des paramètres de prédiction
        col1, col2, col3 = st.columns(3)
        with col1:
            liste_clients = df_brut['id'].unique()
            client_choisi = st.selectbox("Choisissez l'ID du client :", liste_clients)
        with col2:
            jours_a_predire = st.slider("Jours à prédire :", min_value=1, max_value=30, value=14)
        with col3:
            modele_choisi = st.radio("Algorithme :", ["Random Forest", "Réseau de Neurones"])

        #Lancement de la prédiction au clic du bouton
        if st.button("Lancer la prédiction", use_container_width=True):
            with st.spinner(f"Entraînement du {modele_choisi} en cours..."):

                # Filtrage des données pour le client sélectionné
                df_client = df_brut[df_brut['id'] == client_choisi].copy()
                df_client.set_index('horodate', inplace=True)

                # Ré-échantillonnage journalier : on agrège les relevés infra-journaliers en une somme quotidienne de consommation (en kWh/jour)
                df_jour = df_client[['valeur']].resample('D').sum()

                #Feature Engineering pour la série temporelle
                # On crée des variables décalées (lags) qui capturent la corrélation entre la consommation d'un jour et celles des jours précédents.
                df_features_ts = df_jour.copy()
                df_features_ts['J-1'] = df_features_ts['valeur'].shift(1)  # Conso de la veille
                df_features_ts['J-2'] = df_features_ts['valeur'].shift(2)  # Conso d'il y a 2 jours
                df_features_ts['J-7'] = df_features_ts['valeur'].shift(7)  # Conso d'il y a 7 jours (effet semaine)
                df_features_ts['jour_semaine'] = df_features_ts.index.dayofweek  # 0=lundi … 6=dimanche

                # Suppression des lignes avec des valeurs manquantes (dues aux décalages)
                df_features_ts.dropna(inplace=True)

                #Découpage temporel Train / Test
                # On réserve les `jours_a_predire` derniers jours comme jeu de test,
                split_index = len(df_features_ts) - jours_a_predire
                X_ts = df_features_ts.drop(columns=['valeur'])
                y_ts = df_features_ts['valeur']

                X_train, y_train = X_ts.iloc[:split_index], y_ts.iloc[:split_index]
                X_test,  y_test  = X_ts.iloc[split_index:], y_ts.iloc[split_index:]

                scaler_ts = StandardScaler()  # Sera utilisé uniquement pour le MLP

                #Entraînement du modèle de régression
                if modele_choisi == "Random Forest":
                    # RandomForestRegresson est insensible aux échelles donc pas de normalisation
                    modele = RandomForestRegressor(n_estimators=100, random_state=42)
                    modele.fit(X_train, y_train)
                else:
                    # Le MLP est sensible aux échelles donc normalisation obligatoire
                    X_train_sc = scaler_ts.fit_transform(X_train)
                    modele = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=2000, random_state=42)
                    modele.fit(X_train_sc, y_train)

                #Prédiction itérative (forecasting récursif)
                # On prédit jour après jour en utilisant les prédictions précédentes comme nouvelles valeurs de lag c'est la stratégie "récursive".
                dernier_jour_connu  = df_features_ts.iloc[split_index - 1].copy()
                predictions_futures = []
                historique_recent   = list(y_train.iloc[-7:].values)

                for i in range(jours_a_predire):
                    # Construction du vecteur de features pour le jour i+1 à prédire
                    feat_jour = pd.DataFrame([{
                        'J-1': historique_recent[-1],  # Dernier jour connu (ou prédit)
                        'J-2': historique_recent[-2],  # Avant-dernier
                        'J-7': historique_recent[-7],  # Il y a 7 jours
                        'jour_semaine': (dernier_jour_connu.name + pd.Timedelta(days=i+1)).dayofweek
                    }])
                    # Normalisation si MLP
                    if modele_choisi == "Réseau de Neurones":
                        feat_jour = scaler_ts.transform(feat_jour)

                    pred = modele.predict(feat_jour)[0]  # Prédiction scalaire
                    predictions_futures.append(pred)

                    # Mise à jour de la fenêtre : on ajoute la prédiction
                    # et on supprime la valeur la plus ancienne (FIFO)
                    historique_recent.append(pred)
                    historique_recent.pop(0)

            #Affichage des résultats
            st.subheader(f"Résultats pour le client {client_choisi}")

            # MAE (Mean Absolute Error) : erreur moyenne en valeur absolue entre les vraies consommations et les prédictions (en kWh/jour)
            mae = mean_absolute_error(y_test, predictions_futures)
            st.metric(label="Erreur Absolue Moyenne (MAE)", value=f"{mae:.2f} kWh/jour")

            # Graphique comparant l'historique, les vraies valeurs et les prédictions
            fig, ax = plt.subplots(figsize=(12, 5))
            couleur_pred = 'red' if modele_choisi == "Random Forest" else 'green'

            # Historique des 30 derniers jours d'entraînement (contexte visuel)
            ax.plot(y_train.index[-30:], y_train.values[-30:],
                    label="Historique d'entraînement (30 derniers j)", color='gray')
            # Vraie consommation sur la période de test
            ax.plot(y_test.index, y_test.values,
                    label="Vraie Consommation", color='blue', marker='o')
            # Prédictions du modèle
            ax.plot(y_test.index, predictions_futures,
                    label=f"Prédiction ({modele_choisi})", color=couleur_pred,
                    linestyle='--', marker='x')
            # Ligne verticale marquant la frontière entraînement / test
            ax.axvline(x=y_train.index[-1], color='black', linestyle=':')
            ax.set_title("Comparaison Réel vs Prédiction", fontsize=14)
            ax.set_ylabel("Valeur quotidienne (kWh)")
            ax.legend()
            st.pyplot(fig)


# =============================================================================
# PAGE 5 : GÉNÉRATION DE PROFILS SYNTHÉTIQUES
# =============================================================================
elif menu == "5. Génération":
    # Import tardif pour ne pas ralentir le chargement des autres pages :
    # le module 'generation_page' n'est importé que si l'utilisateur navigue ici.
    from generation_page import render_generation_page

    # Délègue tout le rendu au module externe en lui passant le DataFrame complet
    # (features + labels), dont il a besoin pour conditionner la génération.
    render_generation_page(features_avec_labels)
