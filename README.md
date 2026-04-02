# Data-Science
============================================================

PROJET : ANALYSE ET DÉTECTION DES PROFILS DE CONSOMMATION ÉLECTRIQUE (ENEDIS)

============================================================

Établissement       : ESIEE Paris

Formation           : Data Science

Année universitaire : 2025/2026

Équipe projet       : Timéo AMBIBARD, Oscar BRAULT, Victor CHEN

Professeur référent : Machrafi Aboubakr


============================================================

DESCRIPTION DU PROJET

============================================================

Ce projet de Data Science vise à analyser des données de
consommation électrique (type compteurs Enedis / Linky)
afin d'identifier des comportements clients, classifier
les types de résidences (Principales ou Secondaires),
prédire la consommation future et générer de nouvelles
courbes réalistes.

Le projet s'appuie sur le jeu de données open data Enedis
(RES2-6-9kVA).

L'ensemble des résultats est présenté à travers un dashboard
interactif développé avec Streamlit, couvrant une pipeline
complète de machine learning : de l'exploration des données
jusqu'à la génération de profils synthétiques.


============================================================

  OBJECTIFS DU PROJET

============================================================

- Dashboard interactif pour explorer les données et tester les modèles
- Transformation des données (Feature Engineering)
- Identification de groupes de clients (clustering)
- Classification automatique RP / RS
- Prévision de la consommation électrique
- Génération de données synthétiques réalistes


============================================================

FONCTIONNALITÉS ET MÉTHODOLOGIE

============================================================

Le projet est organisé autour de quatre axes principaux :

1) CLUSTERING (apprentissage non supervisé)
   - Utilisation de K-Means pour segmenter les profils
   - Méthode du coude pour choisir le nombre de clusters
   - Feature Engineering :
       ratio week-end/semaine,
       proportion de consommation nulle, etc.
   - Attribution RP/RS basée sur une règle métier

2) CLASSIFICATION (apprentissage supervisé)
   - Modèles utilisés :
       Random Forest, Régression Logistique, SVM, MLPClassifier
   - Objectif : prédire automatiquement RP ou RS
   - Évaluation :
       matrice de confusion + rapport de classification
   - Interprétabilité :
       importance des variables (Random Forest, logistique)

3) PRÉVISION (forecasting)
   - Prédiction de la consommation journalière
   - Variables utilisées : J-1, J-2, J-7
   - Modèles :
       Random Forest Regressor, MLPRegressor
   - Comparaison des modèles
   - Métrique : MAE (Erreur Absolue Moyenne)

4) GÉNÉRATION DE COURBES
   - Génération de données synthétiques conditionnelles (RP/RS)
   - Modèle principal : CVAE (PyTorch)
   - Baseline : Gaussian Mixture Model
   - Évaluation :
       MMD, distance de Wasserstein,
       test de Kolmogorov-Smirnov


============================================================

FONCTIONNALITÉS DU DASHBOARD

============================================================

- Exploration des données et des clusters
- Visualisation avec ACP (Analyse en Composantes Principales)
- Comparaison des modèles de classification
- Analyse des performances (matrices de confusion)
- Interprétation des variables importantes
- Simulateur de prévision :
    sélection d'un client,
    choix de l'horizon de prédiction,
    comparaison modèle vs réalité


============================================================

TECHNOLOGIES UTILISÉES

============================================================

Langage :
    Python

Data :
    Pandas, NumPy

Machine Learning :
    Scikit-Learn

Deep Learning :
    PyTorch

Visualisation :
    Matplotlib, Seaborn

Interface :
    Streamlit


============================================================

STRUCTURE DU PROJET

============================================================

app.py : Script principal Streamlit (navigation, clustering, prévision)

generation_page.py : Interface utilisateur pour la génération

generation_module.py : Logique CVAE (PyTorch, pertes, reconstruction, métriques)

features_pour_classification.csv : Données de features pour les modèles


============================================================

INSTALLATION

============================================================

Prérequis :

  Python 3.8 ou supérieur

1) Cloner le dépôt
   
    git clone <URL_DU_DEPOT>
    cd <NOM_DU_DOSSIER>

2) Créer un environnement virtuel
   
    python -m venv env

   Activation Windows :
   
    env\Scripts\activate

   Activation macOS / Linux :
   
    source env/bin/activate

3) Installer les dépendances
   
pip install streamlit "pandas==2.3.3" numpy matplotlib seaborn scikit-learn torch scipy

Note technique : La version de Pandas doit imperativement etre inférieur aux plus récente (dans notre cas 2.3.3) pour garantir le bon affichage des tableaux de bord
(utilisation de la methode DataFrame.applymap).


============================================================

EXÉCUTION

============================================================

Pour éxécuter le code : 

Commande :  python -m streamlit run app.py

Si le dataset utilisé est supérieur à 200Mo

Commande :
    python -m streamlit run app.py --server.maxUploadSize 1000

Application disponible sur :
    http://localhost:8501
