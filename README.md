# airbnb-price-oracle
### Eden Elfassy & Léonie Chapelle

Predicting Airbnb listing prices using machine learning. From raw text 
and messy data to a tuned gradient boosting model, with NLP-based amenity 
extraction, geospatial feature engineering, and multi-model benchmarking 
across 22k listings.

## Pipeline
```mermaid
flowchart TD
    subgraph P1["🔍 Phase 1 — Exploration"]
        A[Étape 1\nChargement & inspection] --> B[Étape 2\nAnalyse des variables]
        B --> C[Étape 3\nNettoyage des données]
        C --> D[Étape 4\nEDA & visualisations]
    end

    subgraph P2["⚙️ Phase 2 — Feature Engineering"]
        E[Encodage catégoriel] --> F[Parsing amenities]
        F --> G[Features temporelles & géo]
        G --> H[Ratios & interactions]
    end

    subgraph P3["🤖 Phase 3 — Modélisation"]
        I[Baseline\nLinear Regression] --> J[Multi-model\nRF · XGBoost · LightGBM]
        J --> K[Cross-validation\n& GridSearch]
        K --> L[Meilleur modèle\ntuning final]
    end

    subgraph P4["📦 Phase 4 — Rendu"]
        M[Prédictions test set\nCSV id + logpred]
        N[Notebook final\nMarkdown + analyses]
    end

    P1 --> P2 --> P3 --> P4
```

## Résultats

| Modèle | RMSE |
|--------|------|
| Linear Regression (baseline) | 0.4116 |
| Ridge | 0.4136 |
| Lasso | 0.4146 |
| Random Forest | 0.4010 |
| Gradient Boosting | 0.3977 |
| XGBoost | 0.3977 |
| **LightGBM ✅** | **0.3881** |

## Features créées

- **Amenities** : 13 features binaires extraites du champ texte
- **Géographiques** : distance au centre-ville par coordonnées GPS
- **Temporelles** : ancienneté hôte, durée d'activité du listing
- **Ratios** : accommodates_per_bed, beds_per_bedroom
- **Neighbourhood** : target encoding du quartier

## Stack technique
```
Python · Pandas · Scikit-learn · XGBoost · LightGBM · Matplotlib · Seaborn


