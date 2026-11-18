# QualityInsight — démonstrateur qualité clients

Mini-projet personnel exploratoire en lien avec une mission d'analyse des données qualité clients. **Aucune donnée du Groupe Atlantic n'est utilisée** et les résultats ne doivent pas être interprétés comme des résultats industriels.

## Démarrer

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Contenu

- Simulation reproductible de 5 000 produits avec familles, fournisseur, lot, mois de vente, ancienneté, incidents, causes et délais de résolution.
- Tableau de bord interactif : taux d'incident, causes, délai moyen, filtres.
- Comparaison de deux modèles : régression logistique et Random Forest.
- Évaluation hors échantillon : précision, rappel, F1 et PR-AUC (average precision).
- Export CSV des données fictives, importable dans Power BI.

## Démarche scientifique

Les variables de cause et délai de résolution sont **exclues des prédicteurs** afin d'éviter une fuite de données. La simulation crée intentionnellement un lien statistique entre quelques caractéristiques et le risque d'incident. Le modèle apprend donc une relation artificielle ; aucune performance réelle n'est revendiquée. Une vraie étude exigerait des données historiques anonymisées, une définition précise du label, une validation temporelle, une analyse des coûts des faux négatifs et des biais de collecte.

## Prolongements possibles

- Validation chronologique sur données réelles, si accessibles et autorisées.
- Choix du seuil selon le coût métier des faux négatifs.
- KPI par gamme, cohorte de production, fournisseur et délai de résolution.
- Rapport Power BI à partir du CSV exporté.
