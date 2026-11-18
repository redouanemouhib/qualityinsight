import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

st.set_page_config(page_title='QualityInsight | Démonstrateur qualité', layout='wide')
st.title('QualityInsight — Qualité clients & risque d’incident')
st.caption('PROTOTYPE PÉDAGOGIQUE · Données 100 % synthétiques · Aucune donnée du Groupe Atlantic')

@st.cache_data
def simulate(n=5000, seed=42):
    rng=np.random.default_rng(seed)
    families=np.array(['Chauffe-eau','Pompe à chaleur','Radiateur','Chaudière'])
    family=rng.choice(families,n,p=[.32,.26,.25,.17])
    supplier=rng.choice(['A','B','C'],n,p=[.45,.35,.20])
    month=rng.integers(1,13,n)
    age=rng.integers(0,49,n)
    batch=rng.choice(['L1','L2','L3','L4'],n)
    # Les variables explicatives sont supposées connues avant la survenue de l'incident.
    logit=-3.4+np.select([family=='Pompe à chaleur',family=='Chaudière'],[.75,.40],0)+.024*age+(supplier=='C')*.7+(batch=='L3')*.45+np.isin(month,[11,12,1,2])*.2
    prob=1/(1+np.exp(-logit))
    incident=rng.binomial(1,prob)
    causes=np.array(['Aucun incident']*n,dtype=object)
    choices=['Électronique','Installation','Étanchéité','Régulation','Autre']
    causes[incident==1]=rng.choice(choices,int(incident.sum()),p=[.30,.24,.20,.16,.10])
    delay=np.full(n,np.nan)
    delay[incident==1]=np.maximum(1,rng.gamma(3,3,int(incident.sum())).round())
    return pd.DataFrame({'id_produit':np.arange(1,n+1),'famille':family,'fournisseur':supplier,'lot':batch,'mois_vente':month,'anciennete_mois':age,'incident':incident,'cause':causes,'delai_resolution_jours':delay})

df=simulate()
st.sidebar.header('Filtres de visualisation')
selected=st.sidebar.multiselect('Familles',sorted(df.famille.unique()),default=sorted(df.famille.unique()))
view=df[df.famille.isin(selected)]
if view.empty:
    st.warning('Sélectionnez au moins une famille de produits.');st.stop()

c1,c2,c3,c4=st.columns(4)
c1.metric('Produits',f'{len(view):,}'.replace(',',' '))
c2.metric('Incidents',int(view.incident.sum()))
c3.metric('Taux d’incident',f'{view.incident.mean():.1%}')
c4.metric('Délai moyen de résolution',f'{view.delai_resolution_jours.mean():.1f} jours')

left,right=st.columns(2)
with left:
    st.subheader('Taux d’incident par famille')
    st.bar_chart(view.groupby('famille').incident.mean().sort_values(ascending=False))
with right:
    st.subheader('Causes déclarées')
    st.bar_chart(view.loc[view.incident.eq(1),'cause'].value_counts())

st.divider()
st.subheader('Prédiction expérimentale du risque d’incident')
st.write('Deux modèles sont comparés sur **le même jeu de test**, sans utiliser la cause ni le délai de résolution (inconnus avant l’incident).')
features=['famille','fournisseur','lot','mois_vente','anciennete_mois']
cat=['famille','fournisseur','lot']
num=['mois_vente','anciennete_mois']
X=df[features];y=df['incident']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.25,random_state=42,stratify=y)
pre=ColumnTransformer([('cat',OneHotEncoder(handle_unknown='ignore'),cat),('num',StandardScaler(),num)])
models={'Régression logistique':LogisticRegression(max_iter=1000,class_weight='balanced',random_state=42),'Random Forest':RandomForestClassifier(n_estimators=160,min_samples_leaf=8,class_weight='balanced',random_state=42,n_jobs=-1)}
results=[]
for name,est in models.items():
    pipe=Pipeline([('preprocessing',pre),('model',est)])
    pipe.fit(X_train,y_train)
    scores=pipe.predict_proba(X_test)[:,1]
    pred=(scores>=.5).astype(int)
    results.append({'Modèle':name,'PR-AUC (average precision)':average_precision_score(y_test,scores),'Précision':precision_score(y_test,pred,zero_division=0),'Rappel':recall_score(y_test,pred,zero_division=0),'F1':f1_score(y_test,pred,zero_division=0)})
st.dataframe(pd.DataFrame(results).style.format({'PR-AUC (average precision)':'{:.3f}','Précision':'{:.3f}','Rappel':'{:.3f}','F1':'{:.3f}'}),hide_index=True,use_container_width=True)
st.caption(f'Référence aléatoire pour la PR-AUC ≈ prévalence du test : {y_test.mean():.3f}. Séparation train/test stratifiée 75/25, graine 42. Seuil fixe 0,5, non optimisé.')
st.info('Interprétation : ces résultats mesurent uniquement la capacité à retrouver une règle de simulation artificielle. Ils ne démontrent aucune performance sur des données industrielles réelles.')
st.subheader('Exporter le jeu de données simulé')
st.download_button('Télécharger le CSV synthétique',df.to_csv(index=False).encode('utf-8-sig'),'qualityinsight_donnees_synthetiques.csv','text/csv')
