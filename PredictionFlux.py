import streamlit as st
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import smtplib
from email.mime.text import MIMEText
load_dotenv()

signature = os.getenv('SIGNATURE', 'Signature non trouvée')

add_selectbox = st.sidebar.selectbox(
    "Menu",
    ("Accueil", "Rappel de paiement", "Flux à venir", "Afficher")
)

def si():
    xx = os.getenv('introuver', 'introuver')
    print(f"{xx}")
si()
# Le Menu
if add_selectbox == "Accueil":
    
    st.title("Optimisation des flux de trésorerie et gestion des rappels de paiement")
    st.write("L'objectif de cette application c'est de prédire l'analyse de flux de trésorerie")
    
elif add_selectbox == "Rappel de paiement":
    st.title("Prédiction de rappel de paiement")
    fichier = st.file_uploader("Selectionnez un fichier CSV de transactions", type="csv")

    if fichier is not None:
        df = pd.read_csv(fichier, sep=';')
        st.write("Données de la table transaction :", df)
    
        # Conversion des dates en format datetime
        df['datetransaction'] = pd.to_datetime(df['Date transaction'], errors='coerce')
        df['datesaisi'] = pd.to_datetime(df['Date saisi'], errors='coerce')
    
        # Calculer le retard en jours
        df['Retard'] = (df['datetransaction'] - df['datesaisi']).dt.days
        df['Retard'] = df['Retard'].apply(lambda x: max(x, 0))
        
        # Calculer la colonne mois_num pour suivre les mois
        df['mois_num'] = df['datetransaction'].dt.month + (df['datetransaction'].dt.year - datetime.now().year) * 12
        
        # Filtrer les données pertinentes pour la prédiction
        df = df[['mois_num', 'Montant', 'Paiement', 'Retard']]
        
        # Gérer les valeurs manquantes avec SimpleImputer
        imputer = SimpleImputer(strategy='mean')
        
        # Variables indépendantes (features)
        X = df[['mois_num', 'Montant', 'Retard']]
        y = df['Montant']  # Variable cible
        
        # Vérifier les valeurs manquantes
        st.write("Valeurs manquantes avant imputation :")
        st.write(X.isna().sum())
        sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
        plt.show()
        
        if st.checkbox("Resultat de calcul d'analyse"):
            
            # Imputation des données manquantes pour X
            X = imputer.fit_transform(X)
            
            # Vérification de la taille de X et y
            st.write(f"Taille de X avant filtrage : {X.shape}")
            st.write(f"Taille de y avant filtrage : {y.shape}")
            
            
            
            # Supprimer les lignes où `y` est manquant
            non_null_indices = ~y.isna()
            
            # Vérifier que les tailles de X et y sont compatibles
            if len(non_null_indices) == X.shape[0]:
                X = X[non_null_indices]
                y = y[non_null_indices]
            else:
                st.error("Les tailles de X et y ne correspondent pas. Vérifiez vos données.")

            # Afficher la taille de X et y après le filtrage
            st.write(f"Taille de X après filtrage : {X.shape}")
            st.write(f"Taille de y après filtrage : {len(y)}")
            
            # Calcul statistique
            
            st.subheader("Calcul statistique descriptive ")
            
            stat = df[['Montant', 'Retard']].describe()
            st.write(stat)
            
            '''st.subheader("Graphique de retard de paiement")
            fig, ax = plt.subplots()
            sns.boxplot(x=df['Retard'], ax=ax)
            ax.set_title('Retard de paiement')
            ax.set_xlabel('Retard par jour')
            st.pyplot(fig)'''
            
            #Retard et pourcentage de paiement
    

        # Vérification des données restantes
        if len(X) == 0 or len(y) == 0:
            st.error("Données insuffisantes après traitement. Veuillez vérifier les données.")
        else:
            # Diviser les données en ensembles d'entraînement et de test
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Créer et entraîner le modèle de régression linéaire
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            # Faire des prédictions
            y_pred = model.predict(X_test)
            
            # Calculer l'erreur quadratique moyenne
           # mse = mean_squared_error(y_test, y_pred)
            
            # Afficher les résultats
           # st.write("Erreur quadratique moyenne (MSE) :", mse)
           
                       # Calcul de l'EQM et R²
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            st.subheader("Résultats des Prédictions")
            st.write(f"Erreur Quadratique Moyenne (EQM) : {mse:.2f}")
            st.write(f"Coefficient de détermination (R²) : {r2:.2f}")
            if r2 <1:
                st.info(" {r2:.2f} : Le modèle est parfait")
            elif r2 == 0:
                st.info(" {r2:.2f} : Le modèle ne fait pas mieux que la moyenne de données")
            elif r2 == 1:
                st.info(f" (R²) = {r2:.2f} : Le modèle prédit exactement les données")
            else:
                st.info("le modèle n'a pas d'interpretation ce qui est impossible ")
            
            
            # Option pour afficher les détails réels et prédits
            if st.checkbox("Afficher les valeurs réelles et prédites"):
                result_df = pd.DataFrame({'Réel': y_test, 'Prédit': y_pred})
                st.write(result_df)
            
            if st.checkbox("Afficher le graphique de prédiction"):
                
                # Tracer le graphique des prédictions
                st.subheader("Graphique des Prédictions")
                
                # Tracer les résultats réels vs prédits
                fig, ax = plt.subplots()
                ax.plot(y_test.reset_index(drop=True), label='Valeurs réelles')
                ax.plot(y_pred, label='Valeurs prédites', linestyle='--')
                ax.set_xlabel('Index')
                ax.set_ylabel('Paiement')
                ax.set_title('Comparaison des valeurs réelles et prédites')
                ax.legend()
                st.pyplot(fig)
            
            # Recommandation pour optimiser les rappels de paiement
            st.subheader("Optimisation des rappels de paiement")
            mean_delay = df['Retard'].mean()
            st.write(f"Le délai moyen de retard de paiement est de {mean_delay:.2f} jours.")
            percentage_late = (df['Retard'] > 0).mean() * 100
            st.write(f"Pourcentage de paiements en retard : {percentage_late:.2f} %")
            
            if mean_delay > 30:
                st.info("Recommandation : Envoyer des rappels plus tôt pour éviter les retards.")
                
                def send_email(client_email, client_name, amount_due, due_date):
                    
                    msg = MIMEText(f"Bonjour {client_name},\n\n"
                                f"Nous vous rappelons que vous avez un paiement en retard de {amount_due} €.\n"
                                f"Date limite de paiement : {due_date}.\n\n"
                                "Merci de régler cette facture dans les plus brefs délais.\n"
                                "Cordialement,\nVotre entreprise")
                    msg['Subject'] = "Rappel de paiement"
                    msg['From'] = "votre_email@example.com"
                    msg['To'] = client_email

                    # Remplacer par vos informations d'accès au serveur SMTP
                    with smtplib.SMTP("smtp.gmail.com", 587) as server:
                        server.starttls()
                        server.login("kifouanigiresse@gmail.com", "")
                        server.sendmail(msg['From'], [msg['To']], msg.as_string())

            else:
                st.info("Les délais de paiement sont raisonnables.")
                #st.checkbox("Envoyer un rappel automatique de paiement "):
                


    else:
        st.write("Veuillez selectionnez un fichier CSV pour démarrer l'analyse.")

elif add_selectbox == "Flux à venir":
    # Titre de l'application
    st.title("Prédiction des Flux de Trésorerie")

    # Upload du fichier CSV des données financières historiques
    fichier = st.file_uploader("Choisissez un fichier CSV contenant les transactions financières", type="csv")

    if fichier is not None:
        # Lire le fichier CSV
        df = pd.read_csv(fichier, sep=';')
        
        # Aperçu des données
        st.write("Aperçu des données :", df.head())
        
        # Vérifier les colonnes disponibles
        st.write("Colonnes disponibles :", df.columns.tolist())
        
        # Conversion des dates en format datetime
        df['Date_transaction'] = pd.to_datetime(df['Date_transaction'], errors='coerce')
        df['Date_paiement'] = pd.to_datetime(df['Date_paiement'], errors='coerce')
        
        # Calcul du flux de trésorerie (Entrée - Sortie)
        df['flux'] = df['montant_entree'] - df['montant_sortie']
        
        # Créer une colonne représentant le mois de la transaction
        df['mois'] = df['Date_transaction'].dt.to_period('M')

        # Agréger les flux par mois
        df_monthly = df.groupby('mois').agg({'flux': 'sum'}).reset_index()

        # Calculer la colonne des mois numériques pour la prédiction
        df_monthly['mois_num'] = np.arange(len(df_monthly))

        # Afficher les données mensuelles agrégées
        st.write("Flux de trésorerie agrégés par mois :", df_monthly.head())
        
        # Variables indépendantes (mois numérique) et dépendantes (flux)
        X = df_monthly[['mois_num']]  # Variable indépendante
        y = df_monthly['flux']        # Variable cible (flux de trésorerie)
        
        # Diviser les données en ensembles d'entraînement et de test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Créer et entraîner le modèle de régression linéaire
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Faire des prédictions sur l'ensemble de test
        y_pred = model.predict(X_test)
        
        # Calculer l'erreur quadratique moyenne
        mse = mean_squared_error(y_test, y_pred)
        
        # Afficher l'erreur quadratique moyenne
        st.write(f"Erreur quadratique moyenne (MSE) : {mse:.2f}")
        
        # Prédire les flux pour les mois à venir (6 mois supplémentaires)
        future_months = np.array([[i] for i in range(len(df_monthly), len(df_monthly) + 6)])
        future_flux = model.predict(future_months)
        
        # Créer un DataFrame pour les mois futurs
        future_df = pd.DataFrame({
            'mois': pd.period_range(df_monthly['mois'].max() + 1, periods=6, freq='M'),
            'flux_prédit': future_flux
        })
        
        # Afficher les prédictions
        st.subheader("Prédictions des flux de trésorerie pour les 6 mois à venir")
        st.write(future_df)
        
        # Visualisation des flux réels et prédits
        fig, ax = plt.subplots()
        ax.plot(df_monthly['mois'].astype(str), df_monthly['flux'], label='Flux réels')
        ax.plot(future_df['mois'].astype(str), future_df['flux_prédit'], label='Flux prédits', linestyle='--', color='red')
        ax.set_xlabel('Mois')
        ax.set_ylabel('Flux de trésorerie')
        ax.set_title('Prédiction des Flux de Trésorerie')
        ax.legend()
        
        # Afficher le graphique
        st.pyplot(fig)

        
        # Recommandation pour améliorer les flux de trésorerie
        mean_flux = df_monthly['flux'].mean()
        st.subheader("Analyse et recommandations")
        if mean_flux < 0:
            st.write("Attention, vos flux de trésorerie moyens sont négatifs. Il est conseillé de réduire les sorties de trésorerie ou d'accélérer les entrées.")
        else:
            st.write("Vos flux de trésorerie sont en bonne santé. Continuez à surveiller les dépenses.")
    else:
        st.write("Veuillez selectionnez un fichier CSV pour démarrer l'analyse.")
    
elif add_selectbox == "Afficher":
    st.title("Affichage")