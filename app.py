# app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

st.set_page_config(page_title="BioML App", layout="wide")

st.title(" Εφαρμογή Ανάλυσης Μοριακής Βιολογίας με Machine Learning")

tab1, tab2, tab3 = st.tabs(["Ανάλυση", "Οπτικοποιήσεις", "Πληροφορίες Ομάδας"])

with tab1:
    st.header(" Φόρτωση Δεδομένων")
    uploaded_file = st.file_uploader("Εισάγετε dataset (.csv)", type="csv")
    
    use_example = st.checkbox("Χρήση Παραδείγματος Δεδομένων (Breast Cancer)")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write(" Προεπισκόπηση:")
        st.dataframe(df.head())
    elif use_example:
        data = load_breast_cancer(as_frame=True)
        df = data.frame
        st.write("Χρησιμοποιείται το παράδειγμα Breast Cancer dataset")
        st.dataframe(df.head())
    else:
        st.info("Παρακαλώ επιλέξτε ένα dataset για να συνεχίσετε.")
        df = None

    if df is not None:
        st.header(" Προεπεξεργασία")
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        st.success("Κανονικοποίηση Ολοκληρώθηκε")

        st.header(" Μοντέλο Μηχανικής Μάθησης")
        test_size = st.slider("Test size", 0.1, 0.5, 0.2)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42)

        model_option = st.selectbox("Επιλέξτε Αλγόριθμο", ["Random Forest", "SVM", "K-Nearest Neighbors"])

        if model_option == "Random Forest":
            clf = RandomForestClassifier(random_state=42)
        elif model_option == "SVM":
            clf = SVC(random_state=42, probability=True)
        else:
            clf = KNeighborsClassifier()

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        st.text("Classification Report:")
        st.text(classification_report(y_test, y_pred))

        st.session_state["X_scaled"] = X_scaled
        st.session_state["y"] = y
        st.session_state["clf"] = clf
        st.session_state["X_test"] = X_test
        st.session_state["y_test"] = y_test
        st.session_state["model_option"] = model_option

with tab2:
    st.header(" Οπτικοποιήσεις")
    if "X_scaled" in st.session_state and "clf" in st.session_state:
        X_scaled = st.session_state["X_scaled"]
        y = st.session_state["y"]
        clf = st.session_state["clf"]
        X_test = st.session_state["X_test"]
        y_test = st.session_state["y_test"]
        model_option = st.session_state["model_option"]

        # PCA plot
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
        pca_df["label"] = y.values

        fig, ax = plt.subplots()
        sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="label", ax=ax)
        ax.set_title("PCA Scatterplot")
        st.pyplot(fig)

        # Confusion matrix
        st.subheader("Confusion Matrix")
        y_pred = clf.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        fig2, ax2 = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2)
        ax2.set_xlabel('Προβλεπόμενη Κλάση')
        ax2.set_ylabel('Πραγματική Κλάση')
        ax2.set_title('Confusion Matrix')
        st.pyplot(fig2)

        # ROC curve
        st.subheader("ROC Curve")
        if hasattr(clf, "predict_proba"):
            y_score = clf.predict_proba(X_test)[:, 1]
        else:
            y_score = clf.decision_function(X_test)

        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)

        fig3, ax3 = plt.subplots()
        ax3.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        ax3.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax3.set_xlim([0.0, 1.0])
        ax3.set_ylim([0.0, 1.05])
        ax3.set_xlabel('False Positive Rate')
        ax3.set_ylabel('True Positive Rate')
        ax3.set_title('Receiver Operating Characteristic')
        ax3.legend(loc="lower right")
        st.pyplot(fig3)

        # Feature importance (μόνο για Random Forest)
        if model_option == "Random Forest":
            st.subheader("Feature Importance")
            importances = clf.feature_importances_
            features = df.columns[:-1]
            feat_imp_df = pd.DataFrame({'Feature': features, 'Importance': importances})
            feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=False)

            fig4, ax4 = plt.subplots(figsize=(8,6))
            sns.barplot(x='Importance', y='Feature', data=feat_imp_df, ax=ax4)
            ax4.set_title('Feature Importance (Random Forest)')
            st.pyplot(fig4)

    else:
        st.info("Πρέπει να φορτώσετε και να εκπαιδεύσετε το μοντέλο πρώτα στο tab 'Ανάλυση'.")

with tab3:
    st.header(" Πληροφορίες Ομάδας")
    st.markdown("""
    **Ομάδα:**  
    - Ιωάννης Νταιλάκης – ML Υλοποίηση  
    - Ιωάννης Μάζης – Streamlit και Docker  
    - Στάυρος Ρουμελιώτης – Report και UML Διαγράμματα
    """)
