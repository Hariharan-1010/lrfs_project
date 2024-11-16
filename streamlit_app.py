import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns
import pickle

# Define the DataPipeline and model as provided
class DataPipeline:
    def __init__(self):
        self.month_encoder_dict = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'June': 6, 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
        with open('le_vt.pkl', 'rb') as le_file:
            self.le_vt = pickle.load(le_file)
        with open('le_r.pkl', 'rb') as le_file:
            self.le_r = pickle.load(le_file)
        with open('l_ss.pkl', 'rb') as le_file:
            self.l_ss = pickle.load(le_file)
        with open('r_ss.pkl', 'rb') as le_file:
            self.r_ss = pickle.load(le_file)
        with open('f_ss.pkl', 'rb') as le_file:
            self.f_ss = pickle.load(le_file)
        with open('s_ss.pkl', 'rb') as le_file:
            self.s_ss = pickle.load(le_file)

    def calculate_length(self, row):
        if row['VisitorType'] == 2:  # Returning customer
            return row['Month'] - 1
        else:  # New customer
            return 1

    def data_pipeline(self, input_data):
        df = input_data.copy()
        df.dropna(inplace=True)
        df.drop_duplicates(inplace=True)

        df['Month'] = df['Month'].map(self.month_encoder_dict)
        df['VisitorType'] = self.le_vt.transform(df['VisitorType'])
        df['Revenue'] = self.le_r.transform(df['Revenue'])

        l = df.apply(self.calculate_length, axis=1)
        r = 12 - df['Month'] + 1
        f = df['Administrative'] + df['Informational'] + df['ProductRelated']
        s = df['PageValues'] * (1 - df['ExitRates'])

        ret_df = {'L': l, 'R': r, 'F': f, 'S': s, 'Revenue': df['Revenue']}
        ret_df = pd.DataFrame(ret_df)

        ret_df['L'] = self.l_ss.transform(ret_df['L'].values.reshape(-1, 1))
        ret_df['R'] = self.r_ss.transform(ret_df['R'].values.reshape(-1, 1))
        ret_df['F'] = self.f_ss.transform(ret_df['F'].values.reshape(-1, 1))
        ret_df['S'] = self.s_ss.transform(ret_df['S'].values.reshape(-1, 1))

        return ret_df

# Load the model
clf = pickle.load(open('lrfs_model.sav', 'rb'))

data_pipeline = DataPipeline()

# Streamlit App
st.title("🎈 My ML Project App")
tabs = st.tabs(["Home", "Score", "Pred"])

with tabs[0]:
    st.header("Welcome to the ML Project App")
    st.subheader("Objective")
    st.write(
        """
        This application leverages machine learning to score and predict customer behavior in online shopping. 
        By analyzing key features like visitor type, monthly activity, and engagement metrics, 
        we aim to provide insights into purchasing trends and outcomes.
        """
    )
    st.subheader("Features")
    st.write(
        """
        - **Score Tab**: Upload CSV data to evaluate and visualize model performance on real-world data.
        - **Predict Tab**: Input individual customer details to generate predictions.
        """
    )
    st.subheader("Model Details")
    st.write(
        """
        The model is based on the LRFS (Length-Recency-Frequency-Staying Rate) methodology 
        and trained using a Logistic Regression classifier.
        """
    )


from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns

with tabs[1]:
    st.header("Score")
    uploaded_file = st.file_uploader("Upload CSV file for scoring", type="csv")

    if uploaded_file is not None:
        try:
            input_data = pd.read_csv(uploaded_file)
            processed_data = data_pipeline.data_pipeline(input_data)
            scores = clf.predict(processed_data.drop(columns=['Revenue']))
            actual = processed_data['Revenue']

            st.write("### Scored Data")
            processed_data['Scores'] = scores
            st.write(processed_data)

            # Metrics and Confusion Matrix
            st.write("### Classification Metrics")
            st.write("Accuracy:", (scores == actual).mean())

            st.write("### Confusion Matrix")
            cm = confusion_matrix(actual, scores)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title("Confusion Matrix")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)

            # Classification Report
            st.write("### Detailed Report")
            report = classification_report(actual, scores, output_dict=True)
            st.write(pd.DataFrame(report).transpose())

            # ROC Curve
            st.write("### ROC Curve")
            fpr, tpr, _ = roc_curve(actual, scores)
            roc_auc = auc(fpr, tpr)
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
            ax.plot([0, 1], [0, 1], 'k--', label="Random Guess")
            ax.set_title("ROC Curve")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.legend()
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error processing file: {e}")

with tabs[2]:
    st.header("Predict")
    st.write("### Input data for prediction")
    visitor_type = st.selectbox("Visitor Type", ["New_Visitor", "Returning_Visitor"])
    month = st.selectbox("Month", list(data_pipeline.month_encoder_dict.keys()))
    admin = st.text_input("Administrative", value="0")
    info = st.text_input("Informational", value="0")
    product_related = st.text_input("Product Related", value="0")
    page_values = st.text_input("Page Values", value="0.0")
    exit_rates = st.text_input("Exit Rates", value="0.0")

    if st.button("Predict"):
        try:
            # Ensure input values are correctly formatted
            input_data = pd.DataFrame({
                'VisitorType': [visitor_type],
                'Month': [month],
                'Administrative': [int(admin)],
                'Informational': [int(info)],
                'ProductRelated': [int(product_related)],
                'PageValues': [float(page_values)],
                'ExitRates': [float(exit_rates)],
                'Revenue': [0]
            })

            # Process input data
            processed_data = data_pipeline.data_pipeline(input_data)

            features = processed_data.drop(columns=['Revenue'])
            prediction = clf.predict(features)[0]

            st.success(f"Prediction: {'Revenue Generated' if prediction == 1 else 'No Revenue Generated'}")
        except Exception as e:
            st.error(f"Error making prediction: {e}")
