import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
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

# Home Tab
with tabs[0]:
    st.header("Home")
    st.write("**Description of the Project:**")
    st.text_area("Add content here", height=200)

# Score Tab
with tabs[1]:
    st.header("Score")
    uploaded_file = st.file_uploader("Upload CSV file for scoring", type="csv")

    if uploaded_file is not None:
        try:
            input_data = pd.read_csv(uploaded_file)
            processed_data = data_pipeline.data_pipeline(input_data)
            scores = clf.predict(processed_data.drop(columns=['Revenue']))

            st.write("### Scored Data")
            processed_data['Scores'] = scores
            st.write(processed_data)

            st.write("### Visualization")
            fig, ax = plt.subplots()
            ax.hist(scores, bins=20, color='blue', alpha=0.7, edgecolor='black')
            ax.set_title("Score Distribution")
            ax.set_xlabel("Scores")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error processing file: {e}")

# Pred Tab
with tabs[2]:
    st.header("Pred")
    st.write("### Input data for prediction")
    visitor_type = st.selectbox("Visitor Type", ["New_Visitor", "Returning_Visitor"])
    month = st.selectbox("Month", list(data_pipeline.month_encoder_dict.keys()))
    admin = st.number_input("Administrative", min_value=0)
    info = st.number_input("Informational", min_value=0)
    product_related = st.number_input("Product Related", min_value=0)
    page_values = st.number_input("Page Values", min_value=0.0, format="%.2f")
    exit_rates = st.slider("Exit Rates", min_value=0.0, max_value=1.0, step=0.01)

    if st.button("Predict"):
        try:
            input_data = pd.DataFrame({
                'VisitorType': [visitor_type],
                'Month': [month],
                'Administrative': [admin],
                'Informational': [info],
                'ProductRelated': [product_related],
                'PageValues': [page_values],
                'ExitRates': [exit_rates]
            })

            processed_data = data_pipeline.data_pipeline(input_data)
            prediction = clf.predict(processed_data.drop(columns=['Revenue']))[0]

            st.success(f"Prediction: {prediction}")
        except Exception as e:
            st.error(f"Error making prediction: {e}")
