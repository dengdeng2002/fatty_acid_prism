import streamlit as st
import shap
import joblib
import xgboost
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm
import streamlit.components.v1 as components

def main():
    best_model = joblib.load('./lgbm.pkl')

    class Subject:
        def __init__(self, AGE, RACE, BMI, C4_0):
            self.AGE = AGE
            self.BMI = BMI
            self.RACE = RACE
            self.C4_0 = C4_0
 
        def make_predict(self):
            subject_data = {
                "RACE": [self.RACE],
                "BMI": [self.BMI],
                "AGE": [self.AGE],
                "C4_0": [self.C4_0]
            }

            df_subject = pd.DataFrame(subject_data)
            prediction = best_model.predict_proba(df_subject)[:, 1]
            adjusted_prediction = np.round(prediction * 100, 2)
            st.write(f"""
                <div class='all'>
                    <p style='text-align: center; font-size: 20px;'>
                        <b>The model predicts the prevalence of preserved ratio impaired spirometry is {adjusted_prediction} %</b>
                    </p>
                    <p style='text-align: center; font-size: 14px; color: gray;'>
                        Note: The model was trained on adults aged 20–79 years and is not intended for clinical use.
                    </p>
                </div>
            """, unsafe_allow_html=True)

            explainer = shap.Explainer(best_model)
            shap_values = explainer.shap_values(df_subject)
            shap.force_plot(explainer.expected_value[1], shap_values[1][0, :], df_subject.iloc[0, :], matplotlib=True)
            st.pyplot(plt.gcf())

    st.set_page_config(page_title='the prevalence of preserved ratio impaired spirometry')
    st.markdown(f"""
                <div class='all'>
                    <h1 style='text-align: center;'>Predicting the prevalence of preserved ratio impaired spirometry</h1>
                    <p class='intro' style='text-align: center; font-size: 16px;'>This tool estimates the probability of PRISm (Preserved Ratio Impaired Spirometry) based on demographic and dietary input. The model was trained on U.S. adults aged 20–79 years. It is intended for educational and research use only, not for clinical decision-making.</p>
                </div>
                """, unsafe_allow_html=True)

    with st.form("input_form"):
        RACE = st.selectbox("Race (Mexican American = 1, Other Hispanic = 2, Non-Hispanic White = 3, Non-Hispanic Black = 4, Other race = 5)", [1,2,3,4,5], index=3)
        BMI = st.number_input("BMI (kg/m^2)", value=24.51)
        AGE = st.number_input("Age (years)", value=48)
        C4_0 = st.number_input("dietary C4:0 fatty acid intake (g/day)", value=2.411)
        submitted = st.form_submit_button("Submit")
        reset = st.form_submit_button("Reset")

    if submitted:
        user = Subject(AGE, RACE, BMI, C4_0)
        user.make_predict()

    st.markdown("""---""")
    st.markdown("""
    **Acknowledgments**  
    Developed by: Deng et al.  
    Contact: [dengcy0758@163.com](mailto:dengcy0758@163.com)  

    **Terms of Use**  
    This tool is for research and educational use only.  
    It is not intended for clinical diagnosis or treatment decision-making.  
    The model was trained on individuals aged 20–79 years.

    **Data Privacy**  
    No input data is stored on the server.  
    All inputs are processed only temporarily and not used for any other purposes.
    """)

main()