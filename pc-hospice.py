import pickle
import pandas as pd
import streamlit as st
import numpy as np
from propelauth import auth
import sqlite3
from sqlite3 import Error
from datetime import datetime

def create_connection():
    conn = None
    try:
        conn = sqlite3.connect('user.db')
        print(f"SQLite version: {sqlite3.version}")
        return conn
    except Error as e:
        print(e)

def insert_log_user_activity(conn, email, activity_type):
    from datetime import datetime

    # Get the current date and time
    current_datetime = datetime.now()

    sql = ''' INSERT INTO log_user_activity(login_date_time, email, activity_type) VALUES(?,?,?) '''
    cur = conn.cursor()
    try:
        cur.execute(sql, (current_datetime, email, activity_type))
        conn.commit()
        return True
    except Error as e:
        print(e)


def main():
    user = auth.get_user()

    if user == None:
        st.error('Unauthorized')
        st.stop()

    conn = create_connection()

    # Load UI Config file
    ui_df = pd.read_csv('ui~target_transfertohospice.csv')

    # Create Form Title Header
    # st.title("PredictaCare Hospice")

    st.header('PredictaCare Hospice')
    st.header('A Hospice Care Decision Assistant for Home Health Providers')
    st.write('''Discover the power of predictive analytics with our decision assistant, 
    crafted to analyze home health encounter data comprehensively. 
    By identifying patients who may benefit from hospice care, 
    this tool equips home health professionals with vital insights, 
    enhancing their decision-making process for patient care.''')

    # Iterate through the DataFrame and create Streamlit form elements
    ui_question_order_prev = 0
    form_values = {}

    for index, row in ui_df.iterrows():
        ui_question_order = row['ui_question_order']
        ui_type = row['ui_type']
        question = row['question']
        score_mean = row['score_mean']
        score_min = row['score_min']
        score_max = row['score_max']
        question_type = row['question_type']

        # Create Question if question is different from previous
        if ui_question_order_prev != ui_question_order:

            # Get the Options and Var for each question
            filtered_df = ui_df[ui_df['ui_question_order'] == ui_question_order]
            sorted_df = filtered_df.sort_values(by=['ui_option_order'])
            options_df = sorted_df[['question_option']]
            options_value_df = sorted_df[['option_value']]
            options_list = options_df['question_option'].tolist()
            options_value_list = options_value_df['option_value'].tolist()
            var_df = sorted_df[['var']]
            var_list = var_df['var'].tolist()

            if ui_type == 'text_input':

                option_value = st.text_input(question, 'Enter')
                field = var_list[0]
                form_values[field] = option_value

            if ui_type == 'selectbox':

                option_value = st.selectbox(question, options_list)
                option_index = options_list.index(option_value)
                field = var_list[option_index]
                value = options_value_list[option_index]
                if question_type == 'parent':
                    form_values[field] = 1
                else:
                    form_values[field] = value

            elif ui_type == 'slider':
                option_value = st.slider(question, min_value=int(score_min),
                                                   max_value=int(score_max),
                                                   value=int(score_mean))
                field = var_list[0]
                form_values[field] = option_value

            elif ui_type == 'checkbox':

                st.write(question)
                for option in options_list:
                    option_index = options_list.index(option)
                    field = var_list[option_index]
                    form_values[field] = int(st.checkbox(option))

        ui_question_order_prev = ui_question_order

        # Load the predictive model from the pickle file
        with open('XGBoost~target_transfertohospice.pkl', 'rb') as model_file:
            predictive_model = pickle.load(model_file)

    def make_prediction(input_data):

        input_data_2d = np.array(input_data).reshape(1, -1)
        prediction = predictive_model.predict(input_data_2d)
        probability = predictive_model.predict_proba(input_data_2d)[:, 1]

        return prediction, probability

    # Inject custom CSS to set the width of the sidebar
    st.markdown(
        """
        <style>
            section[data-testid="stSidebar"] {
                width: 300px !important; # Set the width to your desired value
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:

        images = ['PredictaCare.jpg', 'hospice.jpg']
        st.image(images, use_column_width=False)

        st.link_button('Account/Logout', auth.get_account_url(), use_container_width=True)

        # Button to trigger the predictive model
        if st.button('Run Hospice Prediction', use_container_width=True):

            insert_log_user_activity(conn, user.email, 'Predict')

            form_df = pd.DataFrame(list(form_values.items()), columns=['form_field', 'form_value'])
            form_df['form_feature'] = form_df['form_field'] + '_' + form_df['form_value'].astype(int).astype(str)
            # form_df.to_csv('form_df.csv', index=False)

            # Create df for storing model inputs and other attribs
            ui_model_df = ui_df
            ui_model_df['model_input'] = np.nan
            ui_model_df = ui_model_df[ui_model_df['feature_order'] >= 0]

            # Map Positive 1 Binary values
            positive_df = form_df.merge(ui_model_df, left_on=['form_field', 'form_feature'],
                                        right_on=['var', 'var_feature'])
            positive_df.loc[positive_df['form_feature'] == positive_df['var_feature'], 'model_input'] = 1
            positive_df = positive_df[positive_df['feature_order'].notna() & (positive_df['feature_order'] != '')]

            # Map Continuous or question type's of 'value' to the actual form value
            filtered_df = ui_model_df[ui_model_df['question_type'] == 'value']
            value_df = form_df.merge(filtered_df, left_on=['form_field'], right_on=['var'])
            value_df['model_input'] = value_df['form_value']

            # Combine positive and value df's with only cols needed for model input
            positive_value_df = pd.concat([value_df, positive_df], axis=0)
            cols = ['var_feature', 'feature_order', 'score_max',
                    'question_type', 'model_input', 'question', 'question_option']
            positive_value_df = positive_value_df[cols]

            # Map Zero Binary Value - any feature left unmapped must be 0
            filtered_df = ui_model_df[ui_model_df['var_feature'].notna()]
            zero_df = filtered_df[~filtered_df['var_feature'].isin(positive_value_df['var_feature'])]
            zero_df = zero_df[cols]
            zero_df['model_input'] = 0

            # Finalize model input df
            model_input_df = pd.concat([positive_value_df, zero_df], axis=0)
            model_input_df = model_input_df.sort_values(by='feature_order')

            # Scale the input value between 0-1
            model_input_df = model_input_df.reset_index(drop=True)
            model_input_df.loc[model_input_df['question_type'] == 'value', 'model_input'] /= model_input_df[
                'score_max']

            # model_input_df.to_csv('model_input_df.csv', index=False)

            # Call the predictive model
            prediction, probability = make_prediction(model_input_df['model_input'])
            prediction = int(prediction)
            probability = float(probability)
            probability_pct = float(probability * 100)
            probability_pct_str = str(round(probability_pct, 2)) + '%'
            # print('Prediction: ', prediction)
            # print('Probability: ', probability_pct_str)

            cols = ['model_input']
            model_log_df = model_input_df[cols]
            model_log_df = model_log_df.T
            rename_cols = [ 'age_calc',
                            'bmi_category_class_1_obesity_1',
                            'bmi_category_class_2_obesity_1',
                            'bmi_category_class_3_obesity_1',
                            'bmi_category_normal_1',
                            'bmi_category_overweight_1',
                            'bmi_category_unknown_1',
                            'distolic',
                            'high_risk_diag_1',
                            'hospitalization_risk_score_11',
                            'm1028_actv_diag_dm_1',
                            'm1028_actv_diag_noa_1',
                            'm1028_actv_diag_pvd_pad_1',
                            'm1030_thh_par_nutrition_1',
                            'm1033_hosp_risk_compliance_1',
                            'm1033_hosp_risk_hstry_falls_1',
                            'm1033_hosp_risk_mntl_bhv_dcln_1',
                            'm1033_hosp_risk_none_above_1',
                            'm1033_hosp_risk_othr_risk_1',
                            'm1200_vision_2',
                            'm1242_pain_freq_actvty_mvmt_0',
                            'm1242_pain_freq_actvty_mvmt_1',
                            'm1242_pain_freq_actvty_mvmt_2',
                            'm1242_pain_freq_actvty_mvmt_3',
                            'm1242_pain_freq_actvty_mvmt_4',
                            'm1330_stas_ulcr_prsnt_2',
                            'm1340_srgcl_wnd_prsnt_0',
                            'm1340_srgcl_wnd_prsnt_1',
                            'm1340_srgcl_wnd_prsnt_2',
                            'm1400_when_dyspneic_0',
                            'm1400_when_dyspneic_1',
                            'm1400_when_dyspneic_2',
                            'm1600_uti_1',
                            'm1610_ur_incont_0',
                            'm1610_ur_incont_2',
                            'm1620_bwl_incont_1',
                            'm1700_cog_function_0',
                            'm1700_cog_function_2',
                            'm1700_cog_function_3',
                            'm1700_cog_function_4',
                            'm1710_when_confused_0',
                            'm1710_when_confused_2',
                            'm1710_when_confused_4',
                            'm1710_when_confused_5',
                            'm1720_when_anxious_0',
                            'm1720_when_anxious_1',
                            'm1720_when_anxious_2',
                            'm1720_when_anxious_3',
                            'm1730_phq2_dprsn_2',
                            'm1730_phq2_dprsn_3',
                            'm1730_phq2_lack_intrst_2',
                            'm1730_phq2_lack_intrst_3',
                            'm1740_bd_mem_deficit_1',
                            'm1745_beh_prob_freq_1',
                            'm1745_beh_prob_freq_2',
                            'm1745_beh_prob_freq_3',
                            'm1745_beh_prob_freq_4',
                            'm1745_beh_prob_freq_5',
                            'm1840_crnt_toiltg_0',
                            'm1840_crnt_toiltg_1',
                            'm1870_crnt_feeding_0',
                            'm1870_crnt_feeding_3',
                            'm1870_crnt_feeding_4',
                            'm1870_crnt_feeding_5',
                            'm2001_drug_rgmn_rvw_1',
                            'm2030_crnt_mgmt_injctn_mdctn_0',
                            'm2030_crnt_mgmt_injctn_mdctn_1',
                            'm2030_crnt_mgmt_injctn_mdctn_2',
                            'm2030_crnt_mgmt_injctn_mdctn_3',
                            'num_of_episodes',
                            'o2_sat',
                            'primary_diag_clinical_group_complex_1',
                            'primary_diag_clinical_group_mmta_after_1',
                            'primary_diag_clinical_group_mmta_endo_1',
                            'primary_diag_clinical_group_mmta_gi_gu_1',
                            'primary_diag_clinical_group_mmta_other_1',
                            'primary_diag_clinical_group_mmta_resp_1',
                            'primary_diag_clinical_group_ms_rehab_1',
                            'primary_diag_clinical_group_nan_1',
                            'primary_diag_clinical_group_neuro_rehab_1',
                            'primary_diag_clinical_group_wound_1',
                            'pulse',
                            'systolic']
            model_log_df.columns = rename_cols
            current_datetime = datetime.now()
            log_data = {'log_datetime': [current_datetime], 'email': [user.email], 'prediction': [prediction], 'probability': [probability]}
            log_df = pd.DataFrame(log_data)

            model_log_df.reset_index(drop=True, inplace=True)
            log_df.reset_index(drop=True, inplace=True)
            model_log_df = pd.concat([log_df, model_log_df], axis=1)
            # model_log_df.to_csv('model_log_df.csv', index=False)
            model_log_df.to_sql('log_user_model', conn, if_exists='append', index=False)



            # Display the prediction result
            if prediction == 1:
                st.success(
                    '❤️ High Probability of Hospice Care: ' + probability_pct_str + '❤️')
            else:
                st.success(
                    '💪 Low Probability of Hospice Care:  ' + probability_pct_str + '💪')


if __name__ == '__main__':
    main()
