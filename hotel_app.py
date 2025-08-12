import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
import random

random.seed(42)
np.random.seed(42)

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Hotel Rate Estimator",
)

st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1f4e79;
    text-align: center;
}
.stButton > button {
    background-color: #2196f3;
    color: white;
}
</style>
""", unsafe_allow_html=True)

if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Home"

@st.cache_data
def load_data():
    import kagglehub
    from kagglehub import KaggleDatasetAdapter
    
    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "jessemostipak/hotel-booking-demand",
        "hotel_bookings.csv")
    
    df_new = df.copy()
    df_new['children'].fillna(0, inplace=True)
    df_new['country'].fillna('Unknown', inplace=True)
    df_new.drop_duplicates(inplace=True)
    df_new.drop(['agent', 'company'], axis=1, inplace=True)
    df_new['reservation_status_date'] = df_new['reservation_status_date'].astype(str)
    df_new.drop_duplicates(inplace=True)

    df_new['total_nights'] = df_new['stays_in_weekend_nights'] + df_new['stays_in_week_nights']
    df_new['total_guests'] = df_new['adults'] + df_new['children'] + df_new['babies']
    
    month_assignment = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
                       'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12}
    df_new['arrival_month_numeric'] = df_new['arrival_date_month'].map(month_assignment)
    
    def get_season(month):
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'
    
    df_new['arrival_season'] = df_new['arrival_month_numeric'].apply(get_season)
    
    categorical_features = ['hotel', 'meal', 'country', 'market_segment', 'distribution_channel',
                               'reserved_room_type', 'assigned_room_type', 'deposit_type', 'customer_type', 
                               'arrival_season', 'reservation_status']
    
    df_new_encoded = pd.get_dummies(df_new, columns=categorical_features, drop_first=True)
    
    return df_new_encoded

def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results = {}
    
    # Linear Regression
    linear_regression = LinearRegression()
    linear_regression.fit(X_train_scaled, y_train)
    
    y_pred_lr_train = linear_regression.predict(X_train_scaled)
    y_pred_lr_test = linear_regression.predict(X_test_scaled)
    
    results['Linear Regression'] = {
        'model': linear_regression,
        'use_scaling': True,
        'scaler': scaler,
        'mae': mean_absolute_error(y_test, y_pred_lr_test),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred_lr_test)),
        'r2': r2_score(y_test, y_pred_lr_test),
        'metrics': {
            'train': {
                'mae': mean_absolute_error(y_train, y_pred_lr_train),
                'rmse': np.sqrt(mean_squared_error(y_train, y_pred_lr_train)),
                'r2': r2_score(y_train, y_pred_lr_train)
            },
            'test': {
                'mae': mean_absolute_error(y_test, y_pred_lr_test),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred_lr_test)),
                'r2': r2_score(y_test, y_pred_lr_test)
            }
        }
    }
    
    # Random Forest
    random_forest = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
    random_forest.fit(X_train, y_train)
    
    y_pred_rf_train = random_forest.predict(X_train)
    y_pred_rf_test = random_forest.predict(X_test)
    
    results['Random Forest'] = {
        'model': random_forest,
        'use_scaling': False,
        'scaler': None,
        'mae': mean_absolute_error(y_test, y_pred_rf_test),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred_rf_test)),
        'r2': r2_score(y_test, y_pred_rf_test),
        'metrics': {
            'train': {
                'mae': mean_absolute_error(y_train, y_pred_rf_train),
                'rmse': np.sqrt(mean_squared_error(y_train, y_pred_rf_train)),
                'r2': r2_score(y_train, y_pred_rf_train)
            },
            'test': {
                'mae': mean_absolute_error(y_test, y_pred_rf_test),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred_rf_test)),
                'r2': r2_score(y_test, y_pred_rf_test)
            }
        }
    }
    
    return results

def prepare_input_features(input_data, feature_names, scaler=None, use_scaling=False):
    input_df = pd.DataFrame([input_data])
    input_df = input_df.reindex(columns=feature_names, fill_value=0)
    
    if use_scaling and scaler:
        return scaler.transform(input_df)
    return input_df
        
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'
        
def main():
    if st.sidebar.button("Home"):
        st.session_state.current_page = "Home"
    if st.sidebar.button("Average Daily Rate"):
        st.session_state.current_page = "Average Daily Rate"
    if st.sidebar.button("Data Exploration"):
        st.session_state.current_page = "Data Exploration"
    if st.sidebar.button("Performance Dashboard"):
        st.session_state.current_page = "Performance Dashboard"
    
    page = st.session_state.current_page
    
    df_new_encoded = load_data()
    
    selected_features = ['lead_time', 'arrival_month_numeric', 'days_in_waiting_list', 
                           'total_of_special_requests', 'total_guests', 'total_nights']
    X = df_new_encoded[selected_features]
    y = df_new_encoded['adr']
    models = train_models(X, y)
    
    chosen_model = max(models, key=lambda m: models[m]['r2'])
    model = models[chosen_model]['model']
    scaler = models[chosen_model]['scaler']
    use_scaling = models[chosen_model]['use_scaling']
                        
    if page == "Home":
        show_Home(df_new_encoded)
    elif page == "Average Daily Rate":
        show_Average_Daily_Rate(df_new_encoded, model, scaler, use_scaling, selected_features)
    elif page == "Data Exploration":
        show_data_exploration(df_new_encoded, models)
    elif page == "Performance Dashboard":
        show_Performance_Dashboard(df_new_encoded)
    

def show_Home(df_new_encoded):
    st.markdown('<h1 class="main-header"> Hotel Average Daily Rate</h1>', unsafe_allow_html=True)
    st.markdown("""
    ### Business Problem
    
    Hotel prices are dynamic and influenced by factors such as lead time and seasonality.
    """)

def show_data_exploration(data, models=None):
    st.markdown('<h1 class="main-header"> Data Exploration</h1>', unsafe_allow_html=True)

    if models is not None:
        rows = []
        for model_name, info in models.items():
            rows.append({'Model': model_name, 'Split': 'Train', **info['metrics']['train']})
            rows.append({'Model': model_name, 'Split': 'Test', **info['metrics']['test']})
        metrics_df = pd.DataFrame(rows)
        st.subheader("Model Performance (Train/Test)")
        st.dataframe(metrics_df.style.format({'mae': '{:.2f}', 'rmse': '{:.2f}', 'r2': '{:.3f}'}))


if __name__ == "__main__":
    main()
