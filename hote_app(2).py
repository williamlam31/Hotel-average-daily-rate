import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Hotel Revenue Optimizer",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1f4e79;
    text-align: center;
    margin-bottom: 2rem;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
}
.sub-header {
    font-size: 1.5rem;
    color: #2c5aa0;
    margin-bottom: 1rem;
}
.metric-card {
    background-color: #f8f9fa;
    padding: 1rem;
    border-radius: 0.5rem;
    border: 1px solid #dee2e6;
    margin: 0.5rem 0;
}
.prediction-box {
    background-color: #e3f2fd;
    padding: 2rem;
    border-radius: 1rem;
    border-left: 5px solid #2196f3;
    margin: 1rem 0;
}
.warning-box {
    background-color: #fff3cd;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 5px solid #ffc107;
    margin: 1rem 0;
}
.stButton > button {
    background-color: #2196f3;
    color: white;
    border-radius: 0.5rem;
    border: none;
    padding: 0.5rem 1rem;
    font-weight: bold;
}
.stButton > button:hover {
    background-color: #1976d2;
    transition: background-color 0.3s;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'feature_names' not in st.session_state:
    st.session_state.feature_names = None
if 'model_results' not in st.session_state:
    st.session_state.model_results = None
if 'best_model_name' not in st.session_state:
    st.session_state.best_model_name = None
if 'best_model' not in st.session_state:
    st.session_state.best_model = None
if 'use_scaling' not in st.session_state:
    st.session_state.use_scaling = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None

@st.cache_data
def load_and_preprocess_data():
    import kagglehub
    from kagglehub import KaggleDatasetAdapter
    
    file_path = "hotel_bookings.csv"
    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "jessemostipak/hotel-booking-demand",
        "hotel_bookings.csv")
    
    # Data preprocessing
    df_new = df.copy()
    
    # Handle missing values
    df_new['children'].fillna(0, inplace=True)
    df_new['country'].fillna('Unknown', inplace=True)
    df_new['agent'].fillna(0, inplace=True)
    df_new['company'].fillna(0, inplace=True)
    df_new['adr'].fillna(df_new['adr'].median(), inplace=True)  # Fill missing ADR with median
    
    # Remove rows with zero adults, children, and babies
    df_new = df_new[(df_new['adults'] > 0) | (df_new['children'] > 0) | (df_new['babies'] > 0)]
    
    # Create derived features
    df_new['total_nights'] = df_new['stays_in_weekend_nights'] + df_new['stays_in_week_nights']
    df_new['total_guests'] = df_new['adults'] + df_new['children'] + df_new['babies']
    df_new['adr_per_guest'] = df_new['adr'] / df_new['total_guests']
    df_new['booking_changes_per_day'] = df_new['booking_changes'] / (df_new['lead_time'] + 1)
    
    # Month encoding and season creation
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
    
    # Drop reservation_status_date if it exists
    if 'reservation_status_date' in df_new.columns:
        df_new = df_new.drop(['reservation_status_date'], axis=1)
    
    # Encode categorical variables for the final model
    categorical_columns_final = ['hotel', 'meal', 'country', 'market_segment', 'distribution_channel',
                               'reserved_room_type', 'assigned_room_type', 'deposit_type', 'customer_type', 
                               'arrival_season', 'arrival_date_month', 'reservation_status']
    
    df_new_encoded = pd.get_dummies(df_new, columns=categorical_columns_final, drop_first=True)
    
    # Filter relevant columns for the model input, ensuring they exist
    model_features = [
        'lead_time', 'arrival_date_year', 'arrival_month_numeric',
        'arrival_date_week_number', 'arrival_date_day_of_month',
        'stays_in_weekend_nights', 'stays_in_week_nights', 'adults',
        'children', 'babies', 'is_repeated_guest', 'previous_cancellations',
        'previous_bookings_not_canceled', 'booking_changes',
        'days_in_waiting_list', 'required_car_parking_spaces',
        'total_of_special_requests', 'total_guests', 'total_nights',
        'booking_changes_per_day', 'adr_per_guest'
    ]
    
    # Add encoded categorical columns to the feature list
    encoded_cols = [col for col in df_new_encoded.columns if col.startswith(tuple(categorical_columns_final)) and col != 'adr']
    model_features.extend(encoded_cols)
    
    # Filter the DataFrame to include only the model features and the target variable
    # Ensure all model_features exist in df_new_encoded before selecting
    existing_model_features = [col for col in model_features if col in df_new_encoded.columns]
    X = df_new_encoded[existing_model_features]
    y = df_new_encoded['adr']
    
    return df_new, X, y  # Return original df_new for data exploration, and X, y for training

def train_models(X, y):
    """Trains multiple regression models and evaluates their performance."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results = {}
    
    # Linear Regression
    linear_regression = LinearRegression()
    linear_regression.fit(X_train_scaled, y_train)  # Use scaled data for Linear Regression
    y_pred_linear_regression = linear_regression.predict(X_test_scaled)
    
    mae_lr = mean_absolute_error(y_test, y_pred_linear_regression)
    rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_linear_regression))
    r2_lr = r2_score(y_test, y_pred_linear_regression)
    
    results['Linear Regression'] = {
        'model': linear_regression,
        'mae': mae_lr,
        'rmse': rmse_lr,
        'r2': r2_lr,
        'predictions': y_pred_linear_regression,
        'use_scaling': True
    }
    
    # Random Forest
    random_forest = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
    random_forest.fit(X_train, y_train)  # Use unscaled data for Random Forest
    y_pred_random_forest = random_forest.predict(X_test)
    
    mae_rf = mean_absolute_error(y_test, y_pred_random_forest)
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_random_forest))
    r2_rf = r2_score(y_test, y_pred_random_forest)
    
    results['Random Forest'] = {
        'model': random_forest,
        'mae': mae_rf,
        'rmse': rmse_rf,
        'r2': r2_rf,
        'predictions': y_pred_random_forest,
        'use_scaling': False
    }
    
    return results, X_test, y_test, scaler

def get_season(month):
    """Helper function to determine season from month"""
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

def prepare_input_features(input_data, feature_names, scaler=None, use_scaling=False):
    """Prepares input data for prediction."""
    input_df = pd.DataFrame([input_data])
    
    # Ensure all expected features are present, fill missing with 0
    input_df = input_df.reindex(columns=feature_names, fill_value=0)
    
    if use_scaling and scaler:
        input_scaled = scaler.transform(input_df)
        return input_scaled
    else:
        return input_df

def main():
    # Header
    st.markdown('<h1 class="main-header">🏨 Hotel Revenue Optimizer</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">AI-Powered Average Daily Rate (ADR) Prediction System</p>', unsafe_allow_html=True)

    
    # Load data only once
    data, X_train_data, y_train_data = load_and_preprocess_data()
    
    if data is None or X_train_data is None or y_train_data is None:
        st.error("Error loading or preprocessing data. Please check the data source and code.")
        return
    
    # Clear the main content area for each page
    # Page routing - each page will be completely separate
    if page == "🏠 Home":
        # Clear sidebar content for this page if needed
        st.sidebar.empty()
        show_home_page(data)
    
    elif page == "📊 Data Exploration":
        # Add page-specific sidebar content if needed
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Data Filters")
        # You can add filters here for the data exploration page
        show_data_exploration(data)
    
    elif page == "🤖 Model Training":
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Training Options")
        # Add training-specific options here
        show_model_training(X_train_data, y_train_data)
    
    elif page == "💰 Price Prediction":
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Quick Actions")
        if st.sidebar.button("Reset Form"):
            st.experimental_rerun()
        show_prediction_page(data)
    
    elif page == "📈 Performance Dashboard":
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Dashboard Options")
        # Add dashboard-specific options here
        show_performance_dashboard(data)
    

def show_Home(data):
    """Display the home page"""
    st.markdown('<h2 class="sub-header">Welcome to Hotel Revenue Optimizer</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Bookings", f"{len(data):,}")
    
    with col2:
        avg_adr = data['adr'].mean()
        st.metric("Average Daily Rate", f"${avg_adr:.2f}")
    
    with col3:
        if 'is_canceled' in data.columns:
            cancellation_rate = data['is_canceled'].mean() * 100
        else:
            cancellation_rate = 15.0  # Default value for sample data
        st.metric("Cancellation Rate", f"{cancellation_rate:.1f}%")
    
    st.markdown("""
    ### 🎯 Business Problem
    
    This AI agent helps hotels optimize their pricing strategy by predicting the Average Daily Rate (ADR)
    based on various booking characteristics and market conditions. By accurately forecasting room rates,
    hotels can maximize revenue while maintaining competitive pricing.
    
    ### 🔍 Key Features
    
    - **Real-time Price Prediction**: Get instant ADR predictions for new bookings
    - **Multiple ML Models**: Compare performance across different algorithms  
    - **Interactive Dashboard**: Explore data insights and model performance
    - **Business Intelligence**: Make data-driven pricing decisions
    """)
    
    st.markdown('<div class="warning-box">💡 <strong>Getting Started:</strong> Navigate to "Model Training" to train your AI models, then use "Price Prediction" for real-time forecasts!</div>', unsafe_allow_html=True)

def show_data_exploration(data):
    """Display data exploration page"""
    st.markdown('<h2 class="sub-header">📊 Data Exploration</h2>', unsafe_allow_html=True)
    
    # Dataset overview
    st.subheader("Dataset Overview")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Dataset Shape:**", data.shape)
        st.write("**Features:**", data.shape[1])
        st.write("**Records:**", data.shape[0])
    
    with col2:
        missing_data = data.isnull().sum().sum()
        st.write("**Missing Values:**", missing_data)
        st.write("**Data Types:**", len(data.dtypes.unique()))
        st.write("**Memory Usage:**", f"{data.memory_usage().sum() / 1024**2:.2f} MB")
    
    # ADR Distribution
    st.subheader("Average Daily Rate (ADR) Distribution")
    fig_adr = px.histogram(data, x='adr', nbins=50, title='ADR Distribution')
    st.plotly_chart(fig_adr, use_container_width=True)
    
    # ADR by Hotel Type
    st.subheader("ADR Analysis by Hotel Type")
    fig_hotel = px.box(data, x='hotel', y='adr', title='ADR by Hotel Type')
    st.plotly_chart(fig_hotel, use_container_width=True)
    
    # Correlation heatmap
    st.subheader("Feature Correlations")
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    corr_matrix = data[numeric_cols].corr()
    fig_corr = px.imshow(corr_matrix,
                        title='Correlation Matrix',
                        color_continuous_scale='RdBu_r',
                        aspect='auto')
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Key insights
    st.subheader("Key Insights")
    avg_adr = data['adr'].mean()
    resort_adr = data[data['hotel'] == 'Resort Hotel']['adr'].mean()
    city_adr = data[data['hotel'] == 'City Hotel']['adr'].mean()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Overall Avg ADR", f"${avg_adr:.2f}")
    
    with col2:
        st.metric("Resort Hotel ADR", f"${resort_adr:.2f}")
    
    with col3:
        st.metric("City Hotel ADR", f"${city_adr:.2f}")

def show_model_training(X, y):
    """Display model training page"""
    st.markdown('<h2 class="sub-header">🤖 Model Training</h2>', unsafe_allow_html=True)
    st.write("Train multiple regression models to predict Average Daily Rate (ADR)")
    
    if st.button("🚀 Train Models", key="train_button"):
        with st.spinner("Training models... This may take a few minutes."):
            try:
                # Remove outliers for better training
                Q1 = y.quantile(0.25)
                Q3 = y.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Corrected mask to keep values within the bounds
                mask = (y >= lower_bound) & (y <= upper_bound)
                X_clean = X[mask]
                y_clean = y[mask]
                
                # Train models
                results, X_test, y_test, scaler = train_models(X_clean, y_clean)
                
                # Store in session state
                st.session_state.model_results = results
                st.session_state.X_test = X_test
                st.session_state.y_test = y_test
                st.session_state.scaler = scaler
                st.session_state.feature_names = X_clean.columns.tolist()  # Store feature names from clean data
                st.session_state.model_trained = True
                
                # Find best model
                best_model_name = min(results.keys(), key=lambda k: results[k]['rmse'])
                st.session_state.best_model_name = best_model_name
                st.session_state.best_model = results[best_model_name]['model']
                st.session_state.use_scaling = results[best_model_name]['use_scaling']
                
                st.success("✅ Models trained successfully!")
                
                # Display results
                st.subheader("Model Performance Comparison")
                
                # Create performance dataframe
                perf_data = []
                for name, result in results.items():
                    perf_data.append({
                        'Model': name,
                        'MAE': result['mae'],
                        'RMSE': result['rmse'],
                        'R²': result['r2']
                    })
                
                perf_df = pd.DataFrame(perf_data)
                st.dataframe(perf_df.round(4))
                
                # Best model highlight
                st.markdown(f'<div class="prediction-box"><strong>🏆 Best Model: {best_model_name}</strong><br>'
                           f'RMSE: {results[best_model_name]["rmse"]:.2f} | '
                           f'R²: {results[best_model_name]["r2"]:.3f}</div>',
                           unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error during training: {str(e)}")
    
    if st.session_state.model_trained:
        st.success("Models are ready! Go to 'Price Prediction' to make forecasts.")

def show_prediction_page(data):
    """Display prediction page"""
    st.markdown('<h2 class="sub-header">💰 Price Prediction</h2>', unsafe_allow_html=True)
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Please train models first in the 'Model Training' section.")
        return
    
    st.write("Enter booking details to get an ADR prediction:")
    
    # Input form
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            hotel = st.selectbox("Hotel Type", options=['Resort Hotel', 'City Hotel'])
            lead_time = st.number_input("Lead Time (days)", min_value=0, max_value=500, value=50)
            arrival_year = st.number_input("Arrival Year", min_value=2015, max_value=2025, value=2024)
            arrival_month = st.selectbox("Arrival Month",
                                       options=['January', 'February', 'March', 'April', 'May', 'June',
                                               'July', 'August', 'September', 'October', 'November', 'December'])
            arrival_week = st.number_input("Week Number", min_value=1, max_value=53, value=26)
        
        with col2:
            arrival_day = st.number_input("Day of Month", min_value=1, max_value=31, value=15)
            weekend_nights = st.number_input("Weekend Nights", min_value=0, max_value=10, value=1)
            week_nights = st.number_input("Week Nights", min_value=0, max_value=20, value=2)
            adults = st.number_input("Adults", min_value=0, max_value=10, value=2)
            children = st.number_input("Children", min_value=0, max_value=5, value=0)
        
        with col3:
            babies = st.number_input("Babies", min_value=0, max_value=3, value=0)
            meal = st.selectbox("Meal Plan", options=['BB', 'FB', 'HB', 'SC', 'Undefined'])
            market_segment = st.selectbox("Market Segment",
                                        options=['Direct', 'Corporate', 'Online TA', 'Offline TA/TO',
                                                'Complementary', 'Groups', 'Aviation'])
            distribution_channel = st.selectbox("Distribution Channel",
                                               options=['Direct', 'Corporate', 'TA/TO', 'Undefined', 'GDS'])
            customer_type = st.selectbox("Customer Type",
                                       options=['Transient', 'Contract', 'Transient-Party', 'Group'])
        
        # Additional features
        st.subheader("Additional Details")
        col4, col5 = st.columns(2)
        
        with col4:
            is_repeated_guest = st.selectbox("Repeated Guest", options=[0, 1])
            previous_cancellations = st.number_input("Previous Cancellations", min_value=0, max_value=20, value=0)
            booking_changes = st.number_input("Booking Changes", min_value=0, max_value=20, value=0)
            days_waiting = st.number_input("Days in Waiting List", min_value=0, max_value=100, value=0)
        
        with col5:
            parking_spaces = st.number_input("Parking Spaces Required", min_value=0, max_value=5, value=0)
            special_requests = st.number_input("Special Requests", min_value=0, max_value=10, value=0)
            previous_bookings = st.number_input("Previous Bookings Not Canceled", min_value=0, max_value=50, value=0)
            deposit_type = st.selectbox("Deposit Type", options=['No Deposit', 'Non Refund', 'Refundable'])
        
        # Get unique values from data for room types
        reserved_room_type = st.selectbox("Reserved Room Type", options=sorted(data['reserved_room_type'].unique()))
        assigned_room_type = st.selectbox("Assigned Room Type", options=sorted(data['assigned_room_type'].unique()))
        reservation_status = st.selectbox("Reservation Status", options=['Check-Out', 'Canceled', 'No-Show'])
        
        submit_button = st.form_submit_button("🔮 Predict ADR")
    
    if submit_button:
        try:
            # Prepare input data
            month_mapping = {
                'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
                'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
            }
            
            input_data = {
                'lead_time': lead_time,
                'arrival_date_year': arrival_year,
                'arrival_month_numeric': month_mapping[arrival_month],
                'arrival_date_week_number': arrival_week,
                'arrival_date_day_of_month': arrival_day,
                'total_of_special_requests': special_requests,
                'total_guests': adults + children + babies,
                'total_nights': weekend_nights + week_nights,
            }
            
            # Add derived features to input_data
            input_data['booking_changes_per_day'] = input_data['booking_changes'] / (input_data['lead_time'] + 1)
            input_data['adr_per_guest'] = 0 if input_data['total_guests'] == 0 else 100 / input_data['total_guests']  # Handle zero guests
            
            # Add encoded categorical features
            categorical_cols_to_encode = {
                'hotel': hotel,
                'meal': meal,
                'country': 'Unknown',  # Country is complex to add as a single input, using 'Unknown' as placeholder
                'market_segment': market_segment,
                'distribution_channel': distribution_channel,
                'reserved_room_type': reserved_room_type,
                'assigned_room_type': assigned_room_type,
                'deposit_type': deposit_type,
                'customer_type': customer_type,
                'arrival_season': get_season(input_data['arrival_month_numeric']),
                'arrival_date_month': arrival_month,
                'reservation_status': reservation_status
            }
            
            # One-hot encode the input categorical features based on the training data's columns
            input_df_encoded = pd.DataFrame([input_data])
            for col, value in categorical_cols_to_encode.items():
                if f'{col}_{value}' in st.session_state.feature_names:
                    input_df_encoded[f'{col}_{value}'] = 1
            
            # Prepare input data using the helper function
            input_prepared = prepare_input_features(input_df_encoded.iloc[0].to_dict(),
                                                   st.session_state.feature_names,
                                                   st.session_state.scaler,
                                                   st.session_state.use_scaling)
            
            # Make prediction
            model = st.session_state.best_model
            prediction = model.predict(input_prepared)[0]
            
            # Display prediction
            st.markdown(f"""
            <div class="prediction-box">
                <h3>💰 Predicted Average Daily Rate</h3>
                <h2 style="color: #2196f3; font-size: 3rem;">${prediction:.2f}</h2>
                <p>Model: {st.session_state.best_model_name}</p>
                <p>Confidence: Based on historical patterns and current market conditions</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Business insights
            avg_adr = data['adr'].mean()
            if prediction > avg_adr * 1.2:
                st.success("🔥 **Premium Pricing Opportunity** - This booking commands above-average rates!")
            elif prediction < avg_adr * 0.8:
                st.info("💡 **Value Pricing** - Consider promotional offers or package deals.")
            else:
                st.info("📊 **Market Rate** - Pricing aligns with typical market conditions.")
            
            # Additional metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("vs. Average ADR", f"${prediction - avg_adr:.2f}", f"{((prediction/avg_adr - 1) * 100):.1f}%")
            
            with col2:
                revenue_estimate = prediction * (weekend_nights + week_nights)
                st.metric("Estimated Revenue", f"${revenue_estimate:.2f}")
            
            with col3:
                revenue_per_guest = revenue_estimate / (adults + children + babies) if (adults + children + babies) > 0 else 0
                st.metric("Revenue per Guest", f"${revenue_per_guest:.2f}")
                
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")

def show_performance_dashboard(data):
    """Display performance dashboard"""
    st.markdown('<h2 class="sub-header">📈 Performance Dashboard</h2>', unsafe_allow_html=True)
    
    if not st.session_state.model_trained or st.session_state.model_results is None:
        st.warning("⚠️ Please train models first to view performance metrics.")
        return
    
    # Model performance metrics
    st.subheader("Model Performance Summary")
    results = st.session_state.model_results
    
    # Create performance comparison chart
    perf_data = []
    for name, result in results.items():
        perf_data.append({
            'Model': name,
            'MAE': result['mae'],
            'RMSE': result['rmse'],
            'R²': result['r2']
        })
    
    perf_df = pd.DataFrame(perf_data)
    
    # Performance visualization
    col1, col2 = st.columns(2)
    
    with col1:
        fig_rmse = px.bar(perf_df, x='Model', y='RMSE',
                         title='Root Mean Square Error by Model',
                         color='RMSE', color_continuous_scale='Viridis_r')
        st.plotly_chart(fig_rmse, use_container_width=True)
    
    with col2:
        fig_r2 = px.bar(perf_df, x='Model', y='R²',
                       title='R² Score by Model',
                       color='R²', color_continuous_scale='Viridis')
        st.plotly_chart(fig_r2, use_container_width=True)
    
    # Detailed metrics table
    st.subheader("Detailed Performance Metrics")
    st.dataframe(perf_df.round(4))
    
    # Feature importance (for tree-based models)
    st.subheader("Feature Importance Analysis")
    if st.session_state.best_model_name in ['Random Forest', 'Gradient Boosting']:
        # Ensure feature_names are available and match model's feature importances
        if (st.session_state.feature_names and 
            hasattr(st.session_state.best_model, 'feature_importances_') and
            len(st.session_state.feature_names) == len(st.session_state.best_model.feature_importances_)):
            
            importance = st.session_state.best_model.feature_importances_
            feature_importance = pd.DataFrame({
                'Feature': st.session_state.feature_names,
                'Importance': importance
            }).sort_values('Importance', ascending=False)
            
            fig_importance = px.bar(feature_importance.head(10),
                                  x='Importance', y='Feature',
                                  orientation='h',
                                  title='Top 10 Feature Importance')
            st.plotly_chart(fig_importance, use_container_width=True)
        else:
            st.warning("Could not display feature importance. Feature names or importance values mismatch.")
    else:
        st.info("Feature importance is available only for tree-based models.")
    
    # Business impact metrics
    st.subheader("Business Impact Analysis")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        best_r2 = results[st.session_state.best_model_name]['r2']
        st.metric("Model Accuracy (R²)", f"{best_r2:.3f}")
    
    with col2:
        best_rmse = results[st.session_state.best_model_name]['rmse']
        st.metric("Prediction Error (RMSE)", f"${best_rmse:.2f}")
    
    with col3:
        avg_adr = data['adr'].mean()
        error_percentage = (best_rmse / avg_adr) * 100 if avg_adr > 0 else 0  # Handle division by zero
        st.metric("Error Percentage", f"{error_percentage:.1f}%")
    
    with col4:
        # Estimate potential revenue impact
        # This is a simplified estimation and might need refinement based on business logic
        potential_improvement = best_r2 * avg_adr * 0.05 if avg_adr > 0 else 0  # Assume 5% improvement potential
        st.metric("Revenue Optimization (Est.)", f"${potential_improvement:.2f}")

# Run the main function
if __name__ == "__main__":
    main()
