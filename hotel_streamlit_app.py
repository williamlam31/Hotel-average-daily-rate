
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

@st.cache_data
def load():
import kagglehub
from kagglehub import KaggleDatasetAdapter
file_path = "hotel_bookings.csv"

df = kagglehub.load_dataset(
 	 KaggleDatasetAdapter.PANDAS,
 	 "jessemostipak/hotel-booking-demand",
 	 "hotel_bookings.csv")        
        # Data preprocessing
        df_new = df.copy()
        
        df_new['children'].fillna(0, inplace=True)
        df_new['country'].fillna('Unknown', inplace=True)
        df_new.drop_duplicates(inplace = True)
        df_new.drop(['agent', 'company'], axis=1, inplace=True)
        df_new['reservation_status_date'] = df_new['reservation_status_date'].astype(str)
        df_new.drop_duplicates(inplace = True)
        
        # Create derived features
        df_new['total_nights'] = df_new['stays_in_weekend_nights'] +    df_new['stays_in_week_nights']
        df_new['total_guests'] = df_new['adults'] + df_new['children'] + df_new['babies']
        # Encode categorical variables
categorical_columns = ['hotel', 'meal', 'country', 'market_segment', 'distribution_channel',
                       'reserved_room_type', 'assigned_room_type', 'deposit_type', 'customer_type', 'arrival_season', 'arrival_date_month', 'reservation_status']

df_new_encoded = pd.get_dummies(df_new, columns=categorical_columns, drop_first=True)
df_new_encoded = df_new_encoded.drop(['reservation_status_date'], axis=1)        
        # Month encoding
        month_assignment = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
             'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12}
        df_new['arrival_month'] = df_new['arrival_date_month'].map(month_assignment)
        def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'
df_new['arrival_season'] = df_new['arrival_month'].apply(get_season)    
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None, None, None, None

def prepare_features(df):
    """Prepare features for modeling"""
    feature_columns = [
        'lead_time', 'arrival_date_year', 'arrival_month_numeric',
        'arrival_date_week_number', 'arrival_date_day_of_month',
        'stays_in_weekend_nights', 'stays_in_week_nights', 'adults',
        'children', 'babies', 'is_repeated_guest', 'previous_cancellations',
        'previous_bookings_not_canceled', 'booking_changes',
        'days_in_waiting_list', 'required_car_parking_spaces',
        'total_of_special_requests', 'hotel_encoded', 'meal_encoded',
        'market_segment_encoded', 'distribution_channel_encoded',
        'customer_type_encoded', 'total_guests', 'total_stay',
        'booking_changes_per_day', 'adr_per_guest'
    ]
    
    # Filter only existing columns
    available_features = [col for col in feature_columns if col in df.columns]
    
    return df[available_features]

def train_models(X, y):
    """Train multiple regression models"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize models
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
   }
    
    results = {}
    
    # Train and evaluate models
    for name, model in models.items():
        if name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression']:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            'model': model,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'predictions': y_pred,
            'use_scaling': name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression']
        }
    
    return results, X_test, y_test, scaler

def main():
    # Header
    st.markdown('<h1 class="main-header">🏨 Hotel Revenue Optimizer</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">AI-Powered Average Daily Rate (ADR) Prediction System</p>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", 
                               ["🏠 Home", "📊 Data Exploration", "🤖 Model Training", "💰 Price Prediction", "📈 Performance Dashboard"])
    
    # Load data
    data, le_hotel, le_meal, le_market_segment, le_distribution_channel, le_customer_type = load_and_preprocess_data()
    
    if data is None:
        st.error("Please ensure 'hotel_bookings_edited.csv' is in the same directory as this app.")
        return
    
    # Page routing
    if page == "🏠 Home":
        show_home_page(data)
    elif page == "📊 Data Exploration":
        show_data_exploration(data)
    elif page == "🤖 Model Training":
        show_model_training(data)
    elif page == "💰 Price Prediction":
        show_prediction_page(data, le_hotel, le_meal, le_market_segment, le_distribution_channel, le_customer_type)
    elif page == "📈 Performance Dashboard":
        show_performance_dashboard(data)

def show_home_page(data):
    """Display the home page"""
    st.markdown('<h2 class="sub-header">Welcome to Hotel Revenue Optimizer</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Bookings", f"{len(data):,}")
    with col2:
        avg_adr = data['adr'].mean()
        st.metric("Average Daily Rate", f"${avg_adr:.2f}")
    with col3:
        cancellation_rate = data['is_canceled'].mean() * 100
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
    
    ### 📈 How It Works
    1. **Data Analysis**: Analyze historical booking patterns and pricing trends
    2. **Model Training**: Train multiple regression models on your data
    3. **Price Prediction**: Input booking details to get ADR predictions
    4. **Performance Monitoring**: Track model accuracy and business impact
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

def show_model_training(data):
    """Display model training page"""
    st.markdown('<h2 class="sub-header">🤖 Model Training</h2>', unsafe_allow_html=True)
    
    st.write("Train multiple regression models to predict Average Daily Rate (ADR)")
    
    if st.button("🚀 Train Models", key="train_button"):
        with st.spinner("Training models... This may take a few minutes."):
            try:
                # Prepare features and target
                X = prepare_features(data)
                y = data['adr']
                
                # Remove outliers for better training
                Q1 = y.quantile(0.25)
                Q3 = y.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
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
                st.session_state.feature_names = X.columns.tolist()
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

def show_prediction_page(data, le_hotel, le_meal, le_market_segment, le_distribution_channel, le_customer_type):
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
            adults = st.number_input("Adults", min_value=1, max_value=10, value=2)
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
        
        submit_button = st.form_submit_button("🔮 Predict ADR")
    
    if submit_button:
        try:
            # Prepare input data
            month_mapping = {
                'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
                'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
            }
            
            # Create input dataframe
            input_data = {
                'lead_time': lead_time,
                'arrival_date_year': arrival_year,
                'arrival_month_numeric': month_mapping[arrival_month],
                'arrival_date_week_number': arrival_week,
                'arrival_date_day_of_month': arrival_day,
                'stays_in_weekend_nights': weekend_nights,
                'stays_in_week_nights': week_nights,
                'adults': adults,
                'children': children,
                'babies': babies,
                'is_repeated_guest': is_repeated_guest,
                'previous_cancellations': previous_cancellations,
                'previous_bookings_not_canceled': previous_bookings,
                'booking_changes': booking_changes,
                'days_in_waiting_list': days_waiting,
                'required_car_parking_spaces': parking_spaces,
                'total_of_special_requests': special_requests,
                'hotel_encoded': 1 if hotel == 'Resort Hotel' else 0,
                'meal_encoded': {'BB': 0, 'FB': 1, 'HB': 2, 'SC': 3, 'Undefined': 4}.get(meal, 0),
                'market_segment_encoded': {'Direct': 0, 'Corporate': 1, 'Online TA': 2, 'Offline TA/TO': 3, 
                                         'Complementary': 4, 'Groups': 5, 'Aviation': 6}.get(market_segment, 0),
                'distribution_channel_encoded': {'Direct': 0, 'Corporate': 1, 'TA/TO': 2, 'Undefined': 3, 'GDS': 4}.get(distribution_channel, 0),
                'customer_type_encoded': {'Transient': 0, 'Contract': 1, 'Transient-Party': 2, 'Group': 3}.get(customer_type, 0),
                'total_guests': adults + children + babies,
                'total_stay': weekend_nights + week_nights,
                'booking_changes_per_day': booking_changes / (lead_time + 1),
                'adr_per_guest': 100 / (adults + children + babies)  # Default assumption
            }
            
            # Create dataframe with correct column order
            input_df = pd.DataFrame([input_data])
            input_df = input_df.reindex(columns=st.session_state.feature_names, fill_value=0)
            
            # Make prediction
            model = st.session_state.best_model
            
            if st.session_state.use_scaling:
                input_scaled = st.session_state.scaler.transform(input_df)
                prediction = model.predict(input_scaled)[0]
            else:
                prediction = model.predict(input_df)[0]
            
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
    
    if not st.session_state.model_trained:
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
        error_percentage = (best_rmse / avg_adr) * 100
        st.metric("Error Percentage", f"{error_percentage:.1f}%")
    
    with col4:
        # Estimate potential revenue impact
        potential_improvement = best_r2 * avg_adr * 0.05  # Assume 5% improvement potential
        st.metric("Revenue Optimization", f"${potential_improvement:.2f}")
