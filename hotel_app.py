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
    
    file_path = "hotel_bookings.csv"
    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "jessemostipak/hotel-booking-demand",
        "hotel_bookings.csv")
    
    df_new = df.copy()

    #Replacing Null values with 0 for numerical and Unknown for Categorical
    df_new['children'].fillna(0, inplace=True)
    df_new['country'].fillna('Unknown', inplace=True)

    #Dropping Duplicates
    df_new.drop_duplicates(inplace = True)
    df_new.drop(['agent', 'company'], axis=1, inplace=True)
    df_new['reservation_status_date'] = df_new['reservation_status_date'].astype(str)
    df_new.drop_duplicates(inplace = True)

    #New Features
    df_new['total_nights'] = df_new['stays_in_weekend_nights'] + df_new['stays_in_week_nights']
    df_new['total_guests'] = df_new['adults'] + df_new['children'] + df_new['babies']
    
    # Month Mapping
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
    
    # One-Hot Encoding
    categorical_features = ['hotel', 'meal', 'country', 'market_segment', 'distribution_channel',
                               'reserved_room_type', 'assigned_room_type', 'deposit_type', 'customer_type', 
                               'arrival_season', 'reservation_status']
    
    df_new_encoded = pd.get_dummies(df_new, columns=categorical_features, drop_first=True)

    
    numerical_features = [
        'lead_time', 'arrival_month_numeric',
        'days_in_waiting_list', 'total_of_special_requests', 'total_guests', 'total_nights']
    
   
    X = df_new[numerical_features]
    y = df_new_encoded['adr']
    
    return df_new_encoded

def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standard Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results = {}
    
    # Linear Regression
    linear_regression = LinearRegression()
    linear_regression.fit(X_train_scaled, y_train) 
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
    random_forest.fit(X_train, y_train) 
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
    
    return results

def prepare_input_features(input_data, feature_names, scaler=None, use_scaling=False):
    input_df = pd.DataFrame([input_data])
    input_df = input_df.reindex(columns=feature_names, fill_value=0)
    
    if use_scaling and scaler:
        input_scaled = scaler.transform(input_df)
        return input_scaled
    else:
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
        best_model_name = max(models, key=lambda m: models[m]['r2'])
        model = models[best_model_name]['model']
        scaler = models[best_model_name]['scaler']
        use_scaling = models[best_model_name]['use_scaling']
                        
    if page == "Home":
        show_Home(df_new_encoded)
    elif page == "Average Daily Rate":
        show_Average_Daily_Rate(df_new_encoded)
    elif page == "Data Exploration":
        show_data_exploration(df_new_encoded)
    elif page == "Performance Dashboard":
        show_Performance_Dashboard(df_new_encoded)
    

def show_Home(df_new_encoded):
    """Display the home page"""
    
    st.markdown('<h1 class="main-header"> Hotel Average Daily Rate</h1>', unsafe_allow_html=True)

    
    st.markdown("""
    ### Business Problem
    
    Hotel prices are dyanmic and influenced by factors such as lead time and seasonality. Since the hospitality industry is very competitive, this AI tool is 
    to assist the hotel in determining an optimal rate that will attract guests and improve their financial outlook.
    
    ### Explanation of Side Pages
    
    - **Average Daily Rate**: Average Daily Rate estimate
    - **Performance Dashboard**: Key metrics and visualiztions
    - **Data Exploration**: Key insights from analysis
    """)
    
def show_data_exploration(data):
    """Display data exploration page"""
    st.markdown('<h1 class="main-header">📊 Data Exploration</h1>', unsafe_allow_html=True)
    
    # ADR vs Month Analysis
    st.subheader("Average Daily Rate (ADR) by Month")
    
    # Create month order for proper sorting
    month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']
    
    # Calculate average ADR by month
    adr_by_month = data.groupby('arrival_date_month')['adr'].agg(['mean', 'median', 'std']).reset_index()
    adr_by_month['arrival_date_month'] = pd.Categorical(adr_by_month['arrival_date_month'], 
                                                        categories=month_order, 
                                                        ordered=True)
    adr_by_month = adr_by_month.sort_values('arrival_date_month')
    
    # Create the visualization
    fig_month = go.Figure()
    
    # Add average ADR line
    fig_month.add_trace(go.Scatter(
        x=adr_by_month['arrival_date_month'],
        y=adr_by_month['mean'],
        mode='lines+markers',
        name='Average ADR',
        line=dict(color='#2196F3', width=3),
        marker=dict(size=8)
    ))
    
    # Add median ADR line
    fig_month.add_trace(go.Scatter(
        x=adr_by_month['arrival_date_month'],
        y=adr_by_month['median'],
        mode='lines+markers',
        name='Median ADR',
        line=dict(color='#FF9800', width=2, dash='dash'),
        marker=dict(size=6)
    ))
    
    # Update layout
    fig_month.update_layout(
        title='Average Daily Rate by Month',
        xaxis_title='Month',
        yaxis_title='ADR ($)',
        hovermode='x unified',
        template='plotly_white'
    )
    
    st.plotly_chart(fig_month, use_container_width=True)
    
    # Show seasonal insights
    col1, col2 = st.columns(2)
    with col1:
        peak_month = adr_by_month.loc[adr_by_month['mean'].idxmax(), 'arrival_date_month']
        peak_adr = adr_by_month['mean'].max()
        st.metric("Peak Month", peak_month, f"${peak_adr}")
    
    with col2:
        low_month = adr_by_month.loc[adr_by_month['mean'].idxmin(), 'arrival_date_month']
        low_adr = adr_by_month['mean'].min()
        st.metric("Lowest Month", low_month, f"${low_adr:.2f}")
    
    st.subheader("Lead Time and ADR by Month")
    
    # Create month order for proper sorting
    month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']
    
    # Calculate average lead time and ADR by month
    monthly_stats = data.groupby('arrival_date_month').agg({
        'lead_time': 'mean',
        'adr': 'mean'
    }).reset_index()
    
    # Reorder by month
    monthly_stats['arrival_date_month'] = pd.Categorical(monthly_stats['arrival_date_month'], 
                                                        categories=month_order, 
                                                        ordered=True)
    monthly_stats = monthly_stats.sort_values('arrival_date_month')
    
    # Create dual-axis chart
    fig_dual = go.Figure()
    
    # Add Lead Time bars
    fig_dual.add_trace(go.Bar(
        x=monthly_stats['arrival_date_month'],
        y=monthly_stats['lead_time'],
        name='Average Lead Time',
        marker_color='#2196F3',
        opacity=0.7,
        yaxis='y',
        hovertemplate='<b>%{x}</b><br>Lead Time: %{y:.0f} days<extra></extra>'
    ))
    
    # Add ADR line on secondary y-axis
    fig_dual.add_trace(go.Scatter(
        x=monthly_stats['arrival_date_month'],
        y=monthly_stats['adr'],
        mode='lines+markers',
        name='Average ADR',
        line=dict(color='#FF9800', width=3),
        marker=dict(size=8),
        yaxis='y2',
        hovertemplate='<b>%{x}</b><br>ADR: $%{y:.2f}<extra></extra>'
    ))
    
    # Update layout with dual y-axes
    fig_dual.update_layout(
        title='Lead Time (Days) and ADR ($) by Month',
        xaxis_title='Month',
        yaxis=dict(
            title='Lead Time (Days)',
            side='left',
            color='#2196F3'
        ),
        yaxis2=dict(
            title='ADR ($)',
            side='right',
            overlaying='y',
            color='#FF9800'
        ),
        template='plotly_white',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    st.plotly_chart(fig_dual, use_container_width=True)
    
    # Add summary insights
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_lead_time = data['lead_time'].mean()
        st.metric("Avg Lead Time", f"{avg_lead_time:.0f} days")
    
    with col2:
        # Find correlation between lead time and ADR
        correlation = data['lead_time'].corr(data['adr'])
        st.metric("Lead Time-ADR Correlation", f"{correlation:.3f}")
    
    with col3:
        # Find month with highest average lead time
        lead_by_month = data.groupby('arrival_date_month')['lead_time'].mean()
        highest_lead_month = lead_by_month.idxmax()
        st.metric("Highest Lead Time Month", highest_lead_month)
    
    # Correlation heatmap
    st.subheader("Feature Correlations")
    # Select only the specified features for correlation analysis
    selected_features = [
        'lead_time', 'total_nights', 'total_guests', 
        'total_of_special_requests', 'days_in_waiting_list', 
        'arrival_month_numeric', 'adr'
    ]
    
    # Filter to only include features that exist in the data
    available_features = [col for col in selected_features if col in data.columns]
    corr_matrix = data[available_features].corr()
    
    # Create custom labels for better readability
    feature_labels = {
        'lead_time': 'Lead Time',
        'total_nights': 'Total Nights',
        'total_guests': 'Total Guests',
        'total_of_special_requests': 'Special Requests',
        'days_in_waiting_list': 'Days in Waiting List',
        'arrival_month_numeric': 'Arrival Month',
        'adr': 'ADR'
    }
    
    # Rename columns and index for display
    corr_display = corr_matrix.copy()
    corr_display.columns = [feature_labels.get(col, col) for col in corr_display.columns]
    corr_display.index = [feature_labels.get(idx, idx) for idx in corr_display.index]
    
    fig_corr = px.imshow(corr_display,
                        title='Feature Correlation Matrix (Select Variables)',
                        color_continuous_scale='RdBu_r',
                        aspect='auto',
                        text_auto=True)
    
    # Update layout for better readability
    fig_corr.update_layout(
        width=600,
        height=500
    )
    
    st.plotly_chart(fig_corr, use_container_width=True)
    
   

def show_Average_Daily_Rate(data):
    """Display prediction page"""
    st.markdown('<h1 class="main-header">💰 Average Daily Rate </h1>', unsafe_allow_html=True)
    
    st.markdown(
    "<style>" +
    ".element-container button.step-up { display: none; } " +
    ".element-container button.step-down { display: none; } " +
    ".element-container div[data-baseweb] { border-radius: 4px; } "
    "</style>",
    unsafe_allow_html=True
    )

    
    st.write("Enter booking details:")
    
    # Input form
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            arrival_month = st.selectbox("Arrival Month",
                                       options=['January', 'February', 'March', 'April', 'May', 'June',
                                               'July', 'August', 'September', 'October', 'November', 'December'])
            total_nights = st.number_input("Total Nights", value = 0)
        
        with col2:
            total_guests = st.number_input("Total Guests", value = 0)
            days_waiting = st.number_input("Days in Waiting List", value = 0)
        with col3:
            special_requests = st.number_input("Total of Special Requests", value = 0)
            lead_time = st.number_input("Lead Time", value = 0)
        
        submit_button = st.form_submit_button("🔮 Predict ADR")
    
    if submit_button:
        try:
            month_mapping = {
                'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
                'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
            }
            
            input_data = {
                'lead_time': lead_time,
                'arrival_month_numeric': month_mapping[arrival_month],
                'total_of_special_requests': special_requests,
                'total_guests': total_guests,
                'total_nights': total_nights,
                'days_in_waiting_list': days_waiting,
            }
            input_processed = prepare_input_features(input_data, selected_features, scaler, use_scaling)

            prediction = model.predict(input_processed)[0]
            
            st.markdown(f"""
            <div style="background-color: #ffffff; color: #000000; padding: 2rem; border-radius: 1rem; border-left: 5px solid #2196f3; margin: 1rem 0;">
                <h3 style="color: #000000;"> Predicted Average Daily Rate</h3>
                <h2 style="color: #000000; font-size: 3rem;">${prediction:.2f}</h2>
       
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
                revenue_estimate = prediction * total_nights
                st.metric("Estimated Revenue", f"${revenue_estimate:.2f}")
            
            with col3:
                revenue_per_guest = revenue_estimate / total_guests if total_guests > 0 else 0
                st.metric("Revenue per Guest", f"${revenue_per_guest:.2f}")
                
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")

def show_Performance_Dashboard(data):
    st.markdown('<h1 class="main-header">📈 Performance Dashboard</h1>', unsafe_allow_html=True)

    
    # Focus on the 6 key features used in the model
    key_features = ['lead_time', 'total_nights', 'total_guests', 
                   'total_of_special_requests', 'days_in_waiting_list', 'arrival_month_numeric']
    
    # SECTION 1: Key Pricing Insights
    st.write("### Pricing Patterns")
    
    # ADR statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        avg_adr = data['adr'].mean()
        st.metric("Average ADR", f"${avg_adr:.2f}")
    
    with col2:
        median_adr = data['adr'].median()
        st.metric("Median ADR", f"${median_adr:.2f}")
    
    with col3:
        adr_std = data['adr'].std()
        st.metric("ADR Std Dev", f"${adr_std:.2f}")
    
    with col4:
        max_adr = data['adr'].max()
        st.metric("Peak ADR", f"${max_adr:.2f}")
    
    # Seasonal pricing insights
    st.write("**Seasonal Pricing Trends:**")
    monthly_adr = data.groupby('arrival_date_month')['adr'].mean().round(2)
    
    peak_season = monthly_adr.idxmax()
    low_season = monthly_adr.idxmin()
    
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"🔥 **Peak Season**: {peak_season} (${monthly_adr[peak_season]:.2f} avg)")
    
    with col2:
        st.info(f"❄️ **Low Season**: {low_season} (${monthly_adr[low_season]:.2f} avg)")
    
    # SECTION 2: Feature Impact Analysis
    st.write("###  Feature Impact Analysis")
    
    # Calculate correlations with ADR for key features
    correlations = {}
    for feature in key_features:
        if feature in data.columns:
            corr = data[feature].corr(data['adr'])
            correlations[feature] = corr
    corr_df = pd.DataFrame(list(correlations.items()), columns=['Feature', 'Correlation with ADR'])
    corr_df = corr_df.sort_values('Correlation with ADR', key=abs, ascending=False)
    
    # Rename features for better display
    feature_display_names = {
        'lead_time': 'Lead Time',
        'total_nights': 'Total Nights',
        'total_guests': 'Total Guests',
        'total_of_special_requests': 'Special Requests',
        'days_in_waiting_list': 'Days in Waiting List',
        'arrival_month_numeric': 'Arrival Month'
    }
    
    corr_df['Feature_Display'] = corr_df['Feature'].map(feature_display_names)
    
    fig_corr = px.bar(corr_df, x='Correlation with ADR', y='Feature_Display',
                     orientation='h',
                     title='Feature Correlation with Average Daily Rate',
                     color='Correlation with ADR',
                     color_continuous_scale='RdBu_r')
    st.plotly_chart(fig_corr, use_container_width=True)   
    # Feature importance insights
    st.write("**Top Predictive Features:**")
    for idx, row in corr_df.head(3).iterrows():
        correlation_strength = "Strong" if abs(row['Correlation with ADR']) > 0.5 else "Moderate" if abs(row['Correlation with ADR']) > 0.3 else "Weak"
        direction = "Positive" if row['Correlation with ADR'] > 0 else "Negative"
        st.write(f"• **{row['Feature_Display']}**: {correlation_strength} {direction.lower()} correlation ({row['Correlation with ADR']:.3f})")
    
    # SECTION 3: Business Intelligence
    st.write("### Business Intelligence")
    
    # Lead time insights
    avg_lead_time = data['lead_time'].mean()
    high_lead_time_adr = data[data['lead_time'] > avg_lead_time]['adr'].mean()
    low_lead_time_adr = data[data['lead_time'] <= avg_lead_time]['adr'].mean()
    
    st.write("**Booking Behavior:**")
    st.write(f"• Average lead time: {avg_lead_time:.0f} days")
    st.write(f"• High lead time bookings (>{avg_lead_time:.0f} days): ${high_lead_time_adr:.2f} avg ADR")
    st.write(f"• Low lead time bookings (≤{avg_lead_time:.0f} days): ${low_lead_time_adr:.2f} avg ADR")
    
    # Guest composition insights
    st.write(" **Guest Composition:**")
    avg_guests = data['total_guests'].mean()
    avg_nights = data['total_nights'].mean()
    
    st.write(f"• Average party size: {avg_guests:.1f} guests")
    st.write(f"• Average stay duration: {avg_nights:.1f} nights")
    
    # Special requests impact
    special_req_correlation = data['total_of_special_requests'].corr(data['adr'])
    st.write("**Service Level:**")
    st.write(f"• Special requests correlation with ADR: {special_req_correlation:.3f}")
    if special_req_correlation > 0:
        st.write("• Higher service requests typically associated with higher rates")
    else:
        st.write("• Special requests show minimal impact on pricing")
    
    
            
# Run the main function
if __name__ == "__main__":
    main()
