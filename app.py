import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as io
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# Page Configuration
st.set_page_config(
    page_title="Real Estate Analytics & Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
    <style>
    .main { padding-top: 1rem; }
    .stMetric { background-color: #f8f9fa; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
""", unsafe_allow_html=True)

# Data Loading with Cache
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('cleaned_house_price1.csv')
        return df
    except FileNotFoundError:
        st.error("Error: 'cleaned_house_price1.csv' file not found in the current directory.")
        return None

df = load_data()

if df is not None:
    # Sidebar Navigation
    st.sidebar.title("📌 Navigation")
    app_mode = st.sidebar.radio(
        "Choose Section:",
        ["📈 Interactive Dashboard & Insights", "🔮 Price & Investment Predictor", "🔍 Raw Data Explorer"]
    )

    # ---------------------------------------------------------
    # SECTION 1: DASHBOARD & INSIGHTS
    # ---------------------------------------------------------
    if app_mode == "📈 Interactive Dashboard & Insights":
        st.title("🏠 Real Estate Market Dashboard & Insights")
        st.markdown("Explore key metrics, distribution patterns, and value drivers from the house price dataset.")

        # Top High-level KPIs
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Properties", f"{len(df):,}")
        col2.metric("Avg Price (Lakhs)", f"₹{df['Price_in_Lakhs'].mean():.2f} L")
        col3.metric("Avg Area (SqFt)", f"{df['Size_in_SqFt'].mean():.0f} sq.ft")
        col4.metric("Good Investment Ratio", f"{(df['Good_Investment'].mean() * 100):.1f}%")

        st.divider()

        # Filters
        st.subheader("🎛️ Filter Analysis")
        f_col1, f_col2, f_col3 = st.columns(3)
        with f_col1:
            selected_states = st.multiselect("Select State(s)", options=df['State'].unique(), default=df['State'].unique()[:3])
        with f_col2:
            available_cities = df[df['State'].isin(selected_states)]['City'].unique() if selected_states else df['City'].unique()
            selected_cities = st.multiselect("Select City(ies)", options=available_cities, default=available_cities[:3])
        with f_col3:
            selected_prop_type = st.multiselect("Property Type", options=df['Property_Type'].unique(), default=df['Property_Type'].unique())

        filtered_df = df[
            (df['State'].isin(selected_states if selected_states else df['State'].unique())) &
            (df['City'].isin(selected_cities if selected_cities else df['City'].unique())) &
            (df['Property_Type'].isin(selected_prop_type if selected_prop_type else df['Property_Type'].unique()))
        ]

        # Charts Row 1
        st.subheader("📊 Price Analysis")
        c1, c2 = st.columns(2)

        with c1:
            fig_price_city = px.box(
                filtered_df, 
                x="City", 
                y="Price_in_Lakhs", 
                color="Property_Type",
                title="Price Distribution by City & Property Type",
                labels={"Price_in_Lakhs": "Price (Lakhs ₹)"}
            )
            st.plotly_chart(fig_price_city, use_container_width=True)

        with c2:
            fig_scatter = px.scatter(
                filtered_df.sample(min(5000, len(filtered_df))), 
                x="Size_in_SqFt", 
                y="Price_in_Lakhs",
                color="BHK", 
                size="Total_Infrastructure_score",
                hover_data=["Furnished_Status", "Age_of_Property"],
                title="Property Size vs Price (Color: BHK, Size: Infra Score)"
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        # Charts Row 2
        st.subheader("🏬 Infrastructure & Amenities Impact")
        c3, c4 = st.columns(2)

        with c3:
            infra_price = filtered_df.groupby("Total_Infrastructure_score")["Price_in_Lakhs"].mean().reset_index()
            fig_infra = px.line(
                infra_price, 
                x="Total_Infrastructure_score", 
                y="Price_in_Lakhs",
                markers=True, 
                title="Avg Price vs Infrastructure Score"
            )
            st.plotly_chart(fig_infra, use_container_width=True)

        with c4:
            fig_invest = px.histogram(
                filtered_df, 
                x="Furnished_Status", 
                color="Good_Investment",
                barmode="group",
                title="Investment Opportunity by Furnishing Status",
                color_discrete_map={0: "#EF553B", 1: "#00CC96"}
            )
            st.plotly_chart(fig_invest, use_container_width=True)

    # ---------------------------------------------------------
    # SECTION 2: PREDICTION ENGINE
    # ---------------------------------------------------------
    elif app_mode == "🔮 Price & Investment Predictor":
        st.title("🔮 House Price & Investment Predictor")
        st.write("Input property details to predict market price and evaluate investment potential.")

        @st.cache_resource
        def train_models(data):
            # Sample data for fast interactive training
            train_df = data.sample(min(30000, len(data)), random_state=42)
            
            features = [
                'State', 'City', 'Property_Type', 'BHK', 'Size_in_SqFt', 
                'Year_Built', 'Furnished_Status', 'Floor_No', 'Total_Floors',
                'Nearby_Schools', 'Nearby_Hospitals', 'Public_Transport_Accessibility',
                'Parking_Space', 'Security', 'Facing', 'Owner_Type', 
                'Availability_Status', 'Amenities_Count', 'Total_Infrastructure_score'
            ]
            
            cat_features = ['State', 'City', 'Property_Type', 'Furnished_Status', 
                            'Public_Transport_Accessibility', 'Parking_Space', 
                            'Security', 'Facing', 'Owner_Type', 'Availability_Status']
            num_features = [c for c in features if c not in cat_features]

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', 'passthrough', num_features),
                    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
                ]
            )

            # Regression model for Price
            reg_pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('model', RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1))
            ])
            
            # Classification model for Investment
            clf_pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('model', RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1))
            ])

            X = train_df[features]
            y_price = train_df['Price_in_Lakhs']
            y_invest = train_df['Good_Investment']

            reg_pipeline.fit(X, y_price)
            clf_pipeline.fit(X, y_invest)

            return reg_pipeline, clf_pipeline, features

        with st.spinner("Training predictive machine learning models..."):
            reg_model, clf_model, feature_cols = train_models(df)

        st.success("Model trained successfully!")

        st.subheader("📋 Enter Property Details")

        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)

            with col1:
                state = st.selectbox("State", sorted(df['State'].unique()))
                city_options = sorted(df[df['State'] == state]['City'].unique())
                city = st.selectbox("City", city_options)
                prop_type = st.selectbox("Property Type", sorted(df['Property_Type'].unique()))
                bhk = st.number_input("BHK (Bedrooms)", min_value=1, max_value=10, value=3)
                sqft = st.number_input("Size in SqFt", min_value=100, max_value=10000, value=2500)
                year_built = st.number_input("Year Built", min_value=1950, max_value=2026, value=2015)

            with col2:
                furnished = st.selectbox("Furnished Status", sorted(df['Furnished_Status'].unique()))
                floor_no = st.number_input("Floor Number", min_value=0, max_value=100, value=5)
                total_floors = st.number_input("Total Floors", min_value=1, max_value=100, value=15)
                schools = st.slider("Nearby Schools (0-10)", 0, 10, 5)
                hospitals = st.slider("Nearby Hospitals (0-10)", 0, 10, 5)
                transport = st.selectbox("Public Transport", sorted(df['Public_Transport_Accessibility'].unique()))

            with col3:
                parking = st.selectbox("Parking Space", sorted(df['Parking_Space'].unique()))
                security = st.selectbox("Security", sorted(df['Security'].unique()))
                facing = st.selectbox("Facing", sorted(df['Facing'].unique()))
                owner_type = st.selectbox("Owner Type", sorted(df['Owner_Type'].unique()))
                avail = st.selectbox("Availability Status", sorted(df['Availability_Status'].unique()))
                amenities_cnt = st.slider("Amenities Count", 1, 10, 4)

            submit_btn = st.form_submit_button("🚀 Predict Valuation & Investment Class")

        if submit_btn:
            infra_score = schools + hospitals
            
            input_data = pd.DataFrame([{
                'State': state,
                'City': city,
                'Property_Type': prop_type,
                'BHK': bhk,
                'Size_in_SqFt': sqft,
                'Year_Built': year_built,
                'Furnished_Status': furnished,
                'Floor_No': floor_no,
                'Total_Floors': total_floors,
                'Nearby_Schools': schools,
                'Nearby_Hospitals': hospitals,
                'Public_Transport_Accessibility': transport,
                'Parking_Space': parking,
                'Security': security,
                'Facing': facing,
                'Owner_Type': owner_type,
                'Availability_Status': avail,
                'Amenities_Count': amenities_cnt,
                'Total_Infrastructure_score': infra_score
            }])

            # Predictions
            pred_price = reg_model.predict(input_data)[0]
            pred_invest = clf_model.predict(input_data)[0]
            pred_invest_prob = clf_model.predict_proba(input_data)[0][1]

            st.divider()
            st.subheader("🎯 Prediction Results")

            p_col1, p_col2 = st.columns(2)

            with p_col1:
                st.metric("Estimated Property Value", f"₹{pred_price:.2f} Lakhs")
                est_sqft_price = (pred_price * 100000) / sqft
                st.caption(f"Estimated Price per SqFt: **₹{est_sqft_price:.2f} / sq.ft**")

            with p_col2:
                if pred_invest == 1:
                    st.success(f"✅ **Good Investment** (Confidence: {pred_invest_prob*100:.1f}%)")
                else:
                    st.warning(f"⚠️ **Moderate/Low Investment Potential** (Confidence: {(1-pred_invest_prob)*100:.1f}%)")
                st.progress(float(pred_invest_prob))

    # ---------------------------------------------------------
    # SECTION 3: RAW DATA EXPLORER
    # ---------------------------------------------------------
    elif app_mode == "🔍 Raw Data Explorer":
        st.title("🔍 Dataset Explorer")
        st.write(f"Displaying data from `cleaned_house_price1.csv` (Total Records: {len(df):,})")

        rows = st.slider("Select number of rows to view:", 10, 500, 50)
        st.dataframe(df.head(rows), use_container_width=True)

        st.subheader("Summary Statistics")
        st.dataframe(df.describe().T, use_container_width=True)