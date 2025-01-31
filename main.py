import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Page configuration MUST be first
st.set_page_config(
    page_title="Weather-Predict",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling (moved after set_page_config)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
        color: #333333;
    }
    
    .main {
        background: linear-gradient(180deg, #f5f9ff 0%, #e6f0ff 100%);
    }
    
    .stButton>button {
        background: #4a90e2;
        color: white;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background: #357abd;
        transform: scale(1.05);
    }
    
    .card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
        color: #333333;
    }
    
    .card h3 {
        color: #333333;
    }
    
    .metric {
        font-size: 1.5rem;
        color: #4a90e2;
        font-weight: 600;
    }
    
    .weather-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
        color: #4a90e2;
    }
</style>
""", unsafe_allow_html=True)

# Rest of the imports
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'features' not in st.session_state:
    st.session_state.features = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'y_pred' not in st.session_state:
    st.session_state.y_pred = None

# Sidebar with animated elements
with st.sidebar:
    st.markdown("# 🌦️ WeatherWise AI")
    st.markdown("### Smart Weather Prediction System")
    st.markdown("---")
    st.markdown("🔮 **Features:**")
    st.markdown("- Interactive Data Exploration")
    st.markdown("- Machine Learning Modeling")
    st.markdown("- Real-time Predictions")
    st.markdown("- Advanced Visual Analytics")
    st.markdown("---")
    st.markdown("🛠️ **Settings:**")
    uploaded_file = st.file_uploader("Upload Dataset", type="csv", key="file_upload")

# Main content area
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Data Cleaning Section with animated progress
    with st.expander("🧹 Data Preparation", expanded=True):
        with st.spinner('Cleaning and preparing data...'):
            # Convert timestamps
            time_cols = ['timestamp_utc', 'datetime', 'timestamp_local']
            for col in time_cols:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            
            # Handle missing values
            initial_rows = len(df)
            df.dropna(subset=['description(output)'], inplace=True)
            final_rows = len(df)
            
            # Categorical encoding
            cat_cols = ['wind_cdir', 'wind_cdir_full', 'pod', 'icon']
            for col in cat_cols:
                df[col] = df[col].astype('category').cat.codes
            
            # Display cleaning results in cards
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("<div class='card'>"
                            "<div class='weather-icon'>📊</div>"
                            "<h3>Data Overview</h3>"
                            f"<p>Initial Rows: {initial_rows}</p>"
                            f"<p>Final Rows: {final_rows}</p>"
                            "</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("<div class='card'>"
                            "<div class='weather-icon'>🧼</div>"
                            "<h3>Cleaning Summary</h3>"
                            f"<p>Rows Removed: {initial_rows - final_rows}</p>"
                            f"<p>Missing Values: 0</p>"
                            "</div>", unsafe_allow_html=True)
            
            st.markdown("#### Sample Data Preview")
            st.dataframe(df.head().style.background_gradient(cmap='Blues'), use_container_width=True)

    # EDA Section with interactive visualizations
    with st.expander("📈 Data Exploration", expanded=True):
        st.markdown("## 📊 Exploratory Data Analysis")
        
        # Metrics cards
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("<div class='card'>"
                        "<h3>Average Temperature</h3>"
                        f"<div class='metric'>{df['temp'].mean():.1f}°C</div>"
                        "</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='card'>"
                        "<h3>Max Wind Speed</h3>"
                        f"<div class='metric'>{df['wind_spd'].max():.1f} km/h</div>"
                        "</div>", unsafe_allow_html=True)
        
        with col3:
            st.markdown("<div class='card'>"
                        "<h3>Average Humidity</h3>"
                        f"<div class='metric'>{df['rh'].mean():.1f}%</div>"
                        "</div>", unsafe_allow_html=True)
        
        # Interactive chart selector
        chart_type = st.selectbox("Choose Visualization", 
                                ["Temperature Distribution", "Humidity vs Temperature", "Wind Speed Analysis"])
        
        fig, ax = plt.subplots(figsize=(10, 5))
        if chart_type == "Temperature Distribution":
            sns.histplot(df['temp'], kde=True, color='#4a90e2', ax=ax)
            ax.set_title('Temperature Distribution 🌡️', fontsize=16)
        elif chart_type == "Humidity vs Temperature":
            sns.scatterplot(x=df['temp'], y=df['rh'], hue=df['description(output)'], 
                          palette='coolwarm', ax=ax)
            ax.set_title('Humidity vs Temperature 💧', fontsize=16)
        else:
            sns.boxplot(x=df['description(output)'], y=df['wind_spd'], 
                       palette='viridis', ax=ax)
            ax.set_title('Wind Speed by Weather Condition 🌪️', fontsize=16)
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        st.pyplot(fig)

    # Modeling Section with modern cards
    with st.expander("🤖 AI Modeling", expanded=True):
        st.markdown("## 🧠 Machine Learning Engine")
        
        # Model configuration cards
        col1, col2 = st.columns(2)
        with col1:
            with st.markdown("<div class='card'>"
                            "<h3>⚙️ Model Configuration</h3>"
                            "</div>", unsafe_allow_html=True):
                test_size = st.slider("Test Set Size", 0.1, 0.5, 0.2)
                n_estimators = st.slider("Number of Trees", 50, 200, 100)
        
        with col2:
            with st.markdown("<div class='card'>"
                            "<h3>📊 Model Performance</h3>"
                            "</div>", unsafe_allow_html=True):
                if st.session_state.model and st.session_state.y_test is not None:
                    accuracy = accuracy_score(st.session_state.y_test, st.session_state.y_pred)
                    st.metric("Accuracy", f"{accuracy:.2%}")
                else:
                    st.info("Train model to see performance metrics")
        
        if st.button("🚀 Train Model", use_container_width=True):
            with st.spinner('Training AI model...'):
                X = df.drop(columns=['description(output)', 'icon', 'code', 
                                    'wind_cdir', 'wind_cdir_full', 
                                    'timestamp_utc', 'datetime', 'timestamp_local'])
                y = df['description(output)']
                st.session_state.features = X.columns.tolist()
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42)
                
                model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
                model.fit(X_train, y_train)
                st.session_state.model = model
                y_pred = model.predict(X_test)
                
                # Store test results in session state
                st.session_state.y_test = y_test
                st.session_state.y_pred = y_pred
                
                st.success("Model trained successfully! ✅")
                
                # Performance visualization
                col1, col2 = st.columns(2)
                with col1:
                    fig, ax = plt.subplots()
                    sns.heatmap(confusion_matrix(y_test, y_pred), 
                               annot=True, fmt='d', cmap='Blues', ax=ax)
                    ax.set_title('Confusion Matrix')
                    st.pyplot(fig)
                
                with col2:
                    report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).T
                    st.dataframe(
                        report_df.style.background_gradient(cmap='Blues', subset=['precision', 'recall', 'f1-score'])
                        .format("{:.2%}", subset=['precision', 'recall', 'f1-score']),
                        use_container_width=True
                    )

    # Prediction Section with animated elements
    with st.expander("🔮 Live Predictions", expanded=True):
        st.markdown("## 🌤️ Real-time Weather Prediction")
        
        # Prediction input cards
        with st.markdown("<div class='card'>"
                        "<h3>⚡ Prediction Parameters</h3>"
                        "</div>", unsafe_allow_html=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                temp = st.slider("Temperature (°C)", -20.0, 50.0, 25.0)
                humidity = st.slider("Humidity (%)", 0.0, 100.0, 50.0)
            
            with col2:
                wind_speed = st.slider("Wind Speed (km/h)", 0.0, 100.0, 10.0)
                solar_rad = st.slider("Solar Radiation (W/m²)", 0.0, 1000.0, 500.0)
            
            with col3:
                precip = st.slider("Precipitation (mm)", 0.0, 50.0, 0.0)
                cloud_cover = st.slider("Cloud Cover (%)", 0.0, 100.0, 30.0)
        
        if st.button("✨ Predict Weather", use_container_width=True):
            if st.session_state.model is None:
                st.error("Please train the model first! ⚠️")
            else:
                with st.spinner('Analyzing weather patterns...'):
                    input_data = pd.DataFrame(columns=st.session_state.features)
                    input_data['temp'] = [temp]
                    input_data['rh'] = [humidity]
                    input_data['wind_spd'] = [wind_speed]
                    input_data['solar_rad'] = [solar_rad]
                    input_data['precip'] = [precip]
                    input_data['clouds'] = [cloud_cover]
                    input_data = input_data.fillna(0)
                    input_data = input_data[st.session_state.features]
                    
                    prediction = st.session_state.model.predict(input_data)[0]
                    
                    # Animated result display
                    st.markdown(f"""
                    <div class='card' style='animation: fadeIn 0.5s ease;'>
                        <h3>🎯 Prediction Result</h3>
                        <div style='font-size: 2rem; color: #4a90e2; margin: 1rem 0;'>
                            {prediction} {'☀️' if 'Clear' in prediction else '⛅' if 'Cloud' in prediction else '🌧️'}
                        </div>
                        <p>Parameters used:</p>
                        <ul>
                            <li>Temperature: {temp}°C</li>
                            <li>Humidity: {humidity}%</li>
                            <li>Wind Speed: {wind_speed} km/h</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)

else:
    # Welcome screen with animations
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("# Welcome to WeatherWise AI 🌦️")
        st.markdown("### Your Intelligent Weather Prediction Platform")
        st.markdown("""
        - Upload your weather dataset
        - Explore interactive visualizations
        - Train machine learning models
        - Get real-time predictions
        """)
        st.markdown("⬅️ Upload your dataset to begin!")
    
    with col2:
        st_lottie = """
        <script src="https://unpkg.com/@lottiefiles/lottie-player@latest/dist/lottie-player.js"></script>
        <lottie-player src="https://assets2.lottiefiles.com/packages/lf20_jcikwtux.json"  
            background="transparent" speed="1" style="width: 100%; height: 400px;" loop autoplay>
        </lottie-player>
        """
        st.components.v1.html(st_lottie, height=400)

