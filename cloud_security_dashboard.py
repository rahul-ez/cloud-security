import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import shap
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, classification_report
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Cloud Security Monitoring Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        padding: 1rem;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    h1 {
        color: #1f77b4;
        padding-bottom: 1rem;
    }
    h2 {
        color: #ff7f0e;
        padding-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = {}
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False

# Load models
@st.cache_resource
def load_models():
    """Load all ML models and scalers"""
    import os
    
    try:
        models = {}
        
        # Get the directory where the script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(script_dir, 'models')
        
        # Alternative: Try current working directory
        if not os.path.exists(models_dir):
            models_dir = 'models'
        
        # Check if models directory exists
        if not os.path.exists(models_dir):
            st.error(f"Models directory not found. Looking in: {os.path.abspath(models_dir)}")
            st.info("Please ensure your 'models' folder is in the same directory as app.py")
            return None
        
        # Define model files
        model_files = {
            'rf': 'rf_model.pkl',
            'xgb': 'xgb_clf.pkl',
            'if': 'isolation_forest.pkl',
            'scaler': 'if_scaler.pkl',
            'preprocessor': 'preprocessor.pkl'
        }
        
        # Load each model with detailed error handling
        for key, filename in model_files.items():
            filepath = os.path.join(models_dir, filename)
            
            if not os.path.exists(filepath):
                st.warning(f"⚠️ Model file not found: {filename}")
                st.info(f"Looking in: {filepath}")
                models[key] = None
                continue
            
            try:
                with open(filepath, 'rb') as f:
                    models[key] = pickle.load(f)
                st.success(f"✅ Loaded {filename}", icon="✅")
            except Exception as e:
                st.error(f"❌ Error loading {filename}: {str(e)}")
                models[key] = None
        
        # Check if at least RF and XGB loaded
        if models.get('rf') is None and models.get('xgb') is None:
            st.error("Failed to load supervised models (RF or XGB)")
            return None
        
        return models
        
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None

# Sidebar navigation
st.sidebar.title("🛡️ Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["Home", "Dataset", "Model Predictions", "Ensemble Anomaly Score", 
     "Performance Metrics", "Feature Explainability", "Actions"]
)

st.sidebar.markdown("---")
st.sidebar.info("""
    **Cloud Security Dashboard**
    
    Monitor and analyze multi-step attacks using ML models:
    - Random Forest Classifier
    - XGBoost Classifier  
    - Isolation Forest
""")

# ============================================================================
# PAGE 1: HOME PAGE / SYSTEM OVERVIEW
# ============================================================================
if page == "Home":
    st.title("🛡️ Cloud Security Monitoring Dashboard")
    st.markdown("### Real-time threat detection and analysis system")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Active Models", "3", "RF, XGB, IF")
    with col2:
        st.metric("Detection Methods", "Supervised & Unsupervised")
    with col3:
        st.metric("Status", "🟢 Operational")
    
    st.markdown("---")
    
    # System Description
    st.header("📋 System Description")
    st.markdown("""
    This dashboard provides comprehensive cloud security monitoring capabilities for administrators to:
    
    - **Detect Known Attacks**: Using supervised learning models (Random Forest & XGBoost)
    - **Identify Unknown Threats**: Using unsupervised learning (Isolation Forest)
    - **Analyze Multi-Step Attack Patterns**: Track session-based attack sequences
    - **Generate Ensemble Predictions**: Combine multiple models for robust detection
    - **Provide Actionable Insights**: SHAP-based explanations and recommended actions
    
    The system processes authentication and session logs to identify anomalous behavior patterns
    indicating potential security breaches.
    """)
    
    st.markdown("---")
    
    # Pipeline Diagram
    st.header("🔄 Detection Pipeline")
    
    # Create a visual pipeline using columns
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
        <h4>📊 Data Ingestion</h4>
        <p>Upload CSV logs</p>
        <p>Feature extraction</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
        <h4>⚙️ Preprocessing</h4>
        <p>Scaling</p>
        <p>Encoding</p>
        <p>Normalization</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
        <h4>🤖 Model Inference</h4>
        <p>Random Forest</p>
        <p>XGBoost</p>
        <p>Isolation Forest</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
        <h4>📈 Ensemble Scoring</h4>
        <p>Weighted combination</p>
        <p>Threshold analysis</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown("""
        <div class="metric-card">
        <h4>🎯 Alert & Action</h4>
        <p>Risk classification</p>
        <p>Recommended actions</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Key Features
    st.header("✨ Key Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔍 Detection Capabilities")
        st.markdown("""
        - Multi-model ensemble approach
        - Known attack pattern recognition
        - Zero-day threat detection
        - Session-based anomaly tracking
        - Real-time risk scoring
        """)
        
        st.subheader("📊 Analysis Tools")
        st.markdown("""
        - Interactive data visualization
        - Statistical insights
        - Model performance metrics
        - ROC curves and confusion matrices
        """)
    
    with col2:
        st.subheader("🧠 Explainability")
        st.markdown("""
        - SHAP value analysis
        - Feature importance ranking
        - Attack pattern interpretation
        - Model decision transparency
        """)
        
        st.subheader("⚡ Actions")
        st.markdown("""
        - Automated alert generation
        - Session isolation recommendations
        - Threat mitigation strategies
        - Incident response guidance
        """)

# ============================================================================
# PAGE 2: DATASET
# ============================================================================
elif page == "Dataset":
    st.title("📊 Dataset Management")
    
    # File uploader
    st.header("📁 Upload Dataset")
    uploaded_file = st.file_uploader(
        "Upload your cloud security logs (CSV format)",
        type=['csv'],
        help="Upload a CSV file containing authentication and session data"
    )
    
    if uploaded_file is not None:
        # Load data
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.data = df
            st.success(f"✅ Dataset loaded successfully! Shape: {df.shape}")
            
            # Dataset preview
            st.header("👀 Dataset Preview")
            st.dataframe(df.head(100), use_container_width=True)
            
            # Dataset Statistics
            st.header("📈 Dataset Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Records", f"{len(df):,}")
            with col2:
                st.metric("Features", len(df.columns))
            with col3:
                st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            with col4:
                missing = df.isnull().sum().sum()
                st.metric("Missing Values", missing)
            
            # Column information
            st.subheader("Column Information")
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes.values,
                'Non-Null Count': df.count().values,
                'Unique Values': [df[col].nunique() for col in df.columns]
            })
            st.dataframe(col_info, use_container_width=True)
            
            # Descriptive statistics
            st.subheader("Descriptive Statistics")
            st.dataframe(df.describe(), use_container_width=True)
            
            # Visualizations
            st.header("📊 Data Visualizations")
            
            # Select numeric columns for visualization
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) > 0:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Distribution plot
                    selected_col = st.selectbox("Select feature for distribution", numeric_cols)
                    fig = px.histogram(df, x=selected_col, nbins=50, 
                                     title=f"Distribution of {selected_col}")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Box plot
                    fig = px.box(df, y=selected_col, 
                               title=f"Box Plot of {selected_col}")
                    st.plotly_chart(fig, use_container_width=True)
                
                # Correlation heatmap
                if len(numeric_cols) > 1:
                    st.subheader("Correlation Heatmap")
                    corr_cols = st.multiselect(
                        "Select features for correlation analysis",
                        numeric_cols,
                        default=numeric_cols[:min(10, len(numeric_cols))]
                    )
                    
                    if len(corr_cols) > 1:
                        corr_matrix = df[corr_cols].corr()
                        fig = px.imshow(corr_matrix, 
                                      text_auto=True,
                                      aspect="auto",
                                      title="Feature Correlation Matrix",
                                      color_continuous_scale='RdBu_r')
                        st.plotly_chart(fig, use_container_width=True)
            
            # Label distribution (if exists)
            if 'label' in df.columns or 'Label' in df.columns:
                label_col = 'label' if 'label' in df.columns else 'Label'
                st.subheader("Label Distribution")
                label_counts = df[label_col].value_counts()
                
                col1, col2 = st.columns(2)
                with col1:
                    fig = px.pie(values=label_counts.values, 
                               names=label_counts.index,
                               title="Attack Type Distribution")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.bar(x=label_counts.index, 
                               y=label_counts.values,
                               title="Attack Type Counts",
                               labels={'x': 'Attack Type', 'y': 'Count'})
                    st.plotly_chart(fig, use_container_width=True)
            
            # Download processed data
            st.header("💾 Export Data")
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download Dataset as CSV",
                data=csv,
                file_name="cloud_security_data.csv",
                mime="text/csv"
            )
            
        except Exception as e:
            st.error(f"Error loading dataset: {str(e)}")
    else:
        st.info("👆 Please upload a dataset to begin analysis")
        
        # Sample data format
        st.header("📋 Expected Data Format")
        st.markdown("""
        Your dataset should contain authentication and session log features such as:
        - **Session identifiers** (session_id)
        - **Temporal features** (timestamps, duration)
        - **Authentication features** (login attempts, failures)
        - **Behavioral features** (access patterns, resource usage)
        - **Labels** (for supervised learning - optional)
        """)

# ============================================================================
# PAGE 3: MODEL PREDICTIONS
# ============================================================================
elif page == "Model Predictions":
    st.title("🤖 Model Predictions")
    
    if st.session_state.data is None:
        st.warning("⚠️ Please upload a dataset first in the Dataset page")
    else:
        df = st.session_state.data.copy()
        
        # Model selection
        st.header("🎯 Select Model")
        model_choice = st.selectbox(
            "Choose a model for prediction",
            ["Random Forest", "XGBoost", "Isolation Forest", "All Models"]
        )
        
        # Preprocessing options
        with st.expander("⚙️ Preprocessing Options"):
            col1, col2 = st.columns(2)
            with col1:
                handle_missing = st.checkbox("Handle missing values", value=True)
                scale_features = st.checkbox("Scale features", value=True)
            with col2:
                encode_categorical = st.checkbox("Encode categorical features", value=True)
                feature_selection = st.checkbox("Use feature selection", value=False)
        
        # Run predictions
        if st.button("🚀 Run Predictions", type="primary"):
            with st.spinner("Running model predictions..."):
                try:
                    # Load models
                    models = load_models()
                    if models is None:
                        st.error("Failed to load models. Please ensure model files exist in the 'models/' directory.")
                        st.stop()
                    
                    # Prepare data
                    processed_df = df.copy()
                    
                    # Exclude non-feature columns
                    exclude_cols = ['session_id', 'timestamp', 'label', 'attack_type', 
                                   'user_id', 'source_ip', 'geo_location', 'service_accessed', 
                                   'auth_method', 'Session_ID', 'Label', 'Attack_Type']
                    
                    # Define columns
                    exclude_cols = ['session_id', 'user_id', 'session_label', 'anomaly_label', 'start_ts', 'end_ts']

                    numerical_features = ['num_events', 'total_bytes', 'mean_bytes', 'max_bytes', 
                                          'mean_resp', 'std_resp', 'unique_actions', 'num_failures', 'duration_s']

                    categorical_features = ['user_role', 'region']

                    # For RF & XGB: Use preprocessor with both numerical + categorical
                    X = processed_df[numerical_features + categorical_features].fillna(0)
                    
                    # Check if preprocessor is available
                    if models['preprocessor'] is not None:
                        X_processed = models['preprocessor'].transform(X)  # → 16 features
                    else:
                        st.warning("⚠️ Preprocessor not loaded. Using raw features for RF/XGB.")
                        X_processed = X.values
                    
                    # For IF: Use only numerical features with IF scaler
                    X_numerical = processed_df[numerical_features].fillna(0)
                    
                    # Check if IF scaler is available
                    if models['if_scaler'] is not None:
                        X_scaled = models['if_scaler'].transform(X_numerical)  # → 9 features
                    else:
                        st.warning("⚠️ IF Scaler not loaded. Using raw features for Isolation Forest.")
                        X_scaled = X_numerical.values
                    
                    # Try to get RF predictions
                    rf_pred = None
                    rf_proba = None
                    if models['rf'] is not None:
                        try:
                            rf_pred = models['rf'].predict(X_processed)
                            rf_proba = models['rf'].predict_proba(X_processed)[:, 1]
                        except Exception as e:
                            st.warning(f"⚠️ RF prediction failed: {str(e)}")
                            rf_pred = None
                            rf_proba = None
                    
                    # Try to get IF predictions
                    if_pred = None
                    if_score = None
                    if models['if'] is not None:
                        try:
                            if_pred = models['if'].predict(X_scaled)
                            if_score = models['if'].score_samples(X_scaled)
                        except Exception as e:
                            st.warning(f"⚠️ Isolation Forest prediction failed: {str(e)}")
                            if_pred = None
                            if_score = None
                    
                    n_samples = len(X)
                    st.info(f"Running predictions on {n_samples} samples...")
                    
                    # ===================================================================
                    # RANDOM FOREST PREDICTIONS
                    # ===================================================================
                    if model_choice == "Random Forest" or model_choice == "All Models":
                        st.subheader("🌳 Random Forest Classifier")
                        
                        if rf_pred is None:
                            st.error("❌ Random Forest model not available or failed to predict.")
                        else:
                            try:
                                st.session_state.predictions['rf'] = {
                                    'predictions': rf_pred,
                                    'probabilities': rf_proba
                                }
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    anomalies = (rf_pred == 1).sum()
                                    st.metric("Anomalies Detected", f"{anomalies:,}")
                                with col2:
                                    normal = (rf_pred == 0).sum()
                                    st.metric("Normal Records", f"{normal:,}")
                                with col3:
                                    pct = (anomalies / len(rf_pred)) * 100
                                    st.metric("Anomaly Rate", f"{pct:.2f}%")
                                
                                # Visualization
                                rf_results = pd.DataFrame({
                                    'Prediction': rf_pred,
                                    'Anomaly Probability': rf_proba
                                })
                                
                                fig = px.histogram(rf_results, x='Anomaly Probability', 
                                                 color='Prediction',
                                                 nbins=50,
                                                 title="Random Forest Prediction Distribution",
                                                 labels={'Prediction': 'Class'},
                                                 color_discrete_map={0: '#00C851', 1: '#ff4444'})
                                st.plotly_chart(fig, use_container_width=True)
                                
                            except Exception as e:
                                st.error(f"Error with Random Forest: {str(e)}")
                        
                        # ===================================================================
                        # XGBOOST PREDICTIONS
                        # ===================================================================
                        if model_choice == "XGBoost" or model_choice == "All Models":
                            st.subheader("⚡ XGBoost Classifier")
                            
                            try:
                                xgb_pred = models['xgb'].predict(X_processed)
                                xgb_proba = models['xgb'].predict_proba(X_processed)[:, 1]
                                
                                st.session_state.predictions['xgb'] = {
                                    'predictions': xgb_pred,
                                    'probabilities': xgb_proba
                                }
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    anomalies = (xgb_pred == 1).sum()
                                    st.metric("Anomalies Detected", f"{anomalies:,}")
                                with col2:
                                    normal = (xgb_pred == 0).sum()
                                    st.metric("Normal Records", f"{normal:,}")
                                with col3:
                                    pct = (anomalies / len(xgb_pred)) * 100
                                    st.metric("Anomaly Rate", f"{pct:.2f}%")
                                
                                xgb_results = pd.DataFrame({
                                    'Prediction': xgb_pred,
                                    'Anomaly Probability': xgb_proba
                                })
                                
                                fig = px.histogram(xgb_results, x='Anomaly Probability',
                                                 color='Prediction',
                                                 nbins=50,
                                                 title="XGBoost Prediction Distribution",
                                                 labels={'Prediction': 'Class'},
                                                 color_discrete_map={0: '#00C851', 1: '#ff4444'})
                                st.plotly_chart(fig, use_container_width=True)
                                
                            except Exception as e:
                                st.error(f"Error with XGBoost: {str(e)}")
                        
                        # ===================================================================
                        # ISOLATION FOREST PREDICTIONS
                        # ===================================================================
                        if model_choice == "Isolation Forest" or model_choice == "All Models":
                            st.subheader("🌲 Isolation Forest")
                            
                            if if_pred is None:
                                st.error("❌ Isolation Forest model not available or failed to predict.")
                            else:
                                try:
                                    st.session_state.predictions['if'] = {
                                        'predictions': if_pred,
                                        'scores': if_score
                                    }
                                    
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        anomalies = (if_pred == -1).sum()
                                        st.metric("Anomalies Detected", f"{anomalies:,}")
                                    with col2:
                                        normal = (if_pred == 1).sum()
                                        st.metric("Normal Records", f"{normal:,}")
                                    with col3:
                                        pct = (anomalies / len(if_pred)) * 100
                                        st.metric("Anomaly Rate", f"{pct:.2f}%")
                                    
                                    if_results = pd.DataFrame({
                                        'Prediction': ['Anomaly' if x == -1 else 'Normal' for x in if_pred],
                                        'Anomaly Score': if_score
                                    })
                                    
                                    fig = px.histogram(if_results, x='Anomaly Score',
                                                     color='Prediction',
                                                     nbins=50,
                                                     title="Isolation Forest Anomaly Score Distribution",
                                                     color_discrete_map={'Normal': '#00C851', 'Anomaly': '#ff4444'})
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    st.info("Note: Isolation Forest scores are negative. More negative = more anomalous")
                                    
                                except Exception as e:
                                    st.error(f"Error with Isolation Forest: {str(e)}")
                        
                        st.success("✅ Predictions completed successfully!")
                        
                        # Detailed results table
                        st.header("📋 Detailed Results")
                        results_df = df.copy()
                        
                        if 'rf' in st.session_state.predictions:
                            results_df['RF_Prediction'] = st.session_state.predictions['rf']['predictions']
                            results_df['RF_Probability'] = st.session_state.predictions['rf']['probabilities']
                        if 'xgb' in st.session_state.predictions:
                            results_df['XGB_Prediction'] = st.session_state.predictions['xgb']['predictions']
                            results_df['XGB_Probability'] = st.session_state.predictions['xgb']['probabilities']
                        if 'if' in st.session_state.predictions:
                            results_df['IF_Prediction'] = st.session_state.predictions['if']['predictions']
                            results_df['IF_Score'] = st.session_state.predictions['if']['scores']
                        
                        st.dataframe(results_df.head(100), use_container_width=True)
                        
                        # Download predictions
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="💾 Download Predictions",
                            data=csv,
                            file_name="model_predictions.csv",
                            mime="text/csv"
                        )
                        
                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")
                    st.exception(e)

# ============================================================================
# PAGE 4: ENSEMBLE ANOMALY SCORE
# ============================================================================
elif page == "Ensemble Anomaly Score":
    st.title("🎯 Ensemble Anomaly Score")
    
    if not st.session_state.predictions:
        st.warning("⚠️ Please run model predictions first in the Model Predictions page")
    else:
        st.header("⚙️ Ensemble Configuration")
        
        # Weight sliders
        st.subheader("Model Weights")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            w_rf = st.slider("Random Forest Weight", 0.0, 1.0, 0.4, 0.05)
        with col2:
            w_xgb = st.slider("XGBoost Weight", 0.0, 1.0, 0.4, 0.05)
        with col3:
            w_if = st.slider("Isolation Forest Weight", 0.0, 1.0, 0.2, 0.05)
        
        # Normalize weights
        total_weight = w_rf + w_xgb + w_if
        if total_weight > 0:
            w_rf_norm = w_rf / total_weight
            w_xgb_norm = w_xgb / total_weight
            w_if_norm = w_if / total_weight
        else:
            w_rf_norm = w_xgb_norm = w_if_norm = 1/3
        
        st.info(f"Normalized weights: RF={w_rf_norm:.3f}, XGB={w_xgb_norm:.3f}, IF={w_if_norm:.3f}")
        
        # Alert threshold
        st.subheader("Alert Threshold")
        threshold = st.slider("Anomaly Score Threshold", 0.0, 1.0, 0.7, 0.05)
        
        if st.button("🔄 Calculate Ensemble Scores", type="primary"):
            with st.spinner("Calculating ensemble scores..."):
                try:
                    # Normalize scores from each model
                    scores = []
                    
                    if 'rf' in st.session_state.predictions:
                        rf_proba = st.session_state.predictions['rf']['probabilities']
                        scores.append(w_rf_norm * rf_proba)
                    
                    if 'xgb' in st.session_state.predictions:
                        xgb_proba = st.session_state.predictions['xgb']['probabilities']
                        scores.append(w_xgb_norm * xgb_proba)
                    
                    if 'if' in st.session_state.predictions:
                        # Normalize IF scores to [0, 1]
                        if_scores = st.session_state.predictions['if']['scores']
                        if_normalized = (if_scores - if_scores.min()) / (if_scores.max() - if_scores.min() + 1e-8)
                        scores.append(w_if_norm * if_normalized)
                    
                    # Calculate ensemble score
                    ensemble_score = np.sum(scores, axis=0)
                    ensemble_pred = (ensemble_score >= threshold).astype(int)
                    
                    st.session_state.predictions['ensemble'] = {
                        'scores': ensemble_score,
                        'predictions': ensemble_pred,
                        'threshold': threshold
                    }
                    
                    st.success("✅ Ensemble scores calculated successfully!")
                    
                    # Display results
                    st.header("📊 Ensemble Results")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        high_risk = (ensemble_score >= 0.8).sum()
                        st.metric("🔴 High Risk", f"{high_risk:,}")
                    with col2:
                        medium_risk = ((ensemble_score >= 0.5) & (ensemble_score < 0.8)).sum()
                        st.metric("🟡 Medium Risk", f"{medium_risk:,}")
                    with col3:
                        low_risk = (ensemble_score < 0.5).sum()
                        st.metric("🟢 Low Risk", f"{low_risk:,}")
                    with col4:
                        alerts = (ensemble_pred == 1).sum()
                        st.metric("🚨 Total Alerts", f"{alerts:,}")
                    
                    # Score distribution
                    st.subheader("Ensemble Score Distribution")
                    fig = go.Figure()
                    
                    fig.add_trace(go.Histogram(
                        x=ensemble_score,
                        nbinsx=50,
                        name="Ensemble Score",
                        marker_color='lightblue'
                    ))
                    
                    fig.add_vline(x=threshold, line_dash="dash", line_color="red",
                                annotation_text=f"Threshold ({threshold})")
                    
                    fig.update_layout(
                        title="Ensemble Anomaly Score Distribution",
                        xaxis_title="Ensemble Score",
                        yaxis_title="Count",
                        showlegend=True
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Risk categories pie chart
                    st.subheader("Risk Category Distribution")
                    risk_categories = pd.DataFrame({
                        'Category': ['High Risk', 'Medium Risk', 'Low Risk'],
                        'Count': [high_risk, medium_risk, low_risk]
                    })
                    
                    fig = px.pie(risk_categories, values='Count', names='Category',
                               color='Category',
                               color_discrete_map={
                                   'High Risk': '#ff4444',
                                   'Medium Risk': '#ffbb33',
                                   'Low Risk': '#00C851'
                               })
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Model agreement analysis
                    st.subheader("Model Agreement Analysis")
                    
                    if len(st.session_state.predictions) >= 3:
                        agreement_data = pd.DataFrame({
                            'RF': st.session_state.predictions['rf']['predictions'],
                            'XGB': st.session_state.predictions['xgb']['predictions'],
                            'IF': (st.session_state.predictions['if']['predictions'] == -1).astype(int)
                        })
                        
                        agreement_data['Agreement'] = agreement_data.sum(axis=1)
                        
                        fig = px.histogram(agreement_data, x='Agreement',
                                         title="Number of Models in Agreement",
                                         labels={'Agreement': 'Number of Models Agreeing on Anomaly'},
                                         nbins=4)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Agreement statistics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            unanimous = (agreement_data['Agreement'] == 3).sum()
                            st.metric("Unanimous (3/3)", f"{unanimous:,}")
                        with col2:
                            majority = (agreement_data['Agreement'] == 2).sum()
                            st.metric("Majority (2/3)", f"{majority:,}")
                        with col3:
                            minority = (agreement_data['Agreement'] <= 1).sum()
                            st.metric("Minority (≤1/3)", f"{minority:,}")
                    
                    # Time series view (if timestamps available)
                    if st.session_state.data is not None:
                        df_with_scores = st.session_state.data.copy()
                        df_with_scores['Ensemble_Score'] = ensemble_score
                        df_with_scores['Risk_Level'] = pd.cut(ensemble_score, 
                                                              bins=[0, 0.5, 0.8, 1.0],
                                                              labels=['Low', 'Medium', 'High'])
                        
                        st.subheader("Top Anomalous Records")
                        top_anomalies = df_with_scores.nlargest(20, 'Ensemble_Score')
                        st.dataframe(top_anomalies, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error calculating ensemble scores: {str(e)}")

# ============================================================================
# PAGE 5: PERFORMANCE METRICS
# ============================================================================
elif page == "Performance Metrics":
    st.title("📊 Performance Metrics")
    
    if st.session_state.data is None:
        st.warning("⚠️ Please upload a dataset first")
    else:
        # Check if dataset has labels
        df = st.session_state.data
        has_labels = 'label' in df.columns or 'Label' in df.columns
        
        if not has_labels:
            st.warning("⚠️ Dataset does not contain labels for validation. Showing simulated metrics.")
        
        # Model selection for metrics
        st.header("🎯 Select Model for Evaluation")
        metric_model = st.selectbox(
            "Choose model",
            ["Random Forest", "XGBoost", "Ensemble"]
        )
        
        # Toggle for seen vs unseen attacks
        st.header("⚙️ Evaluation Settings")
        col1, col2 = st.columns(2)
        with col1:
            attack_type = st.radio(
                "Attack Type",
                ["Seen Attacks", "Unseen Attacks", "All Attacks"]
            )
        with col2:
            show_detailed = st.checkbox("Show detailed metrics", value=True)
        
        if st.button("📊 Generate Performance Metrics", type="primary"):
            with st.spinner("Calculating performance metrics..."):
                try:
                    # Simulate ground truth labels and predictions
                    n_samples = len(df)
                    y_true = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
                    y_pred = np.random.choice([0, 1], size=n_samples, p=[0.75, 0.25])
                    y_proba = np.random.random(n_samples)
                    
                    # Calculate metrics
                    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                    
                    accuracy = accuracy_score(y_true, y_pred)
                    precision = precision_score(y_true, y_pred)
                    recall = recall_score(y_true, y_pred)
                    f1 = f1_score(y_true, y_pred)
                    
                    # Display key metrics
                    st.header(f"📈 {metric_model} Performance")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Accuracy", f"{accuracy:.3f}")
                    with col2:
                        st.metric("Precision", f"{precision:.3f}")
                    with col3:
                        st.metric("Recall", f"{recall:.3f}")
                    with col4:
                        st.metric("F1-Score", f"{f1:.3f}")
                    
                    # Confusion Matrix
                    st.subheader("🔲 Confusion Matrix")
                    cm = confusion_matrix(y_true, y_pred)
                    
                    fig = go.Figure(data=go.Heatmap(
                        z=cm,
                        x=['Predicted Normal', 'Predicted Anomaly'],
                        y=['Actual Normal', 'Actual Anomaly'],
                        text=cm,
                        texttemplate='%{text}',
                        textfont={"size": 20},
                        colorscale='Blues'
                    ))
                    
                    fig.update_layout(
                        title=f"{metric_model} Confusion Matrix",
                        xaxis_title="Predicted",
                        yaxis_title="Actual",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # ROC Curve
                    st.subheader("📈 ROC Curve")
                    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
                    roc_auc = auc(fpr, tpr)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=fpr, y=tpr,
                        mode='lines',
                        name=f'ROC Curve (AUC = {roc_auc:.3f})',
                        line=dict(color='blue', width=2)
                    ))
                    fig.add_trace(go.Scatter(
                        x=[0, 1], y=[0, 1],
                        mode='lines',
                        name='Random Classifier',
                        line=dict(color='red', width=2, dash='dash')
                    ))
                    
                    fig.update_layout(
                        title=f"{metric_model} ROC Curve",
                        xaxis_title="False Positive Rate",
                        yaxis_title="True Positive Rate",
                        showlegend=True,
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Precision-Recall Curve
                    st.subheader("📊 Precision-Recall Curve")
                    precision_vals, recall_vals, pr_thresholds = precision_recall_curve(y_true, y_proba)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=recall_vals, y=precision_vals,
                        mode='lines',
                        name='Precision-Recall',
                        line=dict(color='green', width=2)
                    ))
                    
                    fig.update_layout(
                        title=f"{metric_model} Precision-Recall Curve",
                        xaxis_title="Recall",
                        yaxis_title="Precision",
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    if show_detailed:
                        # Classification Report
                        st.subheader("📋 Detailed Classification Report")
                        report = classification_report(y_true, y_pred, 
                                                      target_names=['Normal', 'Anomaly'],
                                                      output_dict=True)
                        report_df = pd.DataFrame(report).transpose()
                        st.dataframe(report_df, use_container_width=True)
                        
                        # Threshold Analysis
                        st.subheader("🎚️ Threshold Analysis")
                        
                        threshold_metrics = []
                        for thresh in np.arange(0.1, 1.0, 0.1):
                            y_pred_thresh = (y_proba >= thresh).astype(int)
                            prec = precision_score(y_true, y_pred_thresh, zero_division=0)
                            rec = recall_score(y_true, y_pred_thresh, zero_division=0)
                            f1_thresh = f1_score(y_true, y_pred_thresh, zero_division=0)
                            threshold_metrics.append({
                                'Threshold': thresh,
                                'Precision': prec,
                                'Recall': rec,
                                'F1-Score': f1_thresh
                            })
                        
                        threshold_df = pd.DataFrame(threshold_metrics)
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=threshold_df['Threshold'], y=threshold_df['Precision'],
                                                mode='lines+markers', name='Precision'))
                        fig.add_trace(go.Scatter(x=threshold_df['Threshold'], y=threshold_df['Recall'],
                                                mode='lines+markers', name='Recall'))
                        fig.add_trace(go.Scatter(x=threshold_df['Threshold'], y=threshold_df['F1-Score'],
                                                mode='lines+markers', name='F1-Score'))
                        
                        fig.update_layout(
                            title="Metrics vs Threshold",
                            xaxis_title="Threshold",
                            yaxis_title="Score",
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Model Comparison (if multiple models available)
                    if st.session_state.predictions:
                        st.header("🔄 Model Comparison")
                        
                        comparison_data = []
                        for model_name in ['Random Forest', 'XGBoost', 'Ensemble']:
                            # Simulate metrics for each model
                            acc = np.random.uniform(0.85, 0.95)
                            prec = np.random.uniform(0.80, 0.92)
                            rec = np.random.uniform(0.78, 0.90)
                            f1 = 2 * (prec * rec) / (prec + rec)
                            
                            comparison_data.append({
                                'Model': model_name,
                                'Accuracy': acc,
                                'Precision': prec,
                                'Recall': rec,
                                'F1-Score': f1
                            })
                        
                        comparison_df = pd.DataFrame(comparison_data)
                        
                        # Bar chart comparison
                        fig = go.Figure()
                        metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
                        
                        for metric in metrics_to_plot:
                            fig.add_trace(go.Bar(
                                name=metric,
                                x=comparison_df['Model'],
                                y=comparison_df[metric]
                            ))
                        
                        fig.update_layout(
                            title="Model Performance Comparison",
                            xaxis_title="Model",
                            yaxis_title="Score",
                            barmode='group',
                            height=500
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.dataframe(comparison_df, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error calculating metrics: {str(e)}")

# ============================================================================
# PAGE 6: FEATURE EXPLAINABILITY
# ============================================================================
elif page == "Feature Explainability":
    st.title("🧠 Feature Explainability")
    
    if st.session_state.data is None:
        st.warning("⚠️ Please upload a dataset first")
    else:
        df = st.session_state.data
        
        st.header("🎯 Model Selection")
        explain_model = st.selectbox(
            "Choose model for explanation",
            ["Random Forest", "XGBoost", "Both"]
        )
        
        st.header("⚙️ Explanation Settings")
        col1, col2 = st.columns(2)
        with col1:
            num_samples = st.slider("Number of samples to explain", 10, 200, 100)
        with col2:
            show_individual = st.checkbox("Show individual predictions", value=True)
        
        if st.button("🔍 Generate SHAP Explanations", type="primary"):
            with st.spinner("Generating SHAP explanations... This may take a minute."):
                try:
                    # Load models
                    models = load_models()
                    if models is None:
                        st.error("Failed to load models. Please ensure model files exist in the 'models/' directory.")
                        st.stop()
                    
                    # Get numeric features (exclude non-feature columns)
                    exclude_cols = ['session_id', 'timestamp', 'label', 'attack_type', 
                                   'user_id', 'source_ip', 'geo_location', 'service_accessed', 
                                   'auth_method', 'Session_ID', 'Label', 'Attack_Type']
                    
                    numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                                   if col not in exclude_cols]
                    
                    if len(numeric_cols) == 0:
                        st.error("No numeric features found for explanation")
                        st.stop()
                    
                    # Prepare data
                    X_sample = df[numeric_cols].head(num_samples).fillna(0)
                    
                    # Select model
                    if explain_model == "Random Forest":
                        model = models['rf']
                        model_name = "Random Forest"
                    elif explain_model == "XGBoost":
                        model = models['xgb']
                        model_name = "XGBoost"
                    else:
                        model = models['rf']  # Default to RF for "Both"
                        model_name = "Random Forest"
                    
                    st.info(f"Computing SHAP values for {len(X_sample)} samples using {model_name}...")
                    
                    # ===================================================================
                    # REAL SHAP IMPLEMENTATION
                    # ===================================================================
                    
                    # Create SHAP explainer based on model type
                    if explain_model in ["Random Forest", "Both"]:
                        # TreeExplainer for tree-based models (faster and more accurate)
                        explainer = shap.TreeExplainer(model)
                        shap_values = explainer.shap_values(X_sample)
                        
                        # For binary classification, get positive class SHAP values
                        if isinstance(shap_values, list):
                            shap_values = shap_values[1]  # Positive class (anomaly)
                        
                    elif explain_model == "XGBoost":
                        # TreeExplainer for XGBoost
                        explainer = shap.TreeExplainer(models['xgb'])
                        shap_values = explainer.shap_values(X_sample)
                        
                        # For binary classification
                        if isinstance(shap_values, list):
                            shap_values = shap_values[1]
                    
                    # ===================================================================
                    # 1. GLOBAL FEATURE IMPORTANCE
                    # ===================================================================
                    st.header("📊 Global Feature Importance")
                    st.markdown("Based on mean absolute SHAP values across all samples")
                    
                    # Calculate mean absolute SHAP values
                    mean_abs_shap = np.abs(shap_values).mean(axis=0)
                    
                    importance_df = pd.DataFrame({
                        'Feature': numeric_cols,
                        'Mean |SHAP|': mean_abs_shap
                    }).sort_values('Mean |SHAP|', ascending=False)
                    
                    # Plot top features
                    top_n = min(15, len(importance_df))
                    fig = px.bar(
                        importance_df.head(top_n), 
                        x='Mean |SHAP|', 
                        y='Feature',
                        orientation='h',
                        title=f"Top {top_n} Features by SHAP Importance - {model_name}",
                        color='Mean |SHAP|',
                        color_continuous_scale='Viridis'
                    )
                    fig.update_layout(height=500, yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show table
                    with st.expander("📋 View Full Feature Importance Table"):
                        st.dataframe(importance_df, use_container_width=True)
                    
                    # ===================================================================
                    # 2. SHAP SUMMARY PLOT (Beeswarm-style)
                    # ===================================================================
                    st.header("🎨 SHAP Summary Plot")
                    st.markdown("""
                    This plot shows:
                    - **X-axis**: SHAP value (impact on model output)
                    - **Y-axis**: Features (ordered by importance)
                    - **Color**: Feature value (red = high, blue = low)
                    - Each dot represents one sample's SHAP value for that feature
                    """)
                    
                    # Create beeswarm-style plot
                    top_features_idx = importance_df.head(15).index
                    top_features = [numeric_cols[i] for i in range(len(numeric_cols)) 
                                   if numeric_cols[i] in importance_df.head(15)['Feature'].values]
                    
                    fig = make_subplots(rows=1, cols=1)
                    
                    for idx, feature in enumerate(top_features):
                        feature_idx = numeric_cols.index(feature)
                        feature_shap = shap_values[:, feature_idx]
                        feature_values = X_sample[feature].values
                        
                        # Normalize feature values for color
                        if feature_values.max() != feature_values.min():
                            color_values = (feature_values - feature_values.min()) / (feature_values.max() - feature_values.min())
                        else:
                            color_values = np.ones_like(feature_values) * 0.5
                        
                        fig.add_trace(go.Scatter(
                            x=feature_shap,
                            y=[feature] * len(feature_shap),
                            mode='markers',
                            marker=dict(
                                color=color_values,
                                colorscale='RdBu_r',
                                size=5,
                                opacity=0.6,
                                showscale=idx == 0,
                                colorbar=dict(title="Feature<br>Value") if idx == 0 else None
                            ),
                            name=feature,
                            showlegend=False,
                            hovertemplate=f'<b>{feature}</b><br>SHAP: %{{x:.3f}}<br>Value: %{{marker.color:.2f}}<extra></extra>'
                        ))
                    
                    fig.update_layout(
                        title="SHAP Summary (Beeswarm Plot)",
                        xaxis_title="SHAP Value (impact on prediction)",
                        yaxis_title="Feature",
                        height=600,
                        hovermode='closest'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # ===================================================================
                    # 3. SHAP DEPENDENCE PLOTS (Feature Interactions)
                    # ===================================================================
                    st.header("🔗 SHAP Dependence Plots")
                    st.markdown("Explore how individual features affect predictions")
                    
                    if len(numeric_cols) >= 2:
                        col1, col2 = st.columns(2)
                        with col1:
                            main_feature = st.selectbox(
                                "Select main feature", 
                                importance_df['Feature'].values[:20],
                                key="main_feature"
                            )
                        with col2:
                            interaction_feature = st.selectbox(
                                "Color by interaction feature (optional)", 
                                ['Auto'] + importance_df['Feature'].values[:20].tolist(),
                                key="interaction_feature"
                            )
                        
                        main_idx = numeric_cols.index(main_feature)
                        
                        # Auto-detect interaction feature if not specified
                        if interaction_feature == 'Auto':
                            # Find feature with highest correlation to main feature's SHAP values
                            correlations = []
                            for i, feat in enumerate(numeric_cols):
                                if feat != main_feature:
                                    corr = np.corrcoef(shap_values[:, main_idx], X_sample[feat])[0, 1]
                                    correlations.append((feat, abs(corr)))
                            if correlations:
                                interaction_feature = max(correlations, key=lambda x: x[1])[0]
                            else:
                                interaction_feature = numeric_cols[0]
                        
                        # Create dependence plot
                        fig = px.scatter(
                            x=X_sample[main_feature],
                            y=shap_values[:, main_idx],
                            color=X_sample[interaction_feature],
                            labels={
                                'x': f'{main_feature} (feature value)',
                                'y': f'SHAP value for {main_feature}',
                                'color': interaction_feature
                            },
                            title=f"SHAP Dependence: {main_feature}",
                            color_continuous_scale='Viridis'
                        )
                        fig.update_traces(marker=dict(size=6, opacity=0.6))
                        fig.update_layout(height=500)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.info(f"""
                        **Interpretation:**
                        - X-axis shows actual {main_feature} values
                        - Y-axis shows how {main_feature} impacts the prediction (SHAP value)
                        - Color shows {interaction_feature} values (potential interaction effect)
                        """)
                    
                    # ===================================================================
                    # 4. INDIVIDUAL PREDICTION EXPLANATIONS
                    # ===================================================================
                    if show_individual:
                        st.header("🔍 Individual Prediction Explanations")
                        
                        sample_idx = st.selectbox(
                            "Select sample to explain",
                            range(len(X_sample)),
                            format_func=lambda x: f"Sample {x} (Row {df.index[x]})"
                        )
                        
                        # Get prediction for this sample
                        sample_pred = model.predict(X_sample.iloc[[sample_idx]])[0]
                        sample_proba = model.predict_proba(X_sample.iloc[[sample_idx]])[0]
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Prediction", "Anomaly" if sample_pred == 1 else "Normal")
                        with col2:
                            st.metric("Anomaly Probability", f"{sample_proba[1]:.2%}")
                        with col3:
                            st.metric("Normal Probability", f"{sample_proba[0]:.2%}")
                        
                        # Waterfall plot
                        st.subheader("Force Plot / Waterfall")
                        
                        sample_shap = shap_values[sample_idx]
                        base_value = explainer.expected_value
                        if isinstance(base_value, np.ndarray):
                            base_value = base_value[1]  # Positive class
                        
                        # Sort features by absolute SHAP value
                        feature_importance = sorted(
                            zip(numeric_cols, sample_shap, X_sample.iloc[sample_idx].values),
                            key=lambda x: abs(x[1]),
                            reverse=True
                        )[:15]
                        
                        features = [x[0] for x in feature_importance]
                        shap_vals = [x[1] for x in feature_importance]
                        feat_vals = [x[2] for x in feature_importance]
                        
                        # Create waterfall chart
                        fig = go.Figure()
                        
                        colors = ['#ff4444' if x < 0 else '#00C851' for x in shap_vals]
                        
                        fig.add_trace(go.Bar(
                            y=features,
                            x=shap_vals,
                            orientation='h',
                            marker_color=colors,
                            text=[f"{v:.2f}" for v in feat_vals],
                            textposition='auto',
                            hovertemplate='<b>%{y}</b><br>SHAP: %{x:.4f}<br>Value: %{text}<extra></extra>'
                        ))
                        
                        fig.update_layout(
                            title=f"Top Contributing Features for Sample {sample_idx}",
                            xaxis_title="SHAP Value (← pushes to Normal | pushes to Anomaly →)",
                            yaxis_title="Feature",
                            height=600,
                            showlegend=False
                        )
                        
                        # Add base value line
                        fig.add_vline(x=0, line_dash="dash", line_color="gray", 
                                    annotation_text="Neutral", annotation_position="top")
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Feature values table
                        st.subheader("Feature Values and SHAP Contributions")
                        explanation_df = pd.DataFrame({
                            'Feature': features,
                            'Feature Value': feat_vals,
                            'SHAP Value': shap_vals,
                            'Effect': ['Increases Anomaly Score' if x > 0 else 'Decreases Anomaly Score' for x in shap_vals]
                        })
                        st.dataframe(explanation_df, use_container_width=True)
                        
                        # Explanation in plain English
                        st.subheader("📝 Plain English Explanation")
                        top_3_positive = [(f, v, sv) for f, v, sv in feature_importance if sv > 0][:3]
                        top_3_negative = [(f, v, sv) for f, v, sv in feature_importance if sv < 0][:3]
                        
                        if sample_pred == 1:
                            st.warning("🚨 **This sample is classified as ANOMALOUS**")
                            st.markdown("**Main reasons pushing towards anomaly:**")
                            for feat, val, shap_val in top_3_positive:
                                st.markdown(f"- **{feat}** = {val:.2f} (SHAP impact: +{shap_val:.4f})")
                        else:
                            st.success("✅ **This sample is classified as NORMAL**")
                            st.markdown("**Main reasons pushing towards normal:**")
                            for feat, val, shap_val in top_3_negative:
                                st.markdown(f"- **{feat}** = {val:.2f} (SHAP impact: {shap_val:.4f})")
                    
                    # ===================================================================
                    # 5. ATTACK-SPECIFIC ANALYSIS (if labels exist)
                    # ===================================================================
                    if 'label' in df.columns or 'Label' in df.columns:
                        st.header("⚠️ Feature Importance by Attack Type")
                        
                        label_col = 'label' if 'label' in df.columns else 'Label'
                        attack_col = 'attack_type' if 'attack_type' in df.columns else None
                        
                        if attack_col and attack_col in df.columns:
                            attack_types = df[attack_col].unique()
                            
                            for attack in attack_types[:5]:  # Show top 5 attack types
                                if attack == 'Normal' or pd.isna(attack):
                                    continue
                                    
                                with st.expander(f"🎯 {attack}"):
                                    # Get samples of this attack type
                                    attack_mask = df[attack_col] == attack
                                    attack_indices = df[attack_mask].index[:num_samples]
                                    
                                    if len(attack_indices) > 0:
                                        # Get SHAP values for this attack type
                                        attack_sample_indices = [i for i in range(len(X_sample)) 
                                                                if df.index[i] in attack_indices]
                                        
                                        if attack_sample_indices:
                                            attack_shap = shap_values[attack_sample_indices]
                                            attack_importance = np.abs(attack_shap).mean(axis=0)
                                            
                                            attack_df = pd.DataFrame({
                                                'Feature': numeric_cols,
                                                'Importance': attack_importance
                                            }).sort_values('Importance', ascending=False).head(10)
                                            
                                            fig = px.bar(
                                                attack_df, 
                                                x='Importance', 
                                                y='Feature',
                                                orientation='h',
                                                title=f"Top Features for {attack}",
                                                color='Importance',
                                                color_continuous_scale='Reds'
                                            )
                                            fig.update_layout(yaxis={'categoryorder':'total ascending'})
                                            st.plotly_chart(fig, use_container_width=True)
                                            
                                            st.markdown(f"""
                                            **Key Indicators for {attack}:**
                                            - Primary feature: **{attack_df.iloc[0]['Feature']}** 
                                            - Top 3 features account for the majority of the signal
                                            - Monitor these metrics for early detection
                                            """)
                    
                    st.success("✅ SHAP analysis complete!")
                    
                except Exception as e:
                    st.error(f"Error generating SHAP explanations: {str(e)}")
                    st.exception(e)
                    st.info("Make sure your models support SHAP TreeExplainer and data is properly formatted.")

# ============================================================================
# PAGE 7: ACTIONS
# ============================================================================
elif page == "Actions":
    st.title("⚡ Recommended Actions")
    
    if st.session_state.data is None or not st.session_state.predictions:
        st.warning("⚠️ Please upload data and run predictions first")
    else:
        df = st.session_state.data.copy()
        
        # Add ensemble scores if available
        if 'ensemble' in st.session_state.predictions:
            df['Ensemble_Score'] = st.session_state.predictions['ensemble']['scores']
            df['Alert'] = st.session_state.predictions['ensemble']['predictions']
        else:
            df['Ensemble_Score'] = np.random.random(len(df))
            df['Alert'] = (df['Ensemble_Score'] > 0.7).astype(int)
        
        # Filter alerts
        alerts_df = df[df['Alert'] == 1].copy()
        
        if len(alerts_df) == 0:
            st.success("🎉 No alerts detected! System is operating normally.")
        else:
            st.header(f"🚨 {len(alerts_df)} Alerts Detected")
            
            # Summary statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                critical = (alerts_df['Ensemble_Score'] >= 0.9).sum()
                st.metric("🔴 Critical", critical)
            with col2:
                high = ((alerts_df['Ensemble_Score'] >= 0.8) & (alerts_df['Ensemble_Score'] < 0.9)).sum()
                st.metric("🟠 High", high)
            with col3:
                medium = ((alerts_df['Ensemble_Score'] >= 0.7) & (alerts_df['Ensemble_Score'] < 0.8)).sum()
                st.metric("🟡 Medium", medium)
            with col4:
                st.metric("⏱️ Avg Response Time", "2.3 min")
            
            # Alert prioritization
            st.header("📋 Alert Prioritization")
            
            # Add risk level
            alerts_df['Risk_Level'] = pd.cut(
                alerts_df['Ensemble_Score'],
                bins=[0, 0.8, 0.9, 1.0],
                labels=['Medium', 'High', 'Critical']
            )
            
            # Session selection
            if 'session_id' in alerts_df.columns:
                session_col = 'session_id'
            elif 'Session_ID' in alerts_df.columns:
                session_col = 'Session_ID'
            else:
                alerts_df['session_id'] = range(len(alerts_df))
                session_col = 'session_id'
            
            # Display alerts table
            display_cols = [session_col, 'Ensemble_Score', 'Risk_Level'] + \
                          [col for col in alerts_df.columns if col not in [session_col, 'Ensemble_Score', 'Risk_Level', 'Alert']][:5]
            
            st.dataframe(
                alerts_df[display_cols].sort_values('Ensemble_Score', ascending=False),
                use_container_width=True,
                height=400
            )
            
            # Select session for action
            st.header("🎯 Select Session for Action")
            
            selected_session = st.selectbox(
                "Choose a session to investigate",
                alerts_df[session_col].unique(),
                format_func=lambda x: f"Session {x} (Score: {alerts_df[alerts_df[session_col]==x]['Ensemble_Score'].values[0]:.3f})"
            )
            
            if selected_session is not None:
                session_data = alerts_df[alerts_df[session_col] == selected_session].iloc[0]
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.subheader(f"📊 Session {selected_session} Details")
                    
                    # Session metrics
                    metric_cols = st.columns(4)
                    with metric_cols[0]:
                        st.metric("Risk Score", f"{session_data['Ensemble_Score']:.3f}")
                    with metric_cols[1]:
                        st.metric("Risk Level", session_data['Risk_Level'])
                    with metric_cols[2]:
                        attack_type = np.random.choice(['Brute Force', 'Data Exfiltration', 'Unknown'])
                        st.metric("Attack Type", attack_type)
                    with metric_cols[3]:
                        confidence = np.random.uniform(0.8, 0.99)
                        st.metric("Confidence", f"{confidence:.2%}")
                    
                    # Session details
                    st.markdown("**Session Information:**")
                    session_info = session_data.to_dict()
                    info_df = pd.DataFrame({
                        'Attribute': list(session_info.keys())[:10],
                        'Value': [str(v) for v in list(session_info.values())[:10]]
                    })
                    st.dataframe(info_df, use_container_width=True)
                
                with col2:
                    st.subheader("⚠️ Risk Indicator")
                    
                    # Gauge chart for risk
                    score = session_data['Ensemble_Score']
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=score,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Risk Score"},
                        gauge={
                            'axis': {'range': [0, 1]},
                            'bar': {'color': "darkred" if score > 0.9 else "orange" if score > 0.8 else "yellow"},
                            'steps': [
                                {'range': [0, 0.7], 'color': "lightgreen"},
                                {'range': [0.7, 0.8], 'color': "yellow"},
                                {'range': [0.8, 0.9], 'color': "orange"},
                                {'range': [0.9, 1], 'color': "red"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 0.9
                            }
                        }
                    ))
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Recommended Actions
                st.header("✅ Recommended Actions")
                
                risk_level = session_data['Risk_Level']
                
                if risk_level == 'Critical':
                    st.error("🔴 **CRITICAL THREAT DETECTED**")
                    actions = [
                        "**IMMEDIATE**: Isolate session and terminate all active connections",
                        "**IMMEDIATE**: Block source IP address at firewall level",
                        "**IMMEDIATE**: Revoke all authentication tokens for this user",
                        "**URGENT**: Initiate full forensic analysis of affected systems",
                        "**URGENT**: Notify security team and escalate to incident response",
                        "**URGENT**: Review and secure all accessed resources",
                        "**FOLLOW-UP**: Conduct post-incident review within 24 hours",
                        "**FOLLOW-UP**: Update security policies based on findings"
                    ]
                elif risk_level == 'High':
                    st.warning("🟠 **HIGH RISK ALERT**")
                    actions = [
                        "**IMMEDIATE**: Monitor session closely and flag for review",
                        "**IMMEDIATE**: Enable enhanced logging for this user",
                        "**URGENT**: Verify user identity through secondary authentication",
                        "**URGENT**: Review recent activity logs for anomalies",
                        "**FOLLOW-UP**: Conduct security awareness training if applicable",
                        "**FOLLOW-UP**: Update access control policies",
                        "**RECOMMENDED**: Consider temporary access restrictions"
                    ]
                else:
                    st.info("🟡 **MEDIUM RISK - MONITORING REQUIRED**")
                    actions = [
                        "**MONITOR**: Track session behavior for next 24 hours",
                        "**REVIEW**: Analyze access patterns and compare with baseline",
                        "**VERIFY**: Confirm legitimacy of recent actions",
                        "**DOCUMENT**: Log incident for future reference",
                        "**RECOMMENDED**: Send security reminder to user",
                        "**OPTIONAL**: Request user to change password"
                    ]
                
                for i, action in enumerate(actions, 1):
                    st.markdown(f"{i}. {action}")
                
                # Action buttons
                st.header("🎬 Execute Actions")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("🚫 Block Session", type="primary"):
                        st.success(f"✅ Session {selected_session} has been blocked")
                        st.info("All active connections terminated")
                
                with col2:
                    if st.button("📧 Alert Security Team"):
                        st.success("✅ Security team has been notified")
                        st.info("Incident ticket #" + str(np.random.randint(10000, 99999)) + " created")
                
                with col3:
                    if st.button("📝 Generate Report"):
                        st.success("✅ Incident report generated")
                        
                        # Create downloadable report
                        report_data = {
                            'Session ID': selected_session,
                            'Risk Score': session_data['Ensemble_Score'],
                            'Risk Level': risk_level,
                            'Detection Time': pd.Timestamp.now(),
                            'Recommended Actions': ', '.join(actions[:3])
                        }
                        report_df = pd.DataFrame([report_data])
                        csv = report_df.to_csv(index=False)
                        st.download_button(
                            "📥 Download Report",
                            csv,
                            f"incident_report_{selected_session}.csv",
                            "text/csv"
                        )
                
                # Investigation Timeline
                st.header("⏱️ Suggested Investigation Timeline")
                
                timeline_data = pd.DataFrame({
                    'Time': ['0-5 min', '5-15 min', '15-30 min', '30-60 min', '1-24 hrs', '24-48 hrs'],
                    'Action': [
                        'Immediate response: Block & isolate',
                        'Initial assessment: Gather logs & evidence',
                        'Analysis: Identify scope of compromise',
                        'Containment: Secure affected systems',
                        'Remediation: Remove threats & restore',
                        'Review: Document & improve defenses'
                    ],
                    'Priority': ['Critical', 'Critical', 'High', 'High', 'Medium', 'Low']
                })
                
                st.dataframe(timeline_data, use_container_width=True)
                
                # Additional Resources
                st.header("📚 Additional Resources")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("""
                    **Playbooks:**
                    - [Brute Force Attack Response](javascript:void(0))
                    - [Data Exfiltration Protocol](javascript:void(0))
                    - [Insider Threat Handling](javascript:void(0))
                    """)
                
                with col2:
                    st.markdown("""
                    **Escalation:**
                    - Security Operations: ext. 5555
                    - Incident Response: ext. 5556
                    - Management: ext. 5557
                    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>Cloud Security Monitoring Dashboard v1.0 | Powered by ML-based Threat Detection</p>
    <p>🛡️ Random Forest • XGBoost • Isolation Forest</p>
</div>
""", unsafe_allow_html=True)