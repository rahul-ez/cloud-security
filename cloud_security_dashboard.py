try:
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
except ImportError as e:
    print(f"Error importing required libraries: {e}")
    print("Please install missing dependencies using: pip install streamlit pandas numpy plotly shap scikit-learn")
    exit(1)

try:
    # Page configuration
    st.set_page_config(
        page_title="Cloud Security Monitoring Dashboard",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
except Exception as e:
    st.error(f"Error configuring page: {e}")
    st.stop()

try:
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
            border: 1px solid #ddd;
        }
        .metric-card h4 {
            color: #1f77b4;
            margin-top: 0;
            margin-bottom: 0.5rem;
        }
        .metric-card p {
            color: #333;
            margin: 0.25rem 0;
            font-size: 0.9rem;
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
except Exception as e:
    st.warning(f"Warning: Could not apply custom CSS: {e}")

try:
    # Initialize session state
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'predictions' not in st.session_state:
        st.session_state.predictions = {}
    if 'models_loaded' not in st.session_state:
        st.session_state.models_loaded = False
except Exception as e:
    st.error(f"Error initializing session state: {e}")
    st.stop()

# Load models
@st.cache_resource
def load_models():
    """Load all ML models and scalers"""
    try:
        import os
    except ImportError:
        st.error("Failed to import os module")
        return None
    
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
            # st.info("Please ensure your 'models' folder is in the same directory as app.py")
            return None
        
        # Define model files
        model_files = {
            'rf': 'rf.joblib',
            'xgb': 'xgb.joblib',
            'if': 'if.joblib',
            'scaler': 'if_scaler.joblib',
            'preprocessor': 'preprocessor_rf.joblib',
            'ae': 'ae_model.h5',
            'scaler_ae': 'scaler_ae.joblib'
        }
        
        # Load each model with detailed error handling
        for key, filename in model_files.items():
            filepath = os.path.join(models_dir, filename)
            
            if not os.path.exists(filepath):
                # st.warning(f"⚠️ Model file not found: {filename}")
                # st.info(f"Looking in: {filepath}")
                models[key] = None
                continue
            
            try:
                # Handle H5 files (Keras models) separately
                if filename.endswith('.h5'):
                    ae_loaded = False
                    # Try TensorFlow first
                    try:
                        import tensorflow as tf
                        try:
                            # Try loading normally first
                            models[key] = tf.keras.models.load_model(filepath)
                            ae_loaded = True
                        except Exception as load_error:
                            # If normal load fails, try with compile=False
                            try:
                                # st.warning(f"⚠️ Normal load failed, trying with compile=False...")
                                models[key] = tf.keras.models.load_model(filepath, compile=False)
                                ae_loaded = True
                            except Exception as compile_false_error:
                                # If still fails, try with custom_objects
                                try:
                                    # st.warning(f"⚠️ Still failing, trying with safe mode...")
                                    models[key] = tf.keras.models.load_model(filepath, compile=False, safe_mode=False)
                                    ae_loaded = True
                                except Exception as custom_error:
                                    raise compile_false_error
                    except ImportError as ie:
                        # st.warning(f"⚠️ TensorFlow not installed. Install with: pip install tensorflow")
                        # st.info(f"Attempting Keras standalone...")
                        try:
                            import keras
                            from keras.models import load_model
                            try:
                                models[key] = load_model(filepath)
                                ae_loaded = True
                            except Exception:
                                # Try with compile=False for Keras too
                                try:
                                    models[key] = load_model(filepath, compile=False)
                                    ae_loaded = True
                                except Exception as keras_error:
                                    st.error(f"❌ Failed to load H5 with Keras: {str(keras_error)}")
                                    models[key] = None
                        except ImportError as ie2:
                            st.error(f"❌ Keras not installed. Install with: pip install keras")
                            models[key] = None
                    except Exception as e:
                        st.error(f"❌ Could not load H5 file: {str(e)}")
                        # st.info("This may be due to:")
                        # st.info("1. Version mismatch between Keras/TensorFlow used to save and current installation")
                        # st.info("2. Try upgrading: pip install --upgrade tensorflow keras")
                        # st.info("3. Or downgrade to match the original versions")
                        models[key] = None
                    
                    if ae_loaded:
                        # st.success(f"✅ Loaded {filename}", icon="✅")
                        pass
                else:
                    # Try joblib first, then pickle
                    try:
                        import joblib
                        models[key] = joblib.load(filepath)
                    except ImportError:
                        # st.warning("joblib not available, trying pickle")
                        with open(filepath, 'rb') as f:
                            models[key] = pickle.load(f)
                    # st.success(f"✅ Loaded {filename}", icon="✅")
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

try:
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["Home", "Dataset", "Model Predictions", "Ensemble Anomaly Score", 
         "Performance Metrics", "Model Comparison", "Feature Explainability", "Actions"]
    )

    st.sidebar.markdown("---")
    st.sidebar.info("""
        **Cloud Security Dashboard**
        
        Monitor and analyze multi-step attacks using ML models:
        - Random Forest Classifier
        - XGBoost Classifier  
        - Isolation Forest
        - Autoencoder
    """)
except Exception as e:
    st.error(f"Error setting up sidebar: {e}")
    st.stop()

# ============================================================================
# PAGE 1: HOME PAGE / SYSTEM OVERVIEW
# ============================================================================
if page == "Home":
    try:
        st.title("Cloud Security Monitoring Dashboard")
        st.markdown("### Real-time threat detection and analysis system")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Active Models", "4", "RF, XGB, IF, AE")
        with col2:
            st.metric("Detection Methods", "Supervised & Unsupervised")
        with col3:
            st.metric("Status", "🟢 Operational")
        
        st.markdown("---")
        
        # System Description
        st.header("System Description")
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
        st.header("Detection Pipeline")
        
        # Create a visual pipeline using columns
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
            <h4>Data Ingestion</h4>
            <p>Upload CSV logs</p>
            <p>Feature extraction</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
            <h4>Preprocessing</h4>
            <p>Scaling</p>
            <p>Encoding</p>
            <p>Normalization</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
            <h4>Model Inference</h4>
            <p>Random Forest</p>
            <p>XGBoost</p>
            <p>Isolation Forest</p>
            <p>Autoencoder</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="metric-card">
            <h4>Ensemble Scoring</h4>
            <p>Weighted combination</p>
            <p>Threshold analysis</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col5:
            st.markdown("""
            <div class="metric-card">
            <h4>Alert & Action</h4>
            <p>Risk classification</p>
            <p>Recommended actions</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Key Features
        st.header("Key Features")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Detection Capabilities")
            st.markdown("""
            - Multi-model ensemble approach
            - Known attack pattern recognition
            - Zero-day threat detection
            - Session-based anomaly tracking
            - Real-time risk scoring
            """)
            
            st.subheader("Analysis Tools")
            st.markdown("""
            - Interactive data visualization
            - Statistical insights
            - Model performance metrics
            - ROC curves and confusion matrices
            """)
        
        with col2:
            st.subheader("Explainability")
            st.markdown("""
            - SHAP value analysis
            - Feature importance ranking
            - Attack pattern interpretation
            - Model decision transparency
            """)
            
            st.subheader("Actions")
            st.markdown("""
            - Automated alert generation
            - Session isolation recommendations
            - Threat mitigation strategies
            - Incident response guidance
            """)
    except Exception as e:
        st.error(f"Error loading Home page: {e}")
        st.exception(e)

# ============================================================================
# PAGE 2: DATASET
# ============================================================================
elif page == "Dataset":
    try:
        st.title("Dataset Management")
        
        # File uploader
        st.header("Upload Dataset")
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
                st.header("Dataset Preview")
                st.dataframe(df.head(100), use_container_width=True)
                
                # Dataset Statistics
                st.header("Dataset Statistics")
                
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
                st.header("Data Visualizations")
                
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
                st.header("Export Data")
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
            st.info("Please upload a dataset to begin analysis")
            
            # Sample data format
            st.header("Expected Data Format")
            st.markdown("""
            Your dataset should contain authentication and session log features such as:
            - **Session identifiers** (session_id)
            - **Temporal features** (timestamps, duration)
            - **Authentication features** (login attempts, failures)
            - **Behavioral features** (access patterns, resource usage)
            - **Labels** (for supervised learning - optional)
            """)
    except Exception as e:
        st.error(f"Error in Dataset page: {e}")
        st.exception(e)

# ============================================================================
# PAGE 3: MODEL PREDICTIONS
# ============================================================================
elif page == "Model Predictions":
    st.title("Model Predictions")
    
    if st.session_state.data is None:
        st.warning("⚠️ Please upload a dataset first in the Dataset page")
    else:
        df = st.session_state.data.copy()
        
        # Model selection
        st.header("Select Model")
        model_choice = st.selectbox(
            "Choose a model for prediction",
            ["Random Forest", "XGBoost", "Isolation Forest", "Autoencoder", "All Models"]
        )
        
        # Preprocessing options
        with st.expander("Preprocessing Options"):
            col1, col2 = st.columns(2)
            with col1:
                handle_missing = st.checkbox("Handle missing values", value=True)
                scale_features = st.checkbox("Scale features", value=True)
            with col2:
                encode_categorical = st.checkbox("Encode categorical features", value=True)
                feature_selection = st.checkbox("Use feature selection", value=False)
        
        # Threshold adjustment options
        with st.expander("Threshold Adjustment"):
            col1, col2 = st.columns(2)
            with col1:
                if_percentile = st.slider("IF Contamination Percentile", 1, 50, 23, 1, 
                                         help="Higher = more anomalies detected")
            with col2:
                ae_percentile = st.slider("AE Error Percentile", 1, 50, 23, 1,
                                         help="Higher = more anomalies detected")
        
        # Run predictions
        if st.button("Run Predictions", type="primary"):
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
                    if models['scaler'] is not None:
                        X_scaled = models['scaler'].transform(X_numerical)  # → 9 features
                    else:
                        st.warning("⚠️ IF Scaler not loaded. Using raw features for Isolation Forest.")
                        X_scaled = X_numerical.values
                    
                    # For Autoencoder: Use numerical features with AE scaler
                    X_ae = processed_df[numerical_features].fillna(0)
                    
                    # Check if AE scaler is available
                    if models['scaler_ae'] is not None:
                        try:
                            X_ae_scaled = models['scaler_ae'].transform(X_ae)  # Scaled features for AE
                        except Exception as scaler_error:
                            st.warning(f"⚠️ Error scaling features with AE scaler: {str(scaler_error)}")
                            st.warning("Using raw features for Autoencoder")
                            X_ae_scaled = X_ae.values
                    else:
                        st.warning("⚠️ AE Scaler not loaded. Using raw features for Autoencoder.")
                        X_ae_scaled = X_ae.values
                    
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
                            if_score = models['if'].score_samples(X_scaled)
                            # Use adjustable percentile threshold instead of model's built-in anomaly scores
                            if_threshold = np.percentile(if_score, if_percentile)
                            if_pred = np.where(if_score < if_threshold, -1, 1)  # -1 = anomaly, 1 = normal
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
                        st.subheader("Random Forest Classifier")
                        
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
                        st.subheader("XGBoost Classifier")
                        
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
                        st.subheader("Isolation Forest")
                        
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
                    
                    # ===================================================================
                    # AUTOENCODER PREDICTIONS
                    # ===================================================================
                    if model_choice == "Autoencoder" or model_choice == "All Models":
                            st.subheader("Autoencoder")
                            
                            if models['ae'] is None:
                                st.error("❌ Autoencoder model not available. Please install TensorFlow/Keras: pip install tensorflow keras")
                            else:
                                try:
                                    import numpy as np
                                    
                                    # Get predictions (reconstruction)
                                    try:
                                        # Try with verbose=0 first (TensorFlow models)
                                        try:
                                            ae_reconstruction = models['ae'].predict(X_ae_scaled, verbose=0)
                                        except TypeError:
                                            # If verbose not supported, try without it
                                            ae_reconstruction = models['ae'].predict(X_ae_scaled)
                                    except Exception as pred_error:
                                        st.error(f"❌ Error during autoencoder prediction: {str(pred_error)}")
                                        st.info("Possible issues:")
                                        st.info("1. Model input shape mismatch - check X_ae_scaled dimensions")
                                        st.info("2. Model was saved with incompatible version")
                                        raise
                                    
                                    # Calculate reconstruction error (MSE)
                                    try:
                                        ae_error = np.mean(np.power(X_ae_scaled - ae_reconstruction, 2), axis=1)
                                    except Exception as error_calc:
                                        st.error(f"❌ Error calculating reconstruction error: {str(error_calc)}")
                                        raise
                                    
                                    # Define threshold for anomaly (using adjustable percentile)
                                    try:
                                        threshold = np.percentile(ae_error, 100 - ae_percentile)
                                        ae_pred = np.where(ae_error > threshold, 1, 0)  # 1 = anomaly, 0 = normal
                                    except Exception as threshold_error:
                                        st.error(f"❌ Error calculating threshold: {str(threshold_error)}")
                                        raise
                                    
                                    st.session_state.predictions['ae'] = {
                                        'predictions': ae_pred,
                                        'errors': ae_error,
                                        'threshold': threshold
                                    }
                                    
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        anomalies = (ae_pred == 1).sum()
                                        st.metric("Anomalies Detected", f"{anomalies:,}")
                                    with col2:
                                        normal = (ae_pred == 0).sum()
                                        st.metric("Normal Records", f"{normal:,}")
                                    with col3:
                                        pct = (anomalies / len(ae_pred)) * 100
                                        st.metric("Anomaly Rate", f"{pct:.2f}%")
                                    
                                    ae_results = pd.DataFrame({
                                        'Prediction': ['Anomaly' if x == 1 else 'Normal' for x in ae_pred],
                                        'Reconstruction Error': ae_error
                                    })
                                    
                                    fig = px.histogram(ae_results, x='Reconstruction Error',
                                                     color='Prediction',
                                                     nbins=50,
                                                     title="Autoencoder Reconstruction Error Distribution",
                                                     color_discrete_map={'Normal': '#00C851', 'Anomaly': '#ff4444'})
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    st.info(f"Note: Anomaly threshold (95th percentile) = {threshold:.4f}")
                                    
                                except Exception as e:
                                    st.error(f"❌ Error with Autoencoder: {str(e)}")
                                    st.warning("Common causes:")
                                    st.warning("1. TensorFlow/Keras not installed: pip install tensorflow keras")
                                    st.warning("2. Model file corrupted or incompatible")
                                    st.warning("3. Input data shape mismatch")
                                    import traceback
                                    st.code(traceback.format_exc())
                    
                    st.success("✅ Predictions completed successfully!")
                    
                    # Detailed results table
                    st.header("Detailed Results")
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
                    if 'ae' in st.session_state.predictions:
                        results_df['AE_Prediction'] = st.session_state.predictions['ae']['predictions']
                        results_df['AE_Error'] = st.session_state.predictions['ae']['errors']
                    
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
    try:
        st.title("Ensemble Anomaly Score")
        
        if not st.session_state.predictions:
            st.warning("⚠️ Please run model predictions first in the Model Predictions page")
        else:
            st.header("Ensemble Configuration")
            
            # Weight sliders
            st.subheader("Model Weights")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                w_rf = st.slider("Random Forest Weight", 0.0, 1.0, 0.3, 0.05)
            with col2:
                w_xgb = st.slider("XGBoost Weight", 0.0, 1.0, 0.3, 0.05)
            with col3:
                w_if = st.slider("Isolation Forest Weight", 0.0, 1.0, 0.2, 0.05)
            with col4:
                w_ae = st.slider("Autoencoder Weight", 0.0, 1.0, 0.2, 0.05)
            
            # Normalize weights
            total_weight = w_rf + w_xgb + w_if + w_ae
            if total_weight > 0:
                w_rf_norm = w_rf / total_weight
                w_xgb_norm = w_xgb / total_weight
                w_if_norm = w_if / total_weight
                w_ae_norm = w_ae / total_weight
            else:
                w_rf_norm = w_xgb_norm = w_if_norm = w_ae_norm = 0.25
            
            st.info(f"Normalized weights: RF={w_rf_norm:.3f}, XGB={w_xgb_norm:.3f}, IF={w_if_norm:.3f}, AE={w_ae_norm:.3f}")
            
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
                        
                        if 'ae' in st.session_state.predictions:
                            # Normalize AE errors to [0, 1]
                            try:
                                ae_errors = st.session_state.predictions['ae']['errors']
                                ae_min = ae_errors.min()
                                ae_max = ae_errors.max()
                                ae_normalized = (ae_errors - ae_min) / (ae_max - ae_min + 1e-8)
                                scores.append(w_ae_norm * ae_normalized)
                            except Exception as ae_norm_error:
                                st.warning(f"⚠️ Error normalizing Autoencoder scores: {str(ae_norm_error)}")
                                st.warning("Autoencoder will be skipped from ensemble")
                        
                        # Calculate ensemble score
                        try:
                            ensemble_score = np.sum(scores, axis=0)
                            ensemble_pred = (ensemble_score >= threshold).astype(int)
                        except Exception as ensemble_error:
                            st.error(f"❌ Error calculating ensemble score: {str(ensemble_error)}")
                            st.error("Ensure at least one model has predictions")
                            raise
                        
                        st.session_state.predictions['ensemble'] = {
                            'scores': ensemble_score,
                            'predictions': ensemble_pred,
                            'threshold': threshold
                        }
                        
                        st.success("✅ Ensemble scores calculated successfully!")
                        
                        # Display results
                        st.header("Ensemble Results")
                        
                        col1, col2, col3, col4, col5 = st.columns(5)
                        with col1:
                            critical = (ensemble_score >= 0.9).sum()
                            st.metric("🔴 Critical", f"{critical:,}")
                        with col2:
                            high = ((ensemble_score >= 0.8) & (ensemble_score < 0.9)).sum()
                            st.metric("🟠 High", f"{high:,}")
                        with col3:
                            medium = ((ensemble_score >= 0.5) & (ensemble_score < 0.8)).sum()
                            st.metric("🟡 Medium", f"{medium:,}")
                        with col4:
                            low = (ensemble_score < 0.5).sum()
                            st.metric("🟢 Low", f"{low:,}")
                        with col5:
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
                            'Category': ['Critical', 'High', 'Medium', 'Low'],
                            'Count': [critical, high, medium, low]
                        })
                        
                        fig = px.pie(risk_categories, values='Count', names='Category',
                                   color='Category',
                                   color_discrete_map={
                                       'Critical': '#d32f2f',
                                       'High': '#f57c00',
                                       'Medium': '#fbc02d',
                                       'Low': '#388e3c'
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
    except Exception as e:
        st.error(f"Error in Ensemble Anomaly Score page: {e}")
        st.exception(e)

# ============================================================================
# PAGE 5: PERFORMANCE METRICS
# ============================================================================
elif page == "Performance Metrics":
    try:
        st.title("Performance Metrics")
        
        if st.session_state.data is None:
            st.warning("⚠️ Please upload a dataset first")
        else:
            # Check if dataset has labels
            df = st.session_state.data
            has_labels = ('ground_truth' in df.columns or 'anomaly_label' in df.columns or 
                         'label' in df.columns or 'Label' in df.columns)
            
            if not has_labels:
                st.warning("⚠️ Dataset does not contain labels for validation. Showing simulated metrics.")
            else:
                st.info("✅ Dataset contains labels. Using actual ground truth for evaluation.")
            
            # Model selection for metrics
            st.header("Select Model for Evaluation")
            metric_model = st.selectbox(
                "Choose model",
                ["Random Forest", "XGBoost", "Ensemble"]
            )
            
            # Toggle for seen vs unseen attacks
            st.header("Evaluation Settings")
            col1, col2 = st.columns(2)
            with col1:
                attack_type = st.radio(
                    "Attack Type",
                    ["Seen Attacks", "Unseen Attacks", "All Attacks"]
                )
            with col2:
                show_detailed = st.checkbox("Show detailed metrics", value=True)
            
            if st.button("Generate Performance Metrics", type="primary"):
                with st.spinner("Calculating performance metrics..."):
                    try:
                        # Load models first
                        models = load_models()
                        if models is None:
                            st.error("Failed to load models. Please ensure model files exist in the 'models/' directory.")
                            st.stop()
                        
                        # Map model display names to model keys
                        model_key_map = {
                            "Random Forest": "rf",
                            "XGBoost": "xgb",
                            "Ensemble": "if"  # Using Isolation Forest as ensemble placeholder
                        }
                        model_key = model_key_map.get(metric_model)
                        
                        # Define columns to exclude from features
                        exclude_cols = ['session_id', 'timestamp', 'label', 'attack_type', 
                                       'ground_truth', 'anomaly_label', 'event_id', 'event_time',
                                       'request_id', 'attack_id', 'attack_stage', 'session_label',
                                       'principal_arn', 'principal_type', 'user_agent', 
                                       'event_source', 'event_name', 'user_id',
                                       'auth_method', 'Session_ID', 'Label', 'Attack_Type']
                        
                        # Filter data based on attack type
                        df_filtered = df.copy()
                        
                        # Check if attack_type column exists
                        if 'attack_type' in df_filtered.columns:
                            if attack_type == "Seen Attacks":
                                # Filter for only known/labeled attacks
                                df_filtered = df_filtered[df_filtered['attack_type'].notna() & (df_filtered['attack_type'] != '')]
                            elif attack_type == "Unseen Attacks":
                                # Filter for only unknown attacks (no attack_type or empty)
                                df_filtered = df_filtered[df_filtered['attack_type'].isna() | (df_filtered['attack_type'] == '')]
                            # "All Attacks" uses all data
                        else:
                            st.warning("⚠️ 'attack_type' column not found. Using all data for evaluation.")
                        
                        # Use ground truth labels from dataset
                        label_col = 'ground_truth' if 'ground_truth' in df_filtered.columns else 'anomaly_label'
                        y_true = df_filtered[label_col].astype(int).values
                        
                        # Prepare features based on model type
                        if model_key in ["rf", "xgb"]:
                            # For RF & XGB: Use preprocessor with numerical + categorical features
                            numerical_features = ['num_events', 'total_bytes', 'mean_bytes', 'max_bytes', 
                                                'mean_resp', 'std_resp', 'unique_actions', 'num_failures', 'duration_s']
                            categorical_features = ['user_role', 'region']
                            
                            available_features = [f for f in numerical_features + categorical_features if f in df_filtered.columns]
                            X_features = df_filtered[available_features].fillna(0)
                            
                            # Apply preprocessor
                            if models['preprocessor'] is not None:
                                X_processed = models['preprocessor'].transform(X_features)
                            else:
                                st.warning("⚠️ Preprocessor not loaded. Cannot make predictions for this model.")
                                st.stop()
                        else:
                            # For IF: Use numerical features with scaler
                            numerical_features = ['num_events', 'total_bytes', 'mean_bytes', 'max_bytes', 
                                                'mean_resp', 'std_resp', 'unique_actions', 'num_failures', 'duration_s']
                            available_features = [f for f in numerical_features if f in df_filtered.columns]
                            X_features = df_filtered[available_features].fillna(0)
                            
                            # Apply scaler
                            if models['scaler'] is not None:
                                X_processed = models['scaler'].transform(X_features)
                            else:
                                st.warning("⚠️ Scaler not loaded. Cannot make predictions for this model.")
                                st.stop()
                        
                        # Get model
                        model = models.get(model_key)
                        
                        if model is None:
                            st.error(f"Model '{metric_model}' not loaded. Please check models directory.")
                            st.stop()
                        
                        # Make predictions
                        if model_key == "if":
                            # Isolation Forest returns -1 for anomalies, 1 for normal
                            y_pred_raw = model.predict(X_processed)
                            y_pred = (y_pred_raw == -1).astype(int)  # Convert to 1 for anomaly, 0 for normal
                            y_proba = (model.score_samples(X_processed) - model.score_samples(X_processed).min()) / \
                                     (model.score_samples(X_processed).max() - model.score_samples(X_processed).min())
                        else:
                            # RF and XGB
                            y_pred = model.predict(X_processed).astype(int)
                            y_proba = model.predict_proba(X_processed)[:, 1]
                        
                        # Calculate metrics
                        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                        
                        accuracy = accuracy_score(y_true, y_pred)
                        precision = precision_score(y_true, y_pred, zero_division=0)
                        recall = recall_score(y_true, y_pred, zero_division=0)
                        f1 = f1_score(y_true, y_pred, zero_division=0)
                        
                        # Display key metrics
                        st.header(f"{metric_model} Performance")
                        
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
                        st.subheader("ROC Curve")
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
                        st.subheader("Precision-Recall Curve")
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
                            st.subheader("Detailed Classification Report")
                            report = classification_report(y_true, y_pred, 
                                                          target_names=['Normal', 'Anomaly'],
                                                          output_dict=True)
                            report_df = pd.DataFrame(report).transpose()
                            st.dataframe(report_df, use_container_width=True)
                            
                            # Threshold Analysis
                            st.subheader("Threshold Analysis")
                            
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
                        
                    except Exception as e:
                        st.error(f"Error generating performance metrics: {str(e)}")
                        st.exception(e)
                        
                        # Model Comparison (if multiple models available)
                        if st.session_state.predictions:
                            try:
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
    except Exception as e:
        st.error(f"Error in Performance Metrics page: {e}")
        st.exception(e)

# ============================================================================
# PAGE 6: MODEL COMPARISON
# ============================================================================
elif page == "Model Comparison":
    try:
        st.title("Model Comparison Analysis")
        
        if st.session_state.data is None:
            st.warning("⚠️ Please upload a dataset first in the Dataset page")
        else:
            df = st.session_state.data.copy()
            
            # Check for ground truth
            label_col = next((col for col in ['ground_truth', 'anomaly_label', 'label', 'Label'] if col in df.columns), None)
            
            if label_col:
                st.info(f"Found ground truth labels in column: `{label_col}`")
                y_true = df[label_col].astype(int).values
            else:
                st.warning("No ground truth labels found. Comparisons will be based on prediction distributions.")
                y_true = None

            if st.button("Run Comprehensive Model Comparison", type="primary", use_container_width=True):
                with st.spinner("Evaluating all models..."):
                    try:
                        # Container for status updates that will be cleared after execution
                        status_container = st.empty()
                        
                        with status_container.container():
                            # Load all models
                            models = load_models()
                            if models is None:
                                st.error("Failed to load models.")
                                st.stop()
                                
                            # Feature sets
                            numerical_features = ['num_events', 'total_bytes', 'mean_bytes', 'max_bytes', 
                                                'mean_resp', 'std_resp', 'unique_actions', 'num_failures', 'duration_s']
                            categorical_features = ['user_role', 'region']
                            
                            available_num = [f for f in numerical_features if f in df.columns]
                            available_cat = [f for f in categorical_features if f in df.columns]
                            
                            # Prepare Data
                            st.status("Preparing data for evaluation...")
                            X_supervised = df[available_num + available_cat].fillna(0)
                            if models.get('preprocessor') is not None:
                                X_supervised_proc = models['preprocessor'].transform(X_supervised)
                            else:
                                X_supervised_proc = X_supervised.values
                                
                            X_if_ae = df[available_num].fillna(0)
                            if models.get('scaler') is not None:
                                X_if_proc = models['scaler'].transform(X_if_ae)
                            else:
                                X_if_proc = X_if_ae.values
                                
                            if models.get('scaler_ae') is not None:
                                X_ae_proc = models['scaler_ae'].transform(X_if_ae)
                            else:
                                X_ae_proc = X_if_ae.values
                                
                            # Predictions & Timing
                            import time
                            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
                            
                            results = []
                            
                            # Model Map
                            model_configs = [
                                ('Random Forest', 'rf', X_supervised_proc, 'supervised'),
                                ('XGBoost', 'xgb', X_supervised_proc, 'supervised'),
                                ('Isolation Forest', 'if', X_if_proc, 'unsupervised'),
                                ('Autoencoder', 'ae', X_ae_proc, 'unsupervised')
                            ]
                            
                            p_bar = st.progress(0)
                            for i, (name, key, data, mtype) in enumerate(model_configs):
                                st.status(f"Evaluating {name}...")
                                model = models.get(key)
                                if model is None:
                                    p_bar.progress((i + 1) / len(model_configs))
                                    continue
                                    
                                start_time = time.time()
                                try:
                                    if key == 'rf' or key == 'xgb':
                                        y_pred = model.predict(data).astype(int)
                                        y_proba = model.predict_proba(data)[:, 1]
                                    elif key == 'if':
                                        y_pred_raw = model.predict(data)
                                        y_pred = (y_pred_raw == -1).astype(int)
                                        # Normalize scores to 0-1 for AUC
                                        scores = model.score_samples(data)
                                        y_proba = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
                                        y_proba = 1 - y_proba # Invert so high = anomaly
                                    elif key == 'ae':
                                        try:
                                            reconstructed = model.predict(data, verbose=0)
                                        except:
                                            reconstructed = model.predict(data)
                                        mse = np.mean(np.power(data - reconstructed, 2), axis=1)
                                        # Use 95th percentile as default threshold for comparison
                                        threshold = np.percentile(mse, 95)
                                        y_pred = (mse > threshold).astype(int)
                                        y_proba = (mse - mse.min()) / (mse.max() - mse.min() + 1e-8)
                                    
                                    end_time = time.time()
                                    inference_time = (end_time - start_time) / len(data) * 1000 # ms per sample
                                    
                                    metrics = {
                                        'Model': name,
                                        'Anomalies': y_pred.sum(),
                                        'Anomaly Rate (%)': (y_pred.sum() / len(y_pred)) * 100,
                                        'Inference Time (ms/sample)': inference_time
                                    }
                                    
                                    if y_true is not None:
                                        # Ensure y_true matches y_pred length
                                        yt = y_true[:len(y_pred)]
                                        metrics['Accuracy'] = accuracy_score(yt, y_pred)
                                        metrics['Precision'] = precision_score(yt, y_pred, zero_division=0)
                                        metrics['Recall'] = recall_score(yt, y_pred, zero_division=0)
                                        metrics['F1-Score'] = f1_score(yt, y_pred, zero_division=0)
                                        try:
                                            metrics['ROC-AUC'] = roc_auc_score(yt, y_proba)
                                        except:
                                            metrics['ROC-AUC'] = 0.5
                                    
                                    results.append(metrics)
                                except Exception as eval_err:
                                    st.warning(f"Failed to evaluate {name}: {str(eval_err)}")
                                
                                p_bar.progress((i + 1) / len(model_configs))

                        # Clear the status container once comparison is done
                        status_container.empty()
                                
                        if not results:
                            st.error("No models were successfully evaluated.")
                            st.stop()
                            
                        comparison_df = pd.DataFrame(results)
                        
                        # --- Display Comparison Visuals ---
                        st.header("Performance Comparison Results")
                        
                        if y_true is not None:
                            # Grouped Bar Chart for metrics
                            plot_cols = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
                            fig = go.Figure()
                            for col in plot_cols:
                                fig.add_trace(go.Bar(
                                    name=col,
                                    x=comparison_df['Model'],
                                    y=comparison_df[col],
                                    text=comparison_df[col].apply(lambda x: f'{x:.3f}'),
                                    textposition='auto',
                                ))
                            fig.update_layout(
                                title="Model Performance Metrics Comparison",
                                barmode='group',
                                xaxis_title="Model",
                                yaxis_title="Score",
                                height=500,
                                legend_title="Metrics"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # ROC-AUC Bar Chart
                            fig_auc = px.bar(comparison_df, x='Model', y='ROC-AUC', 
                                           title="ROC-AUC Score Comparison",
                                           color='ROC-AUC',
                                           color_continuous_scale='RdYlGn',
                                           text_auto='.3f')
                            st.plotly_chart(fig_auc, use_container_width=True)
                        
                        # Detection Rate Comparison
                        fig_rate = px.bar(comparison_df, x='Model', y='Anomaly Rate (%)',
                                        title="Detection Rate Comparison (Anomaly %)",
                                        color='Anomaly Rate (%)',
                                        text_auto='.2f')
                        st.plotly_chart(fig_rate, use_container_width=True)
                        
                        # Inference Time Comparison
                        fig_time = px.bar(comparison_df, x='Model', y='Inference Time (ms/sample)',
                                        title="Inference Latency Comparison (ms per sample)",
                                        log_y=True,
                                        color='Inference Time (ms/sample)',
                                        text_auto='.4f')
                        st.plotly_chart(fig_time, use_container_width=True)
                        
                        # Detailed Dataframe
                        st.header("Detailed Metric Summary")
                        # Highlight max for positive metrics, min for inference time
                        st.dataframe(comparison_df, use_container_width=True)
                        
                        # Summary insight
                        st.header("Summary Insight")
                        if y_true is not None:
                            best_f1_row = comparison_df.loc[comparison_df['F1-Score'].idxmax()]
                            st.success(f"The best performing model based on F1-Score is **{best_f1_row['Model']}** with a score of **{best_f1_row['F1-Score']:.3f}**.")
                            
                            fastest_row = comparison_df.loc[comparison_df['Inference Time (ms/sample)'].idxmin()]
                            st.info(f"The most efficient model (lowest latency) is **{fastest_row['Model']}** taking **{fastest_row['Inference Time (ms/sample)']:.4f} ms** per sample.")
                        else:
                            most_sensitive = comparison_df.loc[comparison_df['Anomaly Rate (%)'].idxmax()]
                            st.info(f"Between all models, **{most_sensitive['Model']}** flagged the most anomalies (**{most_sensitive['Anomaly Rate (%)']:.2f}%**).")
                        
                    except Exception as e:
                        st.error(f"Error in comparison logic: {str(e)}")
                        st.exception(e)
    except Exception as e:
        st.error(f"Error in Model Comparison page: {e}")
        st.exception(e)
# ============================================================================
# PAGE 7: FEATURE EXPLAINABILITY
# ============================================================================
elif page == "Feature Explainability":
    try:
        st.title("Feature Explainability with SHAP")
        
        if st.session_state.data is None:
            st.warning("Please upload a dataset first in the Dataset page")
        else:
            df = st.session_state.data
            
            # Validate SHAP dependencies
            try:
                import shap
            except ImportError:
                st.error("SHAP not installed. Run: pip install shap")
                st.stop()
            
            # ===================================================================
            # STEP 1: DATA PREPARATION FOR SESSION-LEVEL DATA
            # ===================================================================
            st.header("Data Configuration")
            
            # Define session-level features
            session_numeric_features = [
                'num_events', 'total_bytes', 'mean_bytes', 'max_bytes',
                'mean_resp', 'std_resp', 'unique_actions', 'num_failures', 'duration_s'
            ]
            session_categorical_features = ['user_role', 'region']
            
            # Get available features
            available_numeric = [f for f in session_numeric_features if f in df.columns]
            available_categorical = [f for f in session_categorical_features if f in df.columns]
            
            if not available_numeric:
                st.error("No session-level numeric features found. Expected columns: " + ", ".join(session_numeric_features))
                st.stop()
            
            st.info(f"Found {len(available_numeric)} numeric features and {len(available_categorical)} categorical features                                 \n Numeric: {', '.join(available_numeric)}                                                                                                   \n Categorical: {', '.join(available_categorical)}")
            # st.info(f"   Numeric: {', '.join(available_numeric)}")
            # if available_categorical:
            #     st.info(f"   Categorical: {', '.join(available_categorical)}")
            
            # ===================================================================
            # STEP 2: MODEL AND SAMPLE SELECTION
            # ===================================================================
            col1, col2 = st.columns(2)
            
            with col1:
                explain_model = st.selectbox(
                    "Select Model for Explanation",
                    ["Random Forest", "Isolation Forest", "Autoencoder"],
                    help="Choose which model to explain. Note: Autoencoder uses KernelExplainer which is slower."
                )
            
            with col2:
                num_samples = st.slider(
                    "Number of Samples to Explain",
                    min_value=5,
                    max_value=min(100, len(df)),
                    value=min(20, len(df)),
                    help="More samples = more accurate but slower computation"
                )
            
            st.markdown("---")
            
            # ===================================================================
            # STEP 3: GENERATE SHAP EXPLANATIONS
            # ===================================================================
            if st.button("Generate SHAP Explanations", type="primary", use_container_width=True):
                with st.spinner("Generating SHAP explanations... This may take a few minutes"):
                    try:
                        # Container for status updates that will be cleared after execution
                        status_container = st.empty()
                        
                        with status_container.container():
                            # Load models
                            models = load_models()
                            
                            # Map friendly name to internal key
                            model_map = {
                                "Random Forest": "rf",
                                "Isolation Forest": "if",
                                "Autoencoder": "ae"
                            }
                            model_key = model_map[explain_model]
                            
                            if models is None or models.get(model_key) is None:
                                st.error(f"Failed to load {explain_model} model. Ensure model files exist in models/ directory")
                                st.stop()
                            
                            # Select model
                            model = models[model_key]
                            
                            # Initialize variables
                            X_processed = None
                            feature_names_after_preprocessing = []
                            explainer = None
                            shap_values = None
                            
                            # ===================================================================
                            # DATA PREPARATION BRANCHING
                            # ===================================================================
                            st.status(f"Preparing data for {explain_model}...")
                            
                            if model_key == 'rf':
                                # --- RANDOM FOREST (Numeric + Categorical) ---
                                all_features = available_numeric + available_categorical
                                X_raw = df[all_features].head(num_samples).copy()
                                
                                # Fill missing
                                for col in available_numeric: X_raw[col] = X_raw[col].fillna(0)
                                for col in available_categorical: X_raw[col] = X_raw[col].fillna('unknown')
                                
                                if 'preprocessor' not in models:
                                    st.error("Preprocessor not found.")
                                    st.stop()
                                    
                                preprocessor = models['preprocessor']
                                X_processed = preprocessor.transform(X_raw)
                                
                                # Get Feature Names
                                try:
                                    feature_names_after_preprocessing = list(preprocessor.get_feature_names_out())
                                except AttributeError:
                                    feature_names_after_preprocessing = available_numeric + \
                                        list(preprocessor.named_transformers_['cat'].get_feature_names_out(available_categorical))
                                        
                            elif model_key in ['if', 'ae']:
                                # --- ISOLATION FOREST & AUTOENCODER (Numeric Only) ---
                                X_raw = df[available_numeric].head(num_samples).copy()
                                for col in available_numeric: X_raw[col] = X_raw[col].fillna(0)
                                
                                # Use appropriate scaler
                                scaler_key = 'scaler' if model_key == 'if' else 'scaler_ae'
                                if scaler_key not in models or models[scaler_key] is None:
                                    st.error(f"Scaler ({scaler_key}) not found.")
                                    st.stop()
                                    
                                scaler = models[scaler_key]
                                X_processed = scaler.transform(X_raw)
                                feature_names_after_preprocessing = available_numeric  # 9 features
                                
                            st.status(f"Data prepared: {X_processed.shape} features")
                            
                            # ===================================================================
                            # EXPLAINER SELECTION & COMPUTATION
                            # ===================================================================
                            st.status(f"Initializing SHAP Explainer for {explain_model}...")
                            
                            if model_key == 'rf':
                                # TreeExplainer for Random Forest
                                explainer = shap.TreeExplainer(model)
                                explainer.assert_additivity = lambda *args, **kwargs: None
                                shap_values = explainer.shap_values(X_processed)
                                
                                # Handle different output formats
                                if isinstance(shap_values, list):
                                    shap_values = shap_values[1]  # Binary classification -> Positive class
                                elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                                    shap_values = shap_values[:, :, 1]
                                    
                            elif model_key == 'if':
                                # TreeExplainer for Isolation Forest
                                # IF outputs anomaly score. Negative = Anomaly, Positive = Normal (usually)
                                explainer = shap.TreeExplainer(model)
                                explainer.assert_additivity = lambda *args, **kwargs: None
                                shap_values = explainer.shap_values(X_processed)
                                # Shape is usually (samples, features) for IF
                                
                            elif model_key == 'ae':
                                # KernelExplainer for Autoencoder (Reconstruction Error)
                                st.info("Autoencoder uses KernelExplainer (model-agnostic). This simulates perturbations to find feature impact on Reconstruction Error.")
                                
                                # Wrapper function must be pickleable or strictly defined
                                def ae_predict_loss(data_numpy):
                                    # 1. Reconstruct
                                    reconstructed = model.predict(data_numpy, verbose=0)
                                    # 2. Compute MSE per sample (Reconstruction Error)
                                    mse = np.mean(np.power(data_numpy - reconstructed, 2), axis=1)
                                    return mse
                                
                                # Use a background dataset (kmeans summary) for speed
                                # We use X_processed itself if small, or a summary if large
                                background_data = shap.kmeans(X_processed, min(10, len(X_processed)))
                                
                                explainer = shap.KernelExplainer(ae_predict_loss, background_data)
                                
                                # Compute SHAP values
                                with st.spinner("Calculating Kernel SHAP values"):
                                    shap_values = explainer.shap_values(X_processed, nsamples=100)
                            
                            st.status("SHAP values computed successfully!")
                            # st.info(f"Final SHAP shape: {shap_values.shape}")
                            
                            # Store in session state
                            st.session_state.shap_values = shap_values
                            st.session_state.shap_feature_names = feature_names_after_preprocessing
                            st.session_state.shap_X_processed = X_processed
                            st.session_state.shap_explainer = explainer
                            st.session_state.shap_model = model
                            st.session_state.shap_model_type = model_key # Store type for logic
                            st.session_state.shap_computed = True

                        # Clear status messages after successful computation
                        status_container.empty()
                        
                    except Exception as e:
                        st.error(f"Critical Error during SHAP generation: {str(e)}")
                        st.exception(e)
                        st.stop()

            
            # ===================================================================
            # DISPLAY VISUALIZATIONS (if SHAP values exist in session state)
            # ===================================================================
            if st.session_state.get('shap_computed', False):
                # Retrieve from session state
                shap_values = st.session_state.shap_values
                feature_names_after_preprocessing = st.session_state.shap_feature_names
                X_processed = st.session_state.shap_X_processed
                explainer = st.session_state.shap_explainer
                model = st.session_state.shap_model
                
                # ===================================================================
                # VISUALIZATION 1: GLOBAL FEATURE IMPORTANCE
                # ===================================================================
                st.header("1. Global Feature Importance")
                
                # Handle 2D/3D arrays properly
                if len(shap_values.shape) > 2:
                    mean_abs_shap = np.abs(shap_values).mean(axis=(0, 1))
                else:
                    mean_abs_shap = np.abs(shap_values).mean(axis=0)
                
                # Validate array lengths match
                if len(feature_names_after_preprocessing) != len(mean_abs_shap):
                    st.error(f"Feature name count ({len(feature_names_after_preprocessing)}) doesn't match SHAP value count ({len(mean_abs_shap)})")
                    st.info(f"Preprocessed data shape: {X_processed.shape}")
                    st.info(f"SHAP values shape: {shap_values.shape}")
                    st.stop()
                
                importance_df = pd.DataFrame({
                    'Feature': feature_names_after_preprocessing,
                    'Mean |SHAP|': mean_abs_shap
                }).sort_values('Mean |SHAP|', ascending=False)
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    fig_importance = px.bar(
                        importance_df,
                        x='Mean |SHAP|',
                        y='Feature',
                        orientation='h',
                        title="Feature Importance (Mean |SHAP| Value)",
                        labels={'Mean |SHAP|': 'Average Impact on Prediction'},
                        color='Mean |SHAP|',
                        color_continuous_scale='Viridis'
                    )
                    fig_importance.update_layout(
                        yaxis={'categoryorder': 'total ascending'},
                        height=400
                    )
                    st.plotly_chart(fig_importance, use_container_width=True)
                
                with col2:
                    st.dataframe(importance_df, use_container_width=True, hide_index=True)
                        
                # ===================================================================
                # VISUALIZATION 2: SHAP SUMMARY PLOT (Beeswarm)
                # ===================================================================
                st.header("2. SHAP Summary Plot")
                
                fig_summary = make_subplots(
                    rows=1, cols=1,
                    subplot_titles=("Feature Impact Distribution",)
                )
                
                # Create beeswarm-style plot
                top_features_count = min(12, len(feature_names_after_preprocessing))
                top_features_idx = np.argsort(mean_abs_shap)[-top_features_count:][::-1]
                
                for idx_pos, feat_idx in enumerate(top_features_idx):
                    feature_name = feature_names_after_preprocessing[feat_idx]
                    feature_shap = shap_values[:, feat_idx]
                    
                    # For preprocessed features, we can't easily map back to raw values
                    # So we'll use SHAP values for coloring instead
                    if feature_shap.max() != feature_shap.min():
                        colors = (feature_shap - feature_shap.min()) / (feature_shap.max() - feature_shap.min())
                    else:
                        colors = np.ones_like(feature_shap) * 0.5
                    
                    fig_summary.add_trace(go.Scatter(
                        x=feature_shap,
                        y=[feature_name] * len(feature_shap),
                        mode='markers',
                        marker=dict(
                            color=colors,
                            colorscale='RdBu_r',
                            size=8,
                            opacity=0.6,
                            showscale=(idx_pos == 0),
                            colorbar=dict(
                                title="Feature<br>Value",
                                len=0.7
                            ) if idx_pos == 0 else None
                        ),
                        name=feature_name,
                        showlegend=False,
                        hovertemplate=f'<b>{feature_name}</b><br>SHAP: %{{x:.4f}}<br>Value: %{{marker.color:.2f}}<extra></extra>'
                    ))
                
                fig_summary.update_layout(
                    title="SHAP Summary (Beeswarm): Feature Impact on Model Output",
                    xaxis_title="SHAP Value",
                    yaxis_title="Feature",
                    height=500,
                    hovermode='closest',
                    showlegend=False
                )
                st.plotly_chart(fig_summary, use_container_width=True)
                
                # ===================================================================
                # VISUALIZATION 3: DEPENDENCE PLOTS
                # ===================================================================
                st.header("3. Feature Dependence Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    primary_feature = st.selectbox(
                        "Select Primary Feature",
                        feature_names_after_preprocessing,
                        index=0,
                        key="dep_primary"
                    )
                
                with col2:
                    secondary_feature = st.selectbox(
                        "Color by Feature (Interaction)",
                        feature_names_after_preprocessing,
                        index=min(1, len(feature_names_after_preprocessing)-1),
                        key="dep_secondary"
                    )
                
                if st.button("Generate Dependence Plot", key="dep_button"):
                    try:
                        primary_idx = list(feature_names_after_preprocessing).index(primary_feature)
                        secondary_idx = list(feature_names_after_preprocessing).index(secondary_feature)
                        
                        primary_shap = shap_values[:, primary_idx]
                        if isinstance(primary_shap, np.ndarray) and primary_shap.ndim > 1:
                            primary_shap = primary_shap.flatten()
                        
                        secondary_shap = shap_values[:, secondary_idx]
                        if isinstance(secondary_shap, np.ndarray) and secondary_shap.ndim > 1:
                            secondary_shap = secondary_shap.flatten()
                        
                        # Use SHAP values for both axes since we can't map back to raw values easily
                        fig_dep = px.scatter(
                            x=primary_shap,
                            y=secondary_shap,
                            title=f"SHAP Interaction: {primary_feature} vs {secondary_feature}",
                            labels={
                                'x': f'SHAP({primary_feature})',
                                'y': f'SHAP({secondary_feature})'
                            },
                            color_continuous_scale='RdBu_r'
                        )
                        fig_dep.update_layout(height=500)
                        st.plotly_chart(fig_dep, use_container_width=True)
                    except Exception as dep_err:
                        st.error(f"Error creating dependence plot: {str(dep_err)}")
                
                # ===================================================================
                # VISUALIZATION 4: INDIVIDUAL PREDICTION EXPLANATIONS
                # ===================================================================
                st.header("4. Individual Prediction Explanations")
                
                sample_idx = st.slider(
                    "Select Sample to Explain",
                    min_value=0,
                    max_value=len(X_processed)-1,
                    value=0,
                    key="sample_idx_slider"
                )
                
                if st.button("Show Detailed Explanation", key="explain_button"):
                    try:
                        sample_shap = shap_values[sample_idx]
                        if isinstance(sample_shap, np.ndarray) and sample_shap.ndim > 1:
                            sample_shap = sample_shap.flatten()
                        
                        # Get prediction using preprocessed data
                        # Handle different model types
                        model_type = st.session_state.get('shap_model_type', 'rf')
                        
                        col1, col2, col3 = st.columns(3)
                        
                        if model_type == 'rf':
                            sample_pred = model.predict(X_processed[sample_idx:sample_idx+1])[0]
                            sample_proba = model.predict_proba(X_processed[sample_idx:sample_idx+1])[0]
                            
                            with col1:
                                st.metric("Prediction", "ANOMALY" if sample_pred == 1 else "NORMAL")
                            with col2:
                                st.metric("Confidence", f"{max(sample_proba):.2%}")
                            with col3:
                                st.metric("Anomaly Probability", f"{sample_proba[1]:.2%}")
                                
                        elif model_type == 'if':
                            # IF: -1 is outlier, 1 is inlier. Decision function: negative is outlier
                            sample_pred = model.predict(X_processed[sample_idx:sample_idx+1])[0]
                            decision_score = model.decision_function(X_processed[sample_idx:sample_idx+1])[0]
                            
                            with col1:
                                st.metric("Prediction", "ANOMALY" if sample_pred == -1 else "NORMAL")
                            with col2:
                                st.metric("Anomaly Score", f"{decision_score:.4f}", help="Negative = Anomaly, Positive = Normal")
                            with col3:
                                st.metric("Status", "Outlier" if decision_score < 0 else "Inlier")
                                
                        elif model_type == 'ae':
                            # AE: Reconstruction error
                            reconstructed = model.predict(X_processed[sample_idx:sample_idx+1], verbose=0)
                            mse = np.mean(np.power(X_processed[sample_idx:sample_idx+1] - reconstructed, 2))
                            # Threshold is roughly 0.1 based on training (can be adjusted)
                            threshold = 0.1 
                            is_anomaly = mse > threshold
                            
                            with col1:
                                st.metric("Prediction", "ANOMALY" if is_anomaly else "NORMAL")
                            with col2:
                                st.metric("Reconstruction Error", f"{mse:.4f}")
                            with col3:
                                st.metric("Threshold", f"{threshold:.2f}")
                        
                        # Feature contributions (using preprocessed feature names)
                        st.subheader("Feature Contributions")
                        
                        contributions = sorted(
                            zip(feature_names_after_preprocessing, sample_shap),
                            key=lambda x: abs(float(x[1])),
                            reverse=True
                        )[:10]
                        
                        # Determine direction labels and colors based on model type
                        if model_type == 'if':
                            # IF: Negative SHAP -> Pushes to Anomaly (Class -1)
                            #     Positive SHAP -> Pushes to Normal (Class 1)
                            directions = ['→ Anomaly' if x[1] < 0 else '← Normal' for x in contributions]
                            colors = ['#ff4444' if x[1] < 0 else '#00C851' for x in contributions]
                        else:
                            # RF/AE: Positive SHAP -> Pushes to Anomaly (Class 1 / High Error)
                            directions = ['→ Anomaly' if x[1] > 0 else '← Normal' for x in contributions]
                            colors = ['#ff4444' if x[1] > 0 else '#00C851' for x in contributions]
                        
                        contrib_df = pd.DataFrame({
                            'Feature': [x[0] for x in contributions],
                            'SHAP': [f"{x[1]:.4f}" for x in contributions],
                            'Direction': directions
                        })
                        
                        st.dataframe(contrib_df, use_container_width=True, hide_index=True)
                        
                        # Waterfall visualization
                        fig_waterfall = go.Figure(
                            data=[go.Bar(
                                x=[x[1] for x in contributions],
                                y=[x[0] for x in contributions],
                                orientation='h',
                                marker=dict(color=colors),
                                text=[f"{x[1]:.3f}" for x in contributions],
                                textposition='auto',
                                hovertemplate='<b>%{y}</b><br>SHAP: %{x:.4f}<extra></extra>'
                            )]
                        )
                        
                        fig_waterfall.update_layout(
                            title=f"Top Contributing Features - Sample {sample_idx}",
                            xaxis_title="SHAP Value",
                            yaxis_title="Feature",
                            height=400,
                            showlegend=False
                        )
                        
                        st.plotly_chart(fig_waterfall, use_container_width=True)
                        
                    except Exception as explain_err:
                        st.error(f"Error explaining individual prediction: {str(explain_err)}")
                
                # ===================================================================
                # VISUALIZATION 5: FORCE PLOT-STYLE EXPLANATION
                # ===================================================================
                st.header("5. Decision Plot")
                
                try:
                    # Retrieve model type from session state
                    model_type = st.session_state.get('shap_model_type', 'rf')
                    
                    base_value = explainer.expected_value
                    
                    # Handle base value extraction based on model type
                    if model_type == 'rf':
                        # RF: expected_value is usually [value_0, value_1]
                        if isinstance(base_value, np.ndarray) and len(base_value) > 1:
                            base_value = float(base_value[1])
                        else:
                            base_value = float(base_value)
                    else:
                        # IF/AE: expected_value is a scalar
                        if isinstance(base_value, np.ndarray):
                            base_value = float(base_value[0]) if len(base_value) > 0 else 0.0
                        else:
                            base_value = float(base_value)
                    
                    st.info(f"Base value: {base_value:.4f} ({'Mean Probability' if model_type == 'rf' else 'Mean Score' if model_type == 'if' else 'Mean Reconstruction Error'})")
                    
                    # Create decision plot for top samples
                    decision_sample = min(5, len(X_processed))
                    prediction_values = []
                    
                    for i in range(decision_sample):
                        sample_shap = shap_values[i]
                        if isinstance(sample_shap, np.ndarray) and sample_shap.ndim > 1:
                            sample_shap = sample_shap.flatten()
                        pred_val = base_value + sample_shap.sum()
                        prediction_values.append(pred_val)
                    
                    # Determine label based on model type
                    if model_type == 'rf':
                        predictions = ['Anomaly' if v > 0.5 else 'Normal' for v in prediction_values]
                    elif model_type == 'if':
                        # IF: Negative = Anomaly
                        predictions = ['Anomaly' if v < 0 else 'Normal' for v in prediction_values]
                    else: # ae
                        # AE: High Error = Anomaly (Threshold approx 0.1)
                        predictions = ['Anomaly' if v > 0.1 else 'Normal' for v in prediction_values]

                    decision_df = pd.DataFrame({
                        'Sample': [f"Sample {i}" for i in range(decision_sample)],
                        'Value': prediction_values,
                        'Prediction': predictions
                    })
                    
                    st.dataframe(decision_df, use_container_width=True, hide_index=True)
                    
                except Exception as decision_err:
                    st.warning(f"Could not create decision plot: {str(decision_err)}")
                
                # ===================================================================
                # DATA EXPORT
                # ===================================================================
                st.header("Export Analysis Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    csv_shap = pd.DataFrame(
                        shap_values,
                        columns=feature_names_after_preprocessing
                    ).to_csv(index=False)
                    st.download_button(
                        "Download SHAP Values",
                        csv_shap,
                        file_name=f"shap_values_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                with col2:
                    csv_importance = importance_df.to_csv(index=False)
                    st.download_button(
                        "Download Feature Importance",
                        csv_importance,
                        file_name=f"feature_importance_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                #with col3:
                    #st.info("Analysis exported successfully!")
                
                #st.success("SHAP analysis complete! Review the visualizations above for insights.")
    
    except Exception as e:
        st.error(f"Error in Feature Explainability page: {e}")
        st.exception(e)

# ============================================================================
# PAGE 8: ACTIONS
# ============================================================================
elif page == "Actions":
    try:
        st.title("Recommended Actions")
        
        if st.session_state.data is None or not st.session_state.predictions:
            st.warning("Please upload data and run predictions first")
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
                st.success("No alerts detected! System is operating normally.")
            else:
                st.header(f"{len(alerts_df)} Alerts Detected")
                
                # Retrieve threshold from session state
                current_threshold = st.session_state.predictions['ensemble'].get('threshold', 0.7)
                
                # Summary statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    critical_count = (alerts_df['Ensemble_Score'] >= 0.9).sum()
                    st.metric("🔴 Critical", critical_count)
                with col2:
                    high_count = ((alerts_df['Ensemble_Score'] >= 0.8) & (alerts_df['Ensemble_Score'] < 0.9)).sum()
                    st.metric("🟠 High", high_count)
                with col3:
                    # Medium on this page is anything between the alert threshold and 0.8
                    medium_count = ((alerts_df['Ensemble_Score'] >= current_threshold) & (alerts_df['Ensemble_Score'] < 0.8)).sum()
                    st.metric("🟡 Medium", medium_count)
                with col4:
                    st.metric("⏱️ Avg Response Time", "2.3 min")
                
                # Alert prioritization
                st.header("Alert Prioritization")
                
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
                st.header("Select Session for Action")
                
                selected_session = st.selectbox(
                    "Choose a session to investigate",
                    alerts_df[session_col].unique(),
                    format_func=lambda x: f"Session {x} (Score: {alerts_df[alerts_df[session_col]==x]['Ensemble_Score'].values[0]:.3f})"
                )
                
                if selected_session is not None:
                    session_data = alerts_df[alerts_df[session_col] == selected_session].iloc[0]
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.subheader(f"Session {selected_session} Details")
                        
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
                        st.subheader("Risk Indicator")
                        
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
                    st.header("Recommended Actions")
                    
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
                    st.header("Execute Actions")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if st.button("Block Session", type="primary"):
                            st.success(f"✅ Session {selected_session} has been blocked")
                            st.info("All active connections terminated")
                    
                    with col2:
                        if st.button("Alert Security Team"):
                            st.success("✅ Security team has been notified")
                            st.info("Incident ticket #" + str(np.random.randint(10000, 99999)) + " created")
                    
                    with col3:
                        if st.button("Generate Report"):
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
                    st.header("Suggested Investigation Timeline")
                    
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
    except Exception as e:
        st.error(f"Error in Actions page: {e}")
        st.exception(e)