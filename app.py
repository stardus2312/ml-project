import streamlit as st
import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Drug Quality Classification",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .stApp {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Subheader Styles */
    .stSubheader {
        color: #000000 !important;
    }
    
    h3 {
        color: #000000 !important;
    }
    
    h2 {
        color: #000000 !important;
    }
    
    h1 {
        color: #000000 !important;
    }
    
    /* Streamlit specific elements */
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
        color: #000000 !important;
    }
    
    .stSubheader {
        color: #000000 !important;
    }
    
    .stText {
        color: #000000 !important;
    }
    
    .stSelectbox label, .stTextInput label {
        color: #000000 !important;
    }
    
    /* Sidebar text */
    .css-1d391kg h1, .css-1d391kg h2, .css-1d391kg h3, .css-1d391kg h4, .css-1d391kg h5, .css-1d391kg h6 {
        color: #000000 !important;
    }
    
    .css-1d391kg .stSubheader {
        color: #000000 !important;
    }
    
    /* Additional Streamlit elements */
    .element-container h1, .element-container h2, .element-container h3, .element-container h4, .element-container h5, .element-container h6 {
        color: #000000 !important;
    }
    
    .stApp .main .block-container h1, 
    .stApp .main .block-container h2, 
    .stApp .main .block-container h3, 
    .stApp .main .block-container h4, 
    .stApp .main .block-container h5, 
    .stApp .main .block-container h6 {
        color: #000000 !important;
    }
    
    /* Sidebar specific */
    .stApp .sidebar .block-container h1, 
    .stApp .sidebar .block-container h2, 
    .stApp .sidebar .block-container h3, 
    .stApp .sidebar .block-container h4, 
    .stApp .sidebar .block-container h5, 
    .stApp .sidebar .block-container h6 {
        color: #000000 !important;
    }
    
    /* Main Header */
    .main-header {
        background: linear-gradient(135deg, #2C3E50 0%, #34495E 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 20px 40px rgba(44, 62, 80, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grain" width="100" height="100" patternUnits="userSpaceOnUse"><circle cx="25" cy="25" r="1" fill="rgba(255,255,255,0.1)"/><circle cx="75" cy="75" r="1" fill="rgba(255,255,255,0.1)"/></pattern></defs><rect width="100" height="100" fill="url(%23grain)"/></svg>');
        opacity: 0.3;
    }
    
    .main-header h1 {
        color: #ffffff !important;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
        position: relative;
        z-index: 1;
    }
    
    .main-header p {
        color: #ffffff !important;
        font-size: 1.1rem;
        opacity: 0.9;
        position: relative;
        z-index: 1;
    }
    
    /* Override any global styles for main header */
    .main-header h1, .main-header p {
        color: #ffffff !important;
    }
    
    /* Model Card */
    .model-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 2rem;
        border-radius: 16px;
        border-left: 5px solid #2C3E50;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    .model-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.15);
    }
    
    /* Prediction Result */
    .prediction-result {
        background: linear-gradient(135deg, #34495E 0%, #2C3E50 100%);
        padding: 2rem;
        border-radius: 20px;
        border: 2px solid #2C3E50;
        margin: 1.5rem 0;
        box-shadow: 0 15px 35px rgba(44, 62, 80, 0.2);
        position: relative;
        overflow: hidden;
    }
    
    .prediction-result::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(45deg, transparent 30%, rgba(255,255,255,0.1) 50%, transparent 70%);
        transform: translateX(-100%);
        animation: shimmer 3s infinite;
    }
    
    @keyframes shimmer {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }
    
    .prediction-result h3 {
        font-size: 1.8rem;
        font-weight: 600;
        margin-bottom: 1.5rem;
        text-align: center;
        position: relative;
        z-index: 1;
        color: white;
    }
    
    /* Prediction Field */
    .prediction-field {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 5px solid #2C3E50;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
        transition: all 0.3s ease;
    }
    
    .prediction-field:hover {
        transform: translateX(5px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.12);
    }
    
    .prediction-field h4 {
        color: #000000;
        font-size: 0.9rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.5rem;
    }
    
    .prediction-field p {
        font-size: 1.1rem;
        font-weight: 600;
        margin: 0;
        color: #000000;
    }
    
    /* Metric Card */
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
        text-align: center;
        transition: all 0.3s ease;
        border: 1px solid rgba(102, 126, 234, 0.1);
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.15);
    }
    
    .metric-card h4 {
        color: #000000;
        font-size: 0.9rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.5rem;
    }
    
    .metric-card h2 {
        color: #000000;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    
    /* Button Styles */
    .stButton > button {
        background: linear-gradient(135deg, #2C3E50 0%, #34495E 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(44, 62, 80, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.5s;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(44, 62, 80, 0.4);
    }
    
    .stButton > button:hover::before {
        left: 100%;
    }
    
    /* Input Styles */
    .stTextInput > div > div > input {
        border-radius: 12px;
        border: 2px solid #e1e5e9;
        padding: 0.75rem 1rem;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #2C3E50;
        box-shadow: 0 0 0 3px rgba(44, 62, 80, 0.1);
    }
    
    /* Sidebar Styles */
    .css-1d391kg {
        background: linear-gradient(180deg, #2C3E50 0%, #34495E 100%);
    }
    
    /* Success/Info Messages */
    .stSuccess {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border: 1px solid #c3e6cb;
        border-radius: 12px;
    }
    
    .stInfo {
        background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
        border: 1px solid #bee5eb;
        border-radius: 12px;
    }
    
    .stWarning {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        border: 1px solid #ffeaa7;
        border-radius: 12px;
    }
    
    .stError {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border: 1px solid #f5c6cb;
        border-radius: 12px;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2rem;
        }
        
        .main-header {
            padding: 2rem 1rem;
        }
        
        .prediction-field {
            padding: 1rem;
        }
        
        .metric-card {
            padding: 1.5rem;
        }
    }
    
    /* Loading Animation */
    .loading {
        position: relative;
        overflow: hidden;
    }
    
    .loading::after {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.4), transparent);
        animation: loading 1.5s infinite;
    }
    
    @keyframes loading {
        0% { left: -100%; }
        100% { left: 100%; }
    }
    
    /* Success Animation */
    .success {
        animation: success 0.6s ease-in-out;
    }
    
    @keyframes success {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_models_from_hub():
    models = {}
    vectorizer = None
    try:
        # โหลดโมเดลจาก Hugging Face Hub
        logistic_path = hf_hub_download(repo_id="stardust231/MachineLearning",
                                        filename="logistic_regression_model.pkl")
        naive_path = hf_hub_download(repo_id="stardust231/MachineLearning",
                                     filename="naive_bayes_model.pkl")
        decision_path = hf_hub_download(repo_id="stardust231/MachineLearning",
                                        filename="Drcision_bayes_model.pkl")
        
        # โหลด vectorizer
        try:
            vectorizer_path = hf_hub_download(repo_id="stardust231/MachineLearning",
                                            filename="tfidf_vectorizer.pkl")
            vectorizer = joblib.load(vectorizer_path)
            st.success("✅ TF-IDF Vectorizer loaded successfully")
        except:
            st.warning("⚠️ TF-IDF Vectorizer not found, using fallback method")
        
        models['Logistic Regression'] = joblib.load(logistic_path)
        models['Naive Bayes'] = joblib.load(naive_path)
        models['Decision Tree'] = joblib.load(decision_path)
        
        st.success("✅ All models loaded successfully from Hugging Face Hub")
        
    except Exception as e:
        st.error(f"❌ Error loading models from Hugging Face Hub: {str(e)}")
    
    return models, vectorizer

def get_drug_info(drug_name):
    """Get drug information based on drug name"""
    drug_database = {
        'acetaminophen': {
            'generic_name': 'Acetaminophen',
            'usage_category': 'Analgesic',
            'molecular_weight': 151.16,
            'logp': 0.46,
            'hbd': 2,
            'hba': 2,
            'rotb': 0,
            'tpsa': 46.25,
            'aromatic_rings': 1,
            'heavy_atoms': 11
        },
        'ibuprofen': {
            'generic_name': 'Ibuprofen',
            'usage_category': 'NSAID',
            'molecular_weight': 206.29,
            'logp': 3.97,
            'hbd': 1,
            'hba': 2,
            'rotb': 2,
            'tpsa': 37.30,
            'aromatic_rings': 1,
            'heavy_atoms': 15
        },
        'aspirin': {
            'generic_name': 'Aspirin',
            'usage_category': 'Antiplatelet',
            'molecular_weight': 180.16,
            'logp': 1.19,
            'hbd': 1,
            'hba': 4,
            'rotb': 1,
            'tpsa': 63.32,
            'aromatic_rings': 1,
            'heavy_atoms': 13
        },
        'metformin': {
            'generic_name': 'Metformin',
            'usage_category': 'Antidiabetic',
            'molecular_weight': 129.16,
            'logp': -0.64,
            'hbd': 4,
            'hba': 4,
            'rotb': 0,
            'tpsa': 87.49,
            'aromatic_rings': 0,
            'heavy_atoms': 9
        },
        'lisinopril': {
            'generic_name': 'Lisinopril',
            'usage_category': 'ACE Inhibitor',
            'molecular_weight': 405.49,
            'logp': 1.70,
            'hbd': 3,
            'hba': 6,
            'rotb': 5,
            'tpsa': 138.12,
            'aromatic_rings': 1,
            'heavy_atoms': 29
        },
        'paracetamol': {
            'generic_name': 'Paracetamol',
            'usage_category': 'Analgesic',
            'molecular_weight': 151.16,
            'logp': 0.46,
            'hbd': 2,
            'hba': 2,
            'rotb': 0,
            'tpsa': 46.25,
            'aromatic_rings': 1,
            'heavy_atoms': 11
        },
        'omeprazole': {
            'generic_name': 'Omeprazole',
            'usage_category': 'Proton Pump Inhibitor',
            'molecular_weight': 345.42,
            'logp': 2.0,
            'hbd': 1,
            'hba': 4,
            'rotb': 3,
            'tpsa': 72.55,
            'aromatic_rings': 2,
            'heavy_atoms': 25
        },
        'atorvastatin': {
            'generic_name': 'Atorvastatin',
            'usage_category': 'Statin',
            'molecular_weight': 558.64,
            'logp': 4.1,
            'hbd': 2,
            'hba': 6,
            'rotb': 8,
            'tpsa': 110.38,
            'aromatic_rings': 2,
            'heavy_atoms': 40
        },
        'amlodipine': {
            'generic_name': 'Amlodipine',
            'usage_category': 'Calcium Channel Blocker',
            'molecular_weight': 408.88,
            'logp': 3.0,
            'hbd': 1,
            'hba': 4,
            'rotb': 4,
            'tpsa': 72.55,
            'aromatic_rings': 2,
            'heavy_atoms': 29
        },
        'metoprolol': {
            'generic_name': 'Metoprolol',
            'usage_category': 'Beta Blocker',
            'molecular_weight': 267.36,
            'logp': 1.88,
            'hbd': 2,
            'hba': 3,
            'rotb': 4,
            'tpsa': 50.72,
            'aromatic_rings': 1,
            'heavy_atoms': 19
        }
    }
    
    # Search for drug (case insensitive)
    drug_name_lower = drug_name.lower().strip()
    for key, value in drug_database.items():
        if key in drug_name_lower or drug_name_lower in key:
            return value
    
    # If not found, return default values
    return {
        'generic_name': drug_name.title(),
        'usage_category': 'Therapeutic Agent',
        'molecular_weight': 200.0,
        'logp': 2.0,
        'hbd': 2,
        'hba': 3,
        'rotb': 3,
        'tpsa': 60.0,
        'aromatic_rings': 1,
        'heavy_atoms': 20
    }

def preprocess_input(data):
    required_features = ['molecular_weight','logp','hbd','hba','rotb','tpsa','aromatic_rings','heavy_atoms']
    for f in required_features:
        if f not in data.columns:
            data[f] = 0
    return data[required_features]

def predict_quality_with_tfidf(model, vectorizer, generic_name):
    """Make prediction using TF-IDF vectorizer"""
    try:
        # แปลงชื่อยาเป็น vector
        X_input = vectorizer.transform([generic_name])
        
        # predict
        prediction = model.predict(X_input)
        
        # Get confidence if available
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X_input)
            confidence = np.max(probabilities) * 100
        else:
            confidence = 85.0
        
        return prediction[0], confidence
    except Exception as e:
        st.error(f"TF-IDF Prediction error: {str(e)}")
        return None, 0

def predict_quality(model, data):
    """Fallback prediction method using molecular features"""
    try:
        processed_data = preprocess_input(data)
        prediction = model.predict(processed_data)
        confidence = np.max(model.predict_proba(processed_data))*100 if hasattr(model,'predict_proba') else 85.0
        return prediction[0], confidence
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, 0

def get_drug_quality_group(generic_name, accuracy):
    """Determine drug quality group based on generic name and accuracy"""
    # กำหนดกลุ่มยาตามชื่อและความแม่นยำ
    high_quality_drugs = ['acetaminophen', 'paracetamol', 'ibuprofen', 'aspirin']
    medium_quality_drugs = ['metformin', 'lisinopril', 'omeprazole', 'amlodipine']
    
    generic_lower = generic_name.lower()
    
    # กำหนดกลุ่มตามชื่อยา
    if any(drug in generic_lower for drug in high_quality_drugs):
        if accuracy >= 90:
            return "A", "#4CAF50"  # เขียว
        elif accuracy >= 80:
            return "B", "#FF9800"  # ส้ม
        else:
            return "C", "#F44336"  # แดง
    elif any(drug in generic_lower for drug in medium_quality_drugs):
        if accuracy >= 85:
            return "A", "#4CAF50"  # เขียว
        elif accuracy >= 75:
            return "B", "#FF9800"  # ส้ม
        else:
            return "C", "#F44336"  # แดง
    else:
        # ยาอื่นๆ
        if accuracy >= 95:
            return "A", "#4CAF50"  # เขียว
        elif accuracy >= 85:
            return "B", "#FF9800"  # ส้ม
        else:
            return "C", "#F44336"  # แดง

def display_prediction_results(generic_name, usage_category, accuracy, model_name):
    st.markdown("<div class='prediction-result'><h3>🎯 Prediction Results</h3></div>", unsafe_allow_html=True)
    
    # กำหนดกลุ่มยา
    quality_group, group_color = get_drug_quality_group(generic_name, accuracy)
    
    # Create a more organized grid layout
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    
    with col1:
        st.markdown(f"""
        <div class='prediction-field'>
            <h4>💊 Generic Name</h4>
            <p style='font-size:18px;font-weight:bold;color:#000000;'>{generic_name}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='prediction-field'>
            <h4>📊 Accuracy</h4>
            <p style='font-size:18px;font-weight:bold;color:#000000;'>{accuracy:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class='prediction-field'>
            <h4>🎯 Target</h4>
            <p style='font-size:18px;font-weight:bold;color:#000000;'>Human</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class='prediction-field'>
            <h4>🏆 Quality Group</h4>
            <p style='font-size:24px;font-weight:bold;color:{group_color};'>{quality_group}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Model information in a separate row
    st.markdown(f"""
    <div class='prediction-field'>
        <h4>🤖 Model Used</h4>
        <p style='font-size:16px;color:#000000;'>{model_name}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # เพิ่มคำอธิบายกลุ่ม
    group_explanations = {
        "A": "🟢 กลุ่ม A: คุณภาพสูง - ยาที่มีประสิทธิภาพและความปลอดภัยดี",
        "B": "🟡 กลุ่ม B: คุณภาพปานกลาง - ยาที่มีประสิทธิภาพดีแต่ต้องระวังการใช้",
        "C": "🔴 กลุ่ม C: คุณภาพต่ำ - ยาที่ต้องใช้ด้วยความระมัดระวังสูง"
    }
    
    st.markdown(f"""
    <div class='prediction-field' style='background: {group_color}20; border-left: 5px solid {group_color};'>
        <h4>📋 Quality Assessment</h4>
        <p style='font-size:16px;color:#000000;'>{group_explanations[quality_group]}</p>
    </div>
    """, unsafe_allow_html=True)

def main():
    st.markdown("<div class='main-header'><h1>💊 Drug Quality Classification with ML</h1><p>Predict drug quality using multiple ML models</p></div>", unsafe_allow_html=True)
    
    # Enhanced model loading section
    with st.spinner("🔄 Loading ML models..."):
        models, vectorizer = load_models_from_hub()
    
    if not models:
        st.error("❌ No models found! Please check Hugging Face Hub.")
        st.stop()
    
    # Enhanced sidebar
    st.sidebar.header("⚙️ Model Settings")
    st.sidebar.markdown("---")
    
    selected_model = st.sidebar.selectbox(
        "Select Model:", 
        list(models.keys()),
        help="Choose the machine learning model for prediction"
    )
    
    st.sidebar.markdown(f"""
    <div class='model-card'>
        <h4>🎯 Selected Model</h4>
        <h3 style='color: #000000; margin: 0.5rem 0;'>{selected_model}</h3>
        <p style='color: #000000; margin: 0;'>Type: {type(models[selected_model]).__name__}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show prediction method
    if vectorizer is not None:
        st.sidebar.success("🔤 Using TF-IDF Text Analysis")
    else:
        st.sidebar.info("🧬 Using Molecular Features")
    
    col1, col2 = st.columns([2,1])
    
    with col1:
        st.subheader("📝 Drug Information Input")
        
        # Enhanced drug name input with better styling
        drug_name = st.text_input(
            "Enter Drug Name:", 
            placeholder="e.g., Acetaminophen, Ibuprofen, Aspirin...",
            help="Enter the generic name of the drug you want to analyze"
        )
        
        if st.button("🔮 Predict Quality", use_container_width=True):
            if drug_name:
                # Get drug information
                drug_info = get_drug_info(drug_name)
                generic_name = drug_info['generic_name']
                
                # Choose prediction method
                if vectorizer is not None:
                    # Use TF-IDF method
                    prediction, confidence = predict_quality_with_tfidf(
                        models[selected_model], 
                        vectorizer, 
                        generic_name
                    )
                else:
                    # Use molecular features method
                    input_data = pd.DataFrame({
                        'generic_name': [generic_name],
                        'molecular_weight': [drug_info['molecular_weight']],
                        'logp': [drug_info['logp']],
                        'hbd': [drug_info['hbd']],
                        'hba': [drug_info['hba']],
                        'rotb': [drug_info['rotb']],
                        'tpsa': [drug_info['tpsa']],
                        'aromatic_rings': [drug_info['aromatic_rings']],
                        'heavy_atoms': [drug_info['heavy_atoms']]
                    })
                    prediction, confidence = predict_quality(models[selected_model], input_data)
                
                if prediction is not None:
                    display_prediction_results(
                        generic_name, 
                        drug_info['usage_category'], 
                        confidence, 
                        selected_model
                    )
            else:
                st.warning("⚠️ Please enter a drug name to get started!")
    
    with col2:
        st.subheader("📊 Model Information")
        st.markdown(f"""
        <div class='metric-card'>
            <h4>🎯 Selected Model</h4>
            <h2>{selected_model}</h2>
            <p style='color: #000000; margin: 0.5rem 0;'>Type: {type(models[selected_model]).__name__}</p>
            <div style='margin-top: 1rem; padding: 0.5rem; background: #E8F4F8; border-radius: 8px;'>
                <small style='color: #000000;'>✅ Model Ready for Prediction</small>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("📋 Available Models")
        for model_name in models.keys():
            status = "✅" if model_name==selected_model else "⚪"
            st.markdown(f"""
            <div style='padding: 0.5rem; margin: 0.25rem 0; border-radius: 8px; background: {'#E8F4F8' if model_name==selected_model else '#f8f9fa'};'>
                <strong style='color: #000000;'>{status} {model_name}</strong>
            </div>
            """, unsafe_allow_html=True)
        
        st.subheader("💊 Supported Drugs")
        supported_drugs = [
            "Acetaminophen", "Ibuprofen", "Aspirin", "Metformin", 
            "Lisinopril", "Paracetamol", "Omeprazole", "Atorvastatin", 
            "Amlodipine", "Metoprolol"
        ]
        
        # Create a grid layout for drugs
        drug_cols = st.columns(2)
        for i, drug in enumerate(supported_drugs):
            with drug_cols[i % 2]:
                st.markdown(f"""
                <div style='padding: 0.5rem; margin: 0.25rem 0; border-radius: 8px; background: #f8f9fa; border-left: 3px solid #2C3E50;'>
                    <span style='color: #000000;'>💊 {drug}</span>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Enhanced footer
    st.markdown("""
    <div style='text-align:center; color:#666; padding:2rem; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border-radius: 15px; margin-top: 2rem;'>
        <h4 style='color:#000000; margin-bottom: 0.5rem;'>💊 Drug Quality Classification System</h4>
        <p style='margin: 0; font-size: 0.9rem; color: #000000;'>Powered by Machine Learning | Built with Streamlit</p>
        <p style='margin: 0.5rem 0 0 0; font-size: 0.8rem; opacity: 0.7; color: #000000;'>© 2024 ML Drug Classification</p>
    </div>
    """, unsafe_allow_html=True)

if __name__=="__main__":
    main()