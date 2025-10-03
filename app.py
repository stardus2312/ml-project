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
/* --- ตัดส่วน CSS ยาวออกเพื่อความกระชับ --- */
</style>
""", unsafe_allow_html=True)

# ------------------------- LOAD MODELS FROM HUGGING FACE -------------------------
@st.cache_data
def load_models_from_hub():
    """
    Load ML models and TF-IDF vectorizer from Hugging Face Hub
    """
    models = {}
    vectorizer = None
    try:
        # Paths to Hugging Face Hub
        logistic_path = hf_hub_download(repo_id="stardust231/MachineLearning",
                                        filename="Logistic_regression_model.pkl")
        naive_path = hf_hub_download(repo_id="stardust231/MachineLearning",
                                     filename="Naive_bayes_model_test.pkl")
        decision_path = hf_hub_download(repo_id="stardust231/MachineLearning",
                                        filename="decision_tree_model.pkl")  # แก้ชื่อไฟล์ให้ตรง

        # โหลด vectorizer
        try:
            vectorizer_path = hf_hub_download(repo_id="stardust231/MachineLearning",
                                             filename="tfidf_vectorizer.pkl")
            vectorizer = joblib.load(vectorizer_path)
            st.success("✅ TF-IDF Vectorizer loaded successfully")
        except Exception as e:
            st.warning(f"⚠️ TF-IDF Vectorizer not found: {str(e)}")

        # โหลดโมเดล
        models['Logistic Regression'] = joblib.load(logistic_path)
        models['Naive Bayes'] = joblib.load(naive_path)
        models['Decision Tree'] = joblib.load(decision_path)

        st.success("✅ All models loaded successfully from Hugging Face Hub")

    except Exception as e:
        st.error(f"❌ Error loading models from Hugging Face Hub: {str(e)}")

    return models, vectorizer

# ------------------------- DRUG INFORMATION -------------------------
def get_drug_info(drug_name):
    """Get drug information based on drug name"""
    drug_database = {
        'acetaminophen': {'generic_name':'Acetaminophen','usage_category':'Analgesic','molecular_weight':151.16,'logp':0.46,'hbd':2,'hba':2,'rotb':0,'tpsa':46.25,'aromatic_rings':1,'heavy_atoms':11},
        'ibuprofen': {'generic_name':'Ibuprofen','usage_category':'NSAID','molecular_weight':206.29,'logp':3.97,'hbd':1,'hba':2,'rotb':2,'tpsa':37.30,'aromatic_rings':1,'heavy_atoms':15},
        'aspirin': {'generic_name':'Aspirin','usage_category':'Antiplatelet','molecular_weight':180.16,'logp':1.19,'hbd':1,'hba':4,'rotb':1,'tpsa':63.32,'aromatic_rings':1,'heavy_atoms':13},
        'metformin': {'generic_name':'Metformin','usage_category':'Antidiabetic','molecular_weight':129.16,'logp':-0.64,'hbd':4,'hba':4,'rotb':0,'tpsa':87.49,'aromatic_rings':0,'heavy_atoms':9},
        'lisinopril': {'generic_name':'Lisinopril','usage_category':'ACE Inhibitor','molecular_weight':405.49,'logp':1.70,'hbd':3,'hba':6,'rotb':5,'tpsa':138.12,'aromatic_rings':1,'heavy_atoms':29},
        'paracetamol': {'generic_name':'Paracetamol','usage_category':'Analgesic','molecular_weight':151.16,'logp':0.46,'hbd':2,'hba':2,'rotb':0,'tpsa':46.25,'aromatic_rings':1,'heavy_atoms':11},
        'omeprazole': {'generic_name':'Omeprazole','usage_category':'Proton Pump Inhibitor','molecular_weight':345.42,'logp':2.0,'hbd':1,'hba':4,'rotb':3,'tpsa':72.55,'aromatic_rings':2,'heavy_atoms':25},
        'atorvastatin': {'generic_name':'Atorvastatin','usage_category':'Statin','molecular_weight':558.64,'logp':4.1,'hbd':2,'hba':6,'rotb':8,'tpsa':110.38,'aromatic_rings':2,'heavy_atoms':40},
        'amlodipine': {'generic_name':'Amlodipine','usage_category':'Calcium Channel Blocker','molecular_weight':408.88,'logp':3.0,'hbd':1,'hba':4,'rotb':4,'tpsa':72.55,'aromatic_rings':2,'heavy_atoms':29},
        'metoprolol': {'generic_name':'Metoprolol','usage_category':'Beta Blocker','molecular_weight':267.36,'logp':1.88,'hbd':2,'hba':3,'rotb':4,'tpsa':50.72,'aromatic_rings':1,'heavy_atoms':19}
    }
    
    drug_name_lower = drug_name.lower().strip()
    for key, value in drug_database.items():
        if key in drug_name_lower or drug_name_lower in key:
            return value
    # Default
    return {'generic_name': drug_name.title(), 'usage_category':'Therapeutic Agent','molecular_weight':200.0,'logp':2.0,'hbd':2,'hba':3,'rotb':3,'tpsa':60.0,'aromatic_rings':1,'heavy_atoms':20}

# ------------------------- PREDICTION FUNCTIONS -------------------------
def preprocess_input(data):
    required_features = ['molecular_weight','logp','hbd','hba','rotb','tpsa','aromatic_rings','heavy_atoms']
    for f in required_features:
        if f not in data.columns:
            data[f] = 0
    return data[required_features]

def predict_quality_with_tfidf(model, vectorizer, generic_name):
    try:
        X_input = vectorizer.transform([generic_name])
        prediction = model.predict(X_input)
        confidence = np.max(model.predict_proba(X_input))*100 if hasattr(model,'predict_proba') else 85.0
        return prediction[0], confidence
    except Exception as e:
        st.error(f"TF-IDF Prediction error: {str(e)}")
        return None, 0

def predict_quality(model, data):
    try:
        processed_data = preprocess_input(data)
        prediction = model.predict(processed_data)
        confidence = np.max(model.predict_proba(processed_data))*100 if hasattr(model,'predict_proba') else 85.0
        return prediction[0], confidence
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, 0

def get_drug_quality_group(generic_name, accuracy):
    high_quality_drugs = ['acetaminophen', 'paracetamol', 'ibuprofen', 'aspirin']
    medium_quality_drugs = ['metformin', 'lisinopril', 'omeprazole', 'amlodipine']
    generic_lower = generic_name.lower()
    if any(drug in generic_lower for drug in high_quality_drugs):
        if accuracy >= 90: return "A", "#4CAF50"
        elif accuracy >= 80: return "B", "#FF9800"
        else: return "C", "#F44336"
    elif any(drug in generic_lower for drug in medium_quality_drugs):
        if accuracy >= 85: return "A", "#4CAF50"
        elif accuracy >= 75: return "B", "#FF9800"
        else: return "C", "#F44336"
    else:
        if accuracy >= 95: return "A", "#4CAF50"
        elif accuracy >= 85: return "B", "#FF9800"
        else: return "C", "#F44336"

def display_prediction_results(generic_name, usage_category, accuracy, model_name):
    st.markdown(f"### 🎯 Prediction Results")
    quality_group, group_color = get_drug_quality_group(generic_name, accuracy)
    col1, col2, col3, col4 = st.columns([1,1,1,1])
    with col1: st.write(f"💊 Generic Name: {generic_name}")
    with col2: st.write(f"📊 Accuracy: {accuracy:.1f}%")
    with col3: st.write(f"🎯 Target: Human")
    with col4: st.write(f"🏆 Quality Group: {quality_group}")
    st.write(f"🤖 Model Used: {model_name}")

# ------------------------- MAIN APP -------------------------
def main():
    st.title("💊 Drug Quality Classification with ML")
    
    with st.spinner("🔄 Loading ML models..."):
        models, vectorizer = load_models_from_hub()
    
    if not models:
        st.error("❌ No models found! Please check Hugging Face Hub.")
        st.stop()
    
    selected_model = st.sidebar.selectbox("Select Model:", list(models.keys()))
    if vectorizer is not None:
        st.sidebar.success("🔤 Using TF-IDF Text Analysis")
    else:
        st.sidebar.info("🧬 Using Molecular Features")
    
    drug_name = st.text_input("Enter Drug Name:", placeholder="e.g., Acetaminophen")
    
    if st.button("🔮 Predict Quality"):
        if drug_name:
            drug_info = get_drug_info(drug_name)
            generic_name = drug_info['generic_name']
            if vectorizer:
                prediction, confidence = predict_quality_with_tfidf(models[selected_model], vectorizer, generic_name)
            else:
                input_data = pd.DataFrame([drug_info])
                prediction, confidence = predict_quality(models[selected_model], input_data)
            if prediction: display_prediction_results(generic_name, drug_info['usage_category'], confidence, selected_model)
        else:
            st.warning("⚠️ Please enter a drug name to get started!")

if __name__=="__main__":
    main()