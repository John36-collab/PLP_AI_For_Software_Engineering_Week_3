import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn import tree
import spacy
from collections import Counter
import re
import io
import base64

# Set page config
st.set_page_config(
    page_title="AI Analytics Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set style for matplotlib
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'home'

# Load models and data
@st.cache_resource
def load_iris_model():
    """Load the Iris model if it exists, otherwise train it"""
    model_path = 'iris_decision_tree_model.pkl'
    if os.path.exists(model_path):
        model_data = joblib.load(model_path)
        return model_data
    else:
        # Train a new model
        df = pd.read_csv('Data/Iris.csv')
        
        # Data preprocessing
        label_encoder = LabelEncoder()
        df['Species_encoded'] = label_encoder.fit_transform(df['Species'])
        
        X = df.drop(['Id', 'Species', 'Species_encoded'], axis=1)
        y = df['Species_encoded']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Train model
        dt_classifier = DecisionTreeClassifier(random_state=42, max_depth=3)
        dt_classifier.fit(X_train, y_train)
        
        # Save model
        species_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
        model_data = {
            'model': dt_classifier,
            'label_encoder': label_encoder,
            'feature_names': list(X.columns),
            'species_mapping': species_mapping
        }
        joblib.dump(model_data, model_path)
        
        return model_data

@st.cache_resource
def load_nlp_model():
    """Load spaCy model for NER"""
    try:
        nlp = spacy.load("en_core_web_sm")
        # Set max_length to handle longer texts (up to 10M characters)
        nlp.max_length = 10000000
        return nlp
    except OSError:
        import subprocess
        import sys
        st.warning("Downloading spaCy model...")
        try:
            # Try downloading with more verbose output
            result = subprocess.run(
                [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
                capture_output=True,
                text=True,
                check=True
            )
            st.success("spaCy model downloaded successfully!")
            nlp = spacy.load("en_core_web_sm")
            # Set max_length to handle longer texts (up to 10M characters)
            nlp.max_length = 10000000
            return nlp
        except subprocess.CalledProcessError as e:
            st.error(f"Failed to download spaCy model. Error: {e}")
            st.error(f"Stdout: {e.stdout}")
            st.error(f"Stderr: {e.stderr}")
            st.info("Please try installing the model manually by running:")
            st.code("python -m spacy download en_core_web_sm")
            st.error("Using a basic NLP model instead. Some features may be limited.")
            
            # Return a basic spacy model with minimal functionality
            nlp = spacy.blank("en")
            return nlp

# Sidebar navigation
st.sidebar.title("🤖 AI Analytics Dashboard")
st.session_state.page = st.sidebar.selectbox(
    "Select Analysis",
    ["Home", "Iris Classification", "NER Product Analysis"]
)

# Home Page
def home_page():
    st.title("AI Analytics Dashboard")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image("https://streamlit.io/images/brand/streamlit-logo-secondary-colormark-darktext.png", width=200)
        st.header("About This App")
        st.write("""
        This interactive dashboard combines two powerful AI analyses:
        
        1. **Iris Species Classification** - A machine learning model that predicts Iris species based on flower measurements
        2. **NER Product Analysis** - Natural Language Processing to extract brands and products from product reviews
        """)
    
    with col2:
        st.header("Features")
        st.write("""
        - 🌸 Predict Iris species using Decision Tree algorithm
        - 📊 Visualize feature importance and model performance
        - 🏷️ Extract brands and products from text reviews
        - 😊 Analyze sentiment in product reviews
        - 📈 Generate comprehensive reports and visualizations
        """)
    
    st.markdown("---")
    st.header("How to Use")
    st.write("""
    1. Navigate using the sidebar menu
    2. For Iris Classification:
       - Upload your own data or use the sample
       - Adjust model parameters
       - View predictions and performance metrics
    3. For NER Product Analysis:
       - Enter text or upload review files
       - Extract brands and products automatically
       - Analyze sentiment and view statistics
    """)

# Iris Classification Page
def iris_classification():
    st.title("🌸 Iris Species Classification")
    
    # Load model
    model_data = load_iris_model()
    iris_model = model_data['model']
    label_encoder = model_data['label_encoder']
    feature_names = model_data['feature_names']
    
    # Data upload or use sample
    st.sidebar.header("Data Options")
    use_sample = st.sidebar.checkbox("Use Sample Data", value=True)
    
    if use_sample:
        df = pd.read_csv('Data/Iris.csv')
        st.success("Using sample Iris dataset")
    else:
        uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=['csv'])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.success(f"Loaded {len(df)} rows from uploaded file")
        else:
            st.warning("Please upload a CSV file or use sample data")
            return
    
    # Display data info
    st.subheader("Dataset Information")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Samples", len(df))
    col2.metric("Features", len(feature_names))
    col3.metric("Species", df['Species'].nunique())
    col4.metric("Missing Values", df.isnull().sum().sum())
    
    # Show data preview
    if st.checkbox("Show Data Preview"):
        st.dataframe(df.head())   
    
    # Train/test split
    st.subheader("Model Training")
    test_size = st.slider("Test Size", 0.1, 0.5, 0.3)
    random_state = st.number_input("Random State", 1, 100, 42)
    
    if st.button("Train Model"):
        # Prepare data
        df['Species_encoded'] = label_encoder.fit_transform(df['Species'])
        X = df.drop(['Id', 'Species', 'Species_encoded'], axis=1)
        y = df['Species_encoded']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Train model
        max_depth = st.number_input("Max Depth", 1, 10, 3)
        dt_classifier = DecisionTreeClassifier(random_state=random_state, max_depth=max_depth)
        dt_classifier.fit(X_train, y_train)
        
        # Make predictions
        y_pred = dt_classifier.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{accuracy:.4f}")
        col2.metric("Precision", f"{precision:.4f}")
        col3.metric("Recall", f"{recall:.4f}")
        
        # Show classification report
        st.subheader("Classification Report")
        report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
        st.text(report)
        
        # Confusion matrix
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=label_encoder.classes_,
                   yticklabels=label_encoder.classes_, ax=ax)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title('Confusion Matrix')
        st.pyplot(fig)
        
        # Feature importance
        st.subheader("Feature Importance")
        feature_importance = dt_classifier.feature_importances_
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=importance_df, x='Importance', y='Feature', palette='viridis', ax=ax)
        ax.set_title('Feature Importance')
        st.pyplot(fig)
        
        # Decision tree visualization
        st.subheader("Decision Tree Visualization")
        fig, ax = plt.subplots(figsize=(20, 10))
        tree.plot_tree(dt_classifier, feature_names=feature_names, 
                      class_names=label_encoder.classes_, filled=True, rounded=True, ax=ax)
        st.pyplot(fig)
        
        # Prediction interface
        st.subheader("Make Predictions")
        st.write("Enter flower measurements to predict species:")
        
        # Input sliders
        sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.0)
        sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
        petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.0)
        petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.0)
        
        if st.button("Predict Species"):
            input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
            prediction = dt_classifier.predict(input_data)
            prediction_species = label_encoder.inverse_transform(prediction)[0]
            
            st.success(f"Predicted Species: **{prediction_species}**")
            
            # Show prediction probabilities
            probabilities = dt_classifier.predict_proba(input_data)[0]
            prob_df = pd.DataFrame({
                'Species': label_encoder.classes_,
                'Probability': probabilities
            })
            st.write("Prediction Probabilities:")
            st.dataframe(prob_df.style.format({'Probability': '{:.4f}'}))

# NER Product Analysis Page
def ner_product_analysis():
    st.title("🏷️ NER Product Analysis")
    
    # Load NLP model
    nlp = load_nlp_model()
    
    # Define brand and product lists
    common_brands = [
        'Apple', 'Samsung', 'Sony', 'JVC', 'Canon', 'Nikon', 'Microsoft', 'Google',
        'Amazon', 'Nike', 'Adidas', 'Puma', 'Ford', 'Toyota', 'Honda', 'Apple',
        'Sony', 'Panasonic', 'LG', 'Samsung', 'Xbox', 'PlayStation', 'Nintendo',
        'Coca-Cola', 'Pepsi', 'McDonalds', 'Starbucks', 'Nike', 'Adidas', 'Apple',
        'Sony', 'Microsoft', 'Google', 'Amazon', 'Tesla', 'BMW', 'Mercedes', 'Audi'
    ]
    
    product_keywords = [
        'CD', 'DVD', 'game', 'book', 'charger', 'battery', 'phone', 'laptop', 
        'camera', 'headphones', 'speaker', 'tablet', 'watch', 'car', 'shoes',
        'shirt', 'pants', 'jacket', 'software', 'game', 'movie', 'album',
        'soundtrack', 'player', 'console', 'TV', 'monitor', 'keyboard', 'mouse'
    ]
    
    # Text input or file upload
    st.subheader("Input Text")
    input_method = st.radio("Choose input method:", ["Text Input", "File Upload"])
    
    if input_method == "Text Input":
        text_input = st.text_area("Enter product review text:", height=200, 
                                 placeholder="I love my new iPhone from Apple. The camera is amazing and the battery life is excellent.")
    else:
        uploaded_file = st.file_uploader("Upload text file", type=['txt'])
        if uploaded_file is not None:
            text_input = uploaded_file.read().decode('utf-8')
            st.success(f"Loaded {len(text_input)} characters from file")
        else:
            st.warning("Please upload a text file")
            text_input = ""
    
    if text_input:
        # Extract entities
        def extract_entities(text):
            """Extract product names and brands using multiple techniques"""
            if not text:
                return {'brands': [], 'products': []}
            
            # Process text with spaCy
            doc = nlp(text)
            
            # Extract entities
            brands = []
            products = []
            
            # 1. Named Entity Recognition (if spaCy model has entities)
            if hasattr(doc, 'ents') and doc.ents:
                for ent in doc.ents:
                    if ent.label_ == 'ORG':  # Organizations
                        brands.append(ent.text)
                    elif ent.label_ == 'PRODUCT':  # Products
                        products.append(ent.text)
            
            # 2. Pattern-based extraction for brands
            for brand in common_brands:
                if brand.lower() in text.lower():
                    brands.append(brand)
            
            # 3. Pattern-based extraction for products
            for keyword in product_keywords:
                if keyword.lower() in text.lower():
                    products.append(keyword)
            
            # 4. Extract capitalized words that might be product names
            capitalized_words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
            for word in capitalized_words:
                # Filter out common words and likely brands
                if (len(word.split()) > 1 and
                    word not in brands and
                    not any(brand.lower() in word.lower() for brand in common_brands)):
                    products.append(word)
            
            # Remove duplicates and clean up
            brands = list(set(brands))
            products = list(set(products))
            
            return {'brands': brands, 'products': products}
        
        # Analyze sentiment
        def analyze_sentiment_rule_based(text):
            """Analyze sentiment using rule-based approach"""
            if not text:
                return 'neutral'
            
            # Positive and negative word lists
            positive_words = [
                'great', 'excellent', 'amazing', 'awesome', 'fantastic', 'wonderful',
                'love', 'perfect', 'best', 'brilliant', 'superb', 'outstanding',
                'beautiful', 'gorgeous', 'stunning', 'incredible', 'marvelous', 'splendid',
                'good', 'nice', 'cool', 'happy', 'satisfied', 'pleased', 'delighted',
                'impressed', 'recommend', 'favorite', 'perfect', 'works', 'fine'
            ]
            
            negative_words = [
                'bad', 'terrible', 'awful', 'horrible', 'disgusting', 'hate',
                'worst', 'boring', 'disappointing', 'frustrating', 'annoying', 'useless',
                'broken', 'crapped', 'died', 'stopped', 'quit', 'failed', 'bust',
                'not', 'no', 'never', 'nothing', 'nowhere', 'neither', 'nor',
                'complaint', 'problem', 'issue', 'error', 'bug', 'defect', 'fault'
            ]
            
            # Count positive and negative words
            text_lower = text.lower()
            positive_count = sum(1 for word in positive_words if word in text_lower)
            negative_count = sum(1 for word in negative_words if word in text_lower)
            
            # Determine sentiment
            if positive_count > negative_count:
                return 'positive'
            elif negative_count > positive_count:
                return 'negative'
            else:
                return 'neutral'
        
        # Perform analysis
        st.subheader("Analysis Results")
        
        # Extract entities
        entities = extract_entities(text_input)
        predicted_sentiment = analyze_sentiment_rule_based(text_input)
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Extracted Brands:**")
            if entities['brands']:
                for brand in entities['brands']:
                    st.write(f"- {brand}")
            else:
                st.write("No brands found")
        
        with col2:
            st.write("**Extracted Products:**")
            if entities['products']:
                for product in entities['products']:
                    st.write(f"- {product}")
            else:
                st.write("No products found")
        
        # Sentiment analysis
        st.write(f"**Predicted Sentiment:** {predicted_sentiment}")
        
        # Show spaCy entities
        st.subheader("spaCy Named Entities")
        doc = nlp(text_input)
        
        if hasattr(doc, 'ents') and doc.ents:
            entities_df = []
            for ent in doc.ents:
                # Handle both proper spaCy models and basic models
                label = ent.label_ if hasattr(ent, 'label_') else 'UNKNOWN'
                description = spacy.explain(label) if hasattr(spacy, 'explain') else 'N/A'
                entities_df.append({
                    'Entity': ent.text,
                    'Label': label,
                    'Description': description
                })
            st.dataframe(pd.DataFrame(entities_df))
        else:
            if not isinstance(nlp, spacy.lang.en.English):
                st.write("Using basic NLP model - no advanced entity recognition available")
            else:
                st.write("No named entities found by spaCy")
        
        # Download results
        st.subheader("Export Results")
        
        results_data = {
            'text': [text_input],
            'brands': [', '.join(entities['brands']) if entities['brands'] else 'None'],
            'products': [', '.join(entities['products']) if entities['products'] else 'None'],
            'sentiment': [predicted_sentiment]
        }
        
        results_df = pd.DataFrame(results_data)
        
        csv = results_df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="ner_results.csv">Download Results as CSV</a>'
        st.markdown(href, unsafe_allow_html=True)

# Main app logic
if st.session_state.page == "Home":
    home_page()
elif st.session_state.page == "Iris Classification":
    iris_classification()
elif st.session_state.page == "NER Product Analysis":
    ner_product_analysis()

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit | Powered by scikit-learn and spaCy")