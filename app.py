import streamlit as st
import pandas as pd
import numpy as np
import joblib
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class WaterQualityFeatures:
    '''
    Data class to store water quality feature names and their display information
    '''
    names: List[str]
    display_names: List[str]
    units: List[str]


class FeatureConfiguration:
    '''
    Manages feature configuration and metadata
    '''
    
    @staticmethod
    def get_features() -> WaterQualityFeatures:
        '''
        Retrieves the feature configuration for water quality parameters
        Args: None
        Return: WaterQualityFeatures object containing feature metadata
        '''
        feature_names = [
            'pH', 'Dissolved O2', 'BOD', 'COD', 'Turbidity', 
            'Temperature', 'Conductivity', 'Chlorides', 
            'Total Dissolved Solids', 'Hardness CaCo3', 
            'Nitrate N', 'Phosphate', 'Amonia N', 
            'Total Coliform', 'Fecal Coliform'
        ]
        
        display_names = [
            'pH', 'Dissolved O₂', 'BOD', 'COD', 'Turbidity', 
            'Temperature', 'Conductivity', 'Chlorides', 
            'Total Dissolved Solids', 'Hardness (CaCO₃)', 
            'Nitrate (N)', 'Phosphate', 'Ammonia (N)', 
            'Total Coliform', 'Fecal Coliform'
        ]
        
        units = [
            '', 'mg/L', 'mg/L', 'mg/L', 'NTU', 
            '°C', 'μS/cm', 'mg/L', 'mg/L', 'mg/L', 
            'mg/L', 'mg/L', 'mg/L', 'MPN/100ml', 'MPN/100ml'
        ]
        
        return WaterQualityFeatures(feature_names, display_names, units)


class ClassificationLabels:
    '''
    Manages water quality classification labels and descriptions
    '''
    
    @staticmethod
    def get_labels() -> Dict[int, str]:
        '''
        Returns mapping of class indices to labels
        Args: None
        Return: Dictionary mapping class index to label
        '''
        return {0: "A", 1: "B", 2: "C", 3: "E"}
    
    @staticmethod
    def get_descriptions() -> Dict[str, Dict[str, str]]:
        '''
        Returns detailed descriptions for each water quality class
        Args: None
        Return: Dictionary with class descriptions and use cases
        '''
        return {
            "A": {
                "title": "Class A - Drinking Water Source",
                "description": "Drinking water source after disinfection",
                "criteria": "Low BOD, high DO, low coliform"
            },
            "B": {
                "title": "Class B - Outdoor Bathing",
                "description": "Outdoor bathing (organized)",
                "criteria": "Moderate BOD/DO, controlled coliform"
            },
            "C": {
                "title": "Class C - Drinking Water Source",
                "description": "Drinking water source (conventional treatment)",
                "criteria": "Higher BOD tolerated, treatment required"
            },
            "E": {
                "title": "Class E - Industrial Use",
                "description": "Irrigation, industrial cooling, controlled waste",
                "criteria": "Relaxed standards"
            }
        }


class ModelLoader:
    '''
    Handles loading and caching of the machine learning model
    '''
    
    @staticmethod
    @st.cache_resource
    def load_model(model_path: str):
        '''
        Loads the trained LightGBM model from disk
        Args: model_path (str) - Path to the joblib model file
        Return: Loaded model object
        '''
        return joblib.load(model_path)


class PredictionService:
    '''
    Handles prediction logic using the loaded model
    '''
    
    def __init__(self, model):
        '''
        Initializes prediction service with model
        Args: model - Trained model object
        Return: None
        '''
        self.model = model
    
    def predict(self, features: Dict[str, float]) -> Tuple[str, np.ndarray]:
        '''
        Makes prediction on input features
        Args: features (Dict) - Dictionary of feature names and values
        Return: Tuple of (predicted_class_label, probability_array)
        '''
        feature_values = list(features.values())
        input_array = np.array(feature_values).reshape(1, -1)
        
        prediction = self.model.predict(input_array)[0]
        probabilities = self.model.predict_proba(input_array)[0]
        
        class_labels = ClassificationLabels.get_labels()
        predicted_label = class_labels[prediction]
        
        return predicted_label, probabilities


class UIComponents:
    '''
    Manages UI component rendering
    '''
    
    @staticmethod
    def render_header():
        '''
        Renders the application header
        Args: None
        Return: None
        '''
        st.title("💧 Water Quality Classification System")
        st.markdown("### Predict water quality class based on physicochemical parameters")
        st.divider()
    
    @staticmethod
    def render_input_form(features: WaterQualityFeatures) -> Dict[str, float]:
        '''
        Renders input form for water quality parameters
        Args: features (WaterQualityFeatures) - Feature configuration
        Return: Dictionary of feature names and user input values
        '''
        st.subheader("📊 Enter Water Quality Parameters")
        
        input_values = {}
        
        col1, col2, col3 = st.columns(3)
        columns = [col1, col2, col3]
        
        for idx, (name, display, unit) in enumerate(zip(
            features.names, features.display_names, features.units
        )):
            col_idx = idx % 3
            with columns[col_idx]:
                label = f"{display} {f'({unit})' if unit else ''}"
                input_values[name] = st.number_input(
                    label,
                    value=0.0,
                    format="%.2f",
                    key=name
                )
        
        return input_values
    
    @staticmethod
    def render_prediction_result(predicted_class: str, probabilities: np.ndarray):
        '''
        Renders prediction results with probabilities
        Args: predicted_class (str) - Predicted water quality class
              probabilities (np.ndarray) - Probability distribution
        Return: None
        '''
        st.divider()
        st.subheader("🎯 Prediction Results")
        
        descriptions = ClassificationLabels.get_descriptions()
        class_info = descriptions[predicted_class]
        
        st.success(f"**Predicted Class: {predicted_class}**")
        st.info(f"**{class_info['title']}**")
        st.write(f"**Description:** {class_info['description']}")
        st.write(f"**Criteria:** {class_info['criteria']}")
        
        st.subheader("📈 Class Probabilities")
        
        class_labels = ClassificationLabels.get_labels()
        prob_data = {
            "Class": [f"Class {label}" for label in class_labels.values()],
            "Probability": [f"{prob*100:.2f}%" for prob in probabilities]
        }
        
        prob_df = pd.DataFrame(prob_data)
        st.dataframe(prob_df, hide_index=True, use_container_width=True)
        
        st.bar_chart(
            data=pd.DataFrame({
                "Probability": probabilities * 100,
            }, index=[f"Class {label}" for label in class_labels.values()])
        )
    
    @staticmethod
    def render_sidebar_info():
        '''
        Renders sidebar with additional information
        Args: None
        Return: None
        '''
        st.sidebar.title("ℹ️ About")
        st.sidebar.info(
            "This application classifies water quality into four categories "
            "based on Indian water quality standards."
        )
        
        st.sidebar.subheader("Water Quality Classes")
        descriptions = ClassificationLabels.get_descriptions()
        
        for class_label in ["A", "B", "C", "E"]:
            with st.sidebar.expander(f"Class {class_label}"):
                st.write(descriptions[class_label]["description"])


class WaterQualityApp:
    '''
    Main application controller
    '''
    
    def __init__(self, model_path: str):
        '''
        Initializes the application
        Args: model_path (str) - Path to model file
        Return: None
        '''
        self.model_path = model_path
        self.model = None
        self.prediction_service = None
        self.features = FeatureConfiguration.get_features()
    
    def initialize(self):
        '''
        Initializes model and services
        Args: None
        Return: None
        '''
        try:
            self.model = ModelLoader.load_model(self.model_path)
            self.prediction_service = PredictionService(self.model)
        except FileNotFoundError:
            st.error(f"Model file not found at: {self.model_path}")
            st.stop()
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            st.stop()
    
    def run(self):
        '''
        Runs the main application loop
        Args: None
        Return: None
        '''
        st.set_page_config(
            page_title="Water Quality Classifier",
            page_icon="💧",
            layout="wide"
        )
        
        self.initialize()
        
        UIComponents.render_header()
        UIComponents.render_sidebar_info()
        
        input_values = UIComponents.render_input_form(self.features)
        
        st.divider()
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            predict_button = st.button("🔍 Predict Water Quality Class", use_container_width=True)
        
        if predict_button:
            with st.spinner("Analyzing water quality parameters..."):
                predicted_class, probabilities = self.prediction_service.predict(input_values)
                UIComponents.render_prediction_result(predicted_class, probabilities)


def main():
    '''
    Application entry point
    Args: None
    Return: None
    '''
    model_path = "artifacts\\water_quality_lgbm_model.joblib"
    app = WaterQualityApp(model_path)
    app.run()


if __name__ == "__main__":
    main()