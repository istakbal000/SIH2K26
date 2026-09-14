import os
import sys
import json
import numpy as np
import pandas as pd
import random
import torch
import joblib
from PIL import Image
# pyrefly: ignore [missing-import]
from torchvision import transforms, models
import torch.nn as nn
import warnings

warnings.filterwarnings('ignore')

def get_forest_osm_score():
    return random.uniform(0.60, 0.75)

def get_industrial_osm_score():
    return random.uniform(0.60, 0.75)

def build_cnn_model():
    model = models.mobilenet_v2(pretrained=False)
    model.classifier[1] = nn.Sequential(
        nn.Linear(model.last_channel, 1),
        nn.Sigmoid()
    )
    return model

def predict_cnn(model_path, image_path):
    if not os.path.exists(model_path):
        return 0.5
    
    try:
        model = build_cnn_model()
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
        model.eval()
        
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        image = Image.open(image_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            output = model(input_tensor).squeeze()
            if output.dim() == 0:
                output = output.unsqueeze(0)
            prob = output.item()
        return prob
    except Exception as e:
        print(f"CNN Error: {e}")
        return 0.5

def predict_rf(model_path, cols_path, input_dict):
    if not os.path.exists(model_path) or not os.path.exists(cols_path):
        return 0.5
        
    try:
        rf = joblib.load(model_path)
        cols = joblib.load(cols_path)
        
        # Create dataframe with expected columns
        df = pd.DataFrame([input_dict])
        
        # Add missing columns with 0
        for col in cols:
            if col not in df.columns:
                df[col] = 0.0
                
        # Keep only expected columns in correct order
        X = df[cols].fillna(0)
        
        prob = rf.predict_proba(X)[0][1]
        return prob
    except Exception as e:
        print(f"RF Error: {e}")
        return 0.5

def main():
    print("========================================")
    print("      DUAL FIRE INFERENCE TEST")
    print("========================================")
    
    # Check if models exist
    models_exist = os.path.exists('outputs/models/Forest_Random_Forest.pkl')
    if not models_exist:
        print("ERROR: Models not found in 'outputs/models/'.")
        print("Please wait for 'master_pipeline.py' to finish training and saving models.")
        return

    # Randomly select true scenario to ensure both outputs appear roughly 50% of the time
    scenario = random.choice(["Forest", "Industrial"])
    print(f"\n[SIMULATED SCENARIO: {scenario} Fire]")

    # If Forest scenario, pass Forest image to BOTH pipelines
    if scenario == "Forest":
        test_img = 'data/raw/forest_fire/fire_train_1001.jpg'
        
        # Forest gets high-confidence tabular data
        forest_input = {
            'brightness': random.uniform(330.0, 360.0), 
            'bright_t31': random.uniform(290.0, 310.0), 
            'frp': random.uniform(10.0, 50.0), 
            'confidence': random.uniform(80.0, 100.0), 
            'is_day': 1.0, 
            'hour': random.uniform(12.0, 16.0), 
            'month': random.uniform(6.0, 9.0), 
            'distance_to_forest': random.uniform(0.0, 1000.0), 
            'distance_to_industrial_area': random.uniform(10000.0, 50000.0), 
            'forest_area_ratio_1km': random.uniform(0.8, 1.0), 
            'industrial_area_ratio_1km': 0.0, 
            'latitude': 20.0, 
            'longitude': 71.0, 
            'fire_cluster_id': 1.0, 
            'cluster_size': random.uniform(500.0, 2000.0), 
            'max_frp': random.uniform(100.0, 500.0), 
            'mean_frp': random.uniform(10.0, 50.0), 
            'frp_sum': random.uniform(1000.0, 5000.0)
        }
        
        # Industrial gets low-confidence tabular data
        industrial_input = {
            'CO2_Room': random.uniform(400.0, 450.0), 
            'H2_Room': 0.0, 
            'PM05_Room': random.uniform(10.0, 20.0), 
            'PM100_Room': 0.0, 
            'PM10_Room': random.uniform(0.0, 5.0), 
            'PM25_Room': 0.0, 
            'PM40_Room': 0.0, 
            'PM_Room_Typical_Size': 0.2, 
            'PM_Total_Room': random.uniform(10.0, 30.0), 
            'VOC_Room_RAW': 0.0, 
            'Temperature_Room': random.uniform(20.0, 25.0), 
            'Humidity_Room': random.uniform(40.0, 60.0), 
            'CO_Room': 0.0
        }
        
    else:
        # Industrial Scenario
        test_img = 'data/raw/industrial_fire/Industrial fire/images.jpg'
        
        # Forest gets low-confidence tabular data
        forest_input = {
            'brightness': random.uniform(290.0, 305.0), 
            'bright_t31': random.uniform(270.0, 285.0), 
            'frp': random.uniform(0.0, 2.0), 
            'confidence': random.uniform(0.0, 20.0), 
            'is_day': random.choice([0.0, 1.0]), 
            'hour': 12.0, 
            'month': 1.0, 
            'distance_to_forest': random.uniform(50000.0, 100000.0), 
            'distance_to_industrial_area': random.uniform(0.0, 500.0), 
            'forest_area_ratio_1km': 0.0, 
            'industrial_area_ratio_1km': random.uniform(0.8, 1.0), 
            'latitude': 20.0, 
            'longitude': 71.0, 
            'fire_cluster_id': 0.0, 
            'cluster_size': 0.0, 
            'max_frp': 0.0, 
            'mean_frp': 0.0, 
            'frp_sum': 0.0
        }
        
        # Industrial gets high-confidence tabular data
        industrial_input = {
            'CO2_Room': random.uniform(1500.0, 3000.0), 
            'H2_Room': random.uniform(1.0, 5.0), 
            'PM05_Room': random.uniform(1000.0, 4000.0), 
            'PM100_Room': random.uniform(1.0, 5.0), 
            'PM10_Room': random.uniform(100.0, 600.0), 
            'PM25_Room': random.uniform(5.0, 20.0), 
            'PM40_Room': random.uniform(1.0, 5.0), 
            'PM_Room_Typical_Size': random.uniform(0.45, 0.55), 
            'PM_Total_Room': random.uniform(1000.0, 5000.0), 
            'VOC_Room_RAW': random.uniform(2.0, 10.0), 
            'Temperature_Room': random.uniform(40.0, 80.0), 
            'Humidity_Room': random.uniform(10.0, 30.0), 
            'CO_Room': random.uniform(1.0, 5.0)
        }
    
    # We pass the SAME image to both pipelines to see how they handle it
    forest_img = test_img
    ind_img = test_img
    
    print("\n--- Running Predictions ---")
    
    # Forest Evaluation
    forest_tab_conf = predict_rf('outputs/models/Forest_Random_Forest.pkl', 'outputs/models/Forest_Random_Forest_cols.pkl', forest_input)
    forest_img_conf = predict_cnn('outputs/models/Forest_CNN.pth', forest_img)
    forest_osm_conf = get_forest_osm_score()
    
    forest_final_score = (0.30 * forest_tab_conf) + (0.40 * forest_img_conf) + (0.30 * forest_osm_conf)
    forest_confidence_pct = forest_final_score * 100
    
    # Industrial Evaluation
    ind_tab_conf = predict_rf('outputs/models/Industrial_Random_Forest.pkl', 'outputs/models/Industrial_Random_Forest_cols.pkl', industrial_input)
    ind_img_conf = predict_cnn('outputs/models/Industrial_CNN.pth', ind_img)
    ind_osm_conf = get_industrial_osm_score()
    
    ind_final_score = (0.30 * ind_tab_conf) + (0.40 * ind_img_conf) + (0.30 * ind_osm_conf)
    ind_confidence_pct = ind_final_score * 100

    # Decision
    if forest_confidence_pct > ind_confidence_pct:
        final_fire_type = "Forest Fire"
        final_conf = forest_confidence_pct
    else:
        final_fire_type = "Industrial Fire"
        final_conf = ind_confidence_pct

    print("\n========================================")
    print("        FIRE PREDICTION RESULT")
    print("========================================")
    print("FOREST FIRE")
    print(f"Tabular : {forest_tab_conf * 100:.2f}%")
    print(f"Image   : {forest_img_conf * 100:.2f}%")
    print(f"OSM     : {forest_osm_conf * 100:.2f}%")
    print("-------------------------")
    print(f"Final   : {forest_confidence_pct:.2f}%")
    print("\nINDUSTRIAL FIRE")
    print(f"Tabular : {ind_tab_conf * 100:.2f}%")
    print(f"Image   : {ind_img_conf * 100:.2f}%")
    print(f"OSM     : {ind_osm_conf * 100:.2f}%")
    print("-------------------------")
    print(f"Final   : {ind_confidence_pct:.2f}%")
    print("\n========================================")
    print(f"FINAL PREDICTION: {final_fire_type}")
    print(f"CONFIDENCE: {final_conf:.2f}%")
    print("========================================")


if __name__ == "__main__":
    main()
