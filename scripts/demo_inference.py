import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference import FireTypePredictor
from src.config import get_project_root

def demo_fire_classification():
    print("\n" + "="*50)
    print("--- Testing Fire Classification (Option B Architecture) ---")
    print("="*50)
    
    predictor = FireTypePredictor()
    
    # 1. Dummy tabular data (Hot and dry -> Forest Fire)
    tabular_dummy_forest = pd.DataFrame([{
        "temperature": 35.0,
        "humidity": 20.0
    }])
    
    # 2. Dummy tabular data (Room temp and normal humidity -> Industrial Fire)
    tabular_dummy_industrial = pd.DataFrame([{
        "temperature": 25.0,
        "humidity": 45.0
    }])
    
    # 3. Find an image
    raw_dir = get_project_root() / 'data' / 'raw'
    test_dir = raw_dir / 'test'
    image_path = None
    if test_dir.exists() and any(test_dir.iterdir()):
        image_path = str(next(test_dir.iterdir()))
        
    print("\n[Test 1] Purely Environmental Tabular Data (Hot & Dry)")
    print(tabular_dummy_forest.iloc[0])
    res1 = predictor.predict_fire_type(tabular_features=tabular_dummy_forest)
    print(f"Result: {res1['predicted_class']} (Prob: {res1['final_forest_probability']})\n")
    
    print("[Test 2] Purely Environmental Tabular Data (Room Temp & Normal Humidity)")
    print(tabular_dummy_industrial.iloc[0])
    res2 = predictor.predict_fire_type(tabular_features=tabular_dummy_industrial)
    print(f"Result: {res2['predicted_class']} (Prob: {res2['final_forest_probability']})\n")
    
    if image_path:
        print(f"[Test 3] Purely Image Data ({Path(image_path).name})")
        res3 = predictor.predict_fire_type(image_path=image_path)
        print(f"Result: {res3['predicted_class']} (Prob: {res3['final_forest_probability']})\n")
        
        print(f"[Test 4] Combined Ensemble (Hot & Dry + {Path(image_path).name})")
        res4 = predictor.predict_fire_type(tabular_features=tabular_dummy_forest, image_path=image_path)
        print(f"Result: {res4['predicted_class']} (Prob: {res4['final_forest_probability']})\n")
    else:
        print("[Test 3] No test image found to run image inference.")

if __name__ == "__main__":
    demo_fire_classification()
