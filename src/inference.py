import joblib
import json
import pandas as pd
from pathlib import Path
from .config import get_project_root, config

class FireTypePredictor:
    """
    Classifies a detected fire into FOREST or INDUSTRIAL using two independent models:
    1. Shared Tabular Model (Temperature, Humidity)
    2. CNN Image Model
    """
    def __init__(self):
        models_dir = get_project_root() / config['output']['models']
        
        # Load Tabular Model
        self.tabular_model = None
        self.tabular_meta = None
        tabular_path = models_dir / 'shared_fire_model.pkl'
        meta_path = models_dir / 'shared_features.json'
        
        if tabular_path.exists() and meta_path.exists():
            self.tabular_model = joblib.load(tabular_path)
            with open(meta_path, 'r') as f:
                self.tabular_meta = json.load(f)
                
        # Load CNN Model
        self.device = None
        self.cnn_model = None
        cnn_path = models_dir / 'forest_fire_image_model.pth'
        if cnn_path.exists():
            import torch
            # pyrefly: ignore [missing-import]
            from torchvision import transforms
            from .cnn_model import get_fire_cnn_model
            
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.cnn_model = get_fire_cnn_model(pretrained=False)
            self.cnn_model.load_state_dict(torch.load(cnn_path, map_location=self.device, weights_only=True))
            self.cnn_model.to(self.device)
            self.cnn_model.eval()
            
            self.img_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def predict_tabular(self, features: pd.DataFrame):
        """Returns probability that the fire is a FOREST fire (1.0 = Forest, 0.0 = Industrial)"""
        if not self.tabular_model or not self.tabular_meta:
            raise ValueError("Tabular model not loaded.")
            
        cols = self.tabular_meta['features']
        X = features[cols].fillna(0)
        
        prob = self.tabular_model.predict_proba(X)[:, 1][0]
        return prob
        
    def predict_image(self, image_path: str):
        """Returns probability that the fire is a FOREST fire based on image"""
        if not self.cnn_model:
            raise ValueError("CNN model not loaded.")
            
        from PIL import Image
        import torch
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.img_transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.cnn_model(input_tensor)
            prob = torch.sigmoid(output).item()
        return prob
        
    def predict_fire_type(self, tabular_features: pd.DataFrame = None, image_path: str = None):
        """
        Combines Tabular and Image predictions to classify the fire type.
        If both are provided, they are averaged.
        """
        p_tab = None
        if tabular_features is not None:
            try:
                p_tab = self.predict_tabular(tabular_features)
            except Exception as e:
                print(f"Tabular prediction failed: {e}")
                
        p_img = None
        if image_path is not None:
            try:
                p_img = self.predict_image(image_path)
            except Exception as e:
                print(f"Image prediction failed: {e}")
                
        probs = [p for p in (p_tab, p_img) if p is not None]
        
        if not probs:
            return {"error": "No valid input provided or models failed to load."}
            
        final_prob = sum(probs) / len(probs)
        fire_type = "FOREST_FIRE" if final_prob >= 0.5 else "INDUSTRIAL_FIRE"
        
        return {
            "tabular_forest_probability": round(float(p_tab), 4) if p_tab is not None else None,
            "image_forest_probability": round(float(p_img), 4) if p_img is not None else None,
            "final_forest_probability": round(float(final_prob), 4),
            "predicted_class": fire_type
        }

if __name__ == "__main__":
    print("Inference module ready.")
