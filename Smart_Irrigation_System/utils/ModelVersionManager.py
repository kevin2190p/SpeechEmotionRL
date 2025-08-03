import os
import glob
import re
import json
from datetime import datetime
import pandas as pd

class ModelVersionManager:
    """Managing Versions and Metadata for Reinforcement Learning Models """
    
    def __init__(self, base_dir="./models"):
      
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
    
    def get_next_version(self, model_type):
        
        pattern = f"{self.base_dir}/{model_type}_corn_irrigation_v*.zip"
        existing_models = glob.glob(pattern)
        
        versions = [int(re.search(r'v(\d+)', model).group(1)) 
                    for model in existing_models if re.search(r'v(\d+)', model)]
        
        if not versions:
            return 1
        return max(versions) + 1
    
    def get_model_path(self, model_type, version=None):
        if version is None:
            version = self.get_latest_version(model_type)
            if version == 0:
                raise FileNotFoundError(f"No existing models found for type {model_type}")
    
        return f"{self.base_dir}/{model_type}_corn_irrigation_v{version}"
    
    def get_latest_version(self, model_type):
       
        pattern = f"{self.base_dir}/{model_type}_corn_irrigation_v*.zip"
        existing_models = glob.glob(pattern)
        
        versions = [int(re.search(r'v(\d+)', model).group(1)) 
                    for model in existing_models if re.search(r'v(\d+)', model)]
        
        if not versions:
            return 0
        return max(versions)
    
    def save_model(self, model, model_type, metadata=None):
        version = self.get_next_version(model_type)
        model_path = f"{self.base_dir}/{model_type}_corn_irrigation_v{version}"
        
        # Save model
        model.save(model_path)
        
        if metadata is None:
            metadata = {}
        
        metadata.update({
            "version": version,
            "model_type": model_type,
            "timestamp": datetime.now().isoformat(),
            "model_path": f"{model_path}.zip"
        })
        
        with open(f"{model_path}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=4)
        
        print(f"Model saved to: {model_path}.zip")
        print(f"Metadata saved to: {model_path}_metadata.json")
        
        return model_path
    
    def list_models(self, model_type=None):
        if model_type:
            pattern = f"{self.base_dir}/{model_type}_corn_irrigation_v*.zip"
        else:
            pattern = f"{self.base_dir}/*_corn_irrigation_v*.zip"
        
        model_files = glob.glob(pattern)
        
        models_info = []
        for model_file in model_files:
            metadata_file = model_file.replace('.zip', '_metadata.json')
            
            match = re.search(r'([a-z0-9]+)_corn_irrigation_v(\d+)', model_file)
            if match:
                model_type = match.group(1)
                version = int(match.group(2))
                
                info = {
                    "model_type": model_type,
                    "version": version,
                    "path": model_file
                }
                
                try:
                    if os.path.exists(metadata_file):
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                            info.update(metadata)
                except Exception as e:
                    info["metadata_error"] = str(e)
                
                models_info.append(info)
        
        # Sort by type and version
        models_info.sort(key=lambda x: (x["model_type"], x["version"]))
        return models_info
    
    def get_model_metadata(self, model_type, version=None):
        if version is None:
            version = self.get_latest_version(model_type)
        
        metadata_path = f"{self.base_dir}/{model_type}_corn_irrigation_v{version}_metadata.json"
        
        try:
            with open(metadata_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {"error": "Metadata not found", "model_type": model_type, "version": version}
    
    def compare_models(self, models_list):
        
        results = []
        for model_type, version in models_list:
            metadata = self.get_model_metadata(model_type, version)
            
            # Extraction performance indicators
            record = {
                "model_type": model_type,
                "version": version,
                "timestamp": metadata.get("timestamp", "Unknown")
            }
            
            # Adding assessment indicators
            if "evaluation" in metadata:
                eval_data = metadata["evaluation"]
                record.update(eval_data)
            
            results.append(record)
        
        return pd.DataFrame(results)

if __name__ == "__main__":
    manager = ModelVersionManager()
    
    # List all models
    models = manager.list_models()
    print(f"Found {len(models)} models")
    
    for model in models:
        print(f"{model['model_type']} v{model['version']}")