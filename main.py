
from fastapi import FastAPI

app = FastAPI()

@app.get("/")  # Removed the extra quote
def root():
    return {"message": "Welcome To Tuwaiq Academy"}  # Return a dictionary

#python -m uvicorn main:app --reload

from fastapi import FastAPI, HTTPException

app = FastAPI()

# GET request
@app.get("/")
def read_root():
    return {"message": "Welcome to Tuwaiq Academy"}

# POST request
@app.post("/items/{item_id}")
def create_item(item_id: int):  # Correct function definition
    return {"item": item_id}


import joblib
model = joblib. load ('knn_model.joblib')
scaler = joblib. load ('scaler.joblib')


from pydantic import BaseModel

# Define a Pydantic model for input data validation
# Note: Feel free to change the parameters in this class according to your model input
class InputFeatures(BaseModel):
    appearance: int
    height: float
    goals: float
    name: str  
    position: str
    #options: str



def preprocessing(input_features: InputFeatures):
    """
    Function that applies the same preprocessing steps (used on the training data)
    to a new test row, ensuring consistency with the training data preprocessing pipeline.
    """
    dict_f = {
        'Appearance': input_features.appearance if input_features.appearance is not None else 0,  # Default to 0 if None
        'Assists': input_features.assists if input_features.assists is not None else 0,  # Default to 0 if None
        'Minutes_Played': input_features.minutes_played if input_features.minutes_played is not None else 0,  # Default to 0 if None
        'Days_Injured': input_features.days_injured if input_features.days_injured is not None else 0,  # Default to 0 if None
        'Games_Injured': input_features.games_injured if input_features.games_injured is not None else 0,  # Default to 0 if None
        'Award': 1 if input_features.award is not None else 0,  # 1 if there's an award, else 0
        'Highest_Value': input_features.highest_value if input_features.highest_value is not None else 0,  # Default to 0 if None
        #'Current_Value_Category_Encoded': input_features.current_value_category_encoded if input_features.current_value_category_encoded is not None else 0  # Default to 0 if None
    }
    
    return dict_f



@app.post("/predict")
def predict(input_features: InputFeatures):
    
    return preprocessing(input_features)
    

def preprocessing(input_features: InputFeatures):
    dict_f = {
        'Appearance': input_features.appearance if input_features.appearance is not None else 0,  # Default to 0 if None
        'Assists': input_features.assists if input_features.assists is not None else 0,  # Default to 0 if None
        'Minutes_Played': input_features.minutes_played if input_features.minutes_played is not None else 0,  # Default to 0 if None
        'Days_Injured': input_features.days_injured if input_features.days_injured is not None else 0,  # Default to 0 if None
        'Games_Injured': input_features.games_injured if input_features.games_injured is not None else 0,  # Default to 0 if None
        'Award': 1 if input_features.award is not None else 0,  # 1 if there's an award, else 0
        'Highest_Value': input_features.highest_value if input_features.highest_value is not None else 0,  # Default to 0 if None
    } 
    # Convert to list in correct order
    features_list = [dict_f[key] for key in sorted(dict_f)]

# Ensure scaler is working correctly
    try:
        scaled_features = scaler.transform([features_list])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in preprocessing: {e}")

    return scaled_features

# Prediction endpoint
@app.post("/predict")
async def predict(input_features: InputFeatures):
    try:
        data = preprocessing(input_features)
        y_pred = model.predict(data)
        return {"prediction": y_pred.tolist()[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")





# curl -X POST "http://localhost:8000/predict" \
# -H "Content-Type: application/json" \
# -d '{
#   "Current_Value_Category_Encoded": 1
# }'
