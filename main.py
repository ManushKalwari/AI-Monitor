from fastapi import FastAPI, UploadFile, File, Depends, Header, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from model import MLP
from db import SessionLocal, PredictionLog

from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import Response
import time

# count total predictions
prediction_count = Counter('prediction_total', 'total number of predictions made')

#track model inference latency
latency_histogram = Histogram('model_latency_seconds', 'lateycy for model inference')

app = FastAPI()

model = MLP()
model.load_state_dict(torch.load('mnist_mlp.pt'))
model.eval()

transform = transforms.ToTensor()
API_KEY = 'supersecret123'


def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail='unauthorized')


@app.post('/predict')
async def predict_image(file: UploadFile = File(...), _:str = Depends(verify_api_key)):
    try:
        start = time.time()

        image = Image.open(file.file).convert('L').resize((28,28))
        img_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            logits = model(img_tensor)
            probs = F.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()

        
        latency = time.time() - start         
        latency_histogram.observe(latency)

        prediction_count.inc()
        
        db = SessionLocal()
        log = PredictionLog(
            filename = file.filename,
            prediction = pred,
            probabilities = probs.squeeze().tolist()
        )

        db.add(log)
        db.commit()
        db.close()

        return {
            'prediction': pred,
            'probabilities': probs.squeeze().tolist(),
            'latency_ms': latency * 1000
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={'error': str(e)})



@app.get('/metrics')
def metrics(_: str = Depends(verify_api_key)):
    return Response(generate_latest(), media_type='text/plain')


