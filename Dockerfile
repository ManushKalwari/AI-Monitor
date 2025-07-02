
#Base image: slim python smaller footprint
FROM python:3.12-slim

# set working directory inside container
WORKDIR /app

# copy requirements install dependencies from it
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy all app code into container
COPY . .

# expose port fastapi will run on
EXPOSE 8000

# start FastAPI server when container starts
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
