# Project Introduction

This project uses the data from Kaggle (https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset?resource=download&select=events.csv) to predict a user's next action based on their recent activity (like "view" -> "add to cart" -> ?). The model was fine-tuned and trained using PyTorch, converted to ONNX format, and is deployed through a Java Spring Boot API. This service supports secure HTTPS access and is containerized using Docker with Nginx as a reverse proxy to reproduce.

# Reproduction's Environment Set-up

## Prerequisites

- Docker and Docker Compose installed
- Port 443 open and not pre-occupied by other services on your local machine (used for HTTPS)

## Steps to run

### Git Clone

```bash
git clone https://github.com/westing88/kaggle-ml-predict-api.git
cd kaggle-ml-predict-api
```

### Build Docker Image

```bash
docker-compose up --build
```

-> If everything works successfully, you will see logs like:

```log
ml-predict-api | Tomcat started on port(s): 8443 (https)
nginx-proxy    | Configuration complete; ready for start up
```

### Ready to Test the Prediction Result

Example testing command:

```bash
curl -k -X POST https://localhost/predict_behavior \
  -H "Content-Type: application/json" \
  -d '{"sequence": [0, 1]}'
```

- Input (a JSON format): 0 -> "view", 1 -> "add_to_cart", 2 -> "purchase"
- Estimated Output Example:

```log
{
  "prediction": 2,
  "probabilities": [0.12, 0.17, 0.71],
  "labels": ["view", "cart", "purchase"]
}
```

## Git Branches Intro

```bash
ml-predict-api/
├── README.md
├── docker/ # Docker + HTTPS setup for deploying configuration
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── nginx.conf
│   └── certs/
├── api-java/ # Java Spring Boot API + ONNX runtime
│   ├── pom.xml
│   └── src/...
├── model-dev/ # Pytorch model trained and exported
│   ├── model_training.ipynb
│   └── best_model.onnx
│   └── best_model.pth
```

## About Me

Hi! I am Shelly Wei, a data scientist from UC Berkeley Master of Analytics (IEOR). This project was my first time working with Java Spring Boot and Docker for deploying a machine learning model API. Through this project, I gained hands-on knowledge of Docker builds, SSL certificates, and reverse proxy configuration which is very exciting for me to apply theories into real-world practical exercises.

Author: Shelly Wei
Contact: westingwei88@gmail.com
GitHub: westing88
