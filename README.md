# ml-serving-architecture
End-to-end Random Forest ML pipeline with FastAPI serving and multi-environment cloud deployment (EC2, EKS, KServe).


## get the data

```sh
curl -L -o loan-approval-prediction-dataset.zip https://www.kaggle.com/api/v1/datasets/download/architsharma01/loan-approval-prediction-dataset
unzip loan-approval-prediction-dataset.zip
```

## train the model
```sh
python3 -m venv .venv
source .venv/bin/activate

pip install pandas scikit-learn joblib

python3 pipeline.py
mv loan_random_forest_pipeline.pkl ../loan-model/

```

## serve the model

```sh
aws ecr create-repository --repository-name loan-model

# authenticate docker to ECR
aws ecr get-login-password --region $AWS_REGION \
| docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

# build
docker build -t loan-model .

# test
docker run -p 5000:8000 loan-model:latest

HOST_=localhost:5000

curl -X POST $HOST_/predict \
-H "Content-Type: application/json" \
-d '{
  "no_of_dependents": 2,
  "education": "Graduate",
  "self_employed": "No",
  "income_annum": 9600000,
  "loan_amount": 29900000,
  "loan_term": 12,
  "cibil_score": 778
}'


# tag & push to ECR
docker tag loan-model:latest $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/loan-model:latest
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/loan-model:latest
```

