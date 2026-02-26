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


## EKS

```sh
eksctl create cluster \
  --name $CLUSTER_NAME \
  --region $AWS_REGION \
  --nodegroup-name loan-nodes \
  --node-type t3.small \
  --nodes 1

# eksctl delete cluster --name $CLUSTER_NAME --region $AWS_REGION

cd k8s
# image: <account-id>.dkr.ecr.<region>.amazonaws.com/loan-model:latest
sed -i "s|<account-id>|$AWS_ACCOUNT_ID|g" deployment.yaml
sed -i "s|<region>|$AWS_REGION|g" deployment.yaml

kubectl apply -f deployment.yaml
kubectl apply -f k8s/service.yaml

kubectl get svc loan-model-service

# if NodePort: HOST_=13.220.124.39:32508 (you need to give inbound access in node sg)
# if LoadBalancer: HOST_=<LB_DNS>
## PAYLOAD saved as env. variable
curl -X POST http://$HOST_/predict \
  -H "Content-Type: application/json" \
  -d "$PAYLOAD"


```

## AWS ALB Ingress

```sh
# associate IAM OIDC provider
eksctl utils associate-iam-oidc-provider \
  --region $AWS_REGION \
  --cluster $CLUSTER_NAME \
  --approve

# verify
aws eks describe-cluster \
  --name $CLUSTER_NAME \
  --query "cluster.identity.oidc.issuer"

# create IAM policy for ALB controller

curl -o iam_policy.json \
https://raw.githubusercontent.com/kubernetes-sigs/aws-load-balancer-controller/main/docs/install/iam_policy.json

aws iam create-policy \
  --policy-name AWSLoadBalancerControllerIAMPolicy \
  --policy-document file://iam_policy.json

# create IAM role + service account (IRSA)
SA_NAME=aws-load-balancer-controller
eksctl create iamserviceaccount \
  --cluster $CLUSTER_NAME \
  --namespace kube-system \
  --name $SA_NAME \
  --attach-policy-arn arn:aws:iam::$AWS_ACCOUNT_ID:policy/AWSLoadBalancerControllerIAMPolicy \
  --approve

# install the controller
helm repo add eks https://aws.github.io/eks-charts
helm repo update

helm install aws-lb-controller eks/aws-load-balancer-controller \
  -n kube-system \
  --set clusterName=$CLUSTER_NAME \
  --set serviceAccount.create=false \
  --set serviceAccount.name=$SA_NAME \
  --set region=$AWS_REGION \
  --set v=2

# verify
k get deploy -n kube-system
## aws-lb-controller-aws-load-balancer-controller
```

Create ALB ingress
```sh
kubectl apply -f ingress.yaml

# takes ~2 min for the address to show an ALB DNS name
kubectl get ingress loan-model-ingress

# test:
HOST_=k8s-default-loanmode-22081556d5-899362333.us-east-1.elb.amazonaws.com

curl -X POST http://$HOST_/predict \
  -H "Content-Type: application/json" \
  -d @./k8s/payload.json
```
