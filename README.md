# End-to-end-Car-price-prediction
This project aims to predict the prices of cars based on various features like the car’s make, model, year, mileage, and more. The project follows an end-to-end machine learning pipeline, including data collection, preprocessing, model training, evaluation, and deployment.

git clone https://github.com/Abhishek-guptaaa/End-to-end-Car-price-prediction.git

1.Project Structure
2.Installation
3.Dataset
4.Data Preprocessing
5.Modeling
6.Evaluation
7.Deployment
8.Usage


![Screenshot 2024-07-29 002653](https://github.com/user-attachments/assets/ec0eadc5-d01f-44d7-92b2-179c4e5c6478)



- data collection completed



# AWS-CICD-Deployment-with-Github-Actions
## 1. Login to AWS console.

## 2. Create IAM user for deployment

#with specific access

1. EC2 access : It is virtual machine

2. ECR: Elastic Container registry to save your docker image in aws


#Description: About the deployment

1. Build docker image of the source code

2. Push your docker image to ECR

3. Launch Your EC2 

4. Pull Your image from ECR in EC2

5. Lauch your docker image in EC2

#Policy:

1. AmazonEC2ContainerRegistryFullAccess

2. AmazonEC2FullAccess

## 3. Create ECR repo to store/save docker image
- Save the URI: 730335610052.dkr.ecr.us-east-1.amazonaws.com/mlproject

# 4. Create EC2 machine (Ubuntu)

# 5. Open EC2 and Install docker in EC2 Machine:

#optinal

sudo apt-get update -y

sudo apt-get upgrade

#required

curl -fsSL https://get.docker.com -o get-docker.sh

sudo sh get-docker.sh

sudo usermod -aG docker ubuntu

newgrp docker

# 6. Configure EC2 as self-hosted runner:
setting>actions>runner>new self hosted runner> choose os> then run command one by one

# 7. Setup github secrets:
AWS_ACCESS_KEY_ID=

AWS_SECRET_ACCESS_KEY=

AWS_REGION = us-east-1

AWS_ECR_LOGIN_URI = demo>>  730335610052.dkr.ecr.us-east-1.amazonaws.com
ECR_REPOSITORY_NAME = mlproject

