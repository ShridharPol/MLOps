# Modifications

- Switched from Iris to Wine dataset
- Replaced RandomForestClassifier with GradientBoostingClassifier
- Added accuracy score and classification report output
- Changed Dockerfile base image from `python:3.10` to `python:3.10-slim`

# Build and Run

docker build -t wine_model:v1 .

docker save wine_model:v1 > my_image.tar

docker run wine_model:v1
