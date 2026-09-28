pipeline {
    agent any

    stages {

        stage('Checkout') {
            steps {
                echo 'Checking out project...'
            }
        }

        stage('Install Dependencies') {
            steps {
                bat 'python -m pip install -r requirements.txt'
            }
        }

        stage('Data Validation') {
            steps {
                bat 'python src/data_validation.py'
            }
        }

        stage('Preprocessing') {
            steps {
                bat 'python src/data_preprocessing.py'
            }
        }

        stage('Training') {
            steps {
                bat 'python src/train.py'
            }
        }

        stage('Evaluation') {
            steps {
                bat 'python src/evaluate.py'
            }
        }

        stage('API Tests') {
            steps {
                bat 'python -m pytest tests/test_api.py'
            }
        }
    }

    post {
        success {
            echo 'MLOps Pipeline completed successfully!'
        }

        failure {
            echo 'MLOps Pipeline failed. Check the console output.'
        }
    }
}