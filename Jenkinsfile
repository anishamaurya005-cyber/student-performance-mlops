pipeline {
    agent any

    environment {
        PYTHON = 'C:\\Users\\Admin\\AppData\\Roaming\\uv\\python\\cpython-3.12.14-windows-x86_64-none\\python.exe'
    }

    stages {

        stage('Checkout') {
            steps {
                echo 'Checking out project...'
            }
        }

        stage('Check Python') {
            steps {
                bat '"%PYTHON%" --version'
            }
        }

        stage('Install Dependencies') {
            steps {
                bat '"%PYTHON%" -m pip install -r requirements.txt'
            }
        }

        stage('Data Validation') {
            steps {
                bat '"%PYTHON%" src/data_validation.py'
            }
        }

        stage('Preprocessing') {
            steps {
                bat '"%PYTHON%" src/data_preprocessing.py'
            }
        }

        stage('Training') {
            steps {
                bat '"%PYTHON%" src/train.py'
            }
        }

        stage('Evaluation') {
            steps {
                bat '"%PYTHON%" src/evaluate.py'
            }
        }

        stage('API Tests') {
            steps {
                bat '"%PYTHON%" -m pytest tests/test_api.py'
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