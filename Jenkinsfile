pipeline {
    agent any

    environment {
        PYTHON = 'C:\\Users\\Admin\\AppData\\Roaming\\uv\\python\\cpython-3.12.14-windows-x86_64-none\\python.exe'
        VENV_PYTHON = '.venv\\Scripts\\python.exe'
        DOCKER_HOST = 'npipe:////./pipe/dockerDesktopLinuxEngine'
    }

    stages {

        stage('Checkout') {
            steps {
                echo 'Checking out project...'
            }
        }

        stage('Create Virtual Environment') {
            steps {
                bat '"%PYTHON%" -m venv .venv'
            }
        }

        stage('Check Python') {
            steps {
                bat '"%VENV_PYTHON%" --version'
            }
        }

        stage('Install Dependencies') {
            steps {
                bat '"%VENV_PYTHON%" -m pip install --upgrade pip'
                bat '"%VENV_PYTHON%" -m pip install -r requirements.txt'
            }
        }

        stage('Data Validation') {
            steps {
                bat '"%VENV_PYTHON%" src/data_validation.py'
            }
        }

        stage('Preprocessing') {
            steps {
                bat '"%VENV_PYTHON%" src/data_preprocessing.py'
            }
        }

        stage('Training') {
            steps {
                bat '"%VENV_PYTHON%" src/train.py'
            }
        }

        stage('Evaluation') {
            steps {
                bat '"%VENV_PYTHON%" src/evaluate.py'
            }
        }

        stage('API Tests') {
            steps {
                bat '"%VENV_PYTHON%" -m pytest tests/test_api.py'
            }
        }

        stage('Docker Build') {
            steps {
                bat 'docker build -t student-performance-api:latest .'
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