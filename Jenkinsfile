pipeline {
  agent {
    kubernetes {
      defaultContainer 'test'
      yamlFile 'KubernetesPod.yaml'
    }
  }
  stages {
    stage('Build') {
      steps {
        checkout scm
        echo 'Hello world!'
      }
    }
  }
}
