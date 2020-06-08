pipeline {
  agent {
    kubernetes {
      label 'fv3ser'
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
