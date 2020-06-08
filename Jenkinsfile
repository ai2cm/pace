pipeline {
  agent {
    kubernetes {
      yaml """
kind: Pod
spec:
  containers:
    - name: test
      image: gtclang/dawn-env-ubuntu19.10
      command:
        - cat
      tty: true
"""
      defaultContainer 'test'
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
