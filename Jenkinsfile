pipeline {
    agent {
        kubernetes {
        label fv3ser
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
}
    }
    stages {
        stage('Build') {
            steps {
                container('test') {
                    checkout scm
                    echo 'Hello world!'
                }
            }
        }
    }
}
