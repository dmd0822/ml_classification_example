using '../main.bicep'

param location = 'eastus'
param environmentName = 'dev'
param namePrefix = 'mlclass'

// The container listens on 8080 (see Dockerfile).
param containerPort = 8080

// Adjust as needed (TensorFlow-based models may require more memory).
param containerCpu = '1'
param containerMemory = '2Gi'

param minReplicas = 0
param maxReplicas = 2
param externalIngress = true

// Non-secret env vars.
param nonSecretEnvVars = [
  {
    name: 'MAX_LENGTH'
    value: '64'
  }
]

// Image repository + tag to run from your ACR.
// After deploying infra, push/tag your image into ACR as:
//   <acrLoginServer>/<imageRepository>:<imageTag>
param imageRepository = 'ml-classification-infer-api'
param imageTag = 'latest'

// Key Vault secret mapping (optional).
// These entries reference Key Vault secret URLs (not values).
// Example:
// param keyVaultSecrets = [
//   {
//     secretName: 'openai-api-key'
//     envVarName: 'OPENAI_API_KEY'
//     keyVaultUrl: 'https://<your-vault>.vault.azure.net/secrets/openai-api-key'
//   }
// ]
param keyVaultSecrets = []
