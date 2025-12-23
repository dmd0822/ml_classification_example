targetScope = 'resourceGroup'

@description('Azure region for all resources.')
param location string = resourceGroup().location

@description('Short environment name used in resource naming (e.g., dev, prod).')
param environmentName string = 'dev'

@description('Base name used for resources. Only lowercase letters/numbers/hyphens recommended.')
@minLength(5)
param namePrefix string = 'mlclass'

@description('Container image repository name (in ACR).')
param imageRepository string = 'ml-classification-infer-api'

@description('Container image tag (in ACR).')
param imageTag string = 'latest'

@description('Container app exposed port. Must match the container listener (Dockerfile exposes 8080).')
param containerPort int = 8080

@description('CPU cores for the container (e.g. 0.5, 1). Use string so fractional values are supported.')
param containerCpu string = '1'

@description('Memory for the container (e.g. 1Gi, 2Gi).')
param containerMemory string = '2Gi'

@description('Minimum number of replicas.')
param minReplicas int = 0

@description('Maximum number of replicas.')
param maxReplicas int = 2

@description('Whether the Container App should be publicly accessible (external ingress).')
param externalIngress bool = true

@description('Non-secret environment variables for the container.')
param nonSecretEnvVars array = [
  {
    name: 'MAX_LENGTH'
    value: '64'
  }
]

@description('Optional mapping of Key Vault secrets into container environment variables. No secret values live in IaC; only Key Vault secret URLs are referenced.')
type KeyVaultSecretMapping = {
  @description('The Container App secret name (referenced by envVar.secretRef).')
  secretName: string

  @description('The environment variable name passed into the container.')
  envVarName: string

  @description('The Key Vault secret URL, e.g. https://<vault>.vault.azure.net/secrets/<name> (version optional).')
  keyVaultUrl: string
}

param keyVaultSecrets KeyVaultSecretMapping[] = []

@description('ACR SKU (Basic/Standard/Premium).')
param acrSkuName string = 'Basic'

@description('Key Vault SKU (standard/premium).')
param keyVaultSkuName string = 'standard'

var sanitizedPrefix = toLower(replace(replace(namePrefix, '_', '-'), ' ', '-'))
var envSuffix = toLower(environmentName)

// ACR name must be 5-50 chars, alphanumeric only, and globally unique.
// Use a stable prefix + uniqueString to guarantee length and uniqueness.
var acrName = take('mlacr${uniqueString(resourceGroup().id, sanitizedPrefix, envSuffix)}', 50)
var logAnalyticsName = take('${sanitizedPrefix}-${envSuffix}-law', 63)
var containerAppsEnvName = take('${sanitizedPrefix}-${envSuffix}-cae', 32)
var containerAppName = take('${sanitizedPrefix}-${envSuffix}-api', 32)
var keyVaultName = take('${sanitizedPrefix}-${envSuffix}-kv', 24)
var identityName = take('${sanitizedPrefix}-${envSuffix}-uai', 128)

// Built-in role definition IDs (GUIDs) from Azure built-in roles documentation.
var roleDefinitionIdAcrPull = subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '7f951dda-4ed3-4680-a7ca-43fe172d538d')
var roleDefinitionIdKeyVaultSecretsUser = subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '4633458b-17de-408a-b874-0445c86b69e6')

resource logAnalytics 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: logAnalyticsName
  location: location
  properties: {
    retentionInDays: 30
    sku: {
      name: 'PerGB2018'
    }
  }
}

// Used for Container Apps environment log collection.
var logAnalyticsKeys = logAnalytics.listKeys()

resource acr 'Microsoft.ContainerRegistry/registries@2023-07-01' = {
  name: acrName
  location: location
  sku: {
    name: acrSkuName
  }
  properties: {
    adminUserEnabled: false
    publicNetworkAccess: 'Enabled'
  }
}

resource userIdentity 'Microsoft.ManagedIdentity/userAssignedIdentities@2024-11-30' = {
  name: identityName
  location: location
}

resource keyVault 'Microsoft.KeyVault/vaults@2024-11-01' = {
  name: keyVaultName
  location: location
  properties: {
    tenantId: subscription().tenantId
    sku: {
      family: 'A'
      name: keyVaultSkuName
    }
    // Use RBAC permission model so we can grant the managed identity Key Vault Secrets User.
    enableRbacAuthorization: true
    // Provide an empty accessPolicies array (ignored when RBAC is enabled).
    accessPolicies: []
    enableSoftDelete: true
  }
}

resource containerAppsEnv 'Microsoft.App/managedEnvironments@2024-03-01' = {
  name: containerAppsEnvName
  location: location
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logAnalytics.properties.customerId
        sharedKey: logAnalyticsKeys.primarySharedKey
      }
    }
  }
}

// Allow the managed identity to pull from ACR without any registry passwords.
resource acrPullRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(acr.id, userIdentity.id, 'AcrPull')
  scope: acr
  properties: {
    roleDefinitionId: roleDefinitionIdAcrPull
    principalId: userIdentity.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

// Allow the managed identity to read secrets from Key Vault (data plane) via RBAC.
resource kvSecretsUserRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(keyVault.id, userIdentity.id, 'KeyVaultSecretsUser')
  scope: keyVault
  properties: {
    roleDefinitionId: roleDefinitionIdKeyVaultSecretsUser
    principalId: userIdentity.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

var containerAppSecrets = [
  for s in keyVaultSecrets: {
    name: s.secretName
    identity: userIdentity.id
    keyVaultUrl: s.keyVaultUrl
  }
]

var containerAppSecretEnvVars = [
  for s in keyVaultSecrets: {
    name: s.envVarName
    secretRef: s.secretName
  }
]

var containerAppNonSecretEnvVars = [
  for e in nonSecretEnvVars: {
    name: string(e.name)
    value: string(e.value)
  }
]

resource containerApp 'Microsoft.App/containerApps@2024-03-01' = {
  name: containerAppName
  location: location
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: {
      '${userIdentity.id}': {}
    }
  }
  properties: {
    environmentId: containerAppsEnv.id
    configuration: {
      ingress: {
        external: externalIngress
        targetPort: containerPort
        transport: 'auto'
      }
      // Use managed identity for ACR auth (no username/password in secrets).
      registries: [
        {
          server: acr.properties.loginServer
          identity: userIdentity.id
        }
      ]
      secrets: containerAppSecrets
    }
    template: {
      containers: [
        {
          name: 'api'
          image: '${acr.properties.loginServer}/${imageRepository}:${imageTag}'
          resources: {
            cpu: json(containerCpu)
            memory: containerMemory
          }
          env: union(containerAppNonSecretEnvVars, containerAppSecretEnvVars)
          probes: [
            {
              type: 'Readiness'
              httpGet: {
                path: '/health'
                port: containerPort
              }
              initialDelaySeconds: 5
              periodSeconds: 10
              timeoutSeconds: 5
              failureThreshold: 3
            }
            {
              type: 'Liveness'
              httpGet: {
                path: '/health'
                port: containerPort
              }
              initialDelaySeconds: 15
              periodSeconds: 20
              timeoutSeconds: 5
              failureThreshold: 3
            }
          ]
        }
      ]
      scale: {
        minReplicas: minReplicas
        maxReplicas: maxReplicas
        rules: [
          {
            name: 'http'
            http: {
              metadata: {
                concurrentRequests: '50'
              }
            }
          }
        ]
      }
    }
  }
}

output acrLoginServer string = acr.properties.loginServer
output containerAppFqdn string = containerApp.properties.configuration.ingress.fqdn
output keyVaultUri string = keyVault.properties.vaultUri
