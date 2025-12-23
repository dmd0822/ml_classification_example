# Bicep infrastructure as code: Container Apps + ACR + Key Vault

This folder contains Bicep to deploy the inference API container to Azure using:

- Azure Container Apps (runs the container)
- Azure Container Registry (stores the image)
- Azure Key Vault (stores all application secrets)
- Log Analytics workspace (Container Apps logs)

## What this deployment does

- Provisions an ACR with admin user disabled (no registry password)
- Provisions a user-assigned managed identity (UAI)
- Assigns the UAI:
  - `AcrPull` on the ACR (pull images)
  - `Key Vault Secrets User` on the Key Vault (read secrets)
- Provisions a Container Apps environment with Log Analytics logging
- Provisions a Container App that:
  - Exposes ingress on port `8080`
  - Pulls the container image from ACR using managed identity
  - Optionally maps Key Vault secrets into container env vars via Key Vault references

## Deploy (Azure CLI)

Prerequisites:

- Azure CLI installed (`az`)
- Logged in (`az login`)
- Docker installed (to build/tag/push images)

1. Pick a resource group:

```powershell
az group create --name <rg> --location <location>
```

1. (Optional) Set subscription:

```powershell
az account set --subscription <subscriptionId>
```

1. Deploy Bicep:

```powershell
az deployment group create \
  --resource-group <rg> \
  --template-file infra/bicep/main.bicep \
  --parameters infra/bicep/environments/dev.bicepparam
```

1. Get the ACR login server from outputs:

```powershell
az deployment group show --resource-group <rg> --name <deploymentName> --query properties.outputs
```

1. Push your image to ACR.

Build locally:

```powershell
docker build -t ml-classification-infer-api:local .
```

If you built locally as `ml-classification-infer-api:local`, tag + push:

```powershell
# Replace <acrLoginServer> from the deployment output.
docker tag ml-classification-infer-api:local <acrLoginServer>/ml-classification-infer-api:latest

docker push <acrLoginServer>/ml-classification-infer-api:latest
```

Then either re-run the same deployment (to pick up the image tag), or set a new tag in the parameter file and redeploy.

## Put secrets in Key Vault (no secrets in templates)

This repo’s current inference container doesn’t require any secret env vars by default.
If you add secrets later (API keys, DB connection strings, etc.), store values in Key Vault:

```powershell
az keyvault secret set \
  --vault-name <keyVaultName> \
  --name <secretName> \
  --value <secretValue>
```

Then wire them into the Container App by setting `keyVaultSecrets` in the bicep parameter file. Each entry needs:

- `secretName`: the Container App secret name
- `envVarName`: the container environment variable name
- `keyVaultUrl`: Key Vault secret URL, e.g. `https://<vault>.vault.azure.net/secrets/<name>`

Finally redeploy the Bicep to update the Container App.
