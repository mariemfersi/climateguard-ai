@description('Name of the storage account (globally unique, lowercase alphanumeric only, 3-24 chars)')
param storageAccountName string

@description('Name of the Key Vault (globally unique, 3-24 chars)')
param keyVaultName string

@description('Azure region for all resources')
param location string = resourceGroup().location

@description('Object ID of the user/principal to grant Key Vault secrets access to (get via: az ad signed-in-user show --query id -o tsv)')
param keyVaultAdminObjectId string

resource storageAccount 'Microsoft.Storage/storageAccounts@2023-01-01' = {
  name: storageAccountName
  location: location
  sku: {
    name: 'Standard_LRS'
  }
  kind: 'StorageV2'
  properties: {
    isHnsEnabled: true // enables ADLS Gen2 hierarchical namespace
    minimumTlsVersion: 'TLS1_2'
    allowBlobPublicAccess: false
  }
}

resource blobService 'Microsoft.Storage/storageAccounts/blobServices@2023-01-01' = {
  parent: storageAccount
  name: 'default'
}

resource bronzeContainer 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-01-01' = {
  parent: blobService
  name: 'bronze'
  properties: {
    publicAccess: 'None'
  }
}

resource silverContainer 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-01-01' = {
  parent: blobService
  name: 'silver'
  properties: {
    publicAccess: 'None'
  }
}

resource goldContainer 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-01-01' = {
  parent: blobService
  name: 'gold'
  properties: {
    publicAccess: 'None'
  }
}

resource keyVault 'Microsoft.KeyVault/vaults@2023-07-01' = {
  name: keyVaultName
  location: location
  properties: {
    sku: {
      family: 'A'
      name: 'standard'
    }
    tenantId: subscription().tenantId
    enableRbacAuthorization: true
    accessPolicies: []
  }
}

// Grants you (the deploying user) permission to read/write secrets via RBAC
// (built-in role: Key Vault Secrets Officer).
resource keyVaultSecretsOfficerRoleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: keyVault
  name: guid(keyVault.id, keyVaultAdminObjectId, 'Key Vault Secrets Officer')
  properties: {
    roleDefinitionId: subscriptionResourceId(
      'Microsoft.Authorization/roleDefinitions',
      'b86a8fe4-44ce-4948-aee5-eccb2c155cd7'
    )
    principalId: keyVaultAdminObjectId
    principalType: 'User'
  }
}

// Grants you (the deploying user) permission to read/write blob data
// (built-in role: Storage Blob Data Contributor). Without this, DefaultAzureCredential
// (used by upload_to_azure.py) authenticates successfully but gets a 403
// AuthorizationPermissionMismatch on any blob read/write — confirmed as a
// real first-deployment gap while building Phase 3.
resource storageBlobDataContributorRoleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: storageAccount
  name: guid(storageAccount.id, keyVaultAdminObjectId, 'Storage Blob Data Contributor')
  properties: {
    roleDefinitionId: subscriptionResourceId(
      'Microsoft.Authorization/roleDefinitions',
      'ba92f5b4-2d11-453d-a403-e96b0029c9fe'
    )
    principalId: keyVaultAdminObjectId
    principalType: 'User'
  }
}

output storageAccountName string = storageAccount.name
output storageAccountBlobEndpoint string = storageAccount.properties.primaryEndpoints.blob
output keyVaultName string = keyVault.name
output keyVaultUri string = keyVault.properties.vaultUri