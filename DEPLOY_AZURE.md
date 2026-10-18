# Deploy to Azure App Service (Linux container)

One container runs both parts: Streamlit (public, on `$PORT`) and the FastAPI backend (private, `127.0.0.1:8001`).

## 1. Prerequisites
- Azure CLI installed and logged in: `az login`
- Your Azure OpenAI key, endpoint, and deployment name

## 2. Set names (PowerShell)
```powershell
$RG="hackeval-rg"; $LOC="centralindia"
$ACR="hackevalacr$(Get-Random -Max 9999)"   # must be globally unique, lowercase
$PLAN="hackeval-plan"; $APP="hackeval-$(Get-Random -Max 9999)"
```

## 3. Build the image in Azure (no local Docker needed)
Run from the project folder:
```powershell
az group create -n $RG -l $LOC
az acr create -g $RG -n $ACR --sku Basic --admin-enabled true
az acr build -r $ACR -t hackeval:latest .
```

## 4. Create the web app
```powershell
az appservice plan create -g $RG -n $PLAN --is-linux --sku B1
az webapp create -g $RG -p $PLAN -n $APP --deployment-container-image-name "$ACR.azurecr.io/hackeval:latest"
$PWD_ACR = az acr credential show -n $ACR --query "passwords[0].value" -o tsv
az webapp config container set -g $RG -n $APP `
  --container-image-name "$ACR.azurecr.io/hackeval:latest" `
  --container-registry-url "https://$ACR.azurecr.io" `
  --container-registry-user $ACR --container-registry-password $PWD_ACR
```

## 5. Settings (your keys go here, never in the image)
```powershell
az webapp config appsettings set -g $RG -n $APP --settings `
  WEBSITES_PORT=8000 `
  WEBSITES_ENABLE_APP_SERVICE_STORAGE=true `
  LLM_PROVIDER=azure `
  LLM_MODEL=<your-deployment-name> `
  AZURE_OPENAI_ENDPOINT=<your-endpoint-url> `
  AZURE_OPENAI_API_KEY=<your-key>
az webapp config set -g $RG -n $APP --web-sockets-enabled true --always-on true
az webapp restart -g $RG -n $APP
```
`WEBSITES_ENABLE_APP_SERVICE_STORAGE=true` keeps `/home/data` (the CSV) across restarts.
Web sockets must be on for Streamlit.

## 6. Open it
`https://<APP>.azurewebsites.net`
Logs: `az webapp log tail -g $RG -n $APP`

## Updating after code changes
```powershell
az acr build -r $ACR -t hackeval:latest .
az webapp restart -g $RG -n $APP
```

## Notes
- B1 is the smallest tier with Always On; large ZIPs may need B2+.
- Job status is in memory, so a restart clears running jobs (CSV persists).
- The app has no login. Anyone with the URL can use it and spend your Azure credits.
  Turn on App Service Authentication (Portal > your app > Authentication) to restrict it.
