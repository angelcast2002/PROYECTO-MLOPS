# Guía de Deployment - PROYECTO-MLOPS

## Contenido

1. [Instalación](#instalación)
2. [Configuración](#configuración)
3. [Deployment en Producción](#deployment-en-producción)
4. [Monitoreo](#monitoreo)
5. [Troubleshooting](#troubleshooting)

---

## Instalación

### Opción 1: Desde PyPI (Recomendado)

```bash
pip install proyecto-mlops
```

### Opción 2: Desde Código Fuente

```bash
git clone https://github.com/angelcast2002/PROYECTO-MLOPS.git
cd PROYECTO-MLOPS
pip install -e .
```

### Opción 3: Docker

```bash
docker pull angelcast2002/proyecto-mlops:latest
```

---

## Configuración

### Variables de Entorno

```bash
# Configuración de datos
export DATA_RAW_DIR="data/raw"
export DATA_PROCESSED_DIR="data/processed"
export MODELS_DIR="models"

# Configuración de logging
export LOG_LEVEL="INFO"

# Configuración de producción
export ENVIRONMENT="production"
export DEBUG=False
```

### Archivo config.yaml

```yaml
data:
  raw_dir: "data/raw"
  processed_dir: "data/processed"
  registry_dir: "data/registry"

model:
  type: "LinearSVC"
  vectorizer: "TfidfVectorizer"
  params:
    min_df: 2
    C: 1.0

inference:
  batch_size: 100
  timeout_ms: 200
  cache_enabled: true

monitoring:
  log_level: "INFO"
  drift_threshold: 0.2
  alert_email: "ops@example.com"
```

---

## Deployment en Producción

### 1. Docker Compose (Recomendado para staging)

```yaml
# docker-compose.yml
version: '3.8'

services:
  modelo:
    image: angelcast2002/proyecto-mlops:latest
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
    environment:
      - ENVIRONMENT=production
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9090:9090"

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
```

Ejecutar:
```bash
docker-compose up -d
```

### 2. Kubernetes Deployment

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: proyecto-mlops
spec:
  replicas: 3
  selector:
    matchLabels:
      app: proyecto-mlops
  template:
    metadata:
      labels:
        app: proyecto-mlops
    spec:
      containers:
      - name: modelo
        image: angelcast2002/proyecto-mlops:0.1.0
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        healthCheck:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 30
        volumeMounts:
        - name: data
          mountPath: /app/data
      volumes:
      - name: data
        persistentVolumeClaim:
          claimName: proyecto-mlops-pvc

---
apiVersion: v1
kind: Service
metadata:
  name: proyecto-mlops-service
spec:
  selector:
    app: proyecto-mlops
  type: LoadBalancer
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
```

Deploy:
```bash
kubectl apply -f k8s-deployment.yaml
```

### 3. Cloud Deployment

#### AWS ECS

```bash
# Build y push a ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 123456789.dkr.ecr.us-east-1.amazonaws.com

docker tag proyecto-mlops:latest 123456789.dkr.ecr.us-east-1.amazonaws.com/proyecto-mlops:latest
docker push 123456789.dkr.ecr.us-east-1.amazonaws.com/proyecto-mlops:latest
```

#### Google Cloud Run

```bash
gcloud run deploy proyecto-mlops \
  --image angelcast2002/proyecto-mlops:latest \
  --platform managed \
  --region us-central1 \
  --memory 512Mi \
  --cpu 1
```

#### Azure Container Instances

```bash
az container create \
  --resource-group myResourceGroup \
  --name proyecto-mlops \
  --image angelcast2002/proyecto-mlops:latest \
  --cpu 1 \
  --memory 1 \
  --port 8000
```

---

## Monitoreo

### Métricas Clave

```python
# En aplicación
from prometheus_client import Counter, Histogram

predictions = Counter('modelo_predictions_total', 'Total predictions')
latency = Histogram('modelo_latency_seconds', 'Prediction latency')

# Usar
predictions.inc()
with latency.time():
    prediction = model.predict(text)
```

### Alertas Recomendadas

```yaml
# alertas.yml
groups:
- name: modelo
  rules:
  - alert: HighLatency
    expr: histogram_quantile(0.95, modelo_latency_seconds) > 0.2
    for: 5m
    annotations:
      summary: "Latencia P95 > 200ms"
  
  - alert: HighErrorRate
    expr: rate(modelo_errors_total[5m]) > 0.05
    for: 5m
    annotations:
      summary: "Error rate > 5%"
  
  - alert: DataDrift
    expr: drift_psi > 0.2
    for: 1h
    annotations:
      summary: "Data drift detectado"
```

---

## Troubleshooting

### Problema: ModuleNotFoundError

```bash
# Solución
pip install --upgrade proyecto-mlops
```

### Problema: Out of Memory

```bash
# Solución: Usar batch processing
texts = [...]
batch_size = 100
for i in range(0, len(texts), batch_size):
    batch = texts[i:i+batch_size]
    predictions = model.predict(batch)
```

### Problema: Latencia Alta

```bash
# Optimizaciones
1. Aumentar replicas en K8s
2. Usar model quantization
3. Implementar caching
4. Usar GPU (si disponible)
```

### Problema: Modelo Falla

```bash
# Rollback automático
git revert <commit-hash>
git push origin main
# GitHub Actions rebuildeará y deployará versión anterior
```

---

## Checklist de Deployment

- [ ] Tests pasando (CI)
- [ ] Docker image buildea sin errores
- [ ] Imagen pusheada a Docker Hub
- [ ] Configuración de secrets en GitHub
- [ ] Staging environment validado
- [ ] Monitoring y alertas configuradas
- [ ] Rollback plan documentado
- [ ] Team notificado del deployment
- [ ] Runbook de incidentes preparado

---

## Soporte

- 📖 Documentación: https://github.com/angelcast2002/PROYECTO-MLOPS
- 🐛 Issues: https://github.com/angelcast2002/PROYECTO-MLOPS/issues
- 📧 Email: angelcast2002@gmail.com

---

**Última actualización:** Noviembre 2025
