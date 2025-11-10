# 🚀 Despliegue en DigitalOcean (Docker)

Esta guía explica, paso a paso, cómo desplegar el contenedor de `proyecto-mlops` en un Droplet de DigitalOcean con Ubuntu 22.04+.

## 1) Prerrequisitos
- Cuenta en DigitalOcean
- Llave SSH pública en tu máquina local (Windows PowerShell)
- Droplet Ubuntu 22.04/24.04 (1–2 vCPU, 2–4 GB RAM)
- Puerto 8000 abierto si vas a exponer servicio HTTP (o el que uses)

### 1.1 Generar y registrar tu llave SSH
En tu PC (PowerShell):

```pwsh
# Generar llave (si no tienes una)
ssh-keygen -t ed25519 -C "tu_email@example.com" -f "$env:USERPROFILE\.ssh\id_ed25519"
# Mostrar clave pública para copiarla a DigitalOcean
Get-Content "$env:USERPROFILE\.ssh\id_ed25519.pub"
```

En DigitalOcean:
- Settings → Security → SSH Keys → Add SSH Key → pega el contenido del archivo `.pub`.

## 2) Crear el Droplet
- Create → Droplets → Ubuntu 22.04/24.04
- Authentication: SSH Keys → selecciona tu llave
- Networking: habilita IPv4, opcionalmente un Floating IP
- Finaliza la creación y toma nota de la IP pública

Conéctate por SSH:

```pwsh
ssh root@<IP_PUBLICA>
```

## 3) Instalar Docker y Docker Compose plugin

```bash
apt-get update -y && apt-get upgrade -y
apt-get install -y ca-certificates curl gnupg
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
chmod a+r /etc/apt/keyrings/docker.gpg
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo $VERSION_CODENAME) stable" | \
  tee /etc/apt/sources.list.d/docker.list > /dev/null
apt-get update -y
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Opcional: permitir usar docker sin sudo
groupadd -f docker
usermod -aG docker $SUDO_USER 2>/dev/null || true
newgrp docker <<'EOF'
true
EOF
systemctl enable docker --now
```

Verifica:

```bash
docker --version
docker compose version
```

## 4) Iniciar sesión en Docker Hub

Crea un access token en https://hub.docker.com/settings/security y luego:

```bash
docker login -u angelcast2025
# pega el token cuando lo pida
```

## 5) Descargar y ejecutar la imagen

Usa la versión publicada (ejemplo: `0.1.6`) o `latest`:

```bash
# Descargar imagen
docker pull angelcast2025/proyecto-mlops:0.1.6

# Preparar carpetas persistentes
mkdir -p /opt/proyecto-mlops/data /opt/proyecto-mlops/logs

# Ejecutar (modo interactivo)
docker run --name proyecto-mlops \
  -v /opt/proyecto-mlops/data:/app/data \
  -v /opt/proyecto-mlops/logs:/app/logs \
  -p 8000:8000 \
  angelcast2025/proyecto-mlops:0.1.6 \
  python cli.py all
```

Notas:
- Cambia `python cli.py all` por el comando que necesites (p.ej. `python cli.py train`, `python cli.py evaluate`, etc.).
- Si usas un `config.yaml`, móntalo también: `-v /opt/proyecto-mlops/config.yaml:/app/config.yaml`.

## 6) Ejecutar como servicio (systemd)

Crea el archivo `/etc/systemd/system/proyecto-mlops.service` con:

```ini
[Unit]
Description=Proyecto MLOps Container
After=docker.service
Requires=docker.service

[Service]
Restart=always
RestartSec=10
ExecStart=/usr/bin/docker run --rm --name proyecto-mlops \
  -v /opt/proyecto-mlops/data:/app/data \
  -v /opt/proyecto-mlops/logs:/app/logs \
  -p 8000:8000 \
  angelcast2025/proyecto-mlops:0.1.6 python cli.py all
ExecStop=/usr/bin/docker stop proyecto-mlops

[Install]
WantedBy=multi-user.target
```

Aplica y arranca:

```bash
systemctl daemon-reload
systemctl enable proyecto-mlops --now
systemctl status proyecto-mlops
```

## 7) Firewall (UFW)

```bash
ufw allow OpenSSH
ufw allow 8000/tcp
ufw enable
ufw status
```

## 8) Actualizar a nueva versión

```bash
systemctl stop proyecto-mlops || true
docker pull angelcast2025/proyecto-mlops:latest
systemctl start proyecto-mlops
```

## 9) Solución de problemas
- `permission denied /app/data`: verifica permisos de la carpeta montada en el host
- `port already in use`: cambia `-p 8000:8000` a otro puerto
- `image not found`: revisa el tag publicado en Docker Hub

## 10) Recursos
- Docker Hub: https://hub.docker.com/repository/docker/angelcast2025/proyecto-mlops/general
- Repositorio: https://github.com/angelcast2002/PROYECTO-MLOPS
