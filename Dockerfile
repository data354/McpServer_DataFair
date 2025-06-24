# Utilise une image Python officielle
FROM python:3.12-slim

# Définit le répertoire de travail
WORKDIR /app

# Copie les fichiers nécessaires
COPY . .

# Installe les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Expose le port utilisé par le serveur
EXPOSE 8000

# Commande de démarrage
CMD ["python", "mcp_server.py"]
