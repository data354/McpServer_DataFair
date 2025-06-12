import asyncio
import httpx
from mcp.server.fastmcp import FastMCP
from typing import Dict, List, Any, Optional
import json
import os
import csv
import io
import time
import base64
from urllib.parse import parse_qs, urlparse


# Configuration
API_BASE_URL_TEMPLATE = "https://data.gouv.ci/data-fair/api/v1/datasets/{dataset_id}"
# Endpoint pour lister les datasets
DATASETS_LIST_URL = "https://data.gouv.ci/data-fair/api/v1/datasets"  # "http://10.65.163.221:32000/data-fair/api/v1/datasets"

API_KEY = os.getenv("api_key") or os.getenv("API_KEY")

AVAILABLE_DATASETS = {}  # Dictionnaire pour stocker les datasets disponibles
LAST_DATASETS_UPDATE = 0
DATASETS_CACHE_DURATION = 60  # 1 minutes


mcp = FastMCP("multi-datasets-server")


# Client HTTP réutilisable
async def get_http_client():
    return httpx.AsyncClient(
        headers={
            # "x-apikey": f"{API_KEY}",
            "Content-Type": "application/json",
        },
        timeout=30.0,
    )


# Gestion de la configuration pour Smithery
# def get_config_from_query(query_string: str = None) -> Dict[str, Any]:
#     """Récupère la configuration depuis les paramètres de requête (pour Smithery)"""
#     if not query_string:
#         return {}

#     try:
#         # Parse les paramètres de requête
#         parsed = parse_qs(query_string)
#         if "config" in parsed:
#             # Decode la configuration base64
#             config_data = base64.b64decode(parsed["config"][0]).decode("utf-8")
#             return json.loads(config_data)
#     except Exception as e:
#         print(f"Erreur lors du parsing de la configuration: {e}")

#     return {}


async def fetch_available_datasets() -> Dict[str, str]:
    """Récupération complète des datasets"""
    try:
        client = await get_http_client()
        async with client:
            # Pour recupérer tous les datasets
            params = {"size": 10000}

            response = await client.get(DATASETS_LIST_URL, params=params)
            response.raise_for_status()
            data = response.json()

            # Construire le dictionnaire {nom: id}
            datasets = {dataset["title"]: dataset["id"] for dataset in data["results"]}
            print(f"Total: {len(datasets)} datasets récupérés")
            return datasets
    except Exception as e:
        raise RuntimeError(f"Erreur lors de la récupération des datasets: {str(e)}")


def get_dataset_url(dataset_name: str) -> str:
    """Construit l'URL de base pour un dataset donné"""
    if not AVAILABLE_DATASETS:
        raise RuntimeError(
            "La liste des datasets n'a pas été initialisée. Appelez fetch_available_datasets d'abord."
        )
    if dataset_name not in AVAILABLE_DATASETS:
        raise ValueError(
            f"Dataset '{dataset_name}' non disponible. Datasets disponibles: {list(AVAILABLE_DATASETS.keys())}"
        )
    real_id = AVAILABLE_DATASETS[dataset_name]
    return API_BASE_URL_TEMPLATE.format(dataset_id=real_id)


# Initialisation des datasets au démarrage
async def initialize_datasets():
    # global AVAILABLE_DATASETS
    # if not AVAILABLE_DATASETS:
    #     AVAILABLE_DATASETS = await fetch_available_datasets()
    """Les datasets sont initialisés et les recharge si nécessaire"""
    global AVAILABLE_DATASETS, LAST_DATASETS_UPDATE

    current_time = time.time()

    # Recharger si le cache est vide ou expiré
    if (
        not AVAILABLE_DATASETS
        or (current_time - LAST_DATASETS_UPDATE) > DATASETS_CACHE_DURATION
    ):
        try:
            print("Rechargement des datasets...")
            new_datasets = await fetch_available_datasets()

            # Vérifier s'il y a des changements
            if new_datasets != AVAILABLE_DATASETS:
                print(f"Datasets mis à jour: {len(new_datasets)} datasets trouvés")
                if AVAILABLE_DATASETS:  # Pas la première fois
                    added = set(new_datasets.keys()) - set(AVAILABLE_DATASETS.keys())
                    removed = set(AVAILABLE_DATASETS.keys()) - set(new_datasets.keys())
                    if added:
                        print(f"Nouveaux datasets: {', '.join(added)}")
                    if removed:
                        print(f"Datasets supprimés: {', '.join(removed)}")

            AVAILABLE_DATASETS = new_datasets
            LAST_DATASETS_UPDATE = current_time

        except Exception as e:
            print(f"Erreur lors du rechargement des datasets: {e}")
            # Garder l'ancienne version si le rechargement échoue
            if not AVAILABLE_DATASETS:
                raise


# Les Outils
@mcp.tool()
async def lister_datasets_disponibles() -> Dict[str, List[str]]:
    """Liste tous les datasets disponibles sur le serveur"""
    await initialize_datasets()
    return {
        "datasets_disponibles": list(AVAILABLE_DATASETS.keys()),
        "nombre_total": len(AVAILABLE_DATASETS),
    }


@mcp.tool()
async def lister_fichiers(
    dataset_name: str,
    params: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Liste tous les fichiers disponibles d'un dataset"""
    try:
        await initialize_datasets()
        base_url = get_dataset_url(dataset_name)
        client = await get_http_client()
        async with client:
            response = await client.get(f"{base_url}/data-files", params=params or {})
            response.raise_for_status()
            return response.json()
    except Exception as e:
        return [
            {
                "error": f"Erreur lors de la récupération des fichiers du dataset '{dataset_name}': {str(e)}"
            }
        ]


@mcp.tool()
async def lister_lignes(
    dataset_name: str,
    page: Optional[int] = None,
    after: Optional[str] = None,
    size: Optional[int] = None,
    sort: Optional[str] = None,
    select: Optional[str] = None,
    highlight: Optional[str] = None,
    format: Optional[str] = None,
    html: Optional[bool] = None,
    q: Optional[str] = None,
    q_mode: Optional[str] = None,
    q_fields: Optional[str] = None,
    qs: Optional[str] = None,
    collapse: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Requêter les lignes du jeu de données"""
    try:
        await initialize_datasets()
        base_url = get_dataset_url(dataset_name)
        params = {}
        if page is not None:
            params["page"] = page
        if after is not None:
            params["after"] = after
        if size is not None:
            params["size"] = min(size, 10000)
        if sort is not None:
            params["sort"] = sort
        if select is not None:
            params["select"] = select
        if highlight is not None:
            params["highlight"] = highlight
        if format is not None:
            params["format"] = format
        if html is not None:
            params["html"] = html
        if q is not None:
            params["q"] = q
        if q_mode is not None:
            params["q_mode"] = q_mode
        if q_fields is not None:
            params["q_fields"] = q_fields
        if qs is not None:
            params["qs"] = qs
        if collapse is not None:
            params["collapse"] = collapse
        client = await get_http_client()
        async with client:
            response = await client.get(f"{base_url}/lines", params=params or {})
            response.raise_for_status()
            return response.json()
    except Exception as e:
        return [{"error": f"Erreur lors de la récupération des lignes: {str(e)}"}]


@mcp.tool()
async def obtenir_values_agg(
    dataset_name: str,
    field: str,
    format: Optional[str] = None,
    html: Optional[bool] = None,
    metric: Optional[str] = None,
    metric_field: Optional[str] = None,
    agg_size: Optional[int] = None,
    q: Optional[str] = None,
    q_mode: Optional[str] = None,
    q_fields: Optional[str] = None,
    qs: Optional[str] = None,
    size: Optional[int] = None,
    sort: Optional[str] = None,
    select: Optional[str] = None,
    highlight: Optional[str] = None,
) -> Dict[str, Any]:
    """Récupérer des informations agrégées en fonction des valeurs d'une colonne"""
    try:
        await initialize_datasets()
        base_url = get_dataset_url(dataset_name)
        params = {"field": field}
        if format is not None:
            params["format"] = format
        if html is not None:
            params["html"] = html
        if metric is not None:
            params["metric"] = metric
        if metric_field is not None:
            params["metric_field"] = metric_field
        if agg_size is not None:
            params["agg_size"] = agg_size
        if q is not None:
            params["q"] = q
        if q_mode is not None:
            params["q_mode"] = q_mode
        if q_fields is not None:
            params["q_fields"] = q_fields
        if qs is not None:
            params["qs"] = qs
        if size is not None:
            params["size"] = size
        if sort is not None:
            params["sort"] = sort
        if select is not None:
            params["select"] = select
        if highlight is not None:
            params["highlight"] = highlight

        client = await get_http_client()
        async with client:
            response = await client.get(f"{base_url}/values_agg", params=params or {})
            response.raise_for_status()
            return response.json()
    except Exception as e:
        return {"error": f"Erreur lors de la récupération des agrégats: {str(e)}"}


@mcp.tool()
async def obtenir_words_agg(
    dataset_name: str,
    field: str,
    analysis: Optional[str] = None,
    lang: Optional[str] = None,
    q: Optional[str] = None,
    q_mode: Optional[str] = None,
    q_fields: Optional[str] = None,
    qs: Optional[str] = None,
) -> Dict[str, Any]:
    """Récupérer des mots significatifs dans un jeu de données"""
    try:
        await initialize_datasets()
        base_url = get_dataset_url(dataset_name)
        params = {"field": field}
        if analysis is not None:
            params["analysis"] = analysis
        if lang is not None:
            params["lang"] = lang
        if q is not None:
            params["q"] = q
        if q_mode is not None:
            params["q_mode"] = q_mode
        if q_fields is not None:
            params["q_fields"] = q_fields
        if qs is not None:
            params["qs"] = qs

        client = await get_http_client()
        async with client:
            response = await client.get(f"{base_url}/words_agg", params=params or {})
            response.raise_for_status()
            return response.json()
    except Exception as e:
        return {"error": f"Erreur lors de la récupération des mots: {str(e)}"}


# Point d'entrée principal
if __name__ == "__main__":
    # mcp.run()
    mcp.run(transport="streamable-http")
