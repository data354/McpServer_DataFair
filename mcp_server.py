import os
import sys
import asyncio
import time
from typing import Dict, List, Any, Optional
import httpx
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

# from fastmcp import FastMCP


load_dotenv(dotenv_path=".env")

# Configuration constants
DATASET_URL = os.getenv("DATASET_URL")
DATASET_KOUMOUL_URL = os.getenv("DATASET_KOUMOUL_URL")
API_BASE_URL_TEMPLATE = "{DATASET_URL}/{dataset_id}"
API_BASE_URL_TEMPLATE_KOUMOUL = "{DATASET_KOUMOUL_URL}/{dataset_id}"
API_KEY = os.getenv("API_KEY")

# Add a cache for dataset details
DATASET_DETAILS_CACHE = {}
# Global cache for dataset management
AVAILABLE_DATASETS = {}  # Dictionary to store available datasets {name: id}
AVAILABLE_DATASETS_KOUMOUL = {}  # Dictionary to store available datasets {name: id}
LAST_DATASETS_UPDATE = 0
DATASETS_CACHE_DURATION = 300  # Cache duration in seconds (5 minutes)
DATASETS_INITIALIZED = False


# Initialize FastMCP server
mcp = FastMCP("multi-datasets-server")


async def get_http_client() -> httpx.AsyncClient(timeout=30.0):
    """
    Create and return a configured HTTP client for API requests.

    Returns:
        httpx.AsyncClient: Configured HTTP client with headers and timeout
    """
    return httpx.AsyncClient(
        headers={
            # "x-apikey": f"{API_KEY}",
            "Content-Type": "application/json",
        },
        timeout=30.0,
    )


async def get_http_client_koumoul() -> httpx.AsyncClient(timeout=60.0):
    """
    Create and return a configured HTTP client for API requests.

    Returns:
        httpx.AsyncClient: Configured HTTP client with cookies and timeout
    """
    # Use cookies for authentication instead of headers
    cookies = {"id_token": API_KEY}

    # Basic headers without authentication
    headers = {
        "Content-Type": "application/json",
    }

    return httpx.AsyncClient(
        cookies=cookies,  # Authentication via cookies instead of x-apikey header
        headers=headers,
        timeout=60.0,
    )


async def fetch_available_datasets() -> Dict[str, str]:
    """
    Fetch all available datasets from the API dynamically with intelligent pagination.

    Returns:
        Dict[str, str]: Dictionary mapping dataset titles to their IDs

    Raises:
        RuntimeError: If the API request fails
    """
    try:
        client = await get_http_client()
        async with client:
            # Start with a reasonable size for the first request
            initial_size = 500  # Large enough for most cases

            print("Fetching datasets...", file=sys.stderr)
            response = await client.get(
                DATASET_URL, params={"size": initial_size}, timeout=30.0
            )
            response.raise_for_status()
            data = response.json()

            total_count = data.get("count", 0)
            results = data.get("results", [])

            print(f"Found {total_count} total datasets", file=sys.stderr)

            # If we didn't retrieve everything with the first request
            if len(results) < total_count:
                print(
                    f"Need to fetch remaining datasets ({len(results)}/{total_count} retrieved)",
                    file=sys.stderr,
                )
                # Make a second request with the exact size
                response = await client.get(
                    DATASET_URL, params={"size": total_count}, timeout=45.0
                )
                response.raise_for_status()
                data = response.json()
                results = data.get("results", [])

            # Build dictionary mapping dataset names to IDs
            datasets = {}
            for dataset in results:
                try:
                    datasets[dataset["title"]] = dataset["id"]
                except KeyError:
                    print(f"Skipping malformed dataset: {dataset}", file=sys.stderr)

            print(
                f"Successfully fetched {len(datasets)}/{total_count} datasets",
                file=sys.stderr,
            )

            # Consistency check
            if len(datasets) != total_count:
                print(
                    f"Warning: Expected {total_count} datasets but got {len(datasets)}",
                    file=sys.stderr,
                )

            return datasets

    except Exception as e:
        print(f"Failed to fetch datasets: {str(e)}", file=sys.stderr)
        raise RuntimeError(f"Failed to fetch datasets: {str(e)}")


async def fetch_available_datasets_koumoul() -> Dict[str, str]:
    """
    Fetch all available datasets from the API dynamically with intelligent pagination.

    Returns:
        Dict[str, str]: Dictionary mapping dataset titles to their IDs

    Raises:
        RuntimeError: If the API request fails
    """
    try:
        client = await get_http_client_koumoul()
        async with client:
            # Start with a reasonable size for the first request
            initial_size = 500  # Large enough for most cases

            print("Fetching datasets...", file=sys.stderr)
            response = await client.get(
                DATASET_KOUMOUL_URL, params={"size": initial_size}
            )
            response.raise_for_status()
            data = response.json()

            total_count = data.get("count", 0)
            results = data.get("results", [])

            print(f"Found {total_count} total datasets", file=sys.stderr)

            # If we didn't retrieve everything with the first request
            if len(results) < total_count:
                print(
                    f"Need to fetch remaining datasets ({len(results)}/{total_count} retrieved)",
                    file=sys.stderr,
                )
                # Make a second request with the exact size
                response = await client.get(
                    DATASET_KOUMOUL_URL, params={"size": total_count}
                )
                response.raise_for_status()
                data = response.json()
                results = data.get("results", [])

            # Build dictionary mapping dataset names to IDs
            datasets = {dataset["title"]: dataset["id"] for dataset in results}

            print(
                f"Successfully fetched {len(datasets)}/{total_count} datasets",
                file=sys.stderr,
            )

            # Consistency check
            if len(datasets) != total_count:
                print(
                    f"Warning: Expected {total_count} datasets but got {len(datasets)}",
                    file=sys.stderr,
                )

            return datasets

    except Exception as e:
        print(f"Failed to fetch datasets: {str(e)}", file=sys.stderr)
        raise RuntimeError(f"Failed to fetch datasets: {str(e)}")


def get_dataset_url(dataset_name: str) -> str:
    """
    Construct the base URL for a specific dataset.

    Args:
        dataset_name (str): Name of the dataset

    Returns:
        str: Complete API URL for the dataset

    Raises:
        RuntimeError: If datasets are not initialized
        ValueError: If dataset name is not found
    """
    if not AVAILABLE_DATASETS:
        raise RuntimeError(
            "Dataset list not initialized. Call fetch_available_datasets first."
        )

    if dataset_name not in AVAILABLE_DATASETS:
        available_names = list(AVAILABLE_DATASETS.keys())
        raise ValueError(
            f"Dataset '{dataset_name}' not found. Available datasets: {available_names[:10]}..."
        )

    dataset_id = AVAILABLE_DATASETS[dataset_name]
    return API_BASE_URL_TEMPLATE.format(dataset_id=dataset_id, DATASET_URL=DATASET_URL)


def get_dataset_url_koumoul(dataset_name: str) -> str:
    """
    Construct the base URL for a specific dataset.

    Args:
        dataset_name (str): Name of the dataset

    Returns:
        str: Complete API URL for the dataset

    Raises:
        RuntimeError: If datasets are not initialized
        ValueError: If dataset name is not found
    """
    if not AVAILABLE_DATASETS_KOUMOUL:
        raise RuntimeError(
            "Dataset list not initialized. Call fetch_available_datasets first."
        )

    if dataset_name not in AVAILABLE_DATASETS_KOUMOUL:
        available_names = list(AVAILABLE_DATASETS_KOUMOUL.keys())
        raise ValueError(
            f"Dataset '{dataset_name}' not found. Available datasets: {available_names[:10]}..."
        )

    dataset_id = AVAILABLE_DATASETS_KOUMOUL[dataset_name]
    return API_BASE_URL_TEMPLATE_KOUMOUL.format(
        dataset_id=dataset_id, DATASET_KOUMOUL_URL=DATASET_KOUMOUL_URL
    )


async def get_dataset_details(dataset_id: str) -> Dict[str, Any]:
    """Fetch dataset details with caching"""
    # Check the cache first
    if dataset_id in DATASET_DETAILS_CACHE:
        return DATASET_DETAILS_CACHE[dataset_id]

    try:
        client = await get_http_client()
        async with client:
            response = await client.get(f"{DATASET_URL}/{dataset_id}")
            response.raise_for_status()
            details = response.json()

            # Cache for future requests
            DATASET_DETAILS_CACHE[dataset_id] = details
            return details
    except Exception:
        return {}


async def get_dataset_details_koumoul(dataset_id: str) -> Dict[str, Any]:
    """Fetch dataset details with caching"""
    # Check the cache first
    if dataset_id in DATASET_DETAILS_CACHE_KOUMOUL:
        return DATASET_DETAILS_CACHE_KOUMOUL[dataset_id]

    try:
        client = await get_http_client_koumoul()
        async with client:
            response = await client.get(f"{DATASET_KOUMOUL_URL}/{dataset_id}")
            response.raise_for_status()
            details = response.json()

            # Cache for future requests
            DATASET_DETAILS_CACHE_KOUMOUL[dataset_id] = details
            return details
    except Exception:
        return {}


async def initialize_datasets():
    """
    Initialize or refresh the datasets cache if needed.

    Updates the global AVAILABLE_DATASETS dictionary and tracks changes.
    Cache is refreshed if empty or expired based on DATASETS_CACHE_DURATION.

    Raises:
        Exception: If initial dataset loading fails
    """
    global AVAILABLE_DATASETS, LAST_DATASETS_UPDATE, DATASETS_INITIALIZED

    current_time = time.time()
    cache_expired = (current_time - LAST_DATASETS_UPDATE) > DATASETS_CACHE_DURATION

    # Reload datasets if cache is empty or expired
    if not AVAILABLE_DATASETS or cache_expired:
        try:
            print("Refreshing datasets cache...", file=sys.stderr)
            new_datasets = await fetch_available_datasets()

            # Check for changes and log them
            if new_datasets != AVAILABLE_DATASETS:
                print(
                    f"Datasets updated: {len(new_datasets)} total datasets",
                    file=sys.stderr,
                )

                if AVAILABLE_DATASETS:  # Not the first load
                    added = set(new_datasets.keys()) - set(AVAILABLE_DATASETS.keys())
                    removed = set(AVAILABLE_DATASETS.keys()) - set(new_datasets.keys())

                    if added:
                        print(
                            f"New datasets: {', '.join(list(added)[:5])}...",
                            file=sys.stderr,
                        )
                    if removed:
                        print(
                            f"Removed datasets: {', '.join(list(removed)[:5])}...",
                            file=sys.stderr,
                        )

            AVAILABLE_DATASETS = new_datasets
            LAST_DATASETS_UPDATE = current_time
            DATASETS_INITIALIZED = True

        except Exception as e:
            print(f"Error refreshing datasets: {e}", file=sys.stderr)
            # Keep existing cache if refresh fails, but raise if no cache exists
            if not AVAILABLE_DATASETS:
                raise


async def initialize_datasets_koumoul():
    """
    Initialize or refresh the datasets cache if needed.

    Updates the global AVAILABLE_DATASETS dictionary and tracks changes.
    Cache is refreshed if empty or expired based on DATASETS_CACHE_DURATION.

    Raises:
        Exception: If initial dataset loading fails
    """
    global AVAILABLE_DATASETS_KOUMOUL, LAST_DATASETS_UPDATE, DATASETS_INITIALIZED_KOUMOUL

    current_time = time.time()
    cache_expired = (current_time - LAST_DATASETS_UPDATE) > DATASETS_CACHE_DURATION

    # Reload datasets if cache is empty or expired
    if not AVAILABLE_DATASETS_KOUMOUL or cache_expired:
        try:
            print("Refreshing datasets cache...", file=sys.stderr)
            new_datasets = await fetch_available_datasets_koumoul()

            # Check for changes and log them
            if new_datasets != AVAILABLE_DATASETS:
                print(
                    f"Datasets updated: {len(new_datasets)} total datasets",
                    file=sys.stderr,
                )

                if AVAILABLE_DATASETS_KOUMOUL:  # Not the first load
                    added = set(new_datasets.keys()) - set(
                        AVAILABLE_DATASETS_KOUMOUL.keys()
                    )
                    removed = set(AVAILABLE_DATASETS_KOUMOUL.keys()) - set(
                        new_datasets.keys()
                    )

                    if added:
                        print(
                            f"New datasets: {', '.join(list(added)[:5])}...",
                            file=sys.stderr,
                        )
                    if removed:
                        print(
                            f"Removed datasets: {', '.join(list(removed)[:5])}...",
                            file=sys.stderr,
                        )

            AVAILABLE_DATASETS_KOUMOUL = new_datasets
            LAST_DATASETS_UPDATE = current_time
            DATASETS_INITIALIZED_KOUMOUL = True

        except Exception as e:
            print(f"Error refreshing datasets: {e}", file=sys.stderr)
            # Keep existing cache if refresh fails, but raise if no cache exists
            if not AVAILABLE_DATASETS_KOUMOUL:
                raise


# MCP Tools
@mcp.tool()
async def list_available_datasets() -> Dict[str, Any]:
    """
    List all available datasets on the server.

    Returns:
        Dict[str, Any]: Dictionary containing list of dataset names and total count
    """
    try:
        await initialize_datasets()
        return {
            "available_datasets": list(AVAILABLE_DATASETS.keys()),
            "total_count": len(AVAILABLE_DATASETS),
        }
    except Exception as e:
        return {
            "error": f"Failed to fetch datasets: {str(e)}",
            "available_datasets": [],
            "total_count": 0,
        }


@mcp.tool()
async def list_dataset_files(
    dataset_name: str,
    params: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    List all files available in a specific dataset.

    Args:
        dataset_name (str): Name of the target dataset
        params (Optional[Dict[str, Any]]): Additional query parameters

    Returns:
        List[Dict[str, Any]]: List of file information or error details
    """
    try:
        await initialize_datasets()
        base_url = get_dataset_url(dataset_name)
        print(base_url)

        client = await get_http_client()
        async with client:
            response = await client.get(f"{base_url}/data-files", params=params or {})
            response.raise_for_status()
            return response.json()

    except Exception as e:
        return [
            {
                "error": f"Failed to retrieve files for dataset '{dataset_name}': {str(e)}"
            }
        ]


@mcp.tool()
async def list_datasets(query: Optional[str] = None) -> Dict[str, Any]:
    """
    List available datasets with optimized thematic filtering.
    """
    try:
        await initialize_datasets()

        # If no query, return simply the names
        if not query:
            return {
                "available_datasets": list(AVAILABLE_DATASETS.keys()),
                "total_count": len(AVAILABLE_DATASETS),
            }

        query = query.lower()
        matching_datasets = []

        # Optimized search with priority on the name
        for name, dataset_id in AVAILABLE_DATASETS.items():
            # First check the name (faster)
            if query in name.lower():
                matching_datasets.append({"name": name, "topics": []})
                continue

            # Then check the topics if necessary
            details = await get_dataset_details(dataset_id)
            topics = [topic["title"].lower() for topic in details.get("topics", [])]

            if any(query in topic for topic in topics):
                matching_datasets.append({"name": name, "topics": topics})

        return {
            "available_datasets": [ds["name"] for ds in matching_datasets],
            "topics": [topic for ds in matching_datasets for topic in ds["topics"]],
            "total_count": len(matching_datasets),
        }
    except Exception as e:
        return {
            "error": f"Failed to fetch datasets: {str(e)}",
            "available_datasets": [],
            "total_count": 0,
        }


@mcp.tool()
async def query_dataset_rows(
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
    """
    Query rows from a dataset with various filtering and formatting options.

    Args:
        dataset_name (str): Name of the target dataset
        page (Optional[int]): Page number for pagination
        after (Optional[str]): Cursor for pagination
        size (Optional[int]): Number of results per page (max 10000)
        sort (Optional[str]): Sort criteria
        select (Optional[str]): Fields to select
        highlight (Optional[str]): Fields to highlight
        format (Optional[str]): Output format
        html (Optional[bool]): Include HTML formatting
        q (Optional[str]): Search query
        q_mode (Optional[str]): Query mode
        q_fields (Optional[str]): Fields to search in
        qs (Optional[str]): Query string
        collapse (Optional[str]): Field to collapse on

    Returns:
        List[Dict[str, Any]]: Query results or error details
    """
    try:
        await initialize_datasets()
        base_url = get_dataset_url(dataset_name)

        # Build query parameters
        params = {}
        if page is not None:
            params["page"] = page
        if after is not None:
            params["after"] = after
        if size is not None:
            params["size"] = min(size, 10000)  # Enforce maximum size limit
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
            response = await client.get(f"{base_url}/lines", params=params)
            response.raise_for_status()
            return response.json()

    except Exception as e:
        return [{"error": f"Failed to retrieve rows: {str(e)}"}]


@mcp.tool()
async def get_values_aggregation(
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
    """
    Retrieve aggregated information based on field values.

    Args:
        dataset_name (str): Name of the target dataset
        field (str): Field to aggregate on
        format (Optional[str]): Output format
        html (Optional[bool]): Include HTML formatting
        metric (Optional[str]): Aggregation metric
        metric_field (Optional[str]): Field for metric calculation
        agg_size (Optional[int]): Maximum aggregation results
        q (Optional[str]): Search query
        q_mode (Optional[str]): Query mode
        q_fields (Optional[str]): Fields to search in
        qs (Optional[str]): Query string
        size (Optional[int]): Number of results
        sort (Optional[str]): Sort criteria
        select (Optional[str]): Fields to select
        highlight (Optional[str]): Fields to highlight

    Returns:
        Dict[str, Any]: Aggregation results or error details
    """
    try:
        await initialize_datasets()
        base_url = get_dataset_url(dataset_name)

        # Build query parameters
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
            response = await client.get(f"{base_url}/values_agg", params=params)
            response.raise_for_status()
            return response.json()

    except Exception as e:
        return {"error": f"Failed to retrieve aggregates: {str(e)}"}


@mcp.tool()
async def get_words_aggregation(
    dataset_name: str,
    field: str,
    analysis: Optional[str] = None,
    lang: Optional[str] = None,
    q: Optional[str] = None,
    q_mode: Optional[str] = None,
    q_fields: Optional[str] = None,
    qs: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Retrieve significant words from a dataset field.

    Args:
        dataset_name (str): Name of the target dataset
        field (str): Field to analyze for words
        analysis (Optional[str]): Type of text analysis
        lang (Optional[str]): Language for analysis
        q (Optional[str]): Search query
        q_mode (Optional[str]): Query mode
        q_fields (Optional[str]): Fields to search in
        qs (Optional[str]): Query string

    Returns:
        Dict[str, Any]: Word analysis results or error details
    """
    try:
        await initialize_datasets()
        base_url = get_dataset_url(dataset_name)

        # Build query parameters
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
            response = await client.get(f"{base_url}/words_agg", params=params)
            response.raise_for_status()
            return response.json()

    except Exception as e:
        return {"error": f"Failed to retrieve word analysis: {str(e)}"}


@mcp.tool()
async def get_dataset_field_values(
    dataset_name: str,
    field: str,
    size: Optional[int] = None,
    sort: Optional[str] = None,
    q: Optional[str] = None,
    q_mode: Optional[str] = None,
    q_fields: Optional[str] = None,
    qs: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Retrieve distinct values for a specific field in a dataset.

    Args:
        dataset_name: Name of the dataset
        field: Field name to get values for
        size: Number of values to return (default=all)
        sort: Sorting criteria ("-count" or "+value")
        q: Full-text search query
        q_mode: Search mode ("simple" or "complex")
        q_fields: Fields to search in
        qs: Query string for advanced filtering

    Returns:
        Dictionary containing field values and metadata
    """
    try:
        await initialize_datasets()
        base_url = get_dataset_url(dataset_name)
        params = {}

        # Build query parameters
        if size:
            params["size"] = size
        if sort:
            params["sort"] = sort
        if q:
            params["q"] = q
        if q_mode:
            params["q_mode"] = q_mode
        if q_fields:
            params["q_fields"] = q_fields
        if qs:
            params["qs"] = qs

        client = await get_http_client()
        async with client:
            # Construct URL: {base_url}/values/{field}
            url = f"{base_url}/values/{field}"
            response = await client.get(url, params=params)
            response.raise_for_status()

            return response.json()

    except Exception as e:
        return {"error": f"Failed to retrieve values for field '{field}': {str(e)}"}


@mcp.tool()
async def get_dataset_metric_agg(
    dataset_name: str,
    metric: str,
    field: str,
    percents: Optional[str] = None,
    q: Optional[str] = None,
    q_mode: Optional[str] = None,
    q_fields: Optional[str] = None,
    qs: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Perform metric aggregation on a dataset field.

    Args:
        dataset_name: Name of the dataset
        metric: Metric to compute (e.g., "avg", "sum", "min", "max", "percentiles")
        field: Field name to compute metrics on
        percents: Comma-separated percentiles for percentile metrics (e.g., "25,50,75")
        q: Full-text search query
        q_mode: Search mode ("simple" or "complex")
        q_fields: Fields to search in
        qs: Query string for advanced filtering

    Returns:
        Dictionary containing metric results
    """
    try:
        await initialize_datasets()
        base_url = get_dataset_url(dataset_name)
        params = {"metric": metric, "field": field}

        # Build query parameters
        if percents:
            params["percents"] = percents
        if q:
            params["q"] = q
        if q_mode:
            params["q_mode"] = q_mode
        if q_fields:
            params["q_fields"] = q_fields
        if qs:
            params["qs"] = qs

        client = await get_http_client()
        async with client:
            url = f"{base_url}/metric_agg"
            response = await client.get(url, params=params)
            response.raise_for_status()

            return response.json()

    except Exception as e:
        return {"error": f"Failed to compute metric aggregation: {str(e)}"}


@mcp.tool()
async def get_dataset_simple_metrics_agg(
    dataset_name: str,
    metrics: str,
    fields: str,
    q: Optional[str] = None,
    q_mode: Optional[str] = None,
    q_fields: Optional[str] = None,
    qs: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Perform multiple metric aggregations on dataset fields in a single request.

    Args:
        dataset_name: Name of the dataset
        metrics: Comma-separated metrics to compute (e.g., "avg,min,max")
        fields: Comma-separated fields to compute metrics on
        q: Full-text search query
        q_mode: Search mode ("simple" or "complex")
        q_fields: Fields to search in
        qs: Query string for advanced filtering

    Returns:
        Dictionary containing metric results for each field
    """
    try:
        await initialize_datasets()
        base_url = get_dataset_url(dataset_name)
        params = {"metrics": metrics, "fields": fields}

        # Build query parameters
        if q:
            params["q"] = q
        if q_mode:
            params["q_mode"] = q_mode
        if q_fields:
            params["q_fields"] = q_fields
        if qs:
            params["qs"] = qs

        client = await get_http_client()
        async with client:
            url = f"{base_url}/simple_metrics_agg"
            response = await client.get(url, params=params)
            response.raise_for_status()

            return response.json()

    except Exception as e:
        return {"error": f"Failed to compute simple metrics aggregation: {str(e)}"}


@mcp.tool()
async def create_dataset_row(
    dataset_name: str, row_data: Dict[str, Any], validate: bool = True
) -> Dict[str, Any]:
    """
    Create a new row in the specified dataset using the /lines endpoint.

    Args:
        dataset_name: Name of the target dataset
        row_data: Dictionary containing the row data to insert
        validate: Whether to validate the data before insertion (default: True)

    Returns:
        Dictionary with operation result:
        {
            "success": boolean,
            "data": created row data or None,
            "error": error message or None
        }
    """
    try:
        await initialize_datasets_koumoul()

        # Construct the correct endpoint URL
        dataset_id = AVAILABLE_DATASETS_KOUMOUL[dataset_name]
        url = f"{DATASET_KOUMOUL_URL}/{dataset_id}/lines"

        print(f"Creating row in dataset: {dataset_name}", file=sys.stderr)
        print(f"Using endpoint: {url}", file=sys.stderr)

        # Prepare headers - Remove x-apikey from headers
        headers = {
            "Content-Type": "application/json",
            "x-apiKey": API_KEY,
            "Cache-Control": "no-cache",
        }

        # # Set cookies with the API key as id_token (based on documentation)
        # cookies = {"id_token": API_KEY}

        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                json=row_data,
                headers=headers,
                params={"validate": str(validate).lower()},
            )

            # Debug: Print response details
            print(f"Response status: {response.status_code}", file=sys.stderr)
            print(f"Response headers: {response.headers}", file=sys.stderr)
            print(f"Response content: {response.text}", file=sys.stderr)

            # Successful creation returns 201
            if response.status_code == 201:
                try:
                    response_data = response.json() if response.content else None
                    return {"success": True, "data": response_data, "error": None}
                except Exception as json_error:
                    return {"success": True, "data": response.text, "error": None}

            # For other status codes, try to get error details
            try:
                error_details = response.json() if response.content else response.text
            except:
                error_details = response.text

            return {
                "success": False,
                "data": None,
                "error": f"API returned {response.status_code}: {error_details}",
            }

    except KeyError:
        return {
            "success": False,
            "data": None,
            "error": f"Dataset '{dataset_name}' not found",
        }
    except Exception as e:
        return {"success": False, "data": None, "error": f"Unexpected error: {str(e)}"}


# Main entry point
if __name__ == "__main__":
    print("Starting MCP server...", file=sys.stderr)

    port = int(os.environ.get("PORT", 8000))

    mcp.run(
        transport="streamable-http",
        host="0.0.0.0",  # Pour Cloud Run
        port=port,
        path="/mcp",  # Chemin par défaut
    )

if __name__ == "__main__":
    print("Starting MCP server...", file=sys.stderr)

    port = int(os.environ.get("PORT", 8000))

    mcp.run(
        transport="streamable-http",
        host="0.0.0.0",  # Pour Cloud Run
        port=port,
        path="/mcp",  # Chemin par défaut
    )
