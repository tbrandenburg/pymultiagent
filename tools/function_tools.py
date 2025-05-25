"""
Function tools for the PyMultiAgent system.

This module contains all the function tools that can be used by agents
in the multi-agent system.
"""
import os
import requests
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
from urllib.parse import quote

try:
    from agents import function_tool
except ImportError:
    raise ImportError("Required packages not found. Please install with 'pip install openai openai-agents'")


@function_tool
def fetch_http_url_content(url: str, timeout: int = 10) -> str:
    """
    Fetches the content of an HTTP or HTTPS URL.

    Args:
        url (str): The URL to fetch.
        timeout (int, optional): Timeout in seconds for the request (default: 10).

    Returns:
        str: The content of the URL as text.

    Raises:
        requests.RequestException: If the HTTP request fails or the URL is invalid.
    """
    try:
        headers = {
            "User-Agent": "PyMultiAgent/1.0 (https://github.com/example/pymultiagent)"
        }
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        raise requests.RequestException(f"Failed to fetch URL content: {str(e)}")

@function_tool
def read_file(file_path: Path) -> str:
    """
    Reads the entire content of a file.

    Args:
        file_path (Path): The path to the file to read

    Returns:
        str: The complete content of the file as a string

    Raises:
        FileNotFoundError: If the file does not exist
        PermissionError: If the file cannot be read due to permissions
    """
    return file_path.read_text()

@function_tool
def read_file_lines(file_path: Path, start_line: Optional[int] = None, end_line: Optional[int] = None) -> str:
    """
    Reads specific lines from a file within a given range.

    Args:
        file_path (Path): The path to the file to read
        start_line (int, optional): The starting line number (1-based indexing).
                                  If None, starts from the beginning of the file
        end_line (int, optional): The ending line number (1-based indexing).
                                If None, reads to the end of the file

    Returns:
        str: The content of the specified line range as a string

    Raises:
        FileNotFoundError: If the file does not exist
        PermissionError: If the file cannot be read due to permissions
    """
    lines = file_path.read_text().splitlines()
    if start_line is not None or end_line is not None:
        start = start_line - 1 if start_line else 0
        end = end_line if end_line else len(lines)
        return "\n".join(lines[start:end])
    else:
        return file_path.read_text()


@function_tool
def execute_shell_command(cd: str, command: str) -> str:
    """
    Executes a shell command in a specified directory.

    Args:
        cd (str): The directory path where the command should be executed
        command (str): The shell command to execute

    Returns:
        str: The stdout output from the command execution

    Raises:
        RuntimeError: If the command fails (non-zero exit code)
        FileNotFoundError: If the specified directory does not exist
    """
    from subprocess import Popen, PIPE
    process = Popen(command, cwd=cd, shell=True, stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        raise RuntimeError(f"Command failed with error: {stderr.decode()}")
    return stdout.decode()

@function_tool
def get_current_datetime_rfc(timezone: str = "local") -> str:
    """
    Returns the current date and time in RFC 3339 format.

    Args:
        timezone (str, optional): The timezone to use. Options are "local" or "utc".
                                Defaults to "local"

    Returns:
        str: The current datetime in ISO 8601/RFC 3339 format (YYYY-MM-DDTHH:MM:SS.ffffff)

    Example:
        >>> get_current_datetime_rfc("utc")
        '2023-12-07T14:30:15.123456'
    """
    now = datetime.utcnow() if timezone == "utc" else datetime.now()
    return now.isoformat()

@function_tool
def get_current_date() -> str:
    """
    Returns the current date in a human-readable format.

    Returns:
        str: The current date in YYYY-MM-DD format

    Example:
        >>> get_current_date()
        '2023-12-07'
    """
    return datetime.now().strftime("%Y-%m-%d")

@function_tool
def get_current_time() -> str:
    """
    Returns the current time in a human-readable format.

    Returns:
        str: The current time in HH:MM:SS format (24-hour format)

    Example:
        >>> get_current_time()
        '14:30:15'
    """
    return datetime.now().strftime("%H:%M:%S")

@function_tool
def create_directory(path: str) -> None:
    """
    Creates a new directory at the specified path, including parent directories if needed.

    Args:
        path (str): The path where the directory should be created

    Returns:
        None

    Note:
        This function will create parent directories if they don't exist and will not
        raise an error if the directory already exists.

    Raises:
        PermissionError: If the directory cannot be created due to permissions
    """
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)

@function_tool
def get_user_input(prompt: str) -> str:
    """
    Prompts the user for additional input during agent execution.

    Args:
        prompt (str): The prompt message to display to the user

    Returns:
        str: The user's input as a string

    Note:
        This function will block execution until the user provides input.
        Use this when the agent needs clarification or additional information
        from the user to complete a task.
    """
    return input(prompt+":\n")

@function_tool
def list_directory(path: str) -> list:
    """
    Lists all files and directories in the specified path.

    Args:
        path (str): The directory path to list contents for

    Returns:
        list: A list of strings containing the names of all files and directories
              in the specified path

    Raises:
        FileNotFoundError: If the directory does not exist or is not a directory
        PermissionError: If the directory cannot be accessed due to permissions

    Example:
        >>> list_directory("/home/user/documents")
        ['/home/user/documents/file1.txt', '/home/user/documents/subfolder']
    """
    directory = Path(path)
    if not directory.exists() or not directory.is_dir():
        raise FileNotFoundError(f"Directory {path} does not exist.")
    return [str(item) for item in directory.iterdir()]

@function_tool
def write_file(file_path: Path, content: str, mode: str = "overwrite") -> None:
    """
    Writes content to a file with different modes of operation.

    Args:
        file_path (Path): The path to the file to write to
        content (str): The content to write to the file
        mode (str, optional): The write mode. Options are:
                             - "overwrite": Replace the entire file content (default)
                             - "edit": Append content to existing file content
                             - "create": Create new file only if it doesn't exist

    Returns:
        None

    Raises:
        FileExistsError: If mode is "create" and the file already exists
        PermissionError: If the file cannot be written due to permissions

    Note:
        The function will automatically create parent directories if they don't exist.
        All files are written with UTF-8 encoding.

    Example:
        >>> write_file(Path("example.txt"), "Hello World", "overwrite")
        >>> write_file(Path("log.txt"), "New entry\\n", "edit")
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    if mode == "edit":
        existing_content = file_path.read_text(encoding="utf-8") if file_path.exists() else ""
        file_path.write_text(existing_content + content, encoding="utf-8")
    elif mode == "create" and file_path.exists():
        raise FileExistsError(f"File {file_path} already exists.")
    else:
        file_path.write_text(content, encoding="utf-8")


@function_tool
def search_web_serper(query: str, num_results: int = 10) -> Dict[str, Any]:
    """
    Performs a web search using the Serper API.

    Args:
        query (str): The search query string
        num_results (int, optional): Number of search results to return (default: 10, max: 100)

    Returns:
        Dict[str, Any]: A dictionary containing search results with the following structure:
                       {
                           "organic": [
                               {
                                   "title": "Page title",
                                   "link": "URL",
                                   "snippet": "Description snippet",
                                   "date": "Publication date (if available)"
                               },
                               ...
                           ],
                           "answerBox": {...} (if available),
                           "knowledgeGraph": {...} (if available)
                       }

    Raises:
        ValueError: If SERPER_API_KEY environment variable is not set
        requests.RequestException: If the API request fails
        
    Note:
        Requires SERPER_API_KEY environment variable to be set.
        Get your API key from https://serper.dev/
    """
    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        raise ValueError("SERPER_API_KEY environment variable is required. Get your API key from https://serper.dev/")
    
    url = "https://google.serper.dev/search"
    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json"
    }
    
    payload = {
        "q": query,
        "num": min(num_results, 100)
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        raise requests.RequestException(f"Serper API request failed: {str(e)}")


@function_tool
def search_web_duckduckgo(query: str, max_results: int = 10) -> List[Dict[str, str]]:
    """
    Performs a web search using DuckDuckGo's instant answer API.

    Args:
        query (str): The search query string
        max_results (int, optional): Maximum number of search results to return (default: 10)

    Returns:
        List[Dict[str, str]]: A list of dictionaries containing search results:
                             [
                                 {
                                     "title": "Page title",
                                     "url": "URL",
                                     "description": "Description snippet"
                                 },
                                 ...
                             ]

    Raises:
        requests.RequestException: If the API request fails
        
    Note:
        This uses DuckDuckGo's instant answer API which has rate limits.
        For heavy usage, consider using the Serper API instead.
    """
    try:
        # DuckDuckGo instant answer API
        url = "https://api.duckduckgo.com/"
        params = {
            "q": query,
            "format": "json",
            "no_html": "1",
            "skip_disambig": "1"
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        results = []
        
        # Add instant answer if available
        if data.get("Abstract"):
            results.append({
                "title": data.get("Heading", "DuckDuckGo Instant Answer"),
                "url": data.get("AbstractURL", ""),
                "description": data.get("Abstract", "")
            })
        
        # Add related topics
        for topic in data.get("RelatedTopics", [])[:max_results]:
            if isinstance(topic, dict) and "Text" in topic:
                results.append({
                    "title": topic.get("Text", "").split(" - ")[0] if " - " in topic.get("Text", "") else topic.get("Text", ""),
                    "url": topic.get("FirstURL", ""),
                    "description": topic.get("Text", "")
                })
        
        return results[:max_results]
        
    except requests.RequestException as e:
        raise requests.RequestException(f"DuckDuckGo API request failed: {str(e)}")


@function_tool
def fetch_wikipedia_content(title: str, sentences: int = 3) -> Dict[str, Any]:
    """
    Fetches content from Wikipedia for a given topic.

    Args:
        title (str): The Wikipedia article title or search term
        sentences (int, optional): Number of sentences to extract from the article (default: 3)

    Returns:
        Dict[str, Any]: A dictionary containing Wikipedia content:
                       {
                           "title": "Article title",
                           "extract": "Article extract/summary",
                           "url": "Wikipedia URL",
                           "page_id": "Wikipedia page ID"
                       }

    Raises:
        requests.RequestException: If the API request fails
        ValueError: If no Wikipedia article is found for the given title
        
    Note:
        Uses Wikipedia's REST API. No API key required.
    """
    try:
        # First, search for the article
        search_url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + quote(title)
        
        headers = {
            "User-Agent": "PyMultiAgent/1.0 (https://github.com/example/pymultiagent)"
        }
        
        response = requests.get(search_url, headers=headers, timeout=10)
        
        if response.status_code == 404:
            # Try searching for the title
            search_api_url = "https://en.wikipedia.org/w/api.php"
            search_params = {
                "action": "query",
                "format": "json",
                "list": "search",
                "srsearch": title,
                "srlimit": 1
            }
            
            search_response = requests.get(search_api_url, params=search_params, headers=headers, timeout=10)
            search_response.raise_for_status()
            search_data = search_response.json()
            
            if not search_data.get("query", {}).get("search"):
                raise ValueError(f"No Wikipedia article found for '{title}'")
            
            # Get the first search result title
            actual_title = search_data["query"]["search"][0]["title"]
            summary_url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + quote(actual_title)
            response = requests.get(summary_url, headers=headers, timeout=10)
        
        response.raise_for_status()
        data = response.json()
        
        # Extract sentences from the extract
        extract = data.get("extract", "")
        if extract and sentences > 0:
            # Split into sentences and take the requested number
            import re
            sentences_list = re.split(r'(?<=[.!?])\s+', extract)
            extract = ' '.join(sentences_list[:sentences])
        
        return {
            "title": data.get("title", ""),
            "extract": extract,
            "url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
            "page_id": data.get("pageid", "")
        }
        
    except requests.RequestException as e:
        raise requests.RequestException(f"Wikipedia API request failed: {str(e)}")


@function_tool
def search_wikipedia(query: str, limit: int = 5) -> List[Dict[str, str]]:
    """
    Searches Wikipedia for articles matching the query.

    Args:
        query (str): The search query string
        limit (int, optional): Maximum number of search results to return (default: 5)

    Returns:
        List[Dict[str, str]]: A list of dictionaries containing search results:
                             [
                                 {
                                     "title": "Article title",
                                     "snippet": "Search result snippet",
                                     "page_id": "Wikipedia page ID"
                                 },
                                 ...
                             ]

    Raises:
        requests.RequestException: If the API request fails
        
    Note:
        This function searches for articles and returns basic information.
        Use fetch_wikipedia_content() to get full article content.
    """
    try:
        url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "format": "json",
            "list": "search",
            "srsearch": query,
            "srlimit": limit,
            "srprop": "snippet"
        }
        
        headers = {
            "User-Agent": "PyMultiAgent/1.0 (https://github.com/example/pymultiagent)"
        }
        
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        results = []
        for article in data.get("query", {}).get("search", []):
            # Clean HTML tags from snippet
            import re
            snippet = re.sub(r'<[^>]+>', '', article.get("snippet", ""))
            
            results.append({
                "title": article.get("title", ""),
                "snippet": snippet,
                "page_id": str(article.get("pageid", ""))
            })
        
        return results
        
    except requests.RequestException as e:
        raise requests.RequestException(f"Wikipedia search failed: {str(e)}")
