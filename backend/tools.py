import requests
import os
from typing import Type
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import wikipediaapi

class BraveSearchInput(BaseModel):
    """Input schema for Brave Search Tool."""
    query: str = Field(
        ..., 
        description="The search query to find information about"
    )
    count: int = Field(
        default=10, 
        description="Number of search results to return (default: 10, max: 20)"
    )

class BraveSearchTool(BaseTool):
    name: str = "Brave Search"
    description: str = (
        "Search the web using Brave Search API to find current information, "
        "news, and articles on any topic. Returns search results with titles, "
        "descriptions, and URLs."
    )
    args_schema: Type[BaseModel] = BraveSearchInput

    def _run(self, query: str, count: int = 5) -> str:
        """
        Execute the Brave Search API request.
        
        Args:
            query: The search query string
            count: Number of results to return
        
        Returns:
            Formatted string with search results
        """
        api_key = os.getenv("BRAVE_API_KEY")
        
        if not api_key:
            return "Error: BRAVE_API_KEY environment variable not set. Please set your Brave API key."
        
        url = "https://api.search.brave.com/res/v1/web/search"
        
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": api_key
        }
        
        params = {
            "q": query,
            "count": min(count, 20)  # API max is 20
        }
        
        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()

            results_block = data.get("web", {}).get("results", [])

            results = []
            for idx, result in enumerate(results_block, 1):
                # result is a dict
                formatted_result = f"""
            [Result {idx}]
            Title: {result.get('title', 'N/A')}
            URL: {result.get('url', 'N/A')}
            Description: {result.get('description', 'N/A')}
            """
                results.append(formatted_result)

            if not results:
                return f"No search results found for query: '{query}'"

            summary = f"Found {len(results)} result(s) for '{query}':\n"
            return summary + "\n".join(results)
        except:
            raise ValueError()


class WikipediaInput(BaseModel):
    """Input schema for Wikipedia Tool."""
    query: str = Field(
        ...,
        description="The Wikipedia page title or search query to find article summary"
    )
    language: str = Field(
        default="en",
        description="Wikipedia language code (default: 'en' for English)"
    )


class WikipediaTool(BaseTool):
    name: str = "Wikipedia Search"
    description: str = (
        "Search Wikipedia for article summaries. "
        "Retrieves page title, summary, and URL for any Wikipedia topic. "
        "Supports multiple languages."
    )
    args_schema: Type[BaseModel] = WikipediaInput

    def _run(self, query: str, language: str = "en") -> str:
        """
        Execute Wikipedia search and retrieve article summary.
        
        Args:
            query: The Wikipedia page title or search query
            language: Wikipedia language code (default: 'en')
        
        Returns:
            Formatted string with article title, summary, and URL
        """
        try:
            # Initialize Wikipedia API with user agent
            wiki = wikipediaapi.Wikipedia(
                user_agent="CrewAI-Wikipedia-Tool/1.0",
                language=language
            )

            # Get the page
            page = wiki.page(query)

            # Check if page exists
            if not page.exists():
                return f"No Wikipedia page found for '{query}' in {language} language."

            # Format result
            result = f"Wikipedia Article: {page.title}\n"
            result += f"URL: {page.fullurl}\n\n"
            result += f"Summary:\n{page.summary[:2000]}\n"

            return result.strip()

        except Exception as e:
            return f"Error retrieving Wikipedia article: {str(e)}"

