import wikipediaapi
from typing import Type
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from crewai import LLM
import os

llm = LLM(
    model="gemini/gemini-2.0-flash",
    api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.7
)

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



from crewai import Agent, Task, Crew
# from your_file import WikipediaTool  # Save above code as your_file.py

# Create tool
wiki_tool = WikipediaTool()

# Create agent
researcher = Agent(
    role='Research Analyst',
    goal='Research topics using Wikipedia',
    backstory='Expert at finding reliable information on Wikipedia.',
    tools=[wiki_tool],
    llm = llm,
    verbose=True
)

# Create task
research_task = Task(
    description='Research artificial intelligence on Wikipedia and summarize key points.',
    expected_output='Wikipedia summary of artificial intelligence with key concepts.',
    agent=researcher
)

# Create and run crew
crew = Crew(
    agents=[researcher],
    tasks=[research_task],
    verbose=True
)

result = crew.kickoff()
