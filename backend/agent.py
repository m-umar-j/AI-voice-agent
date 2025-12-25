from crewai import Agent, Task, Crew, Process
from crewai_tools import CodeInterpreterTool
import os
from tools import WikipediaTool, BraveSearchTool
# Initialize the tool WITH libraries_used parameter
code_interpreter = CodeInterpreterTool(
    libraries_used=["numpy", "matplotlib", "scipy"],
    timeout=600,        
    unsafe_mode=True
)

wiki_tool = WikipediaTool()
brave_search = BraveSearchTool()
from crewai import LLM

llm = LLM(
    model="gemini/gemini-2.0-flash",
    api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.7
)

data_analyst = Agent(
    role="Data Analyst",
    goal="Analyze data using Python code, Wikipedia research, and web search",
    llm=llm,
    backstory="""You are an expert data analyst who specializes in using Python
    to analyze and visualize data. You research topics using Wikipedia and web search
    to provide context, then write efficient code to process datasets and extract
    meaningful insights. You combine research findings with data analysis.""",
    tools=[code_interpreter, wiki_tool, brave_search],
    verbose=True,
    allow_code_execution=True,  # Add this for better execution support
    
)


# Create a task for the agent
analysis_task = Task(
    description="""
    Perform comprehensive data analysis on correlation between variables:
    
    1. RESEARCH: Use Wikipedia to research "Pearson correlation coefficient" 
       and summarize key concepts (what it measures, assumptions, interpretation).
    
    2. RESEARCH: Use Brave Search to find real-world examples of correlation 
       analysis in data science (query: "correlation analysis examples data science").
    
    3. ANALYSIS: Write Python code to:
       - Generate a random dataset of 100 points with x and y coordinates 
         (include both positive and negative correlation examples)
       - Calculate the Pearson correlation coefficient between x and y using scipy
       - Create a scatter plot with correlation coefficient displayed on plot
       - Save the plot as 'data/scatter_correlation.png'
    
    4. INTERPRETATION: Combine your research findings with analysis results.
       Interpret the correlation coefficient value based on your Wikipedia research.
       Discuss what the result means in context of real-world examples from search.
    
    Use libraries: numpy, matplotlib, scipy.stats, pandas
    Print the correlation coefficient, research summary, and save the plot.
    Structure output clearly with sections for research, analysis, and interpretation.
    """,
    expected_output="""
    A complete analysis report containing:
    - Wikipedia summary of Pearson correlation (3-5 key points)
    - 2-3 real-world correlation examples from web search
    - Generated dataset correlation coefficient (with interpretation)
    - Confirmation that scatter plot saved as 'data/scatter_correlation.png'
    - Final insights combining research + analysis
    """,
    agent=data_analyst,
)


# Run the task
crew = Crew(
    agents=[data_analyst],
    tasks=[analysis_task],
    verbose=True,
    process=Process.sequential,
)
result = crew.kickoff()
