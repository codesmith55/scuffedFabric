from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from ollama_functions import OllamaFunctions
#from langchain_experimental.llms.ollama_functions import OllamaFunctions

### OpenAI

# Grader prompt
code_gen_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a coding assistant with expertise in LCEL, LangChain expression language. \n 
    Here is a full set of LCEL documentation:  \n ------- \n  {context} \n ------- \n Answer the user 
    question based on the above provided documentation. Ensure any code you provide can be executed \n 
    with all required imports and variables defined. Structure your answer with a description of the code solution. \n
    Then list the imports. And finally list the functioning code block. Here is the user question:""",
        ),
        ("placeholder", "{messages}"),
    ]
)


# Data model
class code(BaseModel):
    """Code output"""

    prefix: str = Field(description="Description of the problem and approach")
    imports: str = Field(description="Code block import statements")
    code: str = Field(description="Code block not including import statements")
    description = "Schema for code solutions to questions about LCEL."


expt_llm = "gpt-4o"
llm = ChatOpenAI(temperature=0, model=expt_llm)
code_gen_chain = code_gen_prompt | llm.with_structured_output(code)
question = "How do I build a RAG chain in LCEL?"

concatenated_content = """Skip to main contentLangChain v0.2 is out! You are currently viewing the old v0.1 docs. View the latest docs here.ComponentsIntegrationsGuidesAPI ReferenceMorePeopleVersioningContributingTemplatesCookbooksTutorialsYouTubev0.1v0.2v0.1🦜️🔗LangSmithLangSmith DocsLangServe GitHubTemplates GitHubTemplates HubLangChain HubJS/TS Docs💬SearchGet startedIntroductionQuickstartInstallationUse casesQ&A with RAGExtracting structured outputChatbotsTool use and agentsQuery analysisQ&A over SQL + CSVMoreExpression LanguageGet startedRunnable interfacePrimitivesAdvantages of LCELStreamingAdd message history (memory)MoreEcosystem🦜🛠️ LangSmith🦜🕸️LangGraph🦜️🏓 LangServeSecurityExpression LanguageLangChain Expression Language (LCEL)LangChain Expression Language, or LCEL, is a declarative way to easily compose chains together.
LCEL was designed from day 1 to support putting prototypes in production, with no code changes, from the simplest “prompt + LLM” chain to the most complex chains (we’ve seen folks successfully run LCEL chains with 100s of steps in production). To highlight a few of the reasons you might want to use LCEL:First-class streaming support
When you build your chains with LCEL you get the best possible time-to-first-token (time elapsed until the first chunk of output comes out). For some chains this means eg. we stream tokens straight from an LLM to a streaming output parser, and you get back parsed, incremental chunks of output at the same rate as the LLM provider outputs the raw tokens.Async support
Any chain built with LCEL can be called both with the synchronous API (eg. in your Jupyter notebook while prototyping) as well as with the asynchronous API (eg. in a LangServe server). This enables using the same code for prototypes and in production, with great performance, and the ability to handle many concurrent requests in the same server.Optimized parallel execution
Whenever your LCEL chains have steps that can be executed in parallel (eg if you fetch documents from multiple retrievers) we automatically do it, both in the sync and the async interfaces, for the smallest possible latency.Retries and fallbacks
Configure retries and fallbacks for any part of your LCEL chain. This is a great way to make your chains more reliable at scale. We’re currently working on adding streaming support for retries/fallbacks, so you can get the added reliability without any latency cost.Access intermediate results
For more complex chains it’s often very useful to access the results of intermediate steps even before the final output is produced. This can be used to let end-users know something is happening, or even just to debug your chain. You can stream intermediate results, and it’s available on every LangServe server.Input and output schemas
Input and output schemas give every LCEL chain Pydantic and JSONSchema schemas inferred from the structure of your chain. This can be used for validation of inputs and outputs, and is an integral part of LangServe.Seamless LangSmith tracing
As your chains get more and more complex, it becomes increasingly important to understand what exactly is happening at every step.
With LCEL, all steps are automatically logged to LangSmith for maximum observability and debuggability.Seamless LangServe deployment
Any chain created with LCEL can be easily deployed using LangServe.Help us out by providing feedback on this documentation page:PreviousWeb scrapingNextGet startedCommunityDiscordTwitterGitHubPythonJS/TSMoreHomepageBlogYouTubeCopyright © 2024 LangChain, Inc.
"""
solution = code_gen_chain.invoke({"context":concatenated_content,"messages":[("user",question)]})
print(solution)