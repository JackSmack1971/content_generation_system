import asyncio
import logging
import os
from typing import override
from dataclasses import dataclass
import inspect
import hashlib

from aiofiles import open as async_open
from pydantic import BaseModel, Field, validator

from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import FAISS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ConfigModel(BaseModel):
    """Configuration model with validation."""
    openai_api_key: str = Field(..., min_length=1)
    faiss_index_path: str = Field(..., min_length=1)
    output_file_path: str = Field(..., min_length=1)

    @validator('openai_api_key', 'faiss_index_path', 'output_file_path')
    def check_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Field cannot be empty")
        return v

@dataclass
class Config:
    """Configuration class for the content generation system."""
    model: ConfigModel

    @classmethod
    def from_env(cls) -> 'Config':
        try:
            return cls(ConfigModel(
                openai_api_key=os.environ['OPENAI_API_KEY'],
                faiss_index_path=os.environ['FAISS_INDEX_PATH'],
                output_file_path=os.environ['OUTPUT_FILE_PATH']
            ))
        except KeyError as e:
            raise EnvironmentError(f"Missing required environment variable: {e}")

class ContentGenerationError(Exception):
    """Base exception for content generation errors."""

class ResearchError(ContentGenerationError):
    """Exception for errors during the research phase."""

class WritingError(ContentGenerationError):
    """Exception for errors during the writing phase."""

class SEOError(ContentGenerationError):
    """Exception for errors during SEO optimization."""

class LinkBuildingError(ContentGenerationError):
    """Exception for errors during link building."""

class ResearchAgent:
    """Agent responsible for keyword research and content retrieval."""

    def __init__(self, retriever: FAISS, llm: OpenAI):
        self.retriever = retriever
        self.llm = llm

    @classmethod
    async def create(cls, config: ConfigModel, llm: OpenAI) -> 'ResearchAgent':
        retriever = await cls.load_faiss_index(config.faiss_index_path)
        return cls(retriever, llm)

    @staticmethod
    async def load_faiss_index(path: str) -> FAISS:
        """Load the FAISS index for similarity search."""
        try:
            return FAISS.load_local(path)
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
            raise ResearchError(f"Failed to initialize research capabilities: {e}")

    async def keyword_research(self, topic: str) -> list[str]:
        """Perform keyword research for a given topic."""
        try:
            prompt = ChatPromptTemplate.from_template(
                "Identify the top 10 relevant keywords for the following topic:\n\n{topic}"
            )
            chain = LLMChain(llm=self.llm, prompt=prompt)
            keywords = await chain.arun({"topic": topic})
            logger.info(f"Keyword Research: Keywords identified: {keywords=}, for topic {topic=}")
            return [kw.strip() for kw in keywords.split(',')]
        except Exception as e:
            logger.error(f"Keyword Research Error: {e}")
            raise ResearchError(f"Failed to generate keywords: {e}")

    async def research(self, keywords: list[str]) -> list[str]:
        """Perform research based on given keywords."""
        try:
            documents = self.retriever.similarity_search(" ".join(keywords))
            logger.info(f"Research completed: Retrieved {len(documents)} documents for keywords: {keywords=}")
            return [doc.page_content for doc in documents]
        except Exception as e:
            logger.error(f"Research Error: {e}")
            raise ResearchError(f"Failed to retrieve research data: {e}")

class ContentWriter:
    """Responsible for writing content based on research and keywords."""

    def __init__(self, llm: OpenAI, memory: ConversationBufferMemory):
        self.llm = llm
        self.memory = memory

    async def write_content(self, research_output: list[str], keywords: list[str]) -> str:
        """Generate content using research output and keywords."""
        try:
            prompt = ChatPromptTemplate.from_template(
                "Write a comprehensive blog post on the topic using the following research and incorporating these keywords:\n\n"
                "Keywords: {keywords}\n\nResearch Data:\n{research_output}"
            )
            chain = LLMChain(llm=self.llm, prompt=prompt, memory=self.memory)
            content = await chain.arun({"keywords": ", ".join(keywords), "research_output": "\n".join(research_output)})
            logger.info(f"Content Writing: Generated content successfully. Length: {len(content)}")
            return content
        except Exception as e:
            logger.error(f"Content Writing Error: {e}")
            raise WritingError(f"Failed to generate content: {e}")

class SEOOptimizer:
    """Responsible for SEO optimization of content."""

    def __init__(self, llm: OpenAI):
        self.llm = llm

    async def optimize(self, content: str) -> str:
        """Optimize content for SEO."""
        try:
            seo_prompt = ChatPromptTemplate.from_template(
                "Analyze the following content and suggest SEO improvements. Please provide detailed reasoning for each suggestion:\n\n{content}"
            )
            chain = LLMChain(llm=self.llm, prompt=seo_prompt)
            optimized_content = await chain.arun({"content": content})
            logger.info(f"SEO Optimization: Suggested improvements successfully. Optimization length: {len(optimized_content)}")
            return optimized_content
        except Exception as e:
            logger.error(f"SEO Optimization Error: {e}")
            raise SEOError(f"Failed to optimize content for SEO: {e}")

class LinkBuilder:
    """Responsible for adding links to the content."""

    def __init__(self, llm: OpenAI):
        self.llm = llm

    async def add_links(self, content: str) -> str:
        """Add relevant internal and external links to the content."""
        try:
            prompt = ChatPromptTemplate.from_template(
                "Enhance the following content by adding relevant internal and external links:\n\n{content}"
            )
            chain = LLMChain(llm=self.llm, prompt=prompt)
            linked_content = await chain.arun({"content": content})
            logger.info(f"Link Building: Added links successfully. New content length: {len(linked_content)}")
            return linked_content
        except Exception as e:
            logger.error(f"Link Building Error: {e}")
            raise LinkBuildingError(f"Failed to add links to content: {e}")

class ContentProcessor:
    """Main processor for the content generation workflow."""

    def __init__(self, research_agent: ResearchAgent, writer: ContentWriter, seo_optimizer: SEOOptimizer, link_builder: LinkBuilder):
        self.research_agent = research_agent
        self.writer = writer
        self.seo_optimizer = seo_optimizer
        self.link_builder = link_builder

    async def process(self, topic: str) -> str:
        """Process the entire content generation workflow."""
        try:
            keywords = await self.research_agent.keyword_research(topic)
            research_output = await self.research_agent.research(keywords)
            content = await self.writer.write_content(research_output, keywords)

            # Parallel SEO and Link Building
            optimized_content, linked_content = await asyncio.gather(
                self.seo_optimizer.optimize(content),
                self.link_builder.add_links(content)
            )

            final_content = self.combine_content(content, optimized_content, linked_content)
            return final_content
        except ContentGenerationError as e:
            logger.error(f"Content Generation Failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in content processing: {e}")
            raise ContentGenerationError(f"An unexpected error occurred during content processing: {e}")

    def combine_content(self, original: str, seo: str, linked: str) -> str:
        """Combine original, SEO-optimized, and linked content."""
        # Implement a more sophisticated merging logic here
        combined = original
        # Apply SEO suggestions
        for suggestion in seo.split('\n'):
            if suggestion.startswith('- '):
                combined = self.apply_seo_suggestion(combined, suggestion[2:])
        # Add links
        combined = self.merge_links(combined, linked)
        return combined

    def apply_seo_suggestion(self, content: str, suggestion: str) -> str:
        # Implement logic to apply each SEO suggestion
        # This is a placeholder implementation
        return content + f"\n[Applied SEO suggestion: {suggestion}]"

    def merge_links(self, content: str, linked_content: str) -> str:
        # Implement logic to merge links from linked_content into the original content
        # This is a placeholder implementation
        return content + f"\n[Merged links from:\n{linked_content}]"

async def save_final_content(content: str, file_path: str) -> None:
    """Save the final content to a file asynchronously."""
    try:
        # Use hashlib for secure file naming if needed
        file_hash = hashlib.sha256(file_path.encode()).hexdigest()[:10]
        secure_file_path = f"{file_path}_{file_hash}"
        
        async with async_open(secure_file_path, "w") as f:
            await f.write(content)
        logger.info(f"Final content saved to {secure_file_path}. Content length: {len(content)}")
    except Exception as e:
        logger.error(f"Error saving final content: {e}")
        raise ContentGenerationError(f"Failed to save final content: {e}")

async def main() -> None:
    """Main function to run the content generation workflow."""
    try:
        config = Config.from_env()
        llm = OpenAI(temperature=0, api_key=config.model.openai_api_key)
        memory = ConversationBufferMemory(memory_key="history", return_messages=True)

        research_agent = await ResearchAgent.create(config.model, llm)
        writer = ContentWriter(llm, memory)
        seo_optimizer = SEOOptimizer(llm)
        link_builder = LinkBuilder(llm)
        processor = ContentProcessor(research_agent, writer, seo_optimizer, link_builder)

        topic = "Artificial Intelligence in Healthcare"
        final_content = await processor.process(topic)
        await save_final_content(final_content, config.model.output_file_path)

        # Example of using inspect.get_annotations()
        logger.info(f"ContentProcessor annotations: {inspect.get_annotations(ContentProcessor)}")

    except ContentGenerationError as e:
        logger.error(f"Workflow failed: {e}")
    except Exception as e:
        logger.error(f"Unexpected error in workflow execution: {e}")

if __name__ == "__main__":
    asyncio.run(main())
