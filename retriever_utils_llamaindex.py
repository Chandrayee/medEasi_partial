from llama_index.core.query_engine import CitationQueryEngine
from llama_index.core.indices.vector_store.base import VectorStoreIndex
from llama_index.core import Settings
from llama_index.core.indices.query.schema import QueryBundle, QueryType
from llama_index.core.prompts.base import Prompt
import re
import logging
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
from llama_index.core import PromptHelper
import requests
from typing import List
import pprint as pp
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.llms.azure_openai import AzureOpenAI
import parse_context as parser
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core import ChatPromptTemplate
import tiktoken
import asyncio
'''
When in test phase we will use GPT3.5 unless we are testing
quality of model output directly.
'''

debug_mode = False

'''
set up credentials for using azure openai with llamaindex
'''
api_key = "" #enter azure openai api key here
azure_endpoint = "" #enter azure openai endpoint here
api_version = "" #enter openai api version here

'''
define and initiate models
'''

llm = AzureOpenAI(
    model="",
    deployment_name="", 
    api_key=api_key,
    temperature=0.0,
    api_version=api_version,
    azure_endpoint=azure_endpoint)

if debug_mode:
   print("Using gpt3.5 as LLM in debug mode")
   llm = AzureOpenAI(
    model="",
    deployment_name="",
    api_key=api_key,
    temperature=0.0,
    api_version=api_version,
    azure_endpoint=azure_endpoint
 )



embedding_llm = AzureOpenAIEmbedding(
    model="",
    deployment_name="",
    api_key=api_key,
    azure_endpoint=azure_endpoint,
    api_version=api_version)


Settings.llm = llm
Settings.embed_model = embedding_llm
Settings.chunk_size = 2048

def isEnglish(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True

class SemanticScholarReader(BaseReader):
    def __init__(self, api_key=""):
        """
        Instantiate the SemanticScholar object
        """
        from semanticscholar import SemanticScholar

        self.s2 = SemanticScholar(
            api_key=api_key)


    async def aload_data(
        self,
        query,
        paper_title,
        limit=10,
        returned_fields=[
            "title",
            "abstract",
            "venue",
            "year",
            "paperId",
            "citationCount",
            "openAccessPdf",
            "authors",
        ],
    ) -> List[Document]:
        try:
            results = self.s2.search_paper(
                query, limit=limit, fields=returned_fields)
        except (requests.HTTPError, requests.ConnectionError, requests.Timeout) as e:
            logging.error(
                "Failed to fetch data from Semantic Scholar with exception: %s", e
            )
            raise
        except Exception as e:
            logging.error("An unexpected error occurred: %s", e)
            raise

        documents = []

        for item in results[:limit]:
            openaccesspdf = getattr(item, "openAccessPdf", None)
            abstract = getattr(item, "abstract", None)
            title = getattr(item, "title", None)
            if isEnglish(title) and title.lower() != paper_title.lower():
                text = None
                # concat title and abstract
                if abstract and title:
                    text = "Title: " + title + " - " + abstract
                elif not abstract:
                    # print(f"{title} doesn't have abstract, {openaccesspdf}")
                    continue
                    text = title

                metadata = {
                    "title": title,
                    "venue": getattr(item, "venue", None),
                    "year": getattr(item, "year", None),
                    "paperId": getattr(item, "paperId", None),
                    "citationCount": getattr(item, "citationCount", None),
                    "openAccessPdf": openaccesspdf.get("url") if openaccesspdf else None,
                    "authors": [author["name"] for author in getattr(item, "authors", [])],
                }
                documents.append(Document(text=text, extra_info=metadata))

        return documents

    def load_data(
        self,
        query,
        paper_title,
        limit=10,
        returned_fields=[
            "title",
            "abstract",
            "venue",
            "year",
            "paperId",
            "citationCount",
            "openAccessPdf",
            "authors",
        ],
    ) -> List[Document]:
        try:
            results = self.s2.search_paper(
                query, limit=limit, fields=returned_fields)
        except (requests.HTTPError, requests.ConnectionError, requests.Timeout) as e:
            logging.error(
                "Failed to fetch data from Semantic Scholar with exception: %s", e
            )
            raise
        except Exception as e:
            logging.error("An unexpected error occurred: %s", e)
            raise

        documents = []

        for item in results[:limit]:
            openaccesspdf = getattr(item, "openAccessPdf", None)
            abstract = getattr(item, "abstract", None)
            title = getattr(item, "title", None)
            if isEnglish(title) and title.lower() != paper_title.lower():
                text = None
                # concat title and abstract
                if abstract and title:
                    text = title + " " + abstract
                elif not abstract:
                    # print(f"{title} doesn't have abstract, {openaccesspdf}")
                    continue
                    text = title

                metadata = {
                    "title": title,
                    "venue": getattr(item, "venue", None),
                    "year": getattr(item, "year", None),
                    "paperId": getattr(item, "paperId", None),
                    "citationCount": getattr(item, "citationCount", None),
                    "openAccessPdf": openaccesspdf.get("url") if openaccesspdf else None,
                    "authors": [author["name"] for author in getattr(item, "authors", [])],
                }
                documents.append(Document(text=text, extra_info=metadata))

        return documents

'''
initialize semantic scholar object
'''
s2reader = SemanticScholarReader()

'''
prompts
'''

CITATION_QA_TEMPLATE = Prompt(
    "Please provide an answer based solely on the provided sources. "
    "When referencing information from a source, "
    "cite the appropriate source(s) using their corresponding numbers. "
    # "cite the appropriate source(s) using their corresponding numbers, in the format like "[1::1]", "[2::2]" to denote the beginning and the end of the reference"
    "Every answer should include at least one source citation. "
    "Only cite a source when you are explicitly referencing it. "
    "If none of the sources are helpful, you should indicate that by ending with [ERROR]. "
    "For example:\n"
    "Source 1:\n"
    "The sky is red in the evening and blue in the morning.\n"
    "Source 2:\n"
    "Water is wet when the sky is red.\n"
    "Query: When is water wet?\n"
    "Answer: Water will be wet when the sky is red [2], "
    "which occurs in the evening [1].\n"
    "Now it's your turn. Below are several numbered sources of information:"
    "\n------\n"
    "{context_str}"
    "\n------\n"
    "Query: {query_str}\n"
    "Answer: "
)

DEFINITION_TEMPLATE_1 = Prompt(
    "Give a simple definition of the concept {query_str} based on the given abstracts of scholarly medical articles."
    "The abstracts are:"
    "\n------\n"
    "{context_str}"
    "\n------\n"
    "Your output should be in the following format. Example citation format: [1, 3] where 1 stands for Abstract 1 and 3 stands for Abstract 3."
    "Definition : [definition here] [cite relevant abstracts here]."
    "If you cannot find any definition, your output should be ERROR: No definition found. "
)

DEFINITION_TEMPLATE_2 = Prompt(
    "Quote verbatim a definition of the term {query_str} from the given abstracts of scholarly medical articles."
    "If different abstracts have different defitions extract the one that is written in the simplest language."
    "Cite the titles of the abstracts from which the definitions were extracted."
    "The abstracts are:"
    "\n------\n"
    "{context_str}"
    "\n------\n"
    "Your output should be in the following format. Example citation format: [1, 3] where 1 stands for Abstract 1 and 3 stands for Abstract 3."
    "Definition : [definition here] [cite relevant abstracts here]."
    "If you cannot find any definition, your output should be ERROR: No definition found. "
)


def get_user_prompt(intent, context, title):
    prompt_part_2 = f"\nThe inputs are:\n<Context>: {context}\n<Intent>: {intent}\n<Title>: {title}\n"
    prompt_part_3 = "The abstracts are:\n------\n{context_str}\n------\n"
    explanation_prompt = "Explain in a few simpler sentences the concept {query_str} in the context of the given text marked as <Context>." + \
        prompt_part_2 + prompt_part_3

    return explanation_prompt

def get_system_prompt_v2():
    system_prompt = """You are a scientific writer. You job is to accurately explain unfamiliar concepts to the readers of scholarly medical articles. 
    You will either be given more information about the reader or the context in which those unfamiliar concepts appear. Additionally, you will be 
    given 3 abstracts that contain explanations of the unfamiliar concepts. In order to provide accurate explanation you will first extract the relevant
    contents from the abstracts and then use them to synthesize a single contextually relevant explanation of the concept. The explanation should be simple.
    Any new jargon introduced in the resulting explanation should also be simplified accurately."""
    return system_prompt


def get_custom_prompt(intent, context, title, type='intent', with_excerpt=True):
    system_prompt = get_system_prompt_v2()
    user_prompt = get_user_prompt(intent, context, title)
    if type == 'intent':
      if with_excerpt:
        user_prompt = create_explanation_prompt_v2_intent(intent, title)
      else:
        user_prompt = create_explanation_prompt_intent(intent, title)
    elif type == 'context':
      if with_excerpt:
        user_prompt = create_explanation_prompt_v2_context(
            intent, context, title)
      else:
        user_prompt = create_explanation_prompt_context(context, title)
    chat_text_qa_msgs = [
        ChatMessage(
            role=MessageRole.SYSTEM,
            content=(
                system_prompt
            ),
        ),

        ChatMessage(
            role=MessageRole.USER,
            content=(
                user_prompt
            ),
        ),
    ]
    text_qa_template = ChatPromptTemplate(chat_text_qa_msgs)
    return text_qa_template


def create_explanation_prompt_context(context, title):
    prompt_1 = f"""You are a medical writer. You are simplifying parts of a medical article titled {title}."""
    prompt_2 = """Define and explain in a few simpler sentences the concept {query_str} in the context of the given text marked as <Text>. 
        Your explanation should be solely based on the three given abstracts of scholarly medical articles. 
        If multiple different explanations are found, provide the explanation that is the most relevant to the context <Text>.
        Provide in text citation. 
    When citing multiple abstracts use the format [1, 2] where 1 is for Abstract 1 and 2 stands for Abstract 2. 
    Any content extracted from the <Text> should start with the phrase 'According to this text."""
    prompt_3 = f'\n<Text>: {context}'
    prompt_4 = """The abstracts are:
    "\n------\n"
    "{context_str}"
    "\n------\n"

    Your output should be in the following format. .
    Explanation : [explanation here] [cite relevant abstracts]. [explanation here] [cite relevant abstracts]. and so on.

    If you cannot find any explanation, your output should be ERROR: No explanation found. """
    explanation_prompt = prompt_1 + prompt_2 + prompt_3 + prompt_4
    return explanation_prompt


def create_explanation_prompt_intent(intent, title):
    prompt_1 = f"""You are a medical writer. You are simplifying parts of a medical article titled {title}. The reader is reading this article with the intent of
                understanding {intent}."""
    prompt_2 = """Explain in a few simpler sentences how the concept {query_str} relates to the <Reader Intent>. 
        Your explanation should be solely based on the three given abstracts of scholarly medical articles. 
        If multiple different explanations are found, provide the explanation that is the most relevant to the <Reader Intent>.
        Provide in text citation. 
    When citing multiple abstracts use the format [1, 2] where 1 is for Abstract 1 and 2 stands for Abstract 2. 
    Any content extracted from the <Text> should start with the phrase 'According to this text."""
    prompt_3 = f'\n<Reader Intent>: {intent}'
    prompt_4 = """The abstracts are:
    "\n------\n"
    "{context_str}"
    "\n------\n"

    Your output should be in the following format. .
    Explanation : [explanation here] [cite relevant abstracts]. [explanation here] [cite relevant abstracts]. and so on.

    If you cannot find any explanation, your output should be ERROR: No explanation found. """
    explanation_prompt = prompt_1 + prompt_2 + prompt_3 + prompt_4
    return explanation_prompt


def create_explanation_prompt_v2_context(intent, context, title):
    prompt_1 = f"""You are a medical writer. You are simplifying parts of a medical article titled {title}. The reader of the article is interested
                in {intent}."""
    prompt_2 = """Define and explain in a few simpler sentences the concept {query_str} in the context of the given text marked as <Text>. 
        This should be done in 2 steps.
        First quote verbatim contents from the three given abstracts that would best explain the concept in the given context.
        Your excerpt should not be the same as the provided context <Text>.
        Then simplify and integrate these contents into a single coherent explanation. Provide in text citation.
        Your explanations should be solely based on the three given abstracts of scholarly medical articles. 
        If multiple different explanations are found, provide the explanation that is the most relevant to the context <Text>. 
    When citing multiple abstracts use the format [1, 2] where 1 is for Abstract 1 and 2 stands for Abstract 2. 
    Any content extracted from the <Text> should start with the phrase 'According to this text."""
    prompt_3 = f'\n<Text>: {context}'
    prompt_4 = """The abstracts are:
    "\n------\n"
    "{context_str}"
    "\n------\n"
 
    First output the excerpts from the abstracts as enumerated list along with the abstract title.
    If any abstract does not contain relevant content then output "no relevant explanation found" next to the abstract number.
    Then output the explanation.

    Your output should be in the following format.

    Excerpts:
    1 (this is the abstract number). [abstract title here in parenthesis] [quote here if relevant explanation found]
    2. [abstract title here in parenthesis] [quote here if relevant explanation found]
    ...

    Explanation : [explanation here] [cite relevant abstracts]. [explanation here] [cite relevant abstracts]. and so on.

    If you cannot find any explanation in any of the three abstracts, your output should be ERROR: No explanation found. """
    explanation_prompt = prompt_1 + prompt_2 + prompt_3 + prompt_4
    return explanation_prompt

def create_explanation_prompt_v2_intent(intent, title):
    prompt_1 = f"""You are a medical writer. You are simplifying parts of a medical article titled {title}. 
    The reader is reading this article with the intent of understanding {intent}."""
    prompt_2 = """Explain in a few simpler sentences how the concept {query_str} relates to the 
        topic of user interest marked as <Reader Intent> above. 
        This should be done in 2 steps.
        First quote verbatim contents from the three given abstracts that would best explain the concept in light of the reader's intent.
        Then simplify and integrate these contents into a single coherent explanation. Provide in text citation.
        Your explanations should be solely based on the three given abstracts of scholarly medical articles. 
        If multiple different explanations are found, provide the explanation that is the most relevant to the context <Text>. 
    When citing multiple abstracts use the format [1, 2] where 1 is for Abstract 1 and 2 stands for Abstract 2. 
    Any content extracted from the <Text> should start with the phrase 'According to this text."""
    prompt_3 = f'\n<Reader Intent>: {intent}'
    prompt_4 = """The abstracts are:
    "\n------\n"
    "{context_str}"
    "\n------\n"
 
    First output the excerpts from the abstracts as enumerated list along with the abstract title.
    If any abstract does not contain relevant content then output "no relevant explanation found" next to the abstract number.
    Then output the explanation.

    Your output should be in the following format.

    Excerpts:
    1 (this is the abstract number). [abstract title here in parenthesis] [quote here if relevant explanation found]
    2. [abstract title here in parenthesis] [quote here if relevant explanation found]
    ...

    Explanation : [explanation here] [cite relevant abstracts]. [explanation here] [cite relevant abstracts]. and so on.

    If you cannot find any explanation in any of the three abstracts, your output should be ERROR: No explanation found. """
    explanation_prompt = prompt_1 + prompt_2 + prompt_3 + prompt_4
    return explanation_prompt

def extract_citations(response):
   pattern = r'Source (\d+):\n(.+)'
   citation_list = []
   author_title_list = []
   citations = {}
   for node in response.source_nodes:
      match = re.search(pattern, node.node.text)
      citations[match.group(1)] = (match.group(2), node.node.metadata)

   for k, (text, metadata) in citations.items():
      citation_pattern = metadata["paperId"]
      citation_title = metadata['title']
      citation_authors = metadata['authors']
      if citation_pattern not in citation_list:
          citation_list.append(citation_pattern)
          author_title_list.append(
          {'title': citation_title, 'authors': citation_authors})
   return citations, citation_list, author_title_list

def test_semantic_scholar(explanations, use_context=False, use_intent=False, query_generator=''):
    term_doc = {}
    intent = explanations['intent'][0]
    context = explanations['context']
    for term, _ in explanations['Concepts'].items():
        # increase limit to get more documents
        query = term
        if use_intent:
            if query_generator == 'llm':
                query = parser.use_gpt_intent(intent, term)
                queries_ordered = parser.query_similarity(query, intent)
                query = [k for k, v in queries_ordered][-1]
            else:
                query += " " + intent
        if use_context:
            if query_generator == 'llm':
                query = parser.use_gpt_context(context, term)
                queries_ordered = parser.query_similarity(query, context)
                query = [k for k, v in queries_ordered][-1]
            else:
                keywords = parser.key_phrase_extraction(
                    context, method='keybert', entity_centric=False)
                query += " " + keywords
        print('final query is: {}'.format(query))
        documents = s2reader.load_data(query=query, limit=5)
        term_doc[term] = documents
    return term_doc

def retrieve_docs(query="", limit=10):
    print(f"query: \n{query}")
    documents = s2reader.load_data(query=query, limit=limit)
    print(len(documents))
    print(type(documents))
    for idx, document in enumerate(documents):
        print(str(idx) + ': ' + document.metadata['title'])
    print('\n')
    return documents

def check_retrieval_test(documents, term, query, intent, context, title, match_type="term", query_type="context", with_excerpt=True):
    index = VectorStoreIndex.from_documents(documents)
    text_qa_template = get_custom_prompt(
        intent, context, title, type=query_type, with_excerpt=with_excerpt)
    citation_qa_template = text_qa_template
    query_engine = CitationQueryEngine.from_args(
        index,
        similarity_top_k=3,
        citation_chunk_size=1024,
        citation_qa_template=citation_qa_template
    )
    if match_type=="term":
      response = query_engine.query(QueryBundle(f'{term}'))
    else:
      response = query_engine.query(QueryBundle(f'{query}'))

    citations, citation_list, author_title_list = extract_citations(response)
    return response.response, citations


def get_single_query(term, context, intent, relevance="context"):
  if relevance == "context":
    query = parser.use_gpt_context_keyword(context, term)
  else:
    query = parser.use_gpt_intent_keyword(intent, term)
  return query

def top_k_w_excerpts_term(term, query, intent, context, title, query_type="context"):
   documents = s2reader.load_data(query=query, limit=10)
   return check_retrieval_test(documents, term, query, intent, context, title, query_type=query_type, match_type="term", with_excerpt=True)

def top_k_w_excerpts_query(term, query, intent, context, title, query_type="context"):
   documents = s2reader.load_data(query=query, limit=10) 
   return check_retrieval_test(documents, term, query, intent, context, title, query_type=query_type, match_type="query", with_excerpt=True)

def top_k_one_step_term(term, query, intent, context, title, query_type="context"):
   documents = s2reader.load_data(query=query, limit=10)
   return check_retrieval_test(documents, term, query, intent, context, title, query_type=query_type, match_type="term", with_excerpt=False)

async def atop_k_one_step_term(term, query, intent, context, title, query_type="context"):
   documents = s2reader.load_data(query=query, limit=10)
   return check_retrieval_test(documents, term, query, intent, context, title, query_type=query_type, match_type="term", with_excerpt=False)

def multiquery_term(term, queries, intent, context, title, query_type="context"):
  documents = get_multiquery_docs(queries)
  query = queries[0]
  return check_retrieval_test(documents, term, query, intent, context, title, query_type=query_type, match_type="term", with_excerpt=False)

def multiquery_top_k_term(term, queries, intent, context, title, query_type="context"):
  documents = get_multiquery_docs(queries)
  query = queries[0]
  return check_retrieval_test(documents, term, query, intent, context, title, query_type=query_type, match_type="term", with_excerpt=True)

def construct_multiqueries(term, context, intent, relevance="context"):
  search_engine="semantic scholar"
  if relevance == "context":
    query_text = parser.use_gpt_context(context, term, search_engine)
    query_arr = parser.get_numbered_items(query_text)
    query_arr += [parser.use_gpt_context_keyword(context, term)]
    print(query_arr)
    queries_ordered = parser.query_similarity(query_arr, context)
  else:
    query_text = parser.use_gpt_intent(intent, term, search_engine)
    query_arr = parser.get_numbered_items(query_text)
    query_arr += [parser.use_gpt_intent_keyword(intent, term)]
    print(query_arr)
    queries_ordered = parser.query_similarity(query_arr, intent)
  queries = [x[0] for x in queries_ordered]
  return queries

def get_multiquery_docs(queries):
    document_lst = []
    for query in queries:
        #query = query[0] #this is the most relevant query
        document_lst += retrieve_docs(query=query, limit=10)
    return document_lst

def citation_2_title(citation):
    c2t = {}
    for k, v in citation.items():
      c2t[k] = v[1]['title']
    return c2t

def citation_2_title_v2(citation_list):
    c2t = {}
    for idx, citation in enumerate(citation_list):
      c2t[str(idx+1)] = citation['title']
    return c2t
         
def update_citation_num(text, citation_title_map, title_list):
   title_list=list(title_list.keys())
   pattern = '\[(.*?)\]'
   matches = re.findall(pattern, text)
   for match in matches:
     if ',' not in match and match != 'Text':
        text = text.replace(f"[{match}]", f"[{title_list.index(citation_title_map[match])+1}]")
     elif ',' in match:
        match = [int(x.strip()) for x in match.split(',')]
        match = list(set(match))
        match = [str(x) for x in sorted(match)]
        updated_match = ','.join([str(title_list.index(citation_title_map[x])+1) for x in match])
        text = text.replace(f"[{match}]", f"[{updated_match}]")
   return text

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


async def async_semantic_call(reader, queries, paper_title):
    results = []
    tasks = [reader.aload_data(query, paper_title) for query in queries]

    for completed_task in asyncio.as_completed(tasks):
        result = await completed_task
        results.append(result)

    return results

if __name__ == "__main__":
    import time
    s2reader = SemanticScholarReader()
    queries = ['LLM abstract thinking', 'convolutional neural network cancer diagnosis', 'blood clot biomechanics']
    starttime = time.time()
    res = asyncio.run(async_semantic_call(s2reader, queries))
    print(len(res))
    print(time.time()-starttime)
    starttime = time.time()
    for query in queries:
       res = s2reader.load_data(query)
    print(time.time()-starttime)