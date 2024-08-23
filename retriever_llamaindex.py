from retriever_utils_llamaindex import (
    SemanticScholarReader,
    top_k_one_step_term,
    update_citation_num,
    async_semantic_call,
    async_semantic_call_v2,
    get_system_prompt_v2,
    citation_2_title,
    citation_2_title_v2)
from async_retrieval import main
import asyncio
import numpy as np
import parse_context as parser
import time

def update_citation(retrieval_json, title_list):
    for concept, data in retrieval_json.items():
        citation_title_map = citation_2_title(data['context_citations'])
        retrieval_json[concept]['context-specific elaboration'] = update_citation_num(data['context-specific elaboration'], citation_title_map, title_list)
    fulldata = [v for k, v in title_list.items()]
    retrieval_json['references'] = [f"[{str(idx+1)}] {v}" for idx, v in enumerate(fulldata)]
    return retrieval_json

def create_retrieval_json(res, concepts, queries, context):
    title_list = {}
    retrieval_json = {}
    retrieval_json['terms'] = concepts
    retrieval_json['queries'] = queries
    for idx, _ in enumerate(concepts):
        for k, v in res[idx][1].items():
            if v[1]['title'] not in title_list:
                title_list[v[1]['title']] = ", ".join(v[1]['authors']) + '; ' + v[1]['title'] + '; ' + v[1]['venue'] + ' (' + str(v[1]['year']) + ')'
    for concept, data in zip(concepts, res):
        citation_title_map = citation_2_title(data[1])
        retrieval_json[concept] = {'context':context}
        retrieval_json[concept]['context-specific elaboration'] = update_citation_num(data[0], citation_title_map, title_list)
    fulldata = [v for k, v in title_list.items()]
    retrieval_json['references'] = "\n".join([f"[{str(idx+1)}] {v}" for idx, v in enumerate(fulldata)])
    return retrieval_json

def postprocess(retrieval_json):
    del retrieval_json['queries']
    for k, v in retrieval_json.items():
        if k != 'references' and k != 'terms':
            del retrieval_json[k]['context']
    return retrieval_json 

def compute_relevance(concepts, intent):
    relevance_list = parser.get_batch_relevance(concepts+[intent])
    fil = [r > 0.79 for r in relevance_list]
    filtered_concepts = [i for i, v in zip(concepts, fil) if v]
    return filtered_concepts

def compute_queries(concepts, context, intent):
    query_dict = {concept:{} for concept in concepts}
    queries = parser.get_batch_context_queries({concept:context for concept in concepts})
    for concept in queries:
        query_dict[concept]["context"]=queries[concept]
    filtered_concepts = compute_relevance(concepts, intent)
    if len(filtered_concepts) > 0:
        intent_queries = parser.get_batch_intent_queries({concept:intent for concept in filtered_concepts})
        for concept in intent_queries:
            query_dict[concept]["intent"]=intent_queries[concept]
    return query_dict 

def compute_queries_no_intent(concepts, context, paper_title):
    query_dict = parser.get_batch_context_queries(
        query_input = {concept:context for concept in concepts}, additional=paper_title)
    return query_dict 

def compute_queries_together(concepts, context, intent):
    query_input = {concept:context for concept in concepts}
    filtered_concepts = compute_relevance(concepts, intent)
    for concept in filtered_concepts:
        query_input[concept] = intent
    query_dict = {}
    queries_output = parser.get_batch_queries(query_input)
    for concept, query in queries_output.items():
        if concept in query_dict:
            query_dict[concept]["intent"]=query
        else:
            query_dict[concept]["context"]=query
    return query_dict  

def retrieval_no_intent_async(
        db_client=None,
        user=None,
        pdf_name=None,
        familiarity_json=None,
        context=None,
        raw_context=None,
        intent=None,
        paper_title=None
        ):
    concepts = list(familiarity_json.keys())
    query_dict = compute_queries_no_intent(concepts, context, intent)
    s2reader = SemanticScholarReader()
    queries = [v for _, v in query_dict.items()]
    concepts = [k for k, _ in query_dict.items()]
    intents = [intent for _ in range(len(concepts))]
    contexts = [context for _ in range(len(concepts))]
    titles = [paper_title for _ in range(len(concepts))]
    docs = asyncio.run(async_semantic_call(s2reader, queries, paper_title))
    res = asyncio.run(main(queries, concepts, docs, intents, contexts, titles))
    retrieval_json = create_retrieval_json(res, concepts, queries, context)
    return postprocess(retrieval_json)

def retrieval_no_intent(
        db_client=None,
        user=None,
        pdf_name=None,
        familiarity_json=None,
        context=None,
        raw_context=None,
        intent=None,
        paper_title=None
        ):
    
    concepts = list(familiarity_json.keys())
    query_dict = compute_queries_no_intent(concepts, context, paper_title)

    retrieval_json = {}
    title_list = {}
    
    for concept, query in query_dict.items():
        retrieval_json[concept] = {'context':context}
        res_context, citations_context = top_k_one_step_term(concept, query, intent, context, paper_title, query_type="context")
        retrieval_json[concept]['context-specific elaboration']=res_context
        for k, v in citations_context.items():
            if v[1]['title'] not in title_list:
                title_list[v[1]['title']] = ", ".join(v[1]['authors']) + '; ' + v[1]['title'] + '; ' + v[1]['venue'] + ' (' + str(v[1]['year']) + ')'
        retrieval_json[concept]['context_citations']=citations_context        
        retrieval_json[concept]['queries'] = query

    retrieval_json = update_citation(retrieval_json, title_list)
    retrieval_json = postprocess(retrieval_json)
    return retrieval_json    


if __name__ == "__main__":
    familiarity_json={"abstract thinking":"", "complex reasoning and planning problems":""}
    context="Abstraction-of-Thought reasoning format draws inspiration from the human application of abstract thinking to solve complex reasoning and planning problems (Saitta et al., 2013; Yang, 2012)."
    intent="The reader is interested in learning about abstraction-of-thought reasoning and its application in hypothesis generation in scientific discovery, specifically focusing on the development of a training dataset for this type of reasoning in LLMs."
    paper_title="Abstraction-of-Thought Makes Language Models Better Reasoners"
    concepts = list(familiarity_json.keys())
    query_dict = compute_queries_no_intent(concepts, context, paper_title)
    print(query_dict)
    starttime = time.time()
    res = retrieval_no_intent_async(
        db_client=None,
        user=None,
        pdf_name=None,
        familiarity_json=familiarity_json,
        context=context,
        raw_context=None,
        intent=intent,
        paper_title=paper_title
        )
    print(time.time()-starttime)
    print(res)
    
    

        

