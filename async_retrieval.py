import retriever_utils as ru
from llama_index.core.indices.vector_store.base import VectorStoreIndex
from llama_index.core.query_engine import CitationQueryEngine
import asyncio
from llama_index.core.query_pipeline import (
    QueryPipeline,
    InputComponent,
    ArgPackComponent,
)
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.indices.query.schema import QueryBundle


async def check_retrieval_test(documents, term, query, intent, context, title, match_type="term", query_type="context"):
    index = VectorStoreIndex.from_documents(documents)
    text_qa_template = ru.get_custom_prompt(
        intent, context, title, type=query_type, with_excerpt=False)
    citation_qa_template = text_qa_template
    query_engine = CitationQueryEngine.from_args(
        index,
        similarity_top_k=3,
        citation_chunk_size=1024,
        citation_qa_template=citation_qa_template
    )
    if match_type=="term":
      response = query_engine.query(QueryBundle(f'{term}'))

    #res_str = response.response + '\n\nReferences\n'
    citations, citation_list, author_title_list = ru.extract_citations(response)
    #citation_str = ""
    #for k, v in citations.items():
    #  citation_str += '['+str(k)+']. ' + ", ".join(v[1]['authors']) + '; ' + v[1]['title'] + '; ' + v[1]['venue'] + ' (' + str(v[1]['year']) + ')'
    #res_str += citation_str
    return response.response, citations

async def main(queries, terms, documents, intents, contexts, titles):
    results = []
    tasks = [check_retrieval_test(docs, term, query, intent, context, title) for docs, term, query, intent, context, title in zip(documents, terms, queries, intents, contexts, titles)]
    for completed_task in asyncio.as_completed(tasks):
        result = await completed_task
        results.append(result)
    return results

if __name__ == "__main__":
    queries = ['LLM abstract thinking', 'convolutional neural network cancer diagnosis', 'blood clot biomechanics']
    terms = ['Abstraction of thought', 'Convolutional Neural Network', 'Blood clot flow']
    contexts = ["Abstraction-of-thought is a novel paradigm for explicitly training LLMs to peform abstract thinking at different levels of granularity.", "The usefulness of convolutional neural network, particularly UNet in medical imaging and resulting cancer diagnosis is profound.", "One of the important biomechanics of blood clot contraction is mechanotransduction."]
    s2reader = ru.SemanticScholarReader()
    intents = ["Improve LLM reasoning", "Application of ML in cancer diagnosis", "Simulation of blood clotting process"]
    titles = ["Abstraction-of-Thought", "UNet:a special type of convolutional neural network", "Cardiogenic Embolic Transport"]

    import time
    '''
    starttime = time.time()
    res = []
    for idx in range(len(queries)):
    res.append(ru.top_k_one_step_term(terms[idx], queries[idx], intents[idx], contexts[idx], titles[idx]))
    print(time.time()-starttime) #70 sec
    print(res)
    '''
    import json
    starttime = time.time()
    docs = asyncio.run(ru.async_semantic_call(s2reader, queries))
    res = asyncio.run(main(queries, terms, docs, intents, contexts, titles))
    print(time.time()-starttime) #20 sec
    print(res)
    res_dict = {'res':res, 'queries':queries}
    with open('async_res.json', 'w') as f:
        json.dump(res_dict, f)

    '''
    query_engines={}
    query_type="context"
    docs = asyncio.run(ru.async_semantic_call(s2reader, queries))
    for idx, _ in enumerate(queries):
    text_qa_template = ru.get_custom_prompt(intents[idx], contexts[idx], titles[idx], type=query_type, with_excerpt=False)
    citation_qa_template = text_qa_template
    index = VectorStoreIndex.from_documents(docs[idx])
    query_engines[str(idx)] = CitationQueryEngine.from_args(
        index,
        similarity_top_k=3,
        citation_chunk_size=1024,
        citation_qa_template=citation_qa_template
    )

    p = QueryPipeline(verbose=True)
    module_dict = {
    **query_engines,
    "input": InputComponent(),
    }
    p.add_modules(module_dict)
    for idx, _ in enumerate(queries):
    p.add_link("input"+str(idx), str(idx))

    async def pipeline_test(terms):
    response = await p.arun(input0=terms[0], input1=terms[1], input2=terms[2])
    print(str(response))

    starttime=time.time()
    asyncio.run(pipeline_test(terms))
    print(time.time()-starttime)
    '''








