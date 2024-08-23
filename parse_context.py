import os
from openai import AzureOpenAI
import json
import numpy as np
import re


def use_gpt_context(context, term):
    """
    This functions returns 3 search queries to retrieve documents explaining term in the connection with
    its surrounding context in the document. This prompt is useful for google search but not for 
    keyword based search engines like semantic scholar.
    Inputs:
        context: sentence context in which term appears
        term: term whose explanation must be retrieved
    Outputs:
        text containing 3 search queries
    """
    prompt = f"""
        Construct 3 search queries. The retrieval should return documents that explain the concept {term} or elaborate
        on the concept "{term}" in the context of "{context}". Do not use quotation symbols in queries.
    """
    client = AzureOpenAI(
	api_key = os.getenv("AZURE_OPENAI_API_KEY"),
	api_version = "",
     	azure_endpoint =os.getenv("AZURE_OPENAI_ENDPOINT")
    )
    deployment_name = 'gpt-4'
    response = client.chat.completions.create(
        model=deployment_name,
        messages=[{"role": "system", "content": "You are an AI assistant. Your job is to construct efficient search engine query for your user."},
                {"role": "user", "content": prompt}],
        temperature=0 
    )
    print("GPT4 usage: {}".format(response.usage))
    return response.choices[0].message.content

def use_gpt_intent(intent, term):
    """
    This functions returns 3 search queries to retrieve documents explaining term in the connection with
    another concept in reader's intent. This prompt is useful for google search but not for 
    keyword based search engines like semantic scholar.
    Inputs:
        intent: parsed reader intent
        term: term whose explanation must be retrieved
    Outputs:
        text explanation of link between term and input
        text containing 3 search queries
    """
    prompt = f"""
        First of all identify what is the link between the {term} and {intent} if any. Write a sentence explaining this connection.
        Then construct 3 search queries. The retrieval should return documents that explain the concept {term} and elaborate
        on this this connection. Do not use quotation symbols in queries. Your answer should be in the following format:
        Link between {term} and {intent}:
	Search queries:
    """
    client = AzureOpenAI(
      api_key = os.getenv("AZURE_OPENAI_API_KEY"),  
      api_version ="",
      azure_endpoint =os.getenv("AZURE_OPENAI_ENDPOINT") 
    )

    deployment_name='gpt-4'
    response = client.chat.completions.create(
        model=deployment_name,
        messages=[{"role": "system", "content": "You are an AI assistant. Your job is to construct efficient search engine query for your user."},
                {"role": "user", "content": prompt}],
        temperature=0 
    )
    print("GPT4 usage: {}".format(response.usage))
    return response.choices[0].message.content

def use_gpt_context_keyword(context, term):
    """
    This functions returns 1 search query of upto 3 keywords to retrieve documents explaining term in the connection with
    its surrounding context in the document. This prompt is useful for keyword based search engines like semantic scholar.
    Inputs:
        context: sentence context in which term appears
        term: term whose explanation must be retrieved
    Outputs:
        text containing 1 search query
    """
    prompt = f"""
        Construct a search query of upto three keywords in addition to the {term} or it's full form, so as to retrieve documents that explain the concept "{term}" or elaborate
        on the concept "{term}" in the context of "{context}". Do not use abbreviations, articles, pronouns, prepositions or quotes in 
        the search query. If the term is an abbreviation use its full form in the query. 
    """
    client = AzureOpenAI(
      api_key = os.getenv("AZURE_OPENAI_API_KEY"),  
      api_version ="", #enter your api_version here
      azure_endpoint =os.getenv("AZURE_OPENAI_ENDPOINT") 
    )
    deployment_name = 'gpt-4'
    response = client.chat.completions.create(
        model=deployment_name,
        messages=[{"role": "system", "content": "You are an AI assistant. Your job is to construct efficient search engine query for your user."},
                {"role": "user", "content": prompt}],
        temperature=0
    )
    query = response.choices[0].message.content.strip()
    if '"' in query:
        query = query.lstrip('"')
        query = query.rstrip('"')
    return query

def use_gpt_intent_keyword(intent, term):
    """
    This functions returns one search query of upto 3 keywords to retrieve documents explaining term in the connection with
    another concept in reader's intent. This prompt is useful for keyword based search engines like semantic scholar.
    Inputs:
        intent: parsed reader intent
        term: term whose explanation must be retrieved
    Outputs:
        text explanation of link between term and input
        text containing 1 search query of upto 3 keywords
    """
    prompt = f"""
        Identify what is the link between {term} and {intent} if any. Write a sentence explaining this connection if any.
        Then construct a search query of upto three keywords taking this connection into account. Do not use abbreviations, articles, pronouns, prepositions or quotes in 
        the search query. If {term} is an abbreviation use its full form in the query.
        Your output should be in the following format:

        Link between {term} and {intent}:[link text here]
        Search query:[keywords here]
    """
    client = AzureOpenAI(
      api_key = os.getenv("AZURE_OPENAI_API_KEY"),  
      api_version ="", #enter your api_version here
      azure_endpoint =os.getenv("AZURE_OPENAI_ENDPOINT") 
    )
    deployment_name = "gpt-4"
    response = client.chat.completions.create(
        model=deployment_name,
        messages=[{"role": "system", "content": "You are an AI assistant. Your job is to construct efficient search engine query for your user."},
                {"role": "user", "content": prompt}],
        temperature=0
    )
    
    query = " ".join(response.choices[0].message.content.split('Search query: ')[1].split(', '))
    return query

def get_batch_queries(query_input):
    """
    This functions returns one search query of upto 3 keywords for a batch of (term, context) pairs.
    Inputs:
        query_input: text containing enumerated pairs of term and context
    Outputs:
        json dict of concept and query pairs.
    """
    prompt = f"""
        You are given an enumerated list of tuples where the first element is a scientific concept and the second element is the context in which it appeared or other related topics i.e. each tuple is a pair like (concept, context or related topics). 
        For each concept and context or topic pair, construct a search query of upto three keywords, so as to retrieve documents that explain the concept in connection with the context or the topics.
        Do not use abbreviations, articles, pronouns, prepositions or quotes in the search query. If any concept is abbreviated, use its full form in the query.
        Return your answer as a JSON object in the following format:

        
        {{
            "1":
            {{
                "concept_1": search query connecting concept_1 and the context or related topics
            }}
            "2":
            {{
                "concept_2": search query connecting concept_2 and the context or related topics
            }}
        }}
        

        Your inputs are:
         {query_input}
    """
    client = AzureOpenAI(
      api_key = os.getenv("AZURE_OPENAI_API_KEY"),  
      api_version ="", #enter your api_version here
      azure_endpoint =os.getenv("AZURE_OPENAI_ENDPOINT") 
    )
    deployment_name = "gpt-4"
    response = client.chat.completions.create(
        model=deployment_name,
        messages=[{"role": "system", "content": "You are an AI assistant. Your job is to construct efficient search engine query for your user."},
                {"role": "user", "content": prompt}],
        temperature=0 # between 0 and 2
    )
    query_dict = response.choices[0].message.content
    return query_dict

def get_batch_context_queries(query_input, additional):
    """
    This functions returns one search query of upto 3 keywords for a batch of (term, context) pairs.
    The query generation uses paper title from which the term and the context is taken as additional information. 
    Inputs:
        query_input: text containing enumerated pairs of term and context
        additional: paper title in which the term and the context appear
    Outputs:
        json dict of concept and query pairs.
    """
    prompt = f"""
        You are given a dictionary where the keys and the values are scientific concepts and contexts in which they appeared respectively. 
        First, if the context has incomplete references to some concepts using phrases like "This method" or "This approach" etc, then use the <additional data> to infer the context fully. 
        For each concept and context pair, construct a search query of upto three keywords, so as to retrieve documents that explain the concept in connection with the context.
        Do not use abbreviations, articles, pronouns, prepositions or quotes in the search query. If any concept is abbreviated, use its full form in the query.
        Return your answer as a JSON object in the following format:

        
        {{
            "concept": keywords based search query to retrieve documents that provide contextually explanation of the concept (keywords corresponding to the concept should be at the beginning)
        }}
        
        Your inputs are:
         {query_input}
         <additional data>: The concepts are taken from the paper titled {additional}
    """

    client = AzureOpenAI(
      api_key = os.getenv("AZURE_OPENAI_API_KEY"),  
      api_version ="", #enter your api_version here
      azure_endpoint =os.getenv("AZURE_OPENAI_ENDPOINT") 
    )
    deployment_name = "gpt-4"
    response = client.chat.completions.create(
        model=deployment_name,
        messages=[{"role": "system", "content": "You are an AI assistant. Your job is to construct efficient search engine query for your user."},
                {"role": "user", "content": prompt}],
        temperature=0 
    )
    query_dict = response.choices[0].message.content
    return json.loads(query_dict)


def get_batch_intent_queries(query_input):
    """
    This functions returns one search query of upto 3 keywords for a batch of (term, intent) pairs.
    Inputs:
        query_input: text containing enumerated pairs of term and reader's parsed intent
    Outputs:
        json dict of concept and query pairs.
    """
    prompt = f"""
        You are given a dictionary where the keys and the values are scientific concepts and some related topics respectively. For each concept and topic pair, identify what is the link between the concept and the topic if any. 
        Then construct a search query of upto four keywords taking this connection into account. Do not use abbreviations, articles, pronouns, prepositions or quotes in 
        the search query. If any concept is abbreviated, use its full form in the query.
        Return your answer as a JSON object in the following format:

        
        {{
            "concept": keywords based search query connecting the concept and the context
        }}
        
        Your inputs are:
         {query_input}
    """
    client = AzureOpenAI(
      api_key = os.getenv("AZURE_OPENAI_API_KEY"),  
      api_version ="", #enter your api_version here
      azure_endpoint =os.getenv("AZURE_OPENAI_ENDPOINT") 
    )
    deployment_name = "gpt-4"
    response = client.chat.completions.create(
        model=deployment_name,
        messages=[{"role": "system", "content": "You are an AI assistant. Your job is to construct efficient search engine query for your user."},
                {"role": "user", "content": prompt}],
        temperature=0 
    )
    query_dict = response.choices[0].message.content
    return json.loads(query_dict)

def test_gpt_query_gen():
    import json
    terms = ["sensitivity", "ROC curve"]
    context = "Not surprisingly, the outcome prediction studies have reported a relatively low specificity and sensitivity with an area under the curve (AUC) for the receiver operating characteristic curve (ROC curve) under 0.76."
    intent = "Performance of 1D CNN for classification"
    
    query_input = {term:context for term in terms}
    query = get_batch_context_queries(query_input)
    print(query)
    query_input = {term:intent for term in terms}
    query = get_batch_intent_queries(query_input)
    print(query)
    
def key_phrase_extraction(text, method='scispacy', entity_centric=True):
    """
    This function extracts key phrase from any sentence using three different methods:
    stanza and keyBERT. This is not directly used in the final pipeline.
    """
    if entity_centric:
        import scispacy
        import spacy
        import spacy_transformers
        nlp = spacy.load("en_core_sci_sm")
        doc = nlp(text)
        return doc.ents
    if method == 'scispacy':
        import scispacy
        import spacy
        import spacy_transformers
        nlp = spacy.load("en_core_sci_sm")
        doc = nlp(text)
        roots = [sent.root for sent in doc.sents]
        key_phrases = []
        for root in roots:
            key_phrase = ""
            for token in doc:
                if token.head.text == root.text:
                    key_phrase += " " + token.text
            key_phrases.append(key_phrase.strip())
        return key_phrases
    elif method == 'stanza':
        import stanza
        stanza.download('en')
        nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma,depparse')
        doc = nlp(text)
        root_ids = [(word.id, word.text) for sent in doc.sentences for word in sent.words if word.head == 0]
        key_phrases = []
        for (root, sent) in zip(root_ids, doc.sentences):
            children = [(word.id, word.text) for word in sent.words if word.head == root[0]]
            key_phrases.append(" ".join([x[1] for x in children if x[0] < root[0]]) + " " + root[1] + " " + " ".join([x[1] for x in children  if x[0] > root[0]]))
        return key_phrases
    elif method == 'keybert':
        from keybert import KeyBERT
        kw_model = KeyBERT()
        keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(2, 3), stop_words=None)
        return [x[0] for x in keywords]
    elif method == "mixed":
        import scispacy
        import spacy
        import spacy_transformers
        from keybert import KeyBERT
        kw_model = KeyBERT()
        nlp = spacy.load("en_core_sci_sm")
        doc = nlp(text)
        keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words=None)
        keywords = [x[0] for x in keywords]
        return [x for x in keywords if x in [str(ent) for ent in list(doc.ents)]]
    
def get_numbered_items(text):
    query_arr = []
    lines = text.strip().split("\n")
    for line in lines:
      if len(line) > 1 and line[0].isnumeric():
        total = re.split(r'^\d. ', line)[1]
        query_arr.append(total)
    query_arr = [query.lstrip('"').rstrip('"') for query in query_arr]
    return query_arr
    
def get_embedding(texts, model="embd"):
   """
    This function returns embeddings of a list of texts using text-embedding-ada-002.
    Inputs:
        texts: List of texts
        model: embedding model name from azure deployments
    Outputs: list of embeddings of len number of texts
   """
   client = AzureOpenAI(
     api_key = os.getenv("AZURE_OPENAI_API_KEY"),  
     api_version = "",  #input api version here
     azure_endpoint = os.getenv( "AZURE_OPENAI_ENDPOINT")
   )
   embeddings = [x.embedding for x in client.embeddings.create(input = texts, model=model).data]
   return embeddings

def get_batch_relevance(texts, model="embd"):
    """
    This function returns a list of cosine similarities between all the embeddings
    and the last embedding in a list of embeddings
    Inputs:
        texts: List of texts
        model: embedding model name from azure deployments
    Outputs: list of cosine similarities of length (1-number of texts)
    """
    text_embds = get_embedding(texts)
    relevance = cosine_similarity(text_embds[:-1], text_embds[-1])
    return relevance

def get_topk(queries, docs, model="embd", k=3):
    """
    Given a batch of (query, list of document) pairs, this function ranks the documents per
    query and for each pair returns the top-k documents 
    Inputs: 
        queries: a list of queries
        docs: list of list of n documents per query, size number of queries x n
    Outputs:
        list of list of document indices, size number of queries x k
    """
    docs = [x for xs in docs for x in xs]
    text_embds = get_embedding(docs+queries)
    n = len(queries)
    doc_embds = np.array(text_embds[:-n]).reshape(n, 10, -1) #(10, x)
    query_embds = np.array(text_embds[-n:]) #(1, x)
    relevances = []
    for query, docs in zip(query_embds, doc_embds):
        relevance = cosine_similarity(query, docs)
        sorted_keys = [i[0] for i in sorted(enumerate(relevance), key=lambda x:x[1])][:k]
        relevances.append(sorted_keys)
    return relevances

def cosine_similarity(x, y):
   """
   This function computes cosine similarity between two embeddings x and y
   """
   x = np.array(x)
   y = np.array(y)
   return np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y))

def get_relevance(x, y):
    """
    This function returns overlapping between x and y in embedding space
    """
    e_x = get_embedding(x)
    e_y = get_embedding(y)
    return cosine_similarity(e_x, e_y)

def noun_phrase_extraction(text):
    import nltk
    nltk.download('maxent_ne_chunker')
    nltk.download('words')
    from nltk import word_tokenize, pos_tag, ne_chunk
    from nltk import RegexpParser
    from nltk import Tree
    chunk_func=ne_chunk
    chunked = chunk_func(pos_tag(word_tokenize(text)))
    continuous_chunk = []
    current_chunk = []

    for subtree in chunked:
        if type(subtree) == Tree:
            current_chunk.append(" ".join([token for token, pos in subtree.leaves()]))
        elif current_chunk:
            named_entity = " ".join(current_chunk)
            if named_entity not in continuous_chunk:
                continuous_chunk.append(named_entity)
                current_chunk = []
        else:
            continue

    return continuous_chunk

def get_paragraph_array(doc):
    paragraphs = []
    sentences = []
    for _, value in doc['content'].items():
        for para in value:
            for x in para:
                sentences.append(x)
            para_text = " ".join(para)
            para_text = para_text.replace("-\n", "")
            para_text = para_text.replace("\n", " ")
            paragraphs.append(para_text)
    return paragraphs, sentences

def get_larger_context(term, doc):
    if term in doc['abstract']:
        context = doc['abstract']
        context = context.replace("\n", " ")
        return context
    else:
        para_array, _ = get_paragraph_array(doc)
        for para in para_array:
            if term in para:
                return para
    
            
def query_similarity(query_arr, statement):
    query_dict = {}
    for query in query_arr:
        query_dict[query] = get_relevance(query, statement)
    queries_ordered = [(k, v) for k, v in sorted(query_dict.items(), key=lambda item: item[1])][::-1]
    return queries_ordered
    
if __name__ == '__main__':
    import time
    starttime = time.time()
    test_gpt_query_gen()
    print(time.time()-starttime)

