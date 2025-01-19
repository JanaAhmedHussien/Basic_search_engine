from flask import Flask,request,render_template,jsonify
import pyterrier as pt
import os
if not pt.started():
      pt.init(boot_packages=["com.github.terrierteam:terrier-prf:-SNAPSHOT"])
base_dir = "C:\\Users\\Janaa\\OneDrive\\Desktop\\Prroject_IR"
index_dir = os.path.join(base_dir, "var", "myFirstIndex")
if not os.path.exists(index_dir):
    os.makedirs(index_dir)

import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import re
import tensorflow as tf
import tensorflow_hub as hub


app=Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/submit', methods=['POST'])

def submit():
        query=request.form['query']
        
        
# preprocessing
        os.environ["JAVA_HOME"] = "C:\Program Files\Java\jdk-22"
        pd.set_option('display.max_colwidth', 150)
        stop_words = set(stopwords.words('english'))
        # print(stopwords.words('english'))
        stemmer = PorterStemmer()
        def Stem_text(text):
            tokens = word_tokenize(text)
            stemmed_tokens = [stemmer.stem(word) for word in tokens]
            # print (tokens)
            return ' '.join(stemmed_tokens)

        def clean(text):
                text = re.sub(r"[\.\,\#_\|\:\?\?\/\=\@]", " ", text) # remove special characters
                text = re.sub(r'\t', ' ', text) # remove tabs
                text = re.sub(r'\n', ' ', text) # remove line jump
                text = re.sub(r"\s+", " ", text) # remove extra white space
                text = text.strip()
                return text

        def remove_stopwords(text):
            tokens = word_tokenize(text)
            filtered_tokens = [word.lower() for word in tokens if word.lower() not in stop_words] #Lower is used to normalize al the words make them in lower case
            # print('Tokens are:',tokens,'\n')
            return ' '.join(filtered_tokens)

        #we need to process the query also as we did for documents
        def preprocess(sentence):
                sentence = clean(sentence)
                sentence = remove_stopwords(sentence)
                sentence = Stem_text(sentence)
                return sentence
        import tarfile 
        
        # collect data
        # open file 
        file = tarfile.open('cran.tar.gz') 
        
        # print file names 
        # print(file.getnames()) 
        
        # extract files 
        file.extractall('cran_dataset') 
        
        # close file 
        file.close()
        def load_cran_dataset(data_dir):
            documents_path = os.path.join(data_dir, 'cran.all.1400')
            queries_path = os.path.join(data_dir, 'cran.qry')
            qrels_path = os.path.join(data_dir, 'cranqrel')

            documents_df = read_documents(documents_path)
            queries_df = read_queries(queries_path)
            qrels_df = read_qrels(qrels_path)
            return documents_df, queries_df, qrels_df

        # Read documents from cran.all.1400 file
        def read_documents(documents_path):
            with open(documents_path, 'r') as file:
                lines = file.readlines()
            documents = []
            current_document = None
            for line in lines:
                if line.startswith('.I'):
                    if current_document is not None:
                        current_document['Text'] = current_document['Text'].split('\t')[0].strip()  # Remove anything after the first tab
                        documents.append(current_document)
                    current_document = {'ID': line.strip().split()[1], 'Text': ''}
                elif line.startswith('.T'):
                    continue
                elif line.startswith('.A') or line.startswith('.B') or line.startswith('.W') or line.startswith('.X'):
                    continue
                else:
                    current_document['Text'] += line.strip() + ' '

            # Append the last document
            if current_document is not None:
                current_document['Text'] = current_document['Text'].split('\t')[0].strip()  # Remove anything after the first tab
                documents.append(current_document)
            documents_df = pd.DataFrame(documents)
            return documents_df

        # Read queries from CISI.QRY file
        def read_queries(queries_path):
            with open(queries_path, 'r') as file:
                lines = file.readlines()
            query_texts = []
            query_ids = []
            current_query_id = None
            current_query_text = []
            for line in lines:
                if line.startswith('.I'):
                    if current_query_id is not None:
                        query_texts.append(' '.join(current_query_text))
                        current_query_text = []
                    current_query_id = line.strip().split()[1]
                    query_ids.append(current_query_id)
                elif line.startswith('.W'):
                    continue
                elif line.startswith('.X'):
                    break
                else:
                    current_query_text.append(line.strip())
            # Append the last query
            query_texts.append(' '.join(current_query_text))
            queries_df = pd.DataFrame({
                'qid': query_ids,
                'raw_query': query_texts})
            return queries_df

        # Read qrels from cranqrel file
        def read_qrels(qrels_path):
            qrels_df = pd.read_csv(qrels_path, sep='\s+', names=['qid','Q0','docno','label'])
            return qrels_df
        data_dir = 'cran_dataset'
        documents_df, queries_df, qrels_df = load_cran_dataset(data_dir)
        # print(documents_df['Text'][0])

        # see the data frame of the documents
        # print("\n\n",documents_df)
# ///////////////////////////////////////////////////////////////
# building indexer from scratch
# def get_terms(data):
#       terms=[]
#       for doc in data:
#           for term in data[doc].split():
#               terms.append(term)
#       return  terms
# def get_unique_terms(terms):
#         unique_terms=[]
#         for d in terms :
#             if d not in unique_terms:
#                unique_terms.append(d)
#         return unique_terms
# def get_document_collection_terms(data):
#         docs_colllection={}
#         for doc in data:
#                 docs_colllection[doc]=get_unique_terms(data[doc].split())
#         return docs_colllection

# def inverted_index(documents_df):
#     term_docs_incid_matrix = {}
#     for index, row in documents_df.iterrows():
#         document_id = row['ID']
#         document_text = row['Text']  # Assuming the column containing text is named 'Text'

#         # Split the document text into terms (words)
#         terms = document_text.split()  # Splitting by whitespace

#         # Update the term-document incidence matrix
#         for term in terms:
#             if term in term_docs_incid_matrix:
#                 term_docs_incid_matrix[term].add(document_id)
#             else:
#                 term_docs_incid_matrix[term] = {document_id}

#     return term_docs_incid_matrix
# def display_dict(dic):
#         print()
#         for i in dic:
#             print (i , " : " ,dic[i])
#         print()
# term_docs_incid_matrix=inverted_index(documents_df)
# print("inverted index")
# #formatted Display
# display_dict(term_docs_incid_matrix)
# ////////////////////////////////////////////////////////////////////

        # see the data frame queries
        # print("\n\n",queries_df)

        # see the data frame of the qrels_df
        # print("\n\n",qrels_df)

        # adding docno to the data frame to work with when indexing
        documents_df["docno"]=documents_df["ID"].astype(str)
        # print("\n\n",documents_df)

        # adding query no to the data frame to with it when indexing
        queries_df["qid"]=queries_df["qid"].astype(str)
        # print("\n\n",queries_df)

        # applying preprocessing to the documents
        documents_df['processed_text'] = documents_df['Text'].apply(preprocess)
        # print("\n\n",documents_df)

        # applying preprocess to the dataframe of the queries
        queries_df["query"]=queries_df["raw_query"].apply(preprocess)
        # print("\n\n",queries_df)



        indexer = pt.DFIndexer(index_dir, overwrite=True)
        index_ref = indexer.index(documents_df['processed_text'], documents_df["docno"])
        index = pt.IndexFactory.of(index_ref)
        # query="stability of rectangular plates"
        query = preprocess(query)
        # ///////tF-IDF
        tfidf_retr = pt.BatchRetrieve(index, controls = {"wmodel": "TF_IDF"},num_results=10)
        res=tfidf_retr.search(query)
        top_answer_2_answers =pd.DataFrame( documents_df[documents_df["docno"].isin([res['docno'][0], res['docno'][1]])]['Text'])

        # ///////////////////
        # //////////////////////////Elmo model
        #load the ELMo model
        # elmo = hub.load("https://tfhub.dev/google/elmo/3")
        
        # ///////////////////////////

# /////////////////////query expnsion
        bm25 = pt.BatchRetrieve(index, wmodel="BM25",num_results=10)

        results = bm25.search(query)
        # # print(results)

        # # print exapanded query
        rm3_expander = pt.rewrite.RM3(index,fb_terms=10, fb_docs=100)

        # #output of the BM25 will be fed into the RM3 expander for query expansion.
        rm3_qe = bm25 >> rm3_expander
        expanded_query = rm3_qe.search(query).iloc[0]["query"]
        # # print(expanded_query)


        # # After that you can search using the expanded query
        expanded_query_formatted = ' '.join(expanded_query.split()[1:])

        results_wqe = bm25.search(expanded_query_formatted)

        # print("   Before Expansion    After Expansion")
        # print(pd.concat([results[['docid','score']][0:5].add_suffix('_1'),
        #             results_wqe[['docid','score']][0:5].add_suffix('_2')], axis=1).fillna(''))

        #Let's check the tweets text for the top 5 retrieved tweets
        df=documents_df[['Text']][documents_df['docno'].isin(results_wqe['docno'].loc[0:5].tolist())]
        # ////////////////////////////////////
        # //////////////Using bert model
        # result_merged = res.merge(documents_df, on="docno")[["score", "Text"]]
        # result_merged = result_merged.sort_values(by="score", ascending=False)
        
        return df.to_html()

if(__name__=='__main__'):
    app.run(debug=True)