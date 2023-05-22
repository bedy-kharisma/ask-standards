import streamlit as st
import os
import pandas as pd
from haystack import Pipeline
from haystack.nodes import TextConverter, PreProcessor
from haystack.document_stores import InMemoryDocumentStore
from haystack.pipelines import ExtractiveQAPipeline
from haystack.nodes import EmbeddingRetriever, FARMReader
from haystack.nodes import PromptNode, PromptTemplate
folder = 'text_files'

def filter_data(keyword):
    # Load the dataframe from a pickle file
    df = pd.read_pickle('./standards.pkl')
    df['num_chars'] = df['text'].apply(lambda x: len(x))
    df = df[df['num_chars'] != 0]
    df = df[['name', 'url', 'text']]
    import re
    filtered_std = df[df['text'].str.contains(keyword, flags=re.IGNORECASE)]
    return filtered_std

def initialize_haystack(filtered_std):
    document_store = InMemoryDocumentStore()
    indexing_pipeline = Pipeline()
    text_converter = TextConverter()
    preprocessor = PreProcessor(
        clean_whitespace=True,
        clean_header_footer=True,
        clean_empty_lines=True,
        split_by="word",
        split_length=1000,
        split_overlap=20,
        split_respect_sentence_boundary=True,
    )
    
    indexing_pipeline.add_node(component=text_converter, name="TextConverter", inputs=["File"])
    indexing_pipeline.add_node(component=preprocessor, name="PreProcessor", inputs=["TextConverter"])
    indexing_pipeline.add_node(component=document_store, name="DocumentStore", inputs=["PreProcessor"])

    for index, row in filtered_std.iterrows():
        # Get the file name and text for this row
        file_name = row['name']
        text = row['text']

        # Create a new text file with the given file name and write the text to it
        file_path = os.path.join(folder, file_name)
        with open(file_path, 'w', errors='ignore') as f:
            f.write(text)
            
    #reads all related files
    files_to_index = [folder + "/" + f for f in os.listdir(folder)]
    indexing_pipeline.run_batch(file_paths=files_to_index)

    retriever = EmbeddingRetriever(
        document_store=document_store, embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1"
    )
    document_store.update_embeddings(retriever)
    reader = FARMReader(model_name_or_path="bert-large-uncased-whole-word-masking-finetuned-squad", use_gpu=True)
    pipe = ExtractiveQAPipeline(reader, retriever)
        
    return pipe

def main():
    st.title("Ask Standards")
    keyword = st.text_input("Enter a keyword", "running dynamic")

    # Filter data based on keyword
    filtered_std = filter_data(keyword)
    st.write(str(filtered_std.shape[0]) + " standards found")
    query = st.text_input("Enter a query", "vehicle at what speed that must perform dynamic performance test?")
    
    if st.button("Process"):
        with st.spinner('Wait for it...'):
            # Initialize Haystack pipeline
            pipe = initialize_haystack(filtered_std)

            # Perform the question answering
            prediction = pipe.run(
                query=query,
                params={
                    "Retriever": {"top_k": 15},
                    "Reader": {"top_k": 5}
                }
            )

            answer_contexts = []
            for i in range(5):
                answer_context = prediction['answers'][i].context
                answer_context = answer_context.replace('\n', ' ')  # Remove line feeds
                answer_contexts.append(answer_context)
            joined_contexts = ' '.join(answer_contexts)

            prompt_node = PromptNode(model_name_or_path="google/flan-t5-base", use_gpu=True)
            prompt_text = "Consider you are a rolling stock consultant provided with this query: {query} provide answer from the following context: {contexts}. Answer:"
            output = prompt_node.prompt(prompt_template=prompt_text, query=query, contexts=joined_contexts)
        st.success('Done!')
        st.subheader("Answer:")
        st.write(output[0])
 
if __name__ == '__main__':
    main()
