import streamlit as st
import openai
import langchain
import langchain_experimental



# Setting up the page configuration
st.set_page_config(
    page_title="AI Scientist",
    page_icon=":brain:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Defining the function to display the home page
def home():
    import streamlit as st
    import streamlit_extras as streamlit_extras
    #from streamlit_extras.badges import badge rn
    from streamlit_extras.colored_header import colored_header

    
    st.title("AI Research Assistant👨🏿‍💻")

    # Displaying information and warnings in the sidebar
    st.sidebar.title(
        "AI Assistant for Scientists and Researchers"
    )
    st.sidebar.info(
        "App by [Olatomiwa Bifarin](https://www.linkedin.com/in/obifarin/)."
    )

    st.sidebar.warning(
        "LLMs are known to [hallucinate](https://www.instagram.com/p/C09veCcpqXC), validate model's output."
    )

    st.sidebar.markdown('<a href="mailto:obifarin@yahoo.com">Any feedbacks?</a>', unsafe_allow_html=True)
    
    st.markdown(
        "#### App features:"
    )
    st.markdown(
        "**📊Data Science Agent**: Chat with your data."
    )
    st.markdown(
        "**💻Reviewer Agent**: Review your manuscript with AI."
    )
    st.markdown(
        "**🤖Research Assistant**: AI text researcher."
    )
    
    st.markdown(
        "**📄Paper Agent**: Chat with a Paper."
    )

    st.image("desk.png", caption='Bif X Midjourney')

    st.markdown(
        "#### How to use:"
    )
    # Displaying markdown text on the page
    st.info("You will need an OpenAI API key 🔑 to use this App. To obtain it, visit [OpenAI](https://platform.openai.com/account/api-keys). All new users get free [$5 worth of credit](https://help.openai.com/en/articles/4936830-what-happens-after-i-use-my-free-tokens-or-the-3-months-is-up-in-the-free-trial). Here is an instruction on [how to get the key](https://help.openai.com/en/articles/4936850-where-do-i-find-my-api-key). For more detailed instruction, see this [Tutorial](https://www.maisieai.com/help/how-to-get-an-openai-api-key-for-chatgpt). The API key is free, but charges will be made to your account once the initial credit runs out. So keep an eye 👀 on that account.")

    st.warning('I built this app over the winter break with very limited time to spare, so I am sure you will encounter more than one bug 🐞 while using it, hopefully it is useable. Feel free to give feedback')

    st.markdown("---")


#home()
def chat_with_dataset():
    import os
    import streamlit as st
    import pandas as pd
    #from langchain import OpenAI
    from langchain.chat_models import ChatOpenAI
    from langchain.callbacks import StreamlitCallbackHandler
    from langchain_experimental.agents import create_pandas_dataframe_agent
    
    file_formats = {
        "csv": pd.read_csv,
        "xls": pd.read_excel,
        "xlsx": pd.read_excel,
        "xlsm": pd.read_excel,
        "xlsb": pd.read_excel,
    }

    def clear_submit():
        """
        A utility function to reset the submit button's state in the Streamlit session.
        Clear the Submit Button State
        Returns:

        """
        st.session_state["submit"] = False

    @st.cache_data()
    def load_data(uploaded_file):
        """
        This function loads data from an uploaded file. It determines the file extension 
        and uses the appropriate pandas function to read the file. If the file format 
        isn't supported, it displays an error. 
        Load data from the uploaded file based on its extension.
        """
        try:
            ext = os.path.splitext(uploaded_file.name)[1][1:].lower()
        except:
            ext = uploaded_file.split(".")[-1]
        if ext in file_formats:
            return file_formats[ext](uploaded_file)
        else:
            st.error(f"Unsupported file format: {ext}")
            return None

    st.subheader("Analyze Your Data with English")
    st.info("NOTES: For better result, ask one question at a time. Plot outputs is not supported in this version. Don't forget to add your OpenAI key. Clear conversation history once you are done.")

    openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")


    uploaded_file = st.file_uploader(
        "Upload a Data file",
        type=list(file_formats.keys()),
        help="Various File formats are Support",
        on_change=clear_submit,
    )

    df = None  # Initialize df to None outside the if block

    if uploaded_file:
        df = load_data(uploaded_file)  # df will be assigned a value if uploaded_file is truthy

    if df is None:  # Check if df is still None before proceeding
        st.warning("No data file uploaded or there was an error in loading the data.")
        return  # Exit the function early if df is None
    
    if not openai_api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()

    st.sidebar.info("If you face a KeyError: 'content' error, Press the clear conversation histroy button")
    if "messages" not in st.session_state or st.sidebar.button("Clear conversation history"):
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

    # Display previous chat messages
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input(placeholder="What is this data about?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        # Check if OpenAI API key is provided
        if not openai_api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()

        llm = ChatOpenAI(
            temperature=0, 
            model="gpt-4-1106-preview", # gpt-4-0613 gpt-3.5-turbo-1106 gpt-4-1106-preview
            openai_api_key=openai_api_key, 
            streaming=True
        )

        pandas_df_agent = create_pandas_dataframe_agent(
            llm,
            df,
            verbose=True,
            handle_parsing_errors=True
        )

        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            response = pandas_df_agent.run(st.session_state.messages, callbacks=[st_cb])
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write(response)


def paper_reviews():
    import tempfile
    import streamlit as st
    from langchain.document_loaders import PyPDFLoader
    from langchain.embeddings.openai import OpenAIEmbeddings
    from langchain.vectorstores import Chroma

    from langchain.chat_models import ChatOpenAI
    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate

    # Function to load and process the database
    def load_db(file, api_key): 

      #Load documents
      loader = PyPDFLoader(file)
      documents = loader.load()

      # Define embedding
      embeddings = OpenAIEmbeddings(api_key=openai_api_key)

      # Create a vectorstore from documents
      db = Chroma.from_documents(documents, embeddings)
      
      #Paper Review, build prompt
      template = """
        Give that the user types the word, 'Review', Here is your task
        Your task:
        Compose a high-quality detailed peer review of this scientific paper submitted to
        a top scientific journal.

        Start by "Review outline:".
        And then:
        "1. Significance and novelty"
        "2. Potential reasons for acceptance"
        "3. Potential reasons for rejection", List 4 key reasons. For each of 4 key
        reasons, use **>=2 sub bullet points** to further clarify and support your
        arguments in painstaking details.
        "4. Suggestions for improvement", List 4 key suggestions.

        Be thoughtful and constructive. Write Outlines only.

        {context}
        Question: {question}
        Helpful Answer:"""
      
      QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

      # LLM
      llm = ChatOpenAI(temperature=0, 
                       model='gpt-4-1106-preview', # gpt-4-1106-preview gpt-4-0613 gpt-3.5-turbo-1106 gpt-4-32k-0613
                       api_key=api_key)
      
      # Run chain
      qa_chain = RetrievalQA.from_chain_type(
            llm,
            retriever=db.as_retriever(),
            #return_source_documents=True,
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
        )
      
      return qa_chain

    # Streamlit app interface for chat_with_papers
    st.subheader('Paper Reviewer')
    st.info("After uploading the PDF file, type the word 'Review' and submit query.")
    st.warning("⚠️If you have uploaded a PDF in either this session or the Paper Agent session that is different from the one you intend to upload here, please refresh the web page before proceeding.")


    # File upload
    uploaded_file = st.file_uploader('Upload a PDF document', type=['pdf'])

    # API Key and Query Input
    query_text = st.text_input('Type Review here:', placeholder='...')

    # Main logic for handling OpenAI API key
    if "openai_api_key" not in st.session_state:
        openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
        if not openai_api_key:
            st.sidebar.warning("Please add your OpenAI API key to continue! Go to the home page to learn more.")
            st.warning("Please add your OpenAI API key to continue!")
            #st.sidebar.info("To obtain your OpenAI API key, please visit [OpenAI](https://platform.openai.com/account/api-keys). They provide a $5 credit to allow you to experiment with their models. If you're unsure about how to get the API key, you can follow this [Tutorial](https://www.maisieai.com/help/how-to-get-an-openai-api-key-for-chatgpt). While obtaining the API key doesn't require a compulsory payment, once your allotted credit is exhausted, a payment will be necessary to continue using their services.")
            st.stop()
        st.session_state["openai_api_key"] = openai_api_key

    # Ensure that openai_api_key is available
    openai_api_key = st.session_state.get("openai_api_key", "")

    # Query Submission
    if st.button('Submit Query'):
        if uploaded_file and openai_api_key and query_text:
            with st.spinner('Reviewing your manuscript, should take a minute...'):
                # Save uploaded file to a temporary location
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
                    tmpfile.write(uploaded_file.getvalue())
                    temp_file_path = tmpfile.name

                # Load and process the database
                qa_chain = load_db(temp_file_path, openai_api_key)
                result = qa_chain({"query": query_text})
                response = result['result']
                st.write(response)  # Display the response
        else:
            st.error('Please upload a file, enter an API key, and a query.')

def chat_with_papers():
    import tempfile
    import streamlit as st
    from langchain.embeddings.openai import OpenAIEmbeddings
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.vectorstores import Chroma
    from langchain.document_loaders import PyPDFLoader
    from langchain.chains import ConversationalRetrievalChain
    from langchain.chat_models import ChatOpenAI
    from langchain.memory import ConversationBufferMemory
    from langchain.prompts import PromptTemplate

    # Function to load and process the database
    def load_db(file, api_key, chain_type='stuff', k=5):
      # Load documents
      loader = PyPDFLoader(file)
      documents = loader.load()

      # Split documents
      text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000,
                                                     chunk_overlap=300)
      docs = text_splitter.split_documents(documents)

      # Define embedding
      embeddings = OpenAIEmbeddings(api_key=openai_api_key)

      # Create vector database from data
      #db = DocArrayInMemorySearch.from_documents(docs, embeddings)

      # Create a vectorstore from documents
      db = Chroma.from_documents(docs, embeddings)

      # Define retriever
      retriever = db.as_retriever(search_type="similarity",
                                  search_kwargs={"k": k})
      # Memory
      memory = ConversationBufferMemory(
          memory_key="chat_history",
          return_messages=True)
      
      #Prompt
      # Build prompt
      template = """Use the following pieces of context to answer the question at the
      end. If you don't know the answer, just say that you don't know, DON'T try to
      make up an answer. 
      {context}
      Question: {question}
      Helpful Answer:"""
      
      conversation_prompt = PromptTemplate.from_template(template)

      # LLM
      llm = ChatOpenAI(temperature=0,
                       model='gpt-3.5-turbo-1106', # gpt-4-0613 gpt-3.5-turbo-1106 gpt-4-1106-preview
                       api_key=api_key)

      # Create a chatbot chain
      qa = ConversationalRetrievalChain.from_llm(
          llm,
          chain_type=chain_type,
          retriever=retriever,
          memory=memory,
          combine_docs_chain_kwargs={'prompt': conversation_prompt}
          #return_source_documents=True,
          #return_generated_question=True
          )
      return qa

    # Streamlit app interface for chat_with_papers
    st.title('Chat with Papers')

    #some information
    st.info("I have not implemented memory 🧠 yet, so include conetxt in all questions asked. Also for best results, ask specific questions about the paper📄")
    st.warning("⚠️If you have uploaded a PDF in either this session or the Reviewer Agent session that is different from the one you intend to upload here, please refresh the web page before proceeding.")

    # Main logic for handling OpenAI API key
    if "openai_api_key" not in st.session_state:
        openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
        if not openai_api_key:
            st.sidebar.warning("Please add your OpenAI API key to continue! Go the home page to learn mre.")
            st.warning("Please add your OpenAI API key to continue!")
            #st.sidebar.info("To obtain your OpenAI API key, please visit [OpenAI](https://platform.openai.com/account/api-keys). They provide a $5 credit to allow you to experiment with their models. If you're unsure about how to get the API key, you can follow this [Tutorial](https://www.maisieai.com/help/how-to-get-an-openai-api-key-for-chatgpt). While obtaining the API key doesn't require a compulsory payment, once your allotted credit is exhausted, a payment will be necessary to continue using their services.")
            st.stop()
        st.session_state["openai_api_key"] = openai_api_key

    # Ensure that openai_api_key is available
    openai_api_key = st.session_state.get("openai_api_key", "")

    # File upload
    uploaded_file = st.file_uploader('Upload a PDF document', type=['pdf'])

    # Check if uploaded_file is available
    if not uploaded_file:
        st.warning("Please upload a PDF document to proceed.")
        return  # Exit the function early if no file is uploaded

    # OpenAI API Key Input
    #openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    #if not openai_api_key:
    #    st.sidebar.warning("Please add your OpenAI API key to continue.")
    #    return  # Exit the function early if no API key is provided


    # Save uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
        tmpfile.write(uploaded_file.getvalue())
        temp_file_path = tmpfile.name

    # Load and process the database
    qa_chain = load_db(temp_file_path, openai_api_key)

    # Chat Interface
    st.sidebar.info("Once you are done, don't forget to clear history.")
    if "messages" not in st.session_state or st.sidebar.button("Clear conversation history"):
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I assist you with the paper?"}]

    # Display previous chat messages
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # Handle chat input
    if prompt := st.chat_input("Ask a question about the paper"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        # Process the chat query
        with st.spinner('Processing...'):
            response = qa_chain.run(prompt)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.chat_message("assistant").write(response)

def general_research_assistant():

  from langchain.chat_models import ChatOpenAI
  from langchain.prompts import ChatPromptTemplate
  from langchain.schema.output_parser import StrOutputParser
  import requests
  from bs4 import BeautifulSoup
  from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
  from langchain.utilities import DuckDuckGoSearchAPIWrapper
  import json

  def general_RA_sub(question: str, openai_api_key: str):
      RESULTS_PER_QUESTION = 3

      ddg_search = DuckDuckGoSearchAPIWrapper()

      def web_search(query: str, num_results: int = RESULTS_PER_QUESTION):
          results = ddg_search.results(query, num_results)
          return [r["link"] for r in results]

      def scrape_text(url: str):
          try:
              response = requests.get(url)
              if response.status_code == 200:
                  soup = BeautifulSoup(response.text, "html.parser")
                  page_text = soup.get_text(separator=" ", strip=True)
                  return page_text
              else:
                  return f"Failed to retrieve the webpage: Status code {response.status_code}"
          except Exception as e:
              return f"Failed to retrieve the webpage: {e}"

      SUMMARY_TEMPLATE = """{text}
      -----------
      Using the above text, answer in short the following question:

      > {question}
      -----------
      if the question cannot be answered using the text, imply summarize the text.
      Include all factual information, numbers, stats etc if available."""

      SUMMARY_PROMPT = ChatPromptTemplate.from_template(SUMMARY_TEMPLATE)

      scrape_and_summarize_chain = RunnablePassthrough.assign(
          summary = RunnablePassthrough.assign(
          text=lambda x: scrape_text(x["url"])[:10000]
      ) | SUMMARY_PROMPT | ChatOpenAI(model="gpt-3.5-turbo-1106",
                                      api_key=openai_api_key) | StrOutputParser()
      ) | (lambda x: f"URL: {x['url']}\n\nSUMMARY: {x['summary']}")

      web_search_chain = RunnablePassthrough.assign(
          urls = lambda x: web_search(x["question"])
      ) | (lambda x: [{"question": x["question"], "url": u} for u in x["urls"]]
          ) | scrape_and_summarize_chain.map()

      SEARCH_PROMPT = ChatPromptTemplate.from_messages(
          [
              (
                  "user",
                  "Write 3 google search queries to search online that form an "
                  "objective opinion from the following: {question}\n"
                  "You must respond with a list of strings in the following format: "
                  '["query 1", "query 2", "query 3"].',
              ),
          ]
      )

      search_question_chain = SEARCH_PROMPT | ChatOpenAI(
          model = "gpt-4-1106-preview",
          api_key=openai_api_key,
          temperature=0) | StrOutputParser() | json.loads

      full_research_chain = search_question_chain | (lambda x: [{"question": q} for q in x]) | web_search_chain.map()

      WRITER_SYSTEM_PROMPT = "You are an AI critical thinker research assistant. Your sole purpose is to write well written, critically acclaimed, objective and structured reports on given text."

      RESEARCH_REPORT_TEMPLATE = """Information:
      --------
      {research_summary}
      --------
      Using the above information, answer the following question or topic: "{question}" in a detailed report -- \
      The report should focus on the answer to the question, should be well structured, informative, \
      in depth, with facts and numbers if available and a minimum of 1,200 words.
      You should strive to write the report as long as you can using all relevant and necessary information provided.
      You must write the report with markdown syntax.
      You MUST determine your own concrete and valid opinion based on the given information.
      Do NOT deter to general and meaningless conclusions.

      Cite all used source including urls at the end of the report, and
      make sure to not add duplicated sources, but only one reference for each.
      You must write the report in APA format.
      CITE in text, When citing sources within the text, use numerical citations in
      chronological order (1, 2, 3, etc.).
      For PubMed and ArXiv Papers, end of report reference should take this format, here is an example: 
      Li, J., Shen, C., Wang, X., Lai, S., Yu, X., & Zhang, X. (2023). Associations between different
      types and intensities of physical activity and health-related quality of life mediated by depression 
      in Chinese older adults. BMC Geriatrics, 23(1). https://bmcgeriatr.biomedcentral.com/articles/10.1186/s12877-023-04452-6

      PLEASE do your best, this is very important to my career."""

      prompt = ChatPromptTemplate.from_messages(
          [
              ("system", WRITER_SYSTEM_PROMPT),
              ("user", RESEARCH_REPORT_TEMPLATE),
          ]
      )

      def collapse_list_of_lists(list_of_lists):
          content = []
          for l in list_of_lists:
              content.append("\n\n".join(l))
          return "\n\n".join(content)

      chain = RunnablePassthrough.assign(
          research_summary= full_research_chain | collapse_list_of_lists
      ) | prompt | ChatOpenAI(model="gpt-4-1106-preview",
                              api_key=openai_api_key) | StrOutputParser()
      
      return chain.invoke({"question": question})
  #streamlit code
  st.subheader("Research Assistant")
  st.info("This feature will generate a **draft** report 📄, approximately 1,200 words in length, tailored to your query❓.")
  st.info("Should you prefer a focus 🔭on academic publications, please specify this in your query, for instance state 'use Pubmed for sources' or 'use Arxiv for sources'. In this cases, there *might* be insufficient citations due to API restrictions, which lead to a less stellar report. I will attempt to work on this on subsequent versions.")  
  #query input
  query_text = st.text_input('Enter your research question:', placeholder='e.g. Role of exercise in aging (use Pubmed as Sources)...')

  # Main logic for handling OpenAI API key
  if "openai_api_key" not in st.session_state:
    openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    if not openai_api_key:
        st.sidebar.warning("Please add your OpenAI API key to continue!!")
        st.warning("Please add your OpenAI API key to continue!!")
        st.sidebar.info("Go to the home page for instructions on how to get OpenAI API key.")
        st.stop()
    st.session_state["openai_api_key"] = openai_api_key

  # Ensure that openai_api_key is available
  openai_api_key = st.session_state.get("openai_api_key", "")

  st.sidebar.info("In case you encounter a **JSON decode error**, a simple rerun of your query might fix it.")

  # Query submission
  if st.button('Submit Query'):
    if openai_api_key and query_text:
        with st.spinner('Doing research, this should take about 2 minutes ...'):
            response = general_RA_sub(query_text, openai_api_key)
            st.write(response) #display the response
    else:
        st.error('Please enter an API key and a query.')



# Close the code with this: 
# Dictonary to store all functions as pages
page_names_to_funcs = {
    "Home 🏠": home,
    "📊Data Science Agent": chat_with_dataset,
    "💻Reviewer Agent": paper_reviews,
    "🤖Research Assistant": general_research_assistant,
    "📄Paper Agent": chat_with_papers
}

# display page by dictionary
demo_name = st.sidebar.selectbox("Choose a page to navigate to", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()