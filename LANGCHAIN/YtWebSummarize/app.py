import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain.schema import Document
from langchain_community.document_loaders import UnstructuredURLLoader
from youtube_transcript_api import YouTubeTranscriptApi

# Streamlit App Configuration
st.set_page_config(page_title="LangChain: Summarize Text From Website and Youtube", page_icon="🌐")
st.title("🌐 LangChain: Summarize Text From Website and Youtube")
st.subheader('Summarize URL')

# Get the Groq API Key and URL (YouTube or website) to be summarized
with st.sidebar:
    groq_api_key = st.text_input("Groq API Key", value="", type="password")

if not groq_api_key:
    st.info("Please add your Groq API key to continue")
    st.stop()

generic_url = st.text_input("URL", label_visibility="collapsed")

# Groq LLM Configuration
llm = ChatGroq(model="Gemma-7b-It", groq_api_key=groq_api_key)

# Prompt Template
prompt_template = """
Provide a summary of the following content in 300 words:
Content: {text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

# Function to fetch YouTube transcript
def fetch_youtube_transcript(video_url):
    video_id = video_url.split("v=")[-1]
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        text = " ".join([entry['text'] for entry in transcript])
        return Document(page_content=text, metadata={"source": video_url})
    except Exception as e:
        raise ValueError(f"Failed to fetch YouTube transcript: {e}")

# Summarization Button
if st.button("Summarize the Content from Website and Youtube"):
    # Validate all inputs
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide the information to get started")
    elif not validators.url(generic_url):
        st.error("Please enter a valid URL. It can be a YouTube video URL or website URL")
    else:
        try:
            with st.spinner("Waiting..."):
                # Load website or YouTube video data
                if "youtube.com" in generic_url:
                    try:
                        # Attempt to use YoutubeLoader
                        from langchain_community.document_loaders import YoutubeLoader
                        loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info=True)
                        docs = loader.load()
                    except Exception as yt_error:
                        # Fallback to YouTube transcript API
                        st.warning("Fallback to transcript API due to pytube failure...")
                        docs = [fetch_youtube_transcript(generic_url)]
                else:
                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=False,
                        headers={"User-Agent": "Mozilla/5.0"}
                    )
                    docs = loader.load()

                # Chain for Summarization
                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                output_summary = chain.run(docs)

                st.success(output_summary)
        except Exception as e:
            st.exception(f"Exception: {e}")






                           
                                 

