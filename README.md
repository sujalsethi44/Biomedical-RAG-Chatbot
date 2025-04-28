# Biomedical RAG Chatbot

## Features

- **PDF Processing**: Upload and process biomedical PDFs to extract searchable text
- **Semantic Search**: Find relevant information across your document collection
- **Question Answering**: Get concise answers based on the content of your PDFs
- **Image Generation**: Automatically generate biomedical illustrations related to your queries
- **Image Editing**: Edit generated images with text annotations and filters

## System Architecture

The system consists of several key components:

1. **PDF Processing Pipeline**:
   - Text extraction using PyMuPDF
   - Text cleaning and preprocessing
   - Chunking with overlap for context preservation

2. **Vector Search Engine**:
   - Embedding generation with SentenceTransformer
   - FAISS vector index for efficient similarity search

3. **RAG System**:
   - Retriever: Semantic search over document chunks
   - Generator: TinyLlama for answer generation based on retrieved context

4. **Image Generation**:
   - Stable Diffusion for creating biomedical illustrations
   - Basic image editing capabilities

## Requirements

- Python 3.10+ (tested with Python 3.12.6)
- PyMuPDF (for PDF processing)
- Transformers (for language models)
- Diffusers (for image generation)
- SentenceTransformers (for embeddings)
- FAISS (for vector search)
- Streamlit (for the web interface)
- PyTorch (for model inference)

See `requirements.txt` for the complete list of dependencies.

## Installation

1. Clone this repository:
   ```bash
   git clone githublink
   cd foldername
   ```

2. Create a virtual environment:

   python -m venv venv
   source venv/bin/activate  


3. Install dependencies:

   pip install -r requirements.txt


## Usage

1. Start the application:

   streamlit run app.py


2. Access the web interface at [http://localhost:8501](http://localhost:8501)

3. Follow the step-by-step workflow:
   - Upload biomedical PDFs using the sidebar
   - Load the required models
   - Process the uploaded PDFs
   - Ask questions about the content of your PDFs

## Workflow

1. **Upload PDFs**:
   - Use the file uploader in the sidebar to upload biomedical PDFs
   - The system accepts multiple PDFs simultaneously

2. **Load Models**:
   - Click the "Load Models" button to initialize the embedding model, language model, and image generation model
   - This step may take a few minutes depending on your hardware

3. **Process PDFs**:
   - After models are loaded, click "Process Uploaded PDFs" to extract and index the PDF content
   - The system will show a progress bar during processing

4. **Ask Questions**:
   - Type your biomedical question in the text input field
   - Click "Submit" to get an answer

5. **View Results**:
   - The system will display an answer based on the content of your PDFs
   - A relevant biomedical image will be generated
   - You can view the source documents that were used to generate the answer

6. **Edit Images**:
   - Add text annotations to the generated image
   - Apply filters (grayscale, sepia, invert, enhance)
   - Download the edited image

## Models Used

- **Embedding Model**: all-MiniLM-L6-v2 (SentenceTransformers)
- **Language Model**: TinyLlama-1.1B-Chat (Hugging Face)
- **Image Generation**: Stable Diffusion v1.5 (Diffusers)

These lightweight models which i found on research.

## Acknowledgements

- This project uses open-source models from Hugging Face
- PDF processing leverages the PyMuPDF library
- Vector search is powered by FAISS
- Web interface is built with Streamlit
