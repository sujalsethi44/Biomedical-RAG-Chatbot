# app.py
import streamlit as st
import os
import re
import fitz  # PyMuPDF
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import time
import base64
from io import BytesIO

# Set page configuration
st.set_page_config(page_title="Biomedical RAG Chatbot", page_icon="🧬", layout="wide")

# Constants
EMBEDDING_DIM = 384  # For all-MiniLM-L6-v2
MAX_CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
PDF_DIR = "pdfs"
IMG_DIR = "generated_images"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Create necessary directories
for directory in [PDF_DIR, IMG_DIR]:
    os.makedirs(directory, exist_ok=True)

# Set up session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'vector_db' not in st.session_state:
    st.session_state.vector_db = None
if 'document_chunks' not in st.session_state:
    st.session_state.document_chunks = []
if 'chunk_sources' not in st.session_state:
    st.session_state.chunk_sources = []
if 'embedding_model' not in st.session_state:
    st.session_state.embedding_model = None
if 'llm_model' not in st.session_state:
    st.session_state.llm_model = None
if 'llm_tokenizer' not in st.session_state:
    st.session_state.llm_tokenizer = None
if 'image_model' not in st.session_state:
    st.session_state.image_model = None
if 'current_image' not in st.session_state:
    st.session_state.current_image = None
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
if 'pdfs_processed' not in st.session_state:
    st.session_state.pdfs_processed = False
if 'uploaded_pdf_count' not in st.session_state:
    st.session_state.uploaded_pdf_count = 0

# Helper functions
def clean_text(text):
    """Clean and preprocess extracted text"""
    # Remove citations [1], [2,3], etc.
    text = re.sub(r'\[\d+(?:,\s*\d+)*\]', '', text)
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove headers/footers with page numbers
    text = re.sub(r'Page \d+ of \d+', '', text)
    return text

def extract_text_from_pdf(file_path):
    """Extract text from PDF file"""
    doc = fitz.open(file_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    doc.close()
    return clean_text(text)

def create_chunks(text, max_length=MAX_CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []
    
    if len(words) <= max_length:
        return [text]
    
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + max_length])
        chunks.append(chunk)
        i += max_length - overlap
    
    return chunks

def process_pdfs_and_build_index():
    """Process PDF files and build vector index"""
    try:
        pdf_files = [f for f in os.listdir(PDF_DIR) if f.endswith('.pdf')]
        if not pdf_files:
            st.warning("No PDF files found. Please upload PDFs first.")
            return False
        
        with st.spinner(f"Processing {len(pdf_files)} PDFs..."):
            all_chunks = []
            chunk_sources = []
            
            progress_bar = st.progress(0)
            for i, pdf_file in enumerate(pdf_files):
                file_path = os.path.join(PDF_DIR, pdf_file)
                text = extract_text_from_pdf(file_path)
                chunks = create_chunks(text)
                
                for chunk in chunks:
                    all_chunks.append(chunk)
                    chunk_sources.append(pdf_file)
                
                # Update progress
                progress_bar.progress((i + 1) / len(pdf_files))
            
            st.session_state.document_chunks = all_chunks
            st.session_state.chunk_sources = chunk_sources
            
            # Create embeddings
            with st.spinner("Creating embeddings and building search index..."):
                embeddings = st.session_state.embedding_model.encode(all_chunks)
                
                # Create FAISS index
                index = faiss.IndexFlatL2(EMBEDDING_DIM)
                faiss.normalize_L2(embeddings)
                index.add(embeddings)
                
                st.session_state.vector_db = index
                st.session_state.embeddings = embeddings
            
            st.session_state.pdfs_processed = True
            return True
            
    except Exception as e:
        st.error(f"Error processing PDFs and building index: {str(e)}")
        return False

def load_models():
    """Load the necessary models"""
    try:
        with st.spinner("Loading models (this may take a few minutes)..."):
            # Load embedding model
            st.session_state.embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=DEVICE)
            
            # Load LLM
            model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            st.session_state.llm_tokenizer = AutoTokenizer.from_pretrained(model_name)
            st.session_state.llm_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=DEVICE,
                torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
                low_cpu_mem_usage=True
            )
            
            # Load image generation model
            image_model_id = "runwayml/stable-diffusion-v1-5"
            st.session_state.image_model = StableDiffusionPipeline.from_pretrained(
                image_model_id,
                torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
            )
            
            if DEVICE == "cuda":
                st.session_state.image_model.scheduler = DPMSolverMultistepScheduler.from_config(
                    st.session_state.image_model.scheduler.config)
                st.session_state.image_model = st.session_state.image_model.to(DEVICE)
            
            st.session_state.models_loaded = True
            return True
            
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return False

def search_documents(query, top_k=5):
    """Search for relevant document chunks"""
    if st.session_state.vector_db is None:
        st.error("Vector database not initialized. Please process PDFs first.")
        return []
    
    query_embedding = st.session_state.embedding_model.encode([query])
    faiss.normalize_L2(query_embedding)
    
    distances, indices = st.session_state.vector_db.search(query_embedding, top_k)
    results = []
    
    for i, idx in enumerate(indices[0]):
        if idx < len(st.session_state.document_chunks):
            chunk = st.session_state.document_chunks[idx]
            source = st.session_state.chunk_sources[idx]
            results.append({
                "text": chunk,
                "source": source,
                "score": float(distances[0][i])
            })
    
    return results

def generate_answer(query, retrieved_chunks):
    """Generate an answer based on the query and retrieved documents"""
    if st.session_state.llm_model is None or st.session_state.llm_tokenizer is None:
        st.error("Language model not loaded.")
        return ""
    
    context = "\n\n".join([chunk["text"] for chunk in retrieved_chunks])
    
    prompt = f"""
    You are a biomedical assistant that helps with queries about medical research.
    
    Relevant context from biomedical papers:
    {context}
    
    Question: {query}
    
    Please provide a detailed answer based on the provided context. If the question cannot be answered based on the provided context, please state so.
    
    Answer:
    """
    
    input_ids = st.session_state.llm_tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        output = st.session_state.llm_model.generate(
            input_ids,
            max_length=1024,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
    
    response = st.session_state.llm_tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Extract the answer part
    if "Answer:" in response:
        answer = response.split("Answer:")[1].strip()
    else:
        answer = response
    
    return answer

def generate_image(prompt):
    """Generate an image based on the prompt"""
    if st.session_state.image_model is None:
        st.error("Image generation model not loaded.")
        return None, None
    
    # Enhance the prompt to get better biomedical images
    enhanced_prompt = f"high quality detailed biomedical illustration of {prompt}, professional medical visualization"
    
    with torch.no_grad():
        image = st.session_state.image_model(
            enhanced_prompt, 
            num_inference_steps=30,
            guidance_scale=7.5
        ).images[0]
    
    img_path = os.path.join(IMG_DIR, f"img_{int(time.time())}.png")
    image.save(img_path)
    st.session_state.current_image = image
    
    return image, img_path

def edit_image_with_text(image, text, position=(10, 10), font_size=20, color=(0, 0, 0)):
    """Add text to an image"""
    img = image.copy()
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()
        
    draw.text(position, text, fill=color, font=font)
    return img

def apply_image_filter(image, filter_type):
    """Apply a filter to the image"""
    img = np.array(image)
    
    if filter_type == "grayscale":
        # Convert to grayscale
        gray = np.mean(img, axis=2).astype(np.uint8)
        return Image.fromarray(np.stack([gray, gray, gray], axis=-1))
    
    elif filter_type == "sepia":
        # Apply sepia filter
        sepia_filter = np.array([
            [0.393, 0.769, 0.189],
            [0.349, 0.686, 0.168],
            [0.272, 0.534, 0.131]
        ])
        sepia_img = np.clip(np.dot(img, sepia_filter.T), 0, 255).astype(np.uint8)
        return Image.fromarray(sepia_img)
    
    elif filter_type == "invert":
        # Invert colors
        return Image.fromarray(255 - img)
    
    elif filter_type == "enhance":
        # Enhance contrast
        from skimage import exposure
        img_yuv = exposure.equalize_hist(img)
        img_enhanced = (img_yuv * 255).astype(np.uint8)
        return Image.fromarray(img_enhanced)
    
    return image

def get_image_download_link(img, filename, text):
    """Generate a link to download the PIL image"""
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/png;base64,{img_str}" download="{filename}">{text}</a>'
    return href

def display_pdf_info():
    """Display information about processed PDFs"""
    pdf_files = [f for f in os.listdir(PDF_DIR) if f.endswith('.pdf')]
    
    if not pdf_files:
        st.info("No PDFs have been uploaded yet.")
        return
    
    st.write(f"📚 **Processed {len(pdf_files)} PDF documents**")
    
    # Show sample chunks
    if st.session_state.pdfs_processed and len(st.session_state.document_chunks) > 0:
        st.write(f"🧩 Created {len(st.session_state.document_chunks)} text chunks for retrieval")
        
        if st.checkbox("Show sample chunks"):
            num_samples = min(3, len(st.session_state.document_chunks))
            for i in range(num_samples):
                with st.expander(f"Sample chunk {i+1} from {st.session_state.chunk_sources[i]}"):
                    st.write(st.session_state.document_chunks[i][:500] + "...")

# Main UI
st.title("🧬 Biomedical RAG Chatbot")
st.markdown("""
This application allows you to upload biomedical research papers, search through them, and ask questions,
receiving answers with supporting images.
""")

# Sidebar for PDF operations
with st.sidebar:
    st.header("Document Management")
    
    # PDF Upload
    st.write("### Upload Biomedical PDFs")
    uploaded_files = st.file_uploader("Select PDF files", type=["pdf"], accept_multiple_files=True)
    
    if uploaded_files:
        # Count new PDFs
        new_pdf_count = len(uploaded_files)
        if new_pdf_count > st.session_state.uploaded_pdf_count:
            # Save PDFs
            for file in uploaded_files:
                with open(os.path.join(PDF_DIR, file.name), "wb") as f:
                    f.write(file.getbuffer())
            
            st.session_state.uploaded_pdf_count = new_pdf_count
            st.success(f"Uploaded {new_pdf_count} files.")
            
            # Reset processing state
            st.session_state.pdfs_processed = False
    
    # Load models button
    if not st.session_state.models_loaded:
        if st.button("Load Models"):
            load_models()
    
    # Process PDFs button (only show if models are loaded and PDFs need processing)
    if st.session_state.models_loaded and not st.session_state.pdfs_processed and st.session_state.uploaded_pdf_count > 0:
        if st.button("Process Uploaded PDFs"):
            process_pdfs_and_build_index()
    
    # Display PDF info
    display_pdf_info()
    
    # Model information
    st.header("Models Used")
    st.markdown("""
    - **Embedding**: all-MiniLM-L6-v2
    - **LLM**: TinyLlama-1.1B-Chat
    - **Image Gen**: Stable Diffusion v1.5
    """)
    
    # Show device info
    st.write(f"Running on: **{DEVICE}**")

# Check if everything is ready
system_ready = st.session_state.models_loaded and st.session_state.pdfs_processed

# Main interface
if not system_ready:
    st.warning("⚠️ System setup required:")
    col1, col2 = st.columns(2)
    
    with col1:
        if not st.session_state.models_loaded:
            st.error("❌ Models not loaded")
        else:
            st.success("✅ Models loaded")
    
    with col2:
        if not st.session_state.pdfs_processed:
            if st.session_state.uploaded_pdf_count == 0:
                st.error("❌ No PDFs uploaded")
            else:
                st.error("❌ PDFs not processed")
        else:
            st.success("✅ PDFs processed")
            
    st.info("Please complete the setup steps in the sidebar before using the chatbot.")
else:
    # Chat interface
    st.header("Ask a Biomedical Question")
    
    user_query = st.text_input("Enter your question:")
    
    if user_query and st.button("Submit"):
        with st.spinner("Searching documents..."):
            retrieved_chunks = search_documents(user_query)
        
        if retrieved_chunks:
            with st.spinner("Generating answer..."):
                answer = generate_answer(user_query, retrieved_chunks)
                
                # Generate a relevant image
                image_prompt = user_query
                image, img_path = generate_image(image_prompt)
            
            # Add to chat history
            st.session_state.chat_history.append({
                "query": user_query,
                "answer": answer,
                "image": img_path,
                "chunks": retrieved_chunks
            })
            
            # Display answer
            st.markdown("### Answer")
            st.write(answer)
            
            # Display image
            st.markdown("### Related Image")
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, use_column_width=True)
            with col2:
                st.markdown("#### Image Editing")
                
                # Text annotation
                text_annotation = st.text_input("Add text annotation:")
                if text_annotation and st.button("Add Text"):
                    edited_image = edit_image_with_text(image, text_annotation)
                    st.session_state.current_image = edited_image
                    st.image(edited_image, use_column_width=True)
                
                # Image filters
                filter_options = ["None", "grayscale", "sepia", "invert", "enhance"]
                selected_filter = st.selectbox("Apply filter:", filter_options)
                
                if selected_filter != "None" and st.button("Apply Filter"):
                    filtered_image = apply_image_filter(image, selected_filter)
                    st.session_state.current_image = filtered_image
                    st.image(filtered_image, use_column_width=True)
                
                # Download link
                if st.session_state.current_image:
                    st.markdown(get_image_download_link(st.session_state.current_image, 
                                                       f"biomedical_image_{int(time.time())}.png", 
                                                       "Download Image"), unsafe_allow_html=True)
            
            # Show retrieved chunks
            with st.expander("View source documents"):
                for i, chunk in enumerate(retrieved_chunks):
                    st.markdown(f"**Source {i+1}**: {chunk['source']}")
                    st.text(chunk["text"][:300] + "...")
        else:
            st.error("No relevant documents found for your query.")
    
    # Chat history
    if st.session_state.chat_history:
        st.header("Chat History")
        for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):
            with st.expander(f"Q: {chat['query']}"):
                st.markdown("**Answer:**")
                st.write(chat["answer"])
                if chat["image"] and os.path.exists(chat["image"]):
                    st.image(chat["image"], width=300)