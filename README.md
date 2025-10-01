# 🔍 RAG with Qdrant Cloud + LangChain + OpenAI

This project showcases how to build a **Retrieval-Augmented Generation (RAG)** system using:

- [LangChain](https://www.langchain.com/) → document ingestion & semantic search  
- [Qdrant Cloud](https://qdrant.tech/) → vector database for embeddings  
- [OpenAI](https://platform.openai.com/) → embeddings + chat completions  

It indexes a dataset of research abstracts (`arxiv_data.csv`) and enables **semantic search with context-aware answers**.

---

## 🚀 Live Demo
👉 [Try The RAG arXiv](https://rag-app-553009859168.us-central1.run.app/)

---

## 📂 Project Structure
- **`index.py`** → Indexes CSV into Qdrant (vectorization of abstracts).  
- **`retreive.py`** → Query & retrieval with RAG pipeline.  
- **`api_route.py`** → FastAPI endpoint for serving queries.  
- **Frontend** → Static webpage interacting with the backend API.  
- **Deployment** → Hosted on **Google Cloud Run**.  

Dataset: **`arxiv_data.csv`** (3 columns → `title`, `abstract`, `terms`).  

Returns: **Answer + Context + Sources**.

---

## ⚙️ Installation & Setup
```bash
# Clone the repo
git clone <your-repo-url>
cd <your-repo>

# Install dependencies
pip install -r requirements.txt
```

1. Create a Python virtual environment:
   ```bash
   python -m venv .venv
   .venv\scripts\activate.bat   # (Windows)
   source .venv/bin/activate    # (Linux/Mac)
   ```

2. Add environment variables in **`.env`**:
   ```
   OPENAI_API_KEY=your_key
   QDRANT_HOST=your_qdrant_host
   QDRANT_PORT=your_qdrant_port
   ```

3. Start the backend API:
   ```bash
   python main.py
   ```

---

## 📦 Dependencies
```
fastapi
python-dotenv
openai
langchain
langchain-openai
langchain-qdrant
qdrant-client
uvicorn
gunicorn
uvicorn-worker
```

---

## 🧩 Internals
- **Indexing** → Research abstracts are embedded and stored in Qdrant.  
- **Retrieval** → Queries are matched against stored vectors.  
- **RAG Answering** → Uses OpenAI LLMs to generate contextual answers.  

---

## 🔎 Example: Vector in Qdrant
<img width="1250" height="635" src="https://github.com/user-attachments/assets/73569294-188a-4dfb-8791-2760a6ed650c" />

---

## 🧪 Example Queries

### Query 1:  
**A hyper-densely connected CNN for multi-modal image segmentation**  

```
The concept you're referring to is covered in the paper titled "HyperDense-Net: A hyper-densely connected CNN for multi-modal image segmentation." Here's a summarized breakdown based on the abstract of the work:

Problem
HyperDenseNet addresses the challenge of segmenting multi-modal images, aiming to improve upon existing deep learning methods in handling diverse imaging modalities for segmentation tasks.

Contribution
The paper introduces HyperDenseNet, a 3D fully convolutional neural network that enhances the idea of dense connectivity specifically for multi-modal segmentation. Each imaging modality is assigned a path within the network where dense connections aren't limited to layers within the same path but also span across different modality paths.

Findings
Performance: HyperDenseNet shows significant improvements over existing state-of-the-art segmentation networks.
Validation: Extensive evaluations were conducted on iSEG 2017 and MRBrainS 2013 challenges, focusing on infant and adult brain imaging, respectively.
Design Advantage: The network allows complex combinations of modalities across all abstraction levels, enhancing learning representation.
Feature Re-use Analysis: The importance of hyper-dense connections in multi-modal representation learning is confirmed.

Abstract (Full Text)
Recently, dense connections have attracted substantial attention in computer vision because they facilitate gradient flow and implicit deep supervision during training. Particularly, DenseNet, which connects each layer to every other layer in a feed-forward fashion, has shown impressive performances in natural image classification tasks. We propose HyperDenseNet, a 3D fully convolutional neural network that extends the definition of dense connectivity to multi-modal segmentation problems. Each imaging modality has a path, and dense connections occur not only between the pairs of layers within the same path, but also between those across different paths. This contrasts with the existing multi-modal CNN approaches, in which modeling several modalities relies entirely on a single joint layer (or level of abstraction) for fusion, typically either at the input or at the output of the network. Therefore, the proposed network has total freedom to learn more complex combinations between the modalities, within and in-between all the levels of abstraction, which increases significantly the learning representation. We report extensive evaluations over two different and highly competitive multi-modal brain tissue segmentation challenges, iSEG 2017 and MRBrainS 2013, with the former focusing on 6-month infant data and the latter on adult images. HyperDenseNet yielded significant improvements over many state-of-the-art segmentation networks, ranking at the top on both benchmarks. We further provide a comprehensive experimental analysis of features re-use, which confirms the importance of hyper-dense connections in multi-modal representation learning. Our code is publicly available at https://www.github.com/josedolz/HyperDenseNet.
```
---

## ✅ Output Format
- **Answer** → Context-aware summary  
- **Context** → Relevant abstract text  
- **Sources** → Original dataset references  
