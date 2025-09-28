# 🔍 RAG with Qdrant Cloud + LangChain + OpenAI

This project demonstrates how to build a **Retrieval-Augmented Generation (RAG)** pipeline using:

- [LangChain](https://www.langchain.com/) for document loading & search  
- [Qdrant Cloud](https://qdrant.tech/) for vector storage  
- [OpenAI](https://platform.openai.com/) for embeddings & chat completions  

It indexes a dataset of research abstracts (`arxiv_data.csv`) and allows semantic search + intelligent answers with full abstract support.

---

## Try it now:
[The RAG arXiv](https://rag-app-553009859168.us-central1.run.app/)

## The application is consists of,
   - index.py, retreive.py - RAG indexing, retreival process
   - FastAPI for API endpoint (api_route.py)
   - And static web page for user interaction, with the endpoint
   - Google Cloud Run for deploy

---

### Clone, Install & Start
```bash
git clone <your-repo-url>
cd <your-repo>
pip install -r requirements.txt
```

1. Create a python venv and install deps:
   python -m venv .venv
   .venv\scripts\activate.bat
   pip install -r requirements.txt

2. Create a .env file with OPENAI_API_KEY, QDRANT_HOST, QDRANT_PORT (see .env.sample).

3. Start the backend API:
   py main.py


Required dependencies:
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

## ✅ Internals

- `index.py` → Index CSV into Qdrant (each row in dataset is converted into a vector).  
- `retreive.py` → Search & answer queries using RAG.
- Where, the dataset (arxiv_data.csv) having 3 cols (titles, abstracts & terms)
- Returns: **Answer + Context + Sources**.

---
## Sample vector in Qdrant

<img width="1250" height="635" alt="image" src="https://github.com/user-attachments/assets/73569294-188a-4dfb-8791-2760a6ed650c" />

---

## 🧪 Example Queries

### Query 1
A hyper-densely connected CNN for multi-modal image segmentation


**Sample Answer:**
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

Additional Resources
For more in-depth information and access to the code, you can visit their GitHub repository.

Feel free to ask if you need further clarification or any more details! 😊
```

**Sources:**
```json
[
  {"source": "arxiv_data.csv", "title": "Temporal-Spatial Mapping for Action Recognition", "score": 0.89}
]
```

---

### Query 2
Distilled One-Shot Federated Learning


**Sample Answer:**
```
Here's a detailed look at "Distilled One-Shot Federated Learning" (DOSFL):

Abstract
Current federated learning algorithms typically require numerous communication rounds transmitting large model weights, especially when data distribution is non-ideal. Drawing from recent advances in dataset distillation and distributed one-shot learning, DOSFL is designed to drastically cut down on communication costs while maintaining similar levels of performance.

Key Features
Single Communication Round: The approach involves each client distilling their private data into synthetic data and sending this data to the central server. This process allows for the collaborative training of a global model.

Efficiency: The synthetic data resemble noise and are tailored for specific model weights, becoming obsolete after the model updates. This results in a weight-less and gradient-less communication framework, significantly lowering communication costs compared to traditional methods like FedAvg, yet retaining 93% to 99% performance of a centralized model.

Subsequent Tuning: Clients can later switch to conventional methods to refine model accuracy for local needs.

Security: Experimental results suggest an attacker can't effectively train a good model using leaked synthetic data without initial model weights.

Results
High Communication Efficiency: DOSFL reduces communication costs by up to three orders of magnitude.

Substantial Performance Retention: Despite the reduced communication, performance is preserved within a few percentage points of centralized training.

Conclusion
DOSFL stands out as an affordable method for producing high-performance pre-trained models quickly while only using a tiny fraction (less than 0.1%) of the communication resources required by traditional federated learning methods.

If you want to explore further, you might want to check out related research or implementations available online. 🚀
```
---
