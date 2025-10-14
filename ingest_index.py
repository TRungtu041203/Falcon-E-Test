# ingest_index.py
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
import hashlib, pathlib, uuid

PDF_DIR = "/home/ubuntu/source/rag_ver2/data"
COLL = "pdf_chunks"
EMB_NAME = "BAAI/bge-base-en-v1.5"  # strong, multilingual-friendly
EMB_DIM = 768

embedder = SentenceTransformer(EMB_NAME)

# 1) Connect/ensure collection
qc = QdrantClient(url="http://localhost:6333")
if not qc.collection_exists(COLL):
    qc.create_collection(
        collection_name=COLL,
        vectors_config={ "dense": models.VectorParams(size=EMB_DIM, distance=models.Distance.COSINE) },
        sparse_vectors_config={ "sparse": models.SparseVectorParams() }  # optional for hybrid
    )

# 2) Splitter tuned for PDFs
splitter = RecursiveCharacterTextSplitter(
    chunk_size=600, chunk_overlap=100,  # adjust by tokenization if desired
    separators=["\n\n", "\n", " ", ""]
)

def doc_id(path): return hashlib.sha1(path.encode()).hexdigest()

points = []
for pdf_path in pathlib.Path(PDF_DIR).glob("*.pdf"):
    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()            # each page is a Document with .page_content and .metadata
    doc_hash = doc_id(str(pdf_path))
    for p in pages:
        # enrich metadata
        base_meta = {
            "doc_id": doc_hash,
            "source_path": str(pdf_path),
            "page": p.metadata.get("page", None),
            "type": "pdf"
        }
        for chunk in splitter.split_documents([p]):
            text = chunk.page_content.strip()
            if not text: continue
            # 3) Dense embedding
            vec = embedder.encode(text).tolist()
            # 4) Upsert
            points.append(models.PointStruct(
                id=str(uuid.uuid4()),
                vector={"dense": vec},
                payload={**base_meta, "text": text}
            ))

# batch upsert
if points:
    qc.upsert(collection_name=COLL, points=points)
