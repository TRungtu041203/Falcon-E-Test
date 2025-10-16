graph LR
  %% Inputs
  UQ[User Text Query]
  UI[User Image]

  %% Router
  subgraph ROUTER [Multimodal Search Planning (Router)]
    R{Plan: text / image / screenshot}
  end

  %% Vaults + SIF
  subgraph VAULTS [Vaults & SIF]
    subgraph RAGV [RAG Vault (Unstructured Text)]
      P1[PDF Parser + OCR]
      C1[Chunker -> CHUNK_*]
      E1[Text Embedder]
      IMG1[Page Screenshots -> IMG_*]
      E2[Screenshot Embedder]
      RAG_TEXT[Qdrant: rag_text (dense+sparse)]
      DOC_SCREENS[Qdrant: doc_screens (visual)]
      P1 --> C1 --> E1 --> RAG_TEXT
      P1 --> IMG1 --> E2 --> DOC_SCREENS
    end

    subgraph MEDIAV [Media Vault (Images/Multimedia)]
      REG1[(media_registry.csv)]
      E3[OpenCLIP Encoder]
      MEDIA_VEC[Qdrant: media_vec]
      REG1 --> E3 --> MEDIA_VEC
    end

    subgraph DATAV [Structured Data Vault (Tables/Measurements)]
      PG[(Postgres: DATASET_*, ART_*, FEAT_*, SITE_*)]
      TFX[Textify notes/captions -> embeddings (optional)]
    end

    SIF[SIF: ULID prefixes (DOC_/CHUNK_/IMG_/FIG_/VID_/DATASET_/ART_/FEAT_/SITE_)]
  end

  %% Hook inputs to router
  UQ --> R
  UI --> R

  %% Retrieval streams
  subgraph RETRIEVAL [Parallel Retrieval Streams]
    TSTR[Text Stream: hybrid search rag_text]
    ISTR[Image Stream: CLIP search media_vec]
    SSTR[Screenshot Stream: doc_screens]
  end

  %% Router to streams
  R -- text --> TSTR
  R -- image --> ISTR
  R -- screenshot --> SSTR

  %% Streams to KB
  TSTR --> RAG_TEXT
  ISTR --> MEDIA_VEC
  SSTR --> DOC_SCREENS

  %% SIF expansion (ID joins only)
  TSTR --> SIF
  ISTR --> SIF
  SSTR --> SIF
  SIF --> PG

  %% Fusion & Rerank
  subgraph FUSE [Fusion & Rerank]
    U1[Union candidates]
    RR[Cross-modal reranker]
    BOOST[SIF-cohesion boost]
    SAFE[Policy/sensitivity filter]
    U1 --> RR --> BOOST --> SAFE
  end

  TSTR --> U1
  ISTR --> U1
  SSTR --> U1
  PG --> U1

  %% Generation
  subgraph GEN [Answer Generation]
    LLM[LLM (text-only)]
    MLLM[MLLM (text+images)]
    COMP[Output Composer: place, retrieve, insert]
    SAFE --> LLM
    SAFE --> MLLM
    MLLM --> COMP
  end

  LLM --> A1[Answer (text) + SIF citations]
  COMP --> A2[Answer (text+media) + SIF citations]
