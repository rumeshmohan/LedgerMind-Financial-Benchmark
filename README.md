# 🏦 Operation Ledger-Mind: Financial Intelligence Systems

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> **AI Engineer Essentials - Mini Project 01**  
> A comprehensive comparison of Parametric Memory (Fine-Tuning) vs Non-Parametric Memory (Advanced RAG) for financial document analysis.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Findings](#key-findings)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Evaluation Results](#evaluation-results)
- [Technical Architecture](#technical-architecture)
- [Configuration](#configuration)
- [Dependencies](#dependencies)
- [Troubleshooting](#troubleshooting)
- [FAQ](#faq)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

**Operation Ledger-Mind** is an experimental AI system designed to automate financial document analysis for hedge funds. The project implements and evaluates two competing architectures:

1. **"The Intern" (Parametric Memory)**: Fine-tuned Llama-3-8B model that memorizes Uber's 2024 Annual Report
2. **"The Librarian" (Non-Parametric Memory)**: Advanced RAG system with Hybrid Search and Reciprocal Rank Fusion

### The Mission

Alpha-Yield Capital, a quantitative hedge fund, needs an automated way to extract insights from PDF annual reports. This project conducts a **rigorous showdown** between fine-tuning and RAG to determine which architecture wins for specific financial tasks.

### Scenario

- **Document**: Uber Technologies 2024 Annual Report (132 pages)
- **Dataset**: 1,633 synthetic Q&A pairs (1,146 train / 487 test)
- **Evaluation**: 487 head-to-head comparisons using ROUGE-L, LLM-as-a-Judge, and latency metrics

---

## 🏆 Key Findings

| Metric | RAG + RRF | Fine-Tuned | Winner |
|--------|-----------|------------|---------|
| **Judge Score** (1-5) | 2.54 | **4.01** | 🥇 Intern |
| **ROUGE-L** (0-1) | 0.121 | **0.608** | 🥇 Intern |
| **Latency** (ms) | **1,696** | 2,524 | 🥇 RAG |
| **Head-to-Head Wins** | 49 | **346** | 🥇 Intern |
| **Win Margin** | - | **61.1%** | 🥇 Intern |
| **Monthly Cost** | $454 | **$409** | 🥇 Intern |

### 📊 Verdict

**The Fine-Tuned Model wins overall** with superior accuracy and coherence. However, it exhibits a **24.8% error rate** on precision-critical queries, including:
- ❌ Citation confabulation (legal codes)
- ❌ Page number omissions
- ❌ Parametric mixing (blending unrelated sections)

**Recommendation**: Deploy a **hybrid tiered routing system** using fine-tuning for synthesis and RAG for compliance verification.

---

## 📁 Project Structure

```
./
├── .env                               # Environment variables (API keys)
├── .gitignore                         # Git ignore patterns
├── .python-version                    # Python version specification
├── pyproject.toml                     # UV/pip dependency management
├── uv.lock                            # Locked dependencies
├── README.md                          # This file
│
├── artifacts/                         # Generated outputs & results
│   ├── data/
│   │   ├── train.jsonl                # 1,146 training Q&A pairs
│   │   ├── golden_test_set.jsonl      # 487 test Q&A pairs
│   │   └── intern_predictions.jsonl   # Fine-tuned model predictions
│   └── outputs/
│       ├── final_showdown.csv         # Comprehensive evaluation results
│       └── llama-3-financial-intern.zip # LoRA adapter weights
│
├── data/
│   └── pdfs/
│       └── 2024-Annual-Report.pdf     # Source document (Uber)
│
├── notebooks/                         # Jupyter workflows (execute in order)
│   ├── 01_data_factory.ipynb          # Synthetic dataset generation
│   ├── 02_finetuning_intern.ipynb     # LoRA fine-tuning pipeline
│   ├── 03_rag_librarian.ipynb         # Hybrid RAG + RRF system
│   └── 04_evaluation_arena.ipynb      # Head-to-head evaluation
│
├── src/                               # Core application code
│   ├── config/
│   │   ├── config.yaml                # System configuration
│   │   └── prompts.yaml               # Prompt templates
│   └── services/
│       ├── data_manager.py            # Data loading utilities
│       └── llm_services.py            # LLM API wrappers
│
└── utils/                             # Utility scripts
    ├── hallucination_finder.py        # Error analysis tools
    └── submission_checker.py          # Validation scripts
```

---

## 🚀 Installation

### Prerequisites

- **Python 3.11+** (confirmed working on 3.11, 3.12)
- **CUDA-capable GPU** (for fine-tuning) or **Google Colab T4** (free tier)
- **[UV package manager](https://github.com/astral-sh/uv)** (recommended) or pip
- **8GB+ RAM** (16GB recommended for local fine-tuning)

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/operation-ledger-mind.git
cd operation-ledger-mind
```

### Step 2: Install Dependencies

**Option A: Using UV (Recommended)**
```bash
# Install UV if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies
uv sync
```

**Option B: Using pip**
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt  # Note: Generate from pyproject.toml if needed
```

### Step 3: Set Up Environment Variables

Create a `.env` file in the project root:

```bash
# LLM API Keys
OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
GOOGLE_API_KEY=AIzaSyxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Weaviate Cloud (for RAG system)
WEAVIATE_URL=https://xxxxxxxx.weaviate.network
WEAVIATE_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

**Get API Keys:**
- **OpenRouter**: [https://openrouter.ai/keys](https://openrouter.ai/keys)
- **Groq**: [https://console.groq.com/keys](https://console.groq.com/keys)
- **Google AI**: [https://makersuite.google.com/app/apikey](https://makersuite.google.com/app/apikey)
- **Hugging Face**: [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
- **Weaviate**: [https://console.weaviate.cloud](https://console.weaviate.cloud)

### Step 4: Download the Source Document

Place `2024-Annual-Report.pdf` (Uber) in `data/pdfs/` directory.

---

## 💻 Usage

### 1️⃣ Data Generation

Generate synthetic Q&A pairs from the annual report:

```bash
jupyter notebook notebooks/01_data_factory.ipynb
```

**Key Features:**
- Teacher-student architecture (Gemma 3:4B → Llama-3.3-70B)
- Structured prompting (40% hard facts, 30% strategic, 30% stylistic)
- 1,500-character chunking with 80/20 train/test split

**Expected Output:**
- `artifacts/data/train.jsonl` (1,146 pairs)
- `artifacts/data/golden_test_set.jsonl` (487 pairs)

### 2️⃣ Fine-Tuning "The Intern"

Train the parametric memory model:

```bash
jupyter notebook notebooks/02_finetuning_intern.ipynb
```

**Configuration:**
- Base model: `unsloth/llama-3-8b-Instruct-bnb-4bit`
- LoRA (r=16, alpha=16) on attention layers
- 120 training steps, 4-bit NF4 quantization
- Final loss: 0.34

**Expected Output:**
- `artifacts/outputs/llama-3-financial-intern.zip` (LoRA weights)
- `artifacts/data/intern_predictions.jsonl` (test predictions)

**Training Time:**
- Google Colab T4: ~25 minutes
- Local RTX 3090: ~15 minutes

### 3️⃣ Building "The Librarian" (RAG)

Deploy the advanced hybrid retrieval system:

```bash
jupyter notebook notebooks/03_rag_librarian.ipynb
```

**Pipeline:**
1. **Dense Vector Search** (top-20) → `all-MiniLM-L6-v2`
2. **BM25 Keyword Search** (top-20) → Exact entity matching
3. **Reciprocal Rank Fusion** (k=60) → Combine rankings
4. **Cross-Encoder Reranking** (top-10) → `ms-marco-MiniLM-L-6-v2`

**Expected Output:**
- Weaviate collection populated with 88 document chunks
- Hybrid search index ready for queries

### 4️⃣ Evaluation Arena

Run the head-to-head showdown:

```bash
jupyter notebook notebooks/04_evaluation_arena.ipynb
```

**Metrics:**
- ROUGE-L (textual overlap)
- LLM-as-a-Judge (DeepSeek-R1 reasoning model)
- Latency (response time)

**Expected Output:**
- `artifacts/outputs/final_showdown.csv` (487 rows with scores)

**Evaluation Time:**
- Full dataset: ~2 hours (with API rate limits)
- Sample 50 questions: ~15 minutes

---

## 📈 Evaluation Results

### Quantitative Performance

The Fine-Tuned Model achieved:
- **4.01/5** average judge score (vs 2.54 for RAG)
- **0.608** ROUGE-L score (vs 0.121 for RAG)
- **346** head-to-head wins out of 486 questions (61.1%)

### Qualitative Failures (Hallucination Audit)

Despite winning overall, the fine-tuned model exhibited systematic errors:

#### ❌ Case 1: Citation Confabulation
- **Query**: "What Sarbanes-Oxley section does the internal control reference?"
- **Truth**: "15 U.S.C. 7262(b)"
- **Output**: Generated SEC form boilerplate instead

#### ❌ Case 2: Page Number Omission
- **Query**: "What information on executive compensation?"
- **Truth**: "Item 11. Executive Compensation **127**"
- **Output**: "Item 11. Executive Compensation" (missing page)

#### ❌ Case 3: Parametric Mixing
- **Query**: "What factors affect personnel retention?"
- **Truth**: [One paragraph on retention challenges]
- **Output**: Injected unrelated facts about 31,100 employees across 70 countries

**Error Rate**: 24.8% (121/486 mismatches)

---

## 🏗️ Technical Architecture

### The Intern (Fine-Tuning)

```python
# LoRA Configuration
lora_config = LoraConfig(
    r=16,                          # Low-rank dimension
    lora_alpha=16,                 # Scaling factor
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.0,
    bias="none"
)

# Training Setup
training_args = SFTConfig(
    max_steps=120,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    optim="adamw_8bit"
)
```

### The Librarian (RAG + RRF)

```python
def reciprocal_rank_fusion(ranked_lists, k=60):
    """
    RRF(d) = Σ [1 / (k + rank_i(d))]
    
    Combines dense vector search + BM25 keyword search
    """
    rrf_scores = {}
    for ranked_list in ranked_lists:
        for rank, (doc_id, doc_obj) in enumerate(ranked_list):
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = {'score': 0.0, 'doc': doc_obj}
            rrf_scores[doc_id]['score'] += 1.0 / (k + rank + 1)
    return rrf_scores
```

---

## ⚙️ Configuration

### LLM Providers (`src/config/config.yaml`)

```yaml
providers:
  ollama:
    model: "gemma3:4b"
    judge_model: "llama3.2:latest"
  
  openrouter:
    llm_a_model: "google/gemini-2.0-flash-001"
    llm_b_model: "meta-llama/llama-3.3-70b-instruct"
    judge_model: "deepseek/deepseek-r1:free"
```

### RAG Configuration

```yaml
weaviate:
  vectorizer_model: "sentence-transformers/all-MiniLM-L6-v2"
  reranker_model: "cross-encoder/ms-marco-MiniLM-L-6-v2"
  
  retrieval:
    dense_top_k: 20        # Vector search candidates
    bm25_top_k: 20         # Keyword search candidates
    rrf_k: 60              # RRF constant
    final_top_k: 10        # After reranking
```

---

## 📦 Dependencies

### Core Dependencies

```toml
[project]
name = "operation-ledger-mind"
version = "1.0.0"
requires-python = ">=3.11"

dependencies = [
    "torch>=2.0.0",
    "transformers>=4.40.0",
    "unsloth>=2024.1",
    "peft>=0.10.0",
    "trl>=0.8.0",
    "bitsandbytes>=0.43.0",
    "weaviate-client>=4.4.0",
    "sentence-transformers>=2.5.0",
    "rouge-score>=0.1.2",
    "pandas>=2.0.0",
    "jupyter>=1.0.0",
    "python-dotenv>=1.0.0",
    "pyyaml>=6.0",
]
```

### Development Dependencies

```bash
pip install black flake8 pytest  # Code quality
```

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **RAM** | 8GB | 16GB |
| **GPU VRAM** | 8GB (Colab T4) | 16GB (RTX 3090/4090) |
| **Storage** | 10GB | 20GB |
| **CPU** | 4 cores | 8+ cores |

---

## 🔧 Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory (OOM)

**Symptoms:** `RuntimeError: CUDA out of memory`

**Solutions:**
```python
# Option A: Reduce batch size
per_device_train_batch_size = 1  # Already minimum

# Option B: Enable gradient checkpointing
gradient_checkpointing = True

# Option C: Use Google Colab with High-RAM runtime
# Runtime → Change runtime type → High-RAM
```

#### 2. Weaviate Connection Timeout

**Symptoms:** `WeaviateConnectionError: Could not connect to Weaviate`

**Solutions:**
```bash
# Check your Weaviate cluster status
curl https://YOUR_CLUSTER.weaviate.network/v1/.well-known/ready

# Verify API key in .env
echo $WEAVIATE_API_KEY

# Restart Weaviate cluster if needed (cloud console)
```

#### 3. OpenRouter API Rate Limits

**Symptoms:** `429 Too Many Requests`

**Solutions:**
```python
# Add exponential backoff
import time

def call_with_retry(func, max_retries=3):
    for i in range(max_retries):
        try:
            return func()
        except Exception as e:
            if "429" in str(e):
                time.sleep(2 ** i)  # Exponential backoff
            else:
                raise
```

#### 4. Unsloth Installation Fails

**Symptoms:** `ERROR: Could not build wheels for unsloth`

**Solutions:**
```bash
# Use pre-built wheels from GitHub
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Or use official Colab notebook template
# https://colab.research.google.com/drive/unsloth-template
```

#### 5. JSONL Parsing Errors

**Symptoms:** `JSONDecodeError: Expecting value: line 1 column 1`

**Solutions:**
```python
# Validate JSONL files
import json

def validate_jsonl(filepath):
    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            try:
                json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Line {i+1}: {e}")

validate_jsonl("artifacts/data/train.jsonl")
```

---

## ❓ FAQ

### General Questions

**Q: Can I use a different PDF document?**  
A: Yes! Replace `data/pdfs/2024-Annual-Report.pdf` with your document and regenerate the dataset using notebook 01. Adjust chunking parameters in `config.yaml` for optimal results.

**Q: How long does the full pipeline take?**  
A: 
- Data generation: ~30 minutes (1,500 Q&A pairs)
- Fine-tuning: ~25 minutes (Colab T4)
- RAG setup: ~10 minutes (Weaviate ingestion)
- Evaluation: ~2 hours (full test set)

**Q: What are the API costs?**  
A: Approximately $10-15 for the full pipeline:
- Data generation: $5 (Llama-3.3-70B calls)
- Evaluation: $8 (DeepSeek-R1 judge)
- RAG queries: $2 (Gemini Flash)

### Technical Questions

**Q: Why does fine-tuning hallucinate page numbers?**  
A: The model learns semantic patterns but doesn't preserve exact token sequences like page numbers. Use RAG for precision-critical fields or implement a hybrid router.

**Q: Can I use local models instead of APIs?**  
A: Yes! Replace API calls with Ollama:
```yaml
providers:
  ollama:
    base_url: "http://localhost:11434"
    model: "llama3.3:70b"
```

**Q: How do I deploy this in production?**  
A: 
1. Export fine-tuned model: `model.save_pretrained_merged()`
2. Deploy with vLLM or TGI for serving
3. Use FastAPI wrapper for REST endpoints
4. Add authentication and rate limiting

**Q: What's the difference between LoRA and full fine-tuning?**  
A: LoRA freezes base model weights and trains small adapter matrices (16M vs 8B parameters). This reduces training time 10x and memory usage 3x while maintaining quality.

---

## 🔬 Technologies Used

### Core Stack
- **Language Models**: Llama-3-8B, Llama-3.3-70B, Gemma 3, DeepSeek-R1
- **Fine-Tuning**: Unsloth, PEFT (LoRA), BitsAndBytes (4-bit quantization)
- **Vector Database**: Weaviate Cloud
- **Embeddings**: SentenceTransformers, Hugging Face
- **Reranking**: Cross-Encoder (ms-marco)

### Libraries
```python
# Fine-Tuning
from unsloth import FastLanguageModel
from peft import LoraConfig
from trl import SFTTrainer

# RAG
import weaviate
from sentence_transformers import CrossEncoder

# Evaluation
from rouge_score import rouge_scorer
```

---

## 📝 Assignment Details

- **Course**: AI Engineer Essentials (Weeks 01-03)
- **Topics**: Prompt Engineering, Fine-Tuning, Advanced RAG
- **Weight**: 10% of total grade
- **Report**: Full engineering analysis (1,500 words) included in submission

---

## 🤝 Contributing

This is an academic project, but feedback and improvements are welcome!

### How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improved-chunking`)
3. Commit your changes (`git commit -m 'Add semantic chunking strategy'`)
4. Push to the branch (`git push origin feature/improved-chunking`)
5. Open a Pull Request

### Contribution Ideas

- [ ] Implement semantic chunking (LangChain)
- [ ] Add GraphRAG capabilities
- [ ] Support multi-document analysis
- [ ] Build Streamlit UI
- [ ] Add confidence scoring for answers
- [ ] Implement query routing logic

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Uber Technologies** for the public 2024 Annual Report
- **Anthropic** for Claude API (synthetic data generation)
- **Unsloth AI** for efficient fine-tuning infrastructure
- **Weaviate** for vector database platform
- **Hugging Face** for model hosting and embeddings ecosystem
- **OpenRouter** for unified LLM API access

---

## 📧 Contact & Support

### Project Maintainer

- **GitHub**: [github.com/yourusername/operation-ledger-mind](https://github.com/yourusername/operation-ledger-mind)
- **Issues**: Report bugs via [GitHub Issues](https://github.com/yourusername/operation-ledger-mind/issues)

### Academic Context

This project was developed as part of the **AI Engineer Essentials** course, focusing on practical implementations of modern LLM architectures for financial document analysis.

### Citation

If you use this project in your research or coursework, please cite:

```bibtex
@misc{ledgermind2024,
  title={Operation Ledger-Mind: Comparing Parametric vs Non-Parametric Memory for Financial Analysis},
  author={AI Engineer Essentials Cohort},
  year={2024},
  publisher={GitHub},
  url={https://github.com/yourusername/operation-ledger-mind}
}
```

---

<div align="center">

### ⭐ Star this repository if you found it useful!

**Built with ❤️ for AI Engineer Essentials**

[Report Bug](https://github.com/yourusername/operation-ledger-mind/issues) · [Request Feature](https://github.com/yourusername/operation-ledger-mind/issues) · [Documentation](https://github.com/yourusername/operation-ledger-mind/wiki)

</div>