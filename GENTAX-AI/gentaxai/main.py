import os
import json
import glob
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

class Config:
    GROQ_MODEL = "llama-3.1-8b-instant"
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    KNOWLEDGE_BASE_PATH = "knowledge_base"  # Can be file or directory

SYSTEM_PROMPT = """You are GenTaxAI, an expert on Indian tax, GST, and investment regulations.

Retrieved Context from Knowledge Base:
{context}
---

Instructions:
- ALWAYS prioritize information from the Retrieved Context when available
- If the context contains relevant information, use it as your primary source
- If context is insufficient, supplement with your general knowledge
- Be specific and detailed
- Use bullet points for lists
- Cite relevant sections when possible
- Never give generic responses

Answer the question directly and completely."""

app = FastAPI(title="GenTaxAI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

llm = None
knowledge_base = []

def load_json_file(filepath):
    """Load and flatten a single JSON file"""
    items = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        items.append({
                            "source": os.path.basename(filepath),
                            "content": json.dumps(item, ensure_ascii=False)
                        })
                    else:
                        items.append({
                            "source": os.path.basename(filepath),
                            "content": str(item)
                        })
            elif isinstance(data, dict):
                for key, value in data.items():
                    items.append({
                        "source": os.path.basename(filepath),
                        "title": key,
                        "content": json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)
                    })
    except Exception as e:
        print(f"⚠️  Error loading {filepath}: {e}")
    
    return items

@app.on_event("startup")
async def startup():
    """Initialize LLM and load knowledge base"""
    global llm, knowledge_base
    
    print("🚀 Starting GenTaxAI...")
    
    llm = ChatGroq(
        model_name=Config.GROQ_MODEL,
        api_key=Config.GROQ_API_KEY,
        temperature=0.3
    )
    print("✅ LLM initialized")
    
    # Load knowledge base
    path = Config.KNOWLEDGE_BASE_PATH
    
    if os.path.isfile(path):
        # Single JSON file
        print(f"📂 Loading knowledge base from file: {path}")
        knowledge_base = load_json_file(path)
        
    elif os.path.isdir(path):
        # Directory of JSON files
        print(f"📂 Loading knowledge base from directory: {path}")
        json_files = glob.glob(os.path.join(path, "*.json"))
        
        for json_file in json_files:
            print(f"  - Loading {os.path.basename(json_file)}...")
            items = load_json_file(json_file)
            knowledge_base.extend(items)
    else:
        print(f"⚠️  Knowledge base path not found: {path}")
    
    if knowledge_base:
        print(f"✅ Knowledge base loaded: {len(knowledge_base)} items from {len(set(item.get('source', '') for item in knowledge_base))} files")
    else:
        print("⚠️  No knowledge base loaded - using LLM knowledge only")
    
    print("✅ GenTaxAI ready!")

def search_knowledge_base(query: str, top_k: int = 3) -> str:
    """Simple keyword-based search"""
    if not knowledge_base:
        return "[No knowledge base available]"
    
    query_lower = query.lower()
    query_words = set(query_lower.split())
    
    scored_items = []
    for item in knowledge_base:
        # Combine all text fields
        text = " ".join([
            str(item.get("title", "")),
            str(item.get("content", "")),
            str(item.get("source", ""))
        ]).lower()
        
        # Score by keyword matches
        score = sum(1 for word in query_words if word in text and len(word) > 2)
        
        # Boost for exact phrase
        if query_lower in text:
            score += 10
        
        if score > 0:
            content = item.get("content", "")[:800]  # Limit size
            source = item.get("source", "unknown")
            title = item.get("title", "")
            
            result = f"[Source: {source}]"
            if title:
                result += f"\n{title}\n"
            result += f"\n{content}"
            
            scored_items.append((score, result))
    
    scored_items.sort(reverse=True, key=lambda x: x[0])
    
    if scored_items:
        top_results = [content for _, content in scored_items[:top_k]]
        return "\n\n---\n\n".join(top_results)
    
    return "[No relevant information found in knowledge base]"

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return FileResponse("static/index.html")

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "mode": "rag-lightweight",
        "knowledge_base_items": len(knowledge_base),
        "memory_optimized": True
    }

@app.post("/chat")
async def chat(request: Request):
    if not llm:
        raise HTTPException(status_code=500, detail="LLM not ready")

    try:
        data = await request.json()
    except:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    query = data.get("message", "")
    if not query:
        raise HTTPException(status_code=400, detail="'message' required")

    print(f"📨 Query: {query}")
    
    try:
        context = search_knowledge_base(query, top_k=3)
        print(f"📚 Context: {len(context)} chars")
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "{question}")
        ])
        
        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({
            "question": query,
            "context": context
        })
        
        print(f"✅ Answer: {len(answer)} chars")
        return JSONResponse({"answer": answer})
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
