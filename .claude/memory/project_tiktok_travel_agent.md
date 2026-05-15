---
name: project-tiktok-travel-agent
description: "TikTok Travel Agent build on the video2text repo — branch, architecture, and key decisions"
metadata: 
  node_type: memory
  type: project
  originSessionId: 63f93662-1e8c-49bc-8622-74a5228e9468
---

Building a TikTok travel guide agent on branch `tiktok-travel-agent` of the video2text repo.

**Why:** Transform existing Whisper transcription tool into a full travel RAG agent powered by downloaded TikTok videos.

**Architecture:**
- **Extraction pipeline** (`process_videos.py`): routes each file through Whisper + Claude (narrated video) or Florence-2 OCR + Claude (silent/image), saves structured chunks to `chunks/` dir
- **Agent** (`chat.py` / `agent/travel_agent.py`): Waypoint persona, loads all chunks as full context, Tavily web search tool, multi-turn CLI REPL
- Florence-2-large (HuggingFace, ~0.77B) used for local OCR on 8GB GPU — NOT Claude Vision
- RAG strategy: full context injection (< 50 videos, no vector store needed)
- Web search: Tavily API

**How to apply:** Any future work on this project should respect these choices: Florence-2 for OCR (not Claude Vision), full-context RAG (not embeddings), Tavily for search, `claude-sonnet-4-6` for both structuring and agent.

**Chunk schema:** `{source_file, extractable, destinations[], chunks[{chunk_id, type, destinations[], text, keywords[], approximate_timestamp_start, approximate_timestamp_end}]}`

**Entry points:**
- `python process_videos.py <input_dir>` — batch extract
- `python chat.py [--chunks-dir chunks/]` — interactive agent
- Needs: `ANTHROPIC_API_KEY`, `TAVILY_API_KEY`
