# video2text / TikTok Travel Agent

## Session memory
Project context and preferences are in `.claude/memory/`. Read those files at the start of any session.

## Active branch
`tiktok-travel-agent` — extraction pipeline + Waypoint chat agent.

## Setup (GPU machine)
```
poetry install
export ANTHROPIC_API_KEY=...
export TAVILY_API_KEY=...
```

## Entry points
```
python process_videos.py <videos-dir> --chunks-dir chunks/
python chat.py --chunks-dir chunks/
```
