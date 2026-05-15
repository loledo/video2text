#!/usr/bin/env python3
"""Waypoint travel agent — powered by Claude + Tavily + TikTok video context."""
import logging
import os
import sys
from collections import defaultdict

import anthropic
from tavily import TavilyClient

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are Waypoint, an expert travel guide and personal advisor built on curated first-hand travel knowledge extracted from TikTok travel creators. Your job is to help travelers plan trips, discover hidden gems, understand local customs, and make confident, specific decisions about destinations worldwide.

You draw from three sources, in strict priority order:

1. TIKTOK CONTEXT (primary) — Transcriptions and text extracted from curated travel TikTok videos. These are first-hand tips from real travelers and local creators. Treat this as your most trusted source. When citing from it, attribute clearly (e.g., "From a creator tip on Barcelona food markets...").

2. WEB SEARCH (supplementary) — Use your tavily_search tool when:
   - The TikTok context doesn't cover the user's question
   - You need current operational info (prices, hours, visa rules, transit schedules)
   - You need to verify or update something that may have changed
   Always prefer recent, authoritative sources.

3. TRAINING KNOWLEDGE (fallback only) — Use sparingly. When you do, explicitly label it: "Based on general knowledge (not from your video library or a live search)..."

Behavioral constraints:
- Be specific and actionable. Never give advice so vague it could apply to any destination.
- Never fabricate prices, addresses, opening hours, or visa requirements. If uncertain, say so and instruct the user to verify before travel.
- Flag potentially outdated information with: "This may have changed — verify before you go."
- If you cannot find reliable information, say so and suggest where the user can look.
- When uncertain between two good options, state your reasoning and give a recommendation rather than hedging indefinitely."""

TOOLS = [
    {
        "name": "tavily_search",
        "description": (
            "Search the web for current travel information. Use when the TikTok context is "
            "insufficient or you need current operational details (prices, hours, visa rules, "
            "schedules, recent reviews)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "The search query. Be specific — include destination name and topic."
                    ),
                }
            },
            "required": ["query"],
        },
    }
]


class TravelAgent:
    def __init__(self, chunks: list[dict], tavily_api_key: str | None = None):
        self.client = anthropic.Anthropic()
        self.tavily = TavilyClient(api_key=tavily_api_key or os.environ["TAVILY_API_KEY"])
        self.chunks = chunks
        self.history: list[dict] = []  # {"role": "user"|"assistant", "content": str}

    def _build_context(self) -> str:
        """Format all chunks as an XML block for injection into the system prompt."""
        if not self.chunks:
            return ""

        lines = [
            "<travel_context>",
            "The following content was extracted from curated TikTok travel videos. "
            "Use this as your primary reference.",
            "",
        ]

        # Group by source_file for cleaner output
        by_source: dict[str, list[dict]] = defaultdict(list)
        for chunk in self.chunks:
            by_source[chunk.get("source_file", "unknown")].append(chunk)

        for source_file, source_chunks in sorted(by_source.items()):
            for chunk in source_chunks:
                chunk_type = chunk.get("type", "transcript")
                destinations = chunk.get("destinations", [])
                dest_str = ", ".join(destinations) if destinations else "unknown"
                text = chunk.get("text", "").strip()

                lines.append("---")
                lines.append(
                    f"Source: {source_file} | Type: {chunk_type} | Destinations: {dest_str}"
                )
                lines.append(text)

        lines.append("---")
        lines.append("</travel_context>")
        return "\n".join(lines)

    def _run_tavily(self, query: str) -> str:
        """Execute a Tavily web search and return a formatted result string."""
        print(f"(Searching the web for: {query}...)", file=sys.stderr)
        log.info("Tavily search: %s", query)
        try:
            response = self.tavily.search(query=query, search_depth="basic", max_results=5)
            results = response.get("results", [])
            if not results:
                return "No results found for this query."

            parts = []
            for r in results:
                title = r.get("title", "")
                url = r.get("url", "")
                content = r.get("content", "")
                parts.append(f"**{title}**\n{url}\n{content}")
            return "\n\n".join(parts)
        except Exception as exc:
            log.warning("Tavily search failed: %s", exc)
            return f"Search failed: {exc}"

    def chat(self, user_message: str) -> str:
        """Send a message and return the assistant's response (with tool use handled)."""
        # Add user message to history
        self.history.append({"role": "user", "content": user_message})

        # Build system prompt with injected travel context
        context = self._build_context()
        system = SYSTEM_PROMPT
        if context:
            system = f"{system}\n\n{context}"

        # Build messages list from full history
        messages = [{"role": m["role"], "content": m["content"]} for m in self.history]

        try:
            final_text = self._agentic_loop(system, messages)
        except anthropic.APIError as exc:
            log.error("Anthropic API error: %s", exc)
            print(f"\n[API error: {exc}]", file=sys.stderr)
            # Remove the user message we added so history stays consistent
            self.history.pop()
            return f"Sorry, I encountered an API error: {exc}"

        # Persist assistant response to history
        self.history.append({"role": "assistant", "content": final_text})
        return final_text

    def _agentic_loop(self, system: str, messages: list[dict]) -> str:
        """Run the Claude agentic loop until stop_reason == 'end_turn'."""
        while True:
            response = self.client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=4096,
                system=system,
                tools=TOOLS,
                messages=messages,
            )

            log.debug("stop_reason=%s", response.stop_reason)

            if response.stop_reason == "end_turn":
                # Extract text from all TextBlock content blocks
                text_parts = []
                for block in response.content:
                    if hasattr(block, "text"):
                        text_parts.append(block.text)
                return "".join(text_parts)

            if response.stop_reason == "tool_use":
                # Add the assistant's response (with tool use blocks) to messages
                messages.append({"role": "assistant", "content": response.content})

                # Process all tool calls and collect results
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        query = block.input.get("query", "")
                        result_text = self._run_tavily(query)
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": result_text,
                            }
                        )

                # Add tool results as a user message and loop
                messages.append({"role": "user", "content": tool_results})
                continue

            # Unexpected stop reason — extract whatever text we have and return
            log.warning("Unexpected stop_reason: %s", response.stop_reason)
            text_parts = []
            for block in response.content:
                if hasattr(block, "text"):
                    text_parts.append(block.text)
            return "".join(text_parts)
