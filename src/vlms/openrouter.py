from __future__ import annotations

import json
from typing import TYPE_CHECKING

from openai import AsyncOpenAI, OpenAI
from openai.types.responses import Response
from PIL import Image

from ..config import (
    MAX_ACTIONS,
    MAX_NOTES_CHARS,
    RunConfig,
    make_notes_tool,
)
from .vlm import Action, VLMClient, encode_image_base64

if TYPE_CHECKING:
    from ..benchmark_task import BenchmarkTask


class OpenRouterVLMClient(VLMClient):
    """Client for calling OpenRouter client directly."""

    def __init__(
        self,
        task: BenchmarkTask,
        run_config: RunConfig,
        tools: list[dict] | None = None,
    ):
        super().__init__(
            model=run_config.model,
            task=task,
            tools=tools,
            canvas_size=run_config.canvas_size,
            log_reasoning=run_config.log_reasoning,
            enable_reset=run_config.enable_reset,
            enable_notes=run_config.enable_notes,
        )

        self.focus_mode = run_config.focus_mode
        self.focus_window = run_config.focus_window
        self.enable_notes = run_config.enable_notes
        self.current_notes: list[tuple[int, str]] = []  # Append-only log: (turn_number, entry)

        # Add notes tool if enabled
        if run_config.enable_notes:
            self.tools = list(self.tools) + [make_notes_tool()]

        base_url = self.dotenv_get("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1"
        site_url = self.dotenv_get("OPENROUTER_SITE_URL")
        app_name = self.dotenv_get("OPENROUTER_APP_NAME")

        extra_headers = {}
        if site_url:
            extra_headers["HTTP-Referer"] = site_url
        if app_name:
            extra_headers["X-Title"] = app_name

        self.client = OpenAI(
            api_key=self.env_vars.get("OPENROUTER_API_KEY"),
            base_url=base_url,
            default_headers=extra_headers or None,
        )
        self.async_client = AsyncOpenAI(
            api_key=self.env_vars.get("OPENROUTER_API_KEY"),
            base_url=base_url,
            default_headers=extra_headers or None,
        )

        self.provider_preferences: dict[str, object] = {}
        if self.model == "moonshotai/kimi-k2.5":
            self.provider_preferences["ignore"] = ["fireworks"]

    def build_messages(
        self,
        target_image: Image.Image | None,
        canvas_image: Image.Image,
        history: list[Action],
        action_count: int,
        model: str,
        include_reasoning_summaries: bool = False,
        show_grid: bool = False,
    ) -> list[dict]:
        """Build message list for VLM API call.

        In focus mode, uses format_history_with_focus() for compact history.
        """
        target_b64 = encode_image_base64(target_image) if target_image else None
        canvas_b64 = encode_image_base64(canvas_image)

        # Use focus-based history formatting when focus mode is enabled
        if self.focus_mode:
            history_text = self.format_history_with_focus(history, window=self.focus_window)
        else:
            history_text = self.format_history(
                history, include_reasoning=include_reasoning_summaries
            )

        # Add notes section if enabled
        if self.enable_notes and self.current_notes:
            notes_text = self._format_notes_for_display()
            history_text = f"<your_notes>\n{notes_text}\n</your_notes>\n\n{history_text}"

        remaining_actions = MAX_ACTIONS - action_count

        user_content = self.task.build_vlm_user_content(
            target_b64=target_b64,
            canvas_b64=canvas_b64,
            history_text=history_text,
            remaining=remaining_actions,
            current=action_count + 1,
            max_actions=MAX_ACTIONS,
        )

        # Convert to OpenAI Responses API format
        formatted_content = []
        for item in user_content:
            if item["type"] == "text":
                formatted_content.append({"type": "input_text", "text": item["text"]})
            elif item["type"] == "image_url":
                formatted_content.append(
                    {
                        "type": "input_image",
                        "detail": "high",
                        "image_url": item["image_url"]["url"],
                    }
                )

        return [
            {"type": "message", "role": "system", "content": self._system_prompt},
            {"type": "message", "role": "user", "content": formatted_content},
        ]

    def get_action(
        self,
        target_image: Image.Image | None,
        canvas_image: Image.Image,
        history: list[Action],
        action_count: int,
        show_grid: bool = False,
        max_actions: int = 1,
    ) -> list[Action]:
        include_reasoning_summaries = self.dotenv_get_bool(
            "BENCHMARK_INCLUDE_REASONING_SUMMARY", default=False
        )
        messages = self.build_messages(
            target_image,
            canvas_image,
            history,
            action_count,
            include_reasoning_summaries=include_reasoning_summaries,
            model=self.model,
            show_grid=show_grid,
        )

        reasoning_payload: dict | None = {"effort": "high"}
        summary_mode = self.dotenv_get("BENCHMARK_REASONING_SUMMARY")
        if summary_mode and reasoning_payload is not None:
            reasoning_payload["summary"] = summary_mode

        # Some providers reject tool_choice="required" for specific models.
        tool_choice = "required"
        if self.model == "moonshotai/kimi-k2.5":
            tool_choice = "auto"
        elif self.model.startswith("qwen/"):
            tool_choice = "auto"

        extra_body: dict[str, object] = {}
        if reasoning_payload is not None:
            extra_body["reasoning"] = reasoning_payload
        if self.provider_preferences:
            extra_body["provider"] = self.provider_preferences

        response = self.client.responses.create(
            model=self.model,
            input=messages,  # type: ignore
            tools=self.tools,  # type: ignore
            tool_choice=tool_choice,
            reasoning=reasoning_payload,  # type: ignore
            parallel_tool_calls=max_actions > 1,
            max_output_tokens=50_000,
            extra_body=extra_body or None,
        )
        return self.parse_response(response, max_actions=max_actions)

    async def get_action_async(
        self,
        target_image: Image.Image | None,
        canvas_image: Image.Image,
        history: list[Action],
        action_count: int,
        show_grid: bool = False,
        max_actions: int = 1,
    ) -> list[Action]:
        include_reasoning_summaries = self.dotenv_get_bool(
            "BENCHMARK_INCLUDE_REASONING_SUMMARY", default=False
        )
        messages = self.build_messages(
            target_image,
            canvas_image,
            history,
            action_count,
            include_reasoning_summaries=include_reasoning_summaries,
            model=self.model,
            show_grid=show_grid,
        )

        reasoning_payload: dict | None = {"effort": "high"}
        summary_mode = self.dotenv_get("BENCHMARK_REASONING_SUMMARY")
        if summary_mode and reasoning_payload is not None:
            reasoning_payload["summary"] = summary_mode

        # Some providers reject tool_choice="required" for specific models.
        tool_choice = "required"
        if self.model == "moonshotai/kimi-k2.5":
            tool_choice = "auto"
        elif self.model.startswith("qwen/"):
            tool_choice = "auto"

        extra_body: dict[str, object] = {}
        if reasoning_payload is not None:
            extra_body["reasoning"] = reasoning_payload
        if self.provider_preferences:
            extra_body["provider"] = self.provider_preferences

        response = await self.async_client.responses.create(
            model=self.model,
            input=messages,  # type: ignore
            tools=self.tools,  # type: ignore
            tool_choice=tool_choice,
            reasoning=reasoning_payload,  # type: ignore
            parallel_tool_calls=max_actions > 1,
            max_output_tokens=50_000,
            extra_body=extra_body or None,
        )
        return self.parse_response(response, max_actions=max_actions)

    def parse_response(
        self, response: Response, verbose: bool = False, max_actions: int = 1
    ) -> list[Action]:
        def _extract_reasoning(resp: Response) -> str | None:
            """Extract reasoning from OpenRouter/OpenAI Responses API.

            OpenRouter returns reasoning_details array with objects of type:
            - "reasoning.summary": has 'summary' string field
            - "reasoning.text": has 'text' string field
            - "reasoning.encrypted": has 'data' field (not readable)

            OpenAI Responses API returns output items of type "reasoning" with:
            - 'summary': array of parts with .text field
            """
            print(f"[DEBUG] Response: {response}")
            parts: list[str] = []

            for item in resp.output or []:
                if getattr(item, "type", None) == "reasoning":
                    for part in getattr(item, "summary", None) or []:
                        text = getattr(part, "text", None)
                        if text:
                            parts.append(text)
                    for part in getattr(item, "content", None) or []:
                        if getattr(part, "type", None) == "reasoning_text":
                            parts.append(getattr(part, "text", None))  # type: ignore

            return "\n".join(parts).strip() or None

        if not response.output or len(response.output) == 0:
            if verbose:
                print(f"[DEBUG] No output in response: {response}")
            return []

        reasoning = _extract_reasoning(response)

        # Collect all function calls (supports parallel tool calls)
        actions: list[Action] = []
        for item in response.output or []:
            if getattr(item, "type", None) == "function_call":
                name = item.name  # type: ignore
                raw_args = item.arguments or "{}"  # type: ignore
                try:
                    args = json.loads(raw_args)
                except json.JSONDecodeError:
                    if verbose:
                        print(f"[DEBUG] Failed to JSON-parse tool args: {raw_args!r}")
                    args = {}

                action = self.create_action(name, args, reasoning if not actions else None)
                if action:
                    actions.append(action)
                    if len(actions) >= max_actions:
                        break

        if not actions and verbose:
            print("[DEBUG] No function_call in response.output.")
            print(f"[DEBUG] output_text: {getattr(response, 'output_text', '')!r}")
            print(f"[DEBUG] output: {getattr(response, 'output', None)}")

        return actions

    def append_notes_entry(self, turn_number: int, entry: str) -> None:
        """Append a new entry to the notes log (called by runner after update_notes action).

        Args:
            turn_number: The current turn number
            entry: The note entry to append
        """
        if entry:
            self.current_notes.append((turn_number, entry))
            self._truncate_notes_if_needed()

    def _truncate_notes_if_needed(self) -> None:
        """Remove oldest entries if total exceeds MAX_NOTES_CHARS."""
        while self.get_total_notes_chars() > MAX_NOTES_CHARS and self.current_notes:
            self.current_notes.pop(0)  # Remove oldest

    def get_total_notes_chars(self) -> int:
        """Get total character count across all notes entries."""
        return sum(len(entry) for _, entry in self.current_notes)

    def _format_notes_for_display(self) -> str:
        """Format all notes entries for display to the model."""
        if not self.current_notes:
            return ""

        lines = []
        for turn_num, entry in self.current_notes:
            lines.append(f"[Turn {turn_num}] {entry}")

        return "\n".join(lines)
