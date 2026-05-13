"""
Standalone local copy of the FOREVER prompter utility.
"""

from __future__ import annotations

import json
import os.path as osp
from pathlib import Path
from typing import Optional


class Prompter:
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        if not template_name:
            template_name = "alpaca"

        # Prefer local templates under our_scripts/templates, fallback to FOREVER/templates.
        here = Path(__file__).resolve().parent
        candidates = [
            here / "templates" / f"{template_name}.json",
            here.parent / "FOREVER" / "templates" / f"{template_name}.json",
        ]
        file_name: Optional[Path] = next((p for p in candidates if p.exists()), None)
        if file_name is None:
            raise ValueError(f"Cannot find template {template_name!r} in local or FOREVER templates.")

        with file_name.open("r", encoding="utf-8") as fp:
            self.template = json.load(fp)
        if self._verbose:
            print(f"Using prompt template {template_name}: {self.template.get('description', '')}")

    def generate_prompt(
        self,
        instruction: str,
        input_text: Optional[str] = None,
        label: Optional[str] = None,
    ) -> str:
        if input_text:
            res = self.template["prompt_input"].format(instruction=instruction, input=input_text)
        else:
            res = self.template["prompt_no_input"].format(instruction=instruction)
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()
