import asyncio
import logging
import pathlib
import time
from dataclasses import dataclass, field

from bs4 import BeautifulSoup
from playwright.async_api import Locator, Page

from openrouter_rankings.utils import click_all_by_name, get_output_directory

logger = logging.getLogger(__name__)


@dataclass
class ButtonGroup:
    """Represents a group of buttons to click with their section ID."""

    section_id: str
    button_labels: list[str]

    gathered_sections: dict[str, str] = field(default_factory=dict)

    output_directory: pathlib.Path = None
    page: Page = None

    def setup_paths(self, root: pathlib.Path):
        self.output_directory = root / self.section_id
        self.output_directory.mkdir(parents=True, exist_ok=True)

    def write_html(self):
        for section_name, section_html in self.gathered_sections.items():
            output_file = self.output_directory / f"{section_name}.html"
            output_file.write_text(section_html)

    async def extract_section_html(self):
        full_section: Locator = self.page.locator(f"#{self.section_id}").first
        if not full_section:
            return None
        section_html = await full_section.inner_html()
        return BeautifulSoup(section_html, "html.parser").prettify()

    async def get_section_locator(self):
        locator: Locator = self.page.locator(f"#{self.section_id}")
        count = await locator.count()
        if count == 0:
            raise ValueError(f"Could not find section '{self.section_id}'")
        return locator.first

    async def screenshot(self, locator: Locator, hint: str = ""):
        screenshot_path = self.output_directory / "screenshots"
        screenshot_path.mkdir(parents=True, exist_ok=True)
        await locator.screenshot(path=(screenshot_path / f"{time.time()}{'_' + hint if hint else ''}.png"))

    async def get_matching_button(self, label: str, is_div: bool = False) -> Locator | None:
        this_section = await self.get_section_locator()
        if is_div:
            # The floating menu is not a child of the section
            button_locator = self.page.locator("div").filter(has=self.page.locator(f"> div:text('{label}')"))
        else:
            button_locator = this_section.locator("button", has=self.page.locator(f"> span:text('{label}')"))
        count = await button_locator.count()
        if count > 0:
            logger.debug(f"Found {'<div>' if is_div else '<button>'} button of label '{label}'")
            first_button = button_locator.first
            return first_button
        else:
            logger.debug(f"Could not find {'<div>' if is_div else '<button>'} button of label '{label}'")
            return None

    async def cycle_buttons(self):
        """
        1. Look for the currently selected label: For each possible label, test if a matching <button> exists until
           one is found. current_label = This.
        2. Start loop:
            a. Expand all sections.
            b. Save the current section's HTML: current_label -> html.
            c. Click on the current_label <button>.
            d. Loop over remaining labels, finding the first matching <div> element. current_label = This.
            e. Click the <div> button.
        """
        self.setup_paths(get_output_directory())

        current_label = None
        for button_text in self.button_labels:
            logger.debug(f"Looking for initially visible button: {button_text}")
            button: Locator | None = await self.get_matching_button(label=button_text, is_div=False)
            if button:
                current_label = button_text
                break
        if not current_label:
            raise ValueError(f"Could not find initially visible button for: {self.section_id}.")

        while True:
            logger.debug(f"Current button label: {current_label}")

            # a. Expand all sections.
            await asyncio.sleep(0.5)
            logger.debug(f"Expanding all sections by clicking 'Show more' button. Current label: '{current_label}'")
            await click_all_by_name(self.page, section_label="Show more")
            await asyncio.sleep(0.1)
            await self.page.mouse.move(0, 0)
            await asyncio.sleep(0.4)

            # b. Save the current section's HTML: current_label -> html.
            logger.debug(f"Extracting section '{self.section_id}' HTML for label: '{current_label}'")
            section_html = await self.extract_section_html()
            if not section_html:
                raise ValueError(
                    f"Section with id '{self.section_id}' not found on current page "
                    f"with current label='{current_label}'."
                )
            self.gathered_sections[current_label] = section_html

            # c. Click on the current_label <button>.
            logger.debug(f"Expanding full button menu by clicking current label: '{current_label}'")
            current_label_button = await self.get_matching_button(label=current_label, is_div=False)
            if not current_label_button:
                raise ValueError(f"Could not find current button for label '{current_label}'.")
            await current_label_button.click()

            # The floating menu is now visible
            await self.screenshot(await self.get_section_locator(), hint=f"{self.section_id}_expanded")

            # d. Loop over remaining labels, finding the first matching <div> element. current_label = This.
            logger.debug(f"Finding next remaining label after '{current_label}'")
            remaining_labels = [label for label in self.button_labels if label not in self.gathered_sections]
            if not remaining_labels:
                logger.debug(f"No remaining labels found. Last label: '{current_label}'. Returning.")
                return

            next_button: Locator | None = None
            for label in remaining_labels:
                next_button: Locator | None = await self.get_matching_button(label=label, is_div=True)
                if next_button:
                    current_label = label
                    break
            if not next_button:
                raise ValueError(f"Could not find next button after label '{current_label}'.")

            # e. Click the <div> button.
            logger.debug(f"Clicking next remaining label: '{current_label}'")
            await next_button.click()


# We export this premade set of button groups
button_groups: dict[str, ButtonGroup] = {
    "leaderboard": ButtonGroup(
        button_labels=["Today", "This Week", "This Month", "Trending"],
        section_id="leaderboard",
    ),
    "benchmarks": ButtonGroup(
        button_labels=[
            "Intelligence Index Score",
            "Coding Index Score",
            "Agentic Index Score",
            "Code Categories ELO",
            "UI Component ELO",
            "Game Development ELO",
            "Data Visualization ELO",
            "3D ELO",
            "Image ELO",
            "SVG ELO",
        ],
        section_id="benchmarks",
    ),
    "performance": ButtonGroup(
        button_labels=["Highest throughput", "Lowest latency"],
        section_id="performance",
    ),
    "categories": ButtonGroup(
        button_labels=[
            "Programming",
            "Roleplay",
            "Marketing",
            "SEO",
            "Technology",
            "Science",
            "Translation",
            "Legal",
            "Finance",
            "Health",
            "Trivia",
            "Academia",
        ],
        section_id="categories",
    ),
    "natural-languages": ButtonGroup(
        button_labels=[
            "Arabic",
            "Armenian",
            "Awadhi",
            "Belarusian",
            "Bengali",
            "Bosnian",
            "Bulgarian",
            "Burmese",
            "Cantonese",
            "Catalan",
            "Chinese (Simplified)",
            "Chinese (Traditional)",
            "Croatian",
            "Czech",
            "Danish",
            "Dutch",
            "English",
            "Estonian",
            "Finnish",
            "French",
            "Galician",
            "German",
            "Greek",
            "Hebrew",
            "Hindi",
            "Hungarian",
            "Icelandic",
            "Indonesian",
            "Italian",
            "Japanese",
            "Kannada",
            "Korean",
            "Lao",
            "Limburgish",
            "Lithuanian",
            "Lombard",
            "Malay",
            "Maltese",
            "Nepali",
            "Norwegian",
            "Persian",
            "Polish",
            "Portuguese",
            "Punjabi",
            "Romanian",
            "Russian",
            "Serbian",
            "Slovak",
            "Slovenian",
            "Spanish",
            "Swedish",
            "Tagalog",
            "Thai",
            "Tosk Albanian",
            "Turkish",
            "Ukrainian",
            "Vietnamese",
            "Welsh",
        ],
        section_id="natural-languages",
    ),
    "programming-languages": ButtonGroup(
        button_labels=[
            "Python",
            "JavaScript",
            "TypeScript",
            "Java",
            "Ruby",
            "C",
            "C++",
            "C#",
            "Go",
            "Rust",
            "SQL",
            "Perl",
            "Visual Basic",
            "Fortran",
            "MATLAB",
            "Swift",
            "Lua",
        ],
        section_id="programming-languages",
    ),
    "context-length": ButtonGroup(
        button_labels=[
            "1K - 10K tokens",
            "10K - 100K tokens",
            "100K - 1M tokens",
            "1M - 10M tokens",
        ],
        section_id="context-length",
    ),
}
