import asyncio
import logging
import pathlib
import shutil
from dataclasses import dataclass, field

from bs4 import BeautifulSoup
from playwright.async_api import Locator, Page, async_playwright

from src.utils import find_idea_root

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class ButtonGroup:
    """Represents a group of buttons to click with their section ID."""

    buttons: list[str]
    section_id: str

    sections: list[str] = field(default_factory=list)

    output_directory: pathlib.Path = None

    def setup_paths(self, root: pathlib.Path):
        self.output_directory = root / self.section_id
        self.output_directory.mkdir(parents=True, exist_ok=True)

    def write_html(self):
        for i, section in enumerate(self.sections):
            output_file = self.output_directory / f"{self.section_id}_{i}.html"
            output_file.write_text(section)

    async def cycle_buttons(self, page: Page):
        button_text = None

        try:
            for i, button_text in enumerate(self.buttons):
                logger.debug(f"Looking for id: #{self.section_id}")
                full_section: Locator = page.locator(f"#{self.section_id}").first
                section_html = await full_section.inner_html()
                self.sections.append(section_html)

                if i == len(self.buttons) - 1:
                    continue

                # Find button to open the submenu
                logger.debug(f"Looking for button to open menu: {button_text}")
                button_element: Locator = page.get_by_text(button_text).first
                button: Locator = button_element.locator("xpath=ancestor::button").first
                await button.click()

                # Find button to select next section
                logger.debug(f"Looking for button to load next section: {self.buttons[i + 1]}")
                button_element: Locator = page.get_by_text(self.buttons[i + 1]).first
                # button: Locator = button_element.locator("xpath=ancestor::button").first
                await button_element.click()

                logger.info(f"Clicked button for: {button_text}")
                await asyncio.sleep(1)
        except Exception:
            logger.exception(f"Could not find/click button for '{button_text}'")

        pretty_sections: list[str] = []
        for section in self.sections:
            pretty_sections.append(BeautifulSoup(section, "html.parser").prettify())
        self.sections = pretty_sections


benchmarks_button_group = ButtonGroup(
    buttons=["Intelligence Index Score", "Coding Index Score", "Agentic Index Score"],
    section_id="benchmarks",
)

performance_button_group = ButtonGroup(
    buttons=["Highest throughput", "Lowest latency"],
    section_id="performance",
)

context_length_button_group = ButtonGroup(
    buttons=["1K - 10K tokens", "10K - 100K tokens", "100K - 1M tokens", "1M - 10M tokens"],
    section_id="context-length",
)

button_groups = [benchmarks_button_group, performance_button_group, context_length_button_group]


async def scroll_to_bottom(page: Page):
    """
    Scrolls to the bottom of an expanding/'infinite' content page incrementally.
    """
    current_scroll_position = 0
    scroll_step = 800  # Distance to scroll each time

    while True:
        last_height = await page.evaluate("document.body.scrollHeight")

        # Scroll incrementally to the current bottom
        while current_scroll_position < last_height:
            current_scroll_position += scroll_step
            await page.evaluate(f"window.scrollTo(0, {current_scroll_position})")
            logger.debug(f"Scrolled to {current_scroll_position}px...")
            await asyncio.sleep(1)  # Short pause for smoother scrolling

        # Wait for potential new content to load
        await asyncio.sleep(2)

        new_height = await page.evaluate("document.body.scrollHeight")
        if new_height == last_height:
            # Check one more time to be sure
            await asyncio.sleep(2)
            new_height = await page.evaluate("document.body.scrollHeight")
            if new_height == last_height:
                break

    logger.info("Reached the bottom of the page.")
    page_content = await page.content()

    for button_group in button_groups:
        logger.info(f"Clicking buttons for: {button_group.section_id}")
        await button_group.cycle_buttons(page)

    return page_content


async def get_page_content(url: str) -> str:
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        logger.info(f"Loading {url}...")
        await page.goto(url, wait_until="networkidle")

        page_content: str
        page_content = await scroll_to_bottom(page)

        await browser.close()

        return page_content


async def main():
    logger.info("Deleting existing output directory")
    root_dir = find_idea_root(pathlib.Path.cwd())
    output_directory = root_dir / "output"
    shutil.rmtree(output_directory, ignore_errors=True)
    output_directory.mkdir(parents=True, exist_ok=True)

    url = "https://openrouter.ai/rankings"
    html_content = await get_page_content(url)

    html_content = BeautifulSoup(html_content, "html.parser").prettify()
    (output_directory / "page_content.html").write_text(html_content)

    for button_group in button_groups:
        button_group.setup_paths(output_directory)
        button_group.write_html()


if __name__ == "__main__":
    asyncio.run(main())
