import asyncio
import logging
import shutil

from bs4 import BeautifulSoup
from playwright.async_api import Page, async_playwright

from src.openrouter_rankings.button_groups import button_groups
from src.openrouter_rankings.utils import get_output_directory

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(funcName)-20s - %(levelname)-8s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


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
            logger.debug(f"Scrolled to {current_scroll_position}px")
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

    for button_group in button_groups.values():
        logger.info(f"Clicking buttons for: {button_group.section_id}")
        button_group.page = page
        await button_group.cycle_buttons()
        button_group.write_html()

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
    output_directory = get_output_directory()
    shutil.rmtree(output_directory, ignore_errors=True)
    output_directory.mkdir(parents=True, exist_ok=True)

    url = "https://openrouter.ai/rankings"
    html_content = await get_page_content(url)

    html_content = BeautifulSoup(html_content, "html.parser").prettify()
    (output_directory / "page_content.html").write_text(html_content)


if __name__ == "__main__":
    asyncio.run(main())
