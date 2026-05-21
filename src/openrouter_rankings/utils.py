import asyncio
import logging
import pathlib

from playwright.async_api import Page

logger = logging.getLogger(__name__)


def find_idea_root(start_path: pathlib.Path | str = ".") -> pathlib.Path | None:
    """
    Walks up directories from start_path until a directory containing a '.idea' folder is found.

    Args:
        start_path: The directory to start the search from. Defaults to the current directory.

    Returns:
        The pathlib.Path of the directory containing '.idea', or None if not found.
    """
    current_path = pathlib.Path(start_path).resolve()

    # Iterate up through parents
    # .parents includes all parent directories, but not the directory itself.
    # We should check the current directory first.

    for path in [current_path] + list(current_path.parents):
        if (path / ".idea").is_dir():
            return path

    raise f"Could not find .idea directory starting from {start_path}"


def get_output_directory() -> pathlib.Path:
    root_dir = find_idea_root(pathlib.Path.cwd())
    output_directory = root_dir / "output"
    return output_directory


async def click_all_by_name(page: Page, section_label: str):
    """
    Expands all sections on a page by clicking on their 'expand' buttons.
    """
    all_matching = await page.get_by_text(section_label).all()
    logger.debug(f"Clicking {len(all_matching)} elements matching '{section_label}'")
    for _ in range(len(all_matching)):
        current_matching = page.get_by_text(section_label).first
        await current_matching.scroll_into_view_if_needed()
        await current_matching.click()
        await asyncio.sleep(0.25)
