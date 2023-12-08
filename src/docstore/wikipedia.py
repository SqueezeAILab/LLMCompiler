"""Wrapper around wikipedia API."""
import ast
import time
from typing import List, Optional, Union

import aiohttp
import requests
from bs4 import BeautifulSoup

from langchain.docstore.base import Docstore
from langchain.docstore.document import Document


def clean_str(p):
    try:
        return p.encode().decode("unicode-escape").encode("latin1").decode("utf-8")
    except Exception as e:
        print(e)
        return p


class ReActWikipedia(Docstore):
    """Wrapper around wikipedia API."""

    def __init__(self, benchmark=False, skip_retry_when_postprocess=False) -> None:
        """Check that wikipedia package is installed."""
        try:
            import requests
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError(
                "Could not import wikipedia python package. "
                "Please install it with `pip install wikipedia`."
            )
        self.page = None
        self.lookup_keyword = None
        self.lookup_list = None
        self.lookup_cnt = None

        self.benchmark = benchmark
        self.all_times = []

        # when True, always skip retry when postprocess
        self.skip_retry_when_postprocess = skip_retry_when_postprocess

    def reset(self):
        self.all_times = []

    def get_stats(self):
        return {
            "all_times": self.all_times,
        }

    @staticmethod
    def _get_page_obs(page):
        # find all paragraphs
        paragraphs = page.split("\n")
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        # find all sentence
        sentences = []
        for p in paragraphs:
            sentences += p.split(". ")
        sentences = [s.strip() + "." for s in sentences if s.strip()]
        return " ".join(sentences[:5])

    def _get_alternative(self, result: str) -> str:
        parsed_alternatives = result.split("Similar: ")[1][:-1]

        alternatives = ast.literal_eval(parsed_alternatives)
        alternative = alternatives[0]
        for alt in alternatives:
            if "film" in alt or "movie" in alt:
                alternative = alt
                break
        return alternative

    def post_process(
        self, response_text: str, entity: str, skip_retry_when_postprocess: bool = False
    ) -> str:
        soup = BeautifulSoup(response_text, features="html.parser")
        result_divs = soup.find_all("div", {"class": "mw-search-result-heading"})

        if result_divs:  # mismatch
            self.result_titles = [
                clean_str(div.get_text().strip()) for div in result_divs
            ]
            obs = f"Could not find {entity}. Similar: {self.result_titles[:5]}."
        else:
            page = [
                p.get_text().strip() for p in soup.find_all("p") + soup.find_all("ul")
            ]
            if any("may refer to:" in p for p in page):
                if skip_retry_when_postprocess or self.skip_retry_when_postprocess:
                    obs = "Could not find " + entity + "."
                else:
                    obs = self.search("[" + entity + "]", is_retry=True)
            else:
                self.page = ""
                for p in page:
                    if len(p.split(" ")) > 2:
                        self.page += clean_str(p)
                        if not p.endswith("\n"):
                            self.page += "\n"
                obs = self._get_page_obs(self.page)
                self.lookup_keyword = self.lookup_list = self.lookup_cnt = None

        obs = obs.replace("\\n", "")
        return obs

    async def apost_process(
        self, response_text: str, entity: str, skip_retry_when_postprocess: bool = False
    ) -> str:
        soup = BeautifulSoup(response_text, features="html.parser")
        result_divs = soup.find_all("div", {"class": "mw-search-result-heading"})

        if result_divs:  # mismatch
            self.result_titles = [
                clean_str(div.get_text().strip()) for div in result_divs
            ]
            obs = f"Could not find {entity}. Similar: {self.result_titles[:5]}."
        else:
            page = [
                p.get_text().strip() for p in soup.find_all("p") + soup.find_all("ul")
            ]
            if any("may refer to:" in p for p in page):
                if skip_retry_when_postprocess or self.skip_retry_when_postprocess:
                    obs = "Could not find " + entity + "."
                else:
                    obs = await self.asearch("[" + entity + "]", is_retry=True)
            else:
                self.page = ""
                for p in page:
                    if len(p.split(" ")) > 2:
                        self.page += clean_str(p)
                        if not p.endswith("\n"):
                            self.page += "\n"
                obs = self._get_page_obs(self.page)
                self.lookup_keyword = self.lookup_list = self.lookup_cnt = None

        obs = obs.replace("\\n", "")
        return obs

    def search(self, entity: str, is_retry: bool = False) -> Union[str, Document]:
        """Try to search for wiki page.

        If page exists, return the page summary, and a PageWithLookups object.
        If page does not exist, return similar entries.

        Args:
            entity: entity string.

        Returns: a Document object or error message.
        """
        s = time.time()
        entity = str(entity)
        entity_ = entity.replace(" ", "+")
        search_url = f"https://en.wikipedia.org/w/index.php?search={entity_}"
        response_text = requests.get(search_url).text

        result = self.post_process(response_text, entity)

        if "Similar:" in result:
            alternative = self._get_alternative(result)
            entity_ = alternative.replace(" ", "+")
            search_url = f"https://en.wikipedia.org/w/index.php?search={entity_}"
            response_text = requests.get(search_url).text

            result = self.post_process(
                response_text, entity, skip_retry_when_postprocess=True
            )

            if "Similar:" in result:
                result = "Could not find " + entity + "."

        if self.benchmark and not is_retry:
            # we only benchmark the outermost call
            self.all_times.append(round(time.time() - s, 2))

        return result

    async def asearch(
        self, entity: str, is_retry: bool = False
    ) -> Union[str, Document]:
        """Try to search for wiki page.

        If page exists, return the page summary, and a PageWithLookups object.
        If page does not exist, return similar entries.

        Args:
            entity: entity string.

        Returns: a Document object or error message.
        """
        s = time.time()
        entity = str(entity)
        entity_ = entity.replace(" ", "+")
        search_url = f"https://en.wikipedia.org/w/index.php?search={entity_}"

        async with aiohttp.ClientSession() as session:
            async with session.get(search_url) as response:
                response_text = await response.text()

        result = await self.apost_process(response_text, entity)

        if "Similar:" in result:
            alternative = self._get_alternative(result)
            entity_ = alternative.replace(" ", "+")
            search_url = f"https://en.wikipedia.org/w/index.php?search={entity_}"
            async with aiohttp.ClientSession() as session:
                async with session.get(search_url) as response:
                    response_text = await response.text()

            result = await self.apost_process(
                response_text, entity, skip_retry_when_postprocess=True
            )

            if "Similar:" in result:
                return "Could not find " + entity + "."

        if self.benchmark and not is_retry:
            # we only benchmark the outermost call
            self.all_times.append(round(time.time() - s, 2))

        return result


# TODO: Move this to proper place
class DocstoreExplorer:
    """Class to assist with exploration of a document store."""

    def __init__(self, docstore: ReActWikipedia, char_limit=None, one_sentence=False):
        """Initialize with a docstore, and set initial document to None."""
        self.docstore = docstore
        self.document: Optional[Document] = None
        self.lookup_str = ""
        self.lookup_index = 0
        self.char_limit = char_limit
        self.one_sentence = one_sentence

    def search(self, term: str) -> str:
        """Search for a term in the docstore, and if found save."""
        result = self.docstore.search(term)
        if self.one_sentence:
            result = result.split(". ")[0]
        if self.char_limit is not None:
            result = result[: self.char_limit]
        if isinstance(result, Document):
            self.document = result
            return self._summary
        else:
            self.document = None
            return result

    async def asearch(self, term: str) -> str:
        """Search for a term in the docstore, and if found save."""
        result = await self.docstore.asearch(term)
        if self.one_sentence:
            result = result.split(". ")[0]
        if self.char_limit is not None:
            result = result[: self.char_limit]
        if isinstance(result, Document):
            self.document = result
            return self._summary
        else:
            self.document = None
            return result

    def lookup(self, term: str) -> str:
        """Lookup a term in document (if saved)."""
        if self.document is None:
            raise ValueError("Cannot lookup without a successful search first")
        if term.lower() != self.lookup_str:
            self.lookup_str = term.lower()
            self.lookup_index = 0
        else:
            self.lookup_index += 1
        lookups = [p for p in self._paragraphs if self.lookup_str in p.lower()]
        if len(lookups) == 0:
            return "No Results"
        elif self.lookup_index >= len(lookups):
            return "No More Results"
        else:
            result_prefix = f"(Result {self.lookup_index + 1}/{len(lookups)})"
            return f"{result_prefix} {lookups[self.lookup_index]}"

    @property
    def _summary(self) -> str:
        return self._paragraphs[0]

    @property
    def _paragraphs(self) -> List[str]:
        if self.document is None:
            raise ValueError("Cannot get paragraphs without a document")
        return self.document.page_content.split("\n\n")
