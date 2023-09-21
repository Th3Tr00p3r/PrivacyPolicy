import json
from itertools import product
from pathlib import Path
from typing import Dict, List
from urllib.parse import urlparse, urlunparse

import scrapy
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from scrapy.downloadermiddlewares.useragent import UserAgentMiddleware
from scrapy.linkextractors import LinkExtractor


class RotateUserAgentMiddleware(UserAgentMiddleware):
    def __init__(self, user_agent="Scrapy"):
        self._user_agent = UserAgent()
        super().__init__(user_agent)

    def process_request(self, request, spider):
        request.headers["User-Agent"] = self._user_agent.random


class SearchEngineTextSpider(scrapy.Spider):
    """
    A Scrapy spider to automatically scrape links from search results.
    """

    file_path = Path("./data/url_text.json")
    name = "search engine"
    start_urls: List[str]
    search_engines = [
        "https://www.google.com/search?q=",
        "https://search.brave.com/search?q=",
        "https://duckduckgo.com/?q=",
        "https://yandex.com/search/?text=",
        "https://www.bing.com/search?q=",
        "https://www.qwant.com/?l=en&q=",
    ]

    def __init__(
        self,
        search_queries,
        *args,
        num_pages: Dict[str, int] = dict(yandex=1, brave=1, bing=1),
        url_patterns=[],
        url_anti_patterns=[],
        **kwargs,
    ):
        """
        Initialize the spider with the search query and the number of search result pages to scrape.

        Args:
            search_query (str): The search query to look for privacy policies on Google.
            num_pages (int, optional): The number of search result pages to scrape. Defaults to 1.
        """
        super().__init__(*args, **kwargs)
        self.start_urls = ["".join(prod) for prod in product(self.search_engines, search_queries)]
        self.num_pages = num_pages
        self.url_patterns: List[str] = url_patterns
        self.url_anti_patterns: List[str] = url_anti_patterns
        self.url_text_dict: Dict[str, str] = {}
        self.brave_offset = 0
        self.yandex_page = 0
        self.bing_first = 1

        try:
            # load the existing file into a dictionary
            with open(self.file_path, "r") as json_file:
                self.current_url_text_dict = json.load(json_file)
        except FileNotFoundError:
            self.current_url_text_dict = {}

    def _clean_url(self, url: str):
        """Doc."""

        return urlunparse(urlparse(url)._replace(query=""))

    def parse(self, response):
        """
        Parses the Google search results page and extracts the URLs of relevant search results.

        Args:
            response (HtmlResponse): The response object from the Google search results page.

        Yields:
            dict: A dictionary containing the URL of a relevant search result.
        """

        # Follow the "Next" button link recursively if more pages need to be scraped
        next_page_url = None
        if "google.com" in response.url:
            # Google
            search_engine_name = "Google"
            next_page_url = response.css("a#pnnext::attr(href)").get()

        elif "search.brave.com" in response.url:
            # Brave
            search_engine_name = "brave"
            self.brave_offset += 1
            original_query_url, *_ = response.url.split("&")
            next_page_url = original_query_url + f"&offset={self.brave_offset}"

        elif "yandex.com/search" in response.url:
            # Yandex
            search_engine_name = "yandex"
            self.yandex_page += 1
            original_query_url, *_ = response.url.split("&")
            next_page_url = original_query_url + f"&p={self.yandex_page}"

        elif response.url.startswith("https://duckduckgo.com/"):
            search_engine_name = "duckduckgo"

        elif response.url.startswith("https://www.bing.com/search"):
            search_engine_name = "bing"
            self.bing_first += 10
            original_query_url, *_ = response.url.split("&")
            next_page_url = original_query_url + f"&first={self.bing_first}"

        elif response.url.startswith("https://www.qwant.com"):
            search_engine_name = "qwant"

        else:
            # Unknown search engine
            search_engine_name = "Unknown?"

        # find potential links
        allowed_links_extractor = LinkExtractor(
            allow=self.url_patterns,
            deny=self.url_anti_patterns,
        )
        allowed_clean_urls = {
            clean_url
            for link in allowed_links_extractor.extract_links(response)
            if (clean_url := self._clean_url(link.url))
            not in set(self.current_url_text_dict.keys()) | set(self.url_text_dict.keys())
        }

        print(
            f"Found {len(allowed_clean_urls)} allowed links in {search_engine_name.capitalize()} search page."
        )

        # Get requests for all relevant links and extract the policies
        for url in allowed_clean_urls:
            yield scrapy.Request(url=url, callback=self.parse_text)

        if next_page_url and self.num_pages[search_engine_name] > 0:
            self.num_pages[search_engine_name] -= 1
            print(f"NEXT {search_engine_name.capitalize()} PAGE! ({self.num_pages} more...)")
            yield scrapy.Request(url=response.urljoin(next_page_url))

    def parse_text(self, response):
        """Doc."""

        # Find all the headlines and paragraphs on the page
        soup = BeautifulSoup(response.body, "html.parser")
        text_elements = ["h1", "h2", "h3", "h4", "h5", "h6", "p"]
        text_contents = [elem.get_text() for elem in soup.find_all(text_elements)]

        # Join the contents into plain text preserving the structure with newlines
        text = "\n".join(text_contents)
        dict_item = {self._clean_url(response.url): text}

        if text:
            self.url_text_dict |= dict_item

        yield dict_item

    def closed(self, reason):
        """
        Save the links list as a JSON file after the crawl is finished.
        Existing URLs are overwritten, new ones are added, and those not found in the latest crawl are kept.
        """

        # update current dict with new dict and overwrite
        self.current_url_text_dict.update(self.url_text_dict)
        with open(self.file_path, "w") as json_file:
            json.dump(self.current_url_text_dict, json_file, indent=4)


# Sample Test
if __name__ == "__main__":
    from scrapy.crawler import CrawlerProcess

    process = CrawlerProcess(
        settings={
            "AUTOTHROTTLE_ENABLED": True,
            "USER_AGENT": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "DOWNLOADER_MIDDLEWARES": {
                "privacy_policy_spider.RotateUserAgentMiddleware": 778
            },  # Use a unique priority}
            "COOKIES_ENABLED": False,
        }
    )

    process.crawl(
        SearchEngineTextSpider,
        search_queries=["privacy policy"],
        num_pages=dict(yandex=10, brave=10, bing=10),
        url_patterns=[r".+/.+privacy.+/?"],
        url_anti_patterns=[
            r"privacy.+(please)|(example)|(template)|(generator)",
            r"(query)|(search).+privacy",
            r".*\..*privacy.*\..*",
        ],
    )
    process.start()

    print("DONE!")
