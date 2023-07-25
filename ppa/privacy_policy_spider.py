import json
from itertools import product
from typing import Dict, List

import numpy as np
import scrapy
from bs4 import BeautifulSoup

# from scrapy.http import HtmlResponse
from scrapy.linkextractors import LinkExtractor


class SearchEngineSpider(scrapy.Spider):
    """
    A Scrapy spider to automatically scrape links from search results.
    """

    name = "search engine"
    start_urls: List[str]
    search_engines = [
        "https://www.google.com/search?q=",
        "https://search.brave.com/search?q=",
    ]

    def __init__(
        self, search_queries, *args, num_pages=1, url_patterns=[], url_anti_patterns=[], **kwargs
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

    def parse(self, response):
        """
        Parses the Google search results page and extracts the URLs of relevant search results.

        Args:
            response (HtmlResponse): The response object from the Google search results page.

        Yields:
            dict: A dictionary containing the URL of a relevant search result.
        """

        #        all_links_extractor = LinkExtractor( # TESTESTEST
        #            allow=r".+", deny=r"google.com"
        #        )  # Allow all links except google.com
        #        all_links = all_links_extractor.extract_links(response) # TESTESTEST
        #        print(f"Found {len(all_links)} links in search page.") # TESTESTEST
        #        print("\n".join([link.url for link in all_links])) # TESTESTEST

        allowed_links_extractor = LinkExtractor(
            allow=self.url_patterns,
            deny=self.url_anti_patterns,
        )
        allowed_links = allowed_links_extractor.extract_links(response)

        print(f"Found {len(allowed_links)} allowed links in search page.")  # TESTESTEST

        # Get requests for all relevant links and extract the policies
        for link in allowed_links:
            if link.url not in self.url_text_dict:
                yield scrapy.Request(url=link.url, callback=self.parse_text)

        # Follow the "Next" button link recursively if more pages need to be scraped
        if "google.com" in response.url:
            # Google
            next_page_url = response.css("a#pnnext::attr(href)").get()
        elif "search.brave.com" in response.url:
            # Brave
            next_page_url = response.css("a.pagination-item--next::attr(href)").get()
        else:
            # Unknown search engine
            next_page_url = None

        if next_page_url and self.num_pages > 0:
            print("NEXT PAGE!")  # TESTESTEST
            self.num_pages -= 1
            yield scrapy.Request(url=response.urljoin(next_page_url))

    def parse_text(self, response):
        """Doc."""

        # Find all the headlines and paragraphs on the page
        soup = BeautifulSoup(response.body, "html.parser")
        text_elements = ["h1", "h2", "h3", "h4", "h5", "h6", "p"]
        text_contents = [elem.get_text() for elem in soup.find_all(text_elements)]

        # Join the contents into plain text preserving the structure with newlines
        text = "\n".join(text_contents)

        dict_item = {response.url: text}
        self.url_text_dict |= dict_item
        yield dict_item

    def closed(self, reason):
        """Save the links list as a JSON file after the crawl is finished"""

        # TODO: add datetime to filename so they do not overwrite (and can be unified later)
        with open("./data/url_text.json", "w") as json_file:
            json.dump(self.url_text_dict, json_file, indent=4)


# Sample Test
if __name__ == "__main__":
    from scrapy.crawler import CrawlerProcess

    process = CrawlerProcess(
        settings={
            "USER_AGENT": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
    )
    process.crawl(
        SearchEngineSpider,
        search_queries=["privacy policy"],
        num_pages=np.inf,
        url_patterns=[r".+/.+privacy.+/?"],
        url_anti_patterns=[
            r"privacy.+(please)|(example)|(template)|(generator)",
            r"(query)|(search).+privacy",
            r".*\..*privacy.*\..*",
        ],
    )
    process.start()

    print("DONE!")
