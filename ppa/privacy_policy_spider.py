from typing import List

import scrapy
from scrapy.linkextractors import LinkExtractor

# from scrapy.http import HtmlResponse


class GooglePrivacyPolicySpider(scrapy.Spider):
    """
    A Scrapy spider to automatically scrape privacy policies from Google search results.
    """

    name = "google_privacy_policy"
    start_urls: List[str]

    def __init__(self, search_query, *args, **kwargs):
        """
        Initialize the spider with the search query.

        Args:
            search_query (str): The search query to look for privacy policies on Google.
        """

        super().__init__(*args, **kwargs)
        self.start_urls = ["https://www.google.com/search?q=" + search_query]

    def parse(self, response):
        """
        Parses the Google search results page and extracts the URLs of relevant search results.

        Args:
            response (HtmlResponse): The response object from the Google search results page.

        Yields:
            dict: A dictionary containing the URL of a relevant search result.
        """

        link_extractor = LinkExtractor()
        for link in link_extractor.extract_links(response):
            yield {"url": link.url}

    def parse_privacy_policy(self, response):
        """
        Parses the privacy policy page and extracts the content.

        Args:
            response (HtmlResponse): The response object from the privacy policy page.

        Returns:
            str: The extracted privacy policy content as a string.
        """

        # Your logic to extract the privacy policy content goes here
        pass


if __name__ == "__main__":
    from scrapy.crawler import CrawlerProcess

    search_query = "privacy policy"
    process = CrawlerProcess(
        settings={
            "USER_AGENT": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
    )

    process.crawl(GooglePrivacyPolicySpider, search_query=search_query)
    a = process.start()
    print("DONE!")
