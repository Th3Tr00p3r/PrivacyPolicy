import json
from typing import List

import scrapy

# from scrapy.http import HtmlResponse
from scrapy.linkextractors import LinkExtractor


class GooglePrivacyPolicySpider(scrapy.Spider):
    """
    A Scrapy spider to automatically scrape privacy policies from Google search results.
    """

    name = "google_privacy_policy"
    start_urls: List[str]
    links: List[str]

    def __init__(self, search_query, num_pages=1, *args, **kwargs):
        """
        Initialize the spider with the search query and the number of search result pages to scrape.

        Args:
            search_query (str): The search query to look for privacy policies on Google.
            num_pages (int, optional): The number of search result pages to scrape. Defaults to 1.
        """
        super(GooglePrivacyPolicySpider, self).__init__(*args, **kwargs)
        self.start_urls = ["https://www.google.com/search?q=" + search_query]
        self.num_pages = num_pages
        self.links = []

    def parse(self, response):
        """
        Parses the Google search results page and extracts the URLs of relevant search results.

        Args:
            response (HtmlResponse): The response object from the Google search results page.

        Yields:
            dict: A dictionary containing the URL of a relevant search result.
        """
        link_extractor = LinkExtractor(
            allow=r".+", deny=r"google.com"
        )  # Allow all links except google.com
        links = link_extractor.extract_links(response)

        for link in links:
            self.links.append(link.url)
            yield {"url": link.url}

        # Follow the "Next" button link and continue to the next search result page
        next_page_link = response.css("a#pnnext::attr(href)").get()
        if next_page_link and self.num_pages > 1:
            yield scrapy.Request(
                url=response.urljoin(next_page_link), callback=self.parse_next_page
            )

    def parse_next_page(self, response):
        """
        Callback function to parse the next search result page and continue extracting URLs.

        Args:
            response (HtmlResponse): The response object from the next search result page.

        Yields:
            dict: A dictionary containing the URL of a relevant search result.
        """
        link_extractor = LinkExtractor(
            allow=r".+", deny=r"google.com"
        )  # Allow all links except google.com
        links = link_extractor.extract_links(response)

        for link in links:
            self.links.append(link.url)
            yield {"url": link.url}

        # Follow the "Next" button link recursively if more pages need to be scraped
        next_page_link = response.css("a#pnnext::attr(href)").get()
        if next_page_link and self.num_pages > 1:
            yield scrapy.Request(
                url=response.urljoin(next_page_link), callback=self.parse_next_page
            )

    def closed(self, reason):
        """
        The closed method is called when the spider is closed. It will be executed after the crawl is finished.

        Args:
            reason (str): The reason the spider was closed.
        """
        # Save the links list as a JSON file after the crawl is finished

        # TODO: add datetime to filename so they do not overwrite (and can be unified later)
        with open("./data/links.json", "w") as json_file:
            json.dump(self.links, json_file, indent=4)


# Sample Test
if __name__ == "__main__":
    from scrapy.crawler import CrawlerProcess

    search_query = "privacy policy"  # Replace example.com with the domain you want to search
    num_pages_to_scrape = 3  # Specify the number of search result pages to scrape
    process = CrawlerProcess(
        settings={
            "USER_AGENT": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
    )

    process.crawl(
        GooglePrivacyPolicySpider, search_query=search_query, num_pages=num_pages_to_scrape
    )
    process.start()
    print("DONE!")
