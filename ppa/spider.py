import asyncio
import logging
import re
import ssl
from collections import OrderedDict
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple
from urllib.parse import urljoin, urlparse

import hishel
import httpx
import trafilatura as trf
from bs4 import BeautifulSoup

from ppa.utils import timer

NEWLINE = "\n"


@dataclass
class Worker:
    """Doc."""

    idx: int
    raw_data_fpath: Path
    failed_domains: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.policies_found = 0
        self.is_alive = True

    async def work(
        self, domain_queue: asyncio.LifoQueue, async_client: hishel.AsyncCacheClient, **httpx_kwargs
    ):
        """Work until queue is finished or manually stopped"""

        logging.info(f"[worker {self.idx}] Started working...")
        while True:
            domain = await domain_queue.get()
            if domain is None:
                # Signal to exit when all tasks are done
                logging.info(f"[worker {self.idx}] Killed by 'None' poisoning.")
                domain_queue.task_done()
                break
            else:
                # fetch, extract and save policy to disk, if found
                try:
                    doc, issue = await self.extract_policy_to_disk(
                        async_client, domain, **httpx_kwargs
                    )
                except Exception as exc:
                    # [AttributeError] 'NoneType' object has no attribute 'text'
                    logging.info(
                        f"[worker {self.idx}] Destroyed by unhandled 'extract_policy_to_disk' error: [{exc.__class__.__name__}] {exc}"
                    )
                    domain_queue.task_done()
                    break

                # if a privacy policy was found, create a new file
                if doc is not None:
                    # Success!
                    self.policies_found += 1
                    logging.info(
                        f"[worker {self.idx}] '{domain}' - Found new privacy policy! ({self.policies_found:,} found)"
                    )
                    try:
                        await asyncio.to_thread(self._write_policy, domain, doc)
                    except Exception as exc:
                        logging.info(
                            f"[worker {self.idx}] Destroyed by unhandled '_write_policy' error: [{exc.__class__.__name__}] {exc}"
                        )
                        domain_queue.task_done()
                        break
                else:
                    # Failure!
                    try:
                        domain.encode("utf-8").decode("utf-8")
                    except UnicodeEncodeError:
                        # domain is somehow not in utf-8 - don't write to file!
                        logging.info(
                            f"[worker {self.idx}] Bad domain {domain}"
                        )  # TESTESTEST - see if able to print
                        domain_queue.task_done()
                        break
                    except Exception as exc:
                        logging.info(
                            f"[worker {self.idx}] Destroyed by unhandled 'encode/decode' error: [{exc.__class__.__name__}] {exc}"
                        )
                        domain_queue.task_done()
                        break
                    else:
                        self.failed_domains.append(
                            f"{domain}, {issue.replace(',', '').replace(NEWLINE, ' ')}\n"
                        )
                #                                     logging.info(f"[worker {self.idx}] Failed getting policy from ('{domain}') - {issue}.")

                # mark task as done
                domain_queue.task_done()

        # mark task as done
        domain_queue.task_done()

        # mark as dead
        self.is_alive = False

    async def extract_policy_to_disk(  # NOQA # C901
        self, async_client: hishel.AsyncCacheClient, domain: str
    ) -> Tuple[str, str]:
        """Doc."""

        doc: str = None
        issue: str = ""
        # fetch domain URL
        try:
            response = await self.fetch_url(  # type: ignore
                async_client,
                f"https://{domain}",
            )
        except (ssl.SSLError, httpx.HTTPError) as exc:
            issue = f"Domain fetching failed: [{exc.__class__.__name__}] {exc}"
        except Exception as exc:  # TESTESTEST
            issue = f"FIXME - error in line 129 [{exc.__class__.__name__}] {exc}"
        else:
            try:
                fetched_links = self.extract_links(response)
                privacy_policy_link = self._filter_links(fetched_links)

            except IndexError as exc:
                issue = f"No privacy links found: [{exc.__class__.__name__}] {exc}"
            except AttributeError as exc:
                issue = f"'response' is None: [{exc.__class__.__name__}] {exc}"
            except Exception as exc:  # TESTESTEST
                issue = f"FIXME - error in lines 137 or 138 [{exc.__class__.__name__}] {exc}"

            else:
                # now fetch the privacy policy URL
                try:
                    pp_response = await self.fetch_url(  # type: ignore
                        async_client,
                        privacy_policy_link,
                        archived=False,  # don't allow looking at internet archive for sublinks
                    )
                except (ssl.SSLError, httpx.HTTPError) as exc:
                    issue = f"Privacy policy link fetching failed: [{exc.__class__.__name__}] {exc}"
                except Exception as exc:  # TESTESTEST
                    issue = f"FIXME - error in line 152 [{exc.__class__.__name__}] {exc}"
                else:
                    # extract the response main-text with Trafilatura
                    doc = trf.extract(
                        pp_response.text,
                        url=str(pp_response.url),
                        include_tables=True,
                        include_links=True,
                    )
                    # try getting privacy sub-links (see 'europa.eu' for example) if 'policy' text is too short (find threshold!)
                    if not doc or len(doc) < 2000:
                        try:
                            fetched_links = self.extract_links(pp_response)
                            privacy_policy_link = self._filter_links(fetched_links)
                        except IndexError as exc:
                            issue = f"No privacy sublinks found: [{exc.__class__.__name__}] {exc}"
                        except AttributeError as exc:
                            issue = f"'pp_response' is None: [{exc.__class__.__name__}] {exc}"
                        except Exception as exc:  # TESTESTEST
                            issue = f"FIXME - error in lines 167 or 168 [{exc.__class__.__name__}] {exc}"

                        else:
                            # getting new response with sub-link
                            try:
                                pp_response = await self.fetch_url(  # type: ignore
                                    async_client,
                                    privacy_policy_link,
                                    archived=False,  # don't allow looking at internet archive for sublinks
                                )
                            except (ssl.SSLError, httpx.HTTPError, ValueError) as exc:
                                issue = f"Privacy policy sublink fetching failed: [{exc.__class__.__name__}] {exc}"
                            except Exception as exc:  # TESTESTEST
                                issue = (
                                    f"FIXME - error in line 185 [{exc.__class__.__name__}] {exc}"
                                )
                            else:
                                # extract the response main-text with Trafilatura
                                doc = trf.extract(
                                    pp_response.text,
                                    url=str(pp_response.url),
                                    include_tables=True,
                                    include_links=True,
                                )

        # return the potential document
        return doc, issue

    async def fetch_url(
        self, async_client: hishel.AsyncCacheClient, url: str, archived=True
    ) -> httpx.Response:
        """Get a single response, asynchronously. Falls back to Internet Archive if can't access page."""

        try:
            response = await async_client.get(url, follow_redirects=True)
            #             return response
            # deal with language selection
            if "/select-language" in (redirected_url := str(response.url)):
                url = redirected_url.split("/select-language")[0] + "/en"
                response = await async_client.get(url, follow_redirects=True)

            # If successful call
            if response.status_code == 200:
                return response
            elif archived:
                # try archived version
                return await self.fetch_url(
                    async_client,
                    "https://web.archive.org/web/20/" + url,
                    archived=False,
                )
            else:
                response.raise_for_status()

        except (httpx.TimeoutException, httpx.ConnectError, hishel.ValidationError) as exc:
            # try archived version
            if archived:
                return await self.fetch_url(
                    async_client,
                    "https://web.archive.org/web/20/" + url,
                    archived=False,
                )
            else:
                raise exc

        except Exception as exc:  # TESTESTEST
            raise RuntimeError(f"FIXME - error in lines 201-219 [{exc.__class__.__name__}] {exc}")

    def extract_links(
        self,
        response: httpx.Response,
        sublinks_only: bool = False,
    ) -> List[str]:
        """Extracts links from the given response."""

        soup = BeautifulSoup(response.content, "html.parser")
        base_url = str(response.url)

        # get visible links (Ignore original URL and its variations with #)
        links = [
            urljoin(base_url, link.get("href"))
            for link in soup.find_all("a")
            if link.get("href") and link.text.strip()
        ]

        # Ignore original URL and its variations with #
        links = [link for link in links if urlparse(link).fragment == "" and link != base_url]

        if sublinks_only:
            base_url_len = len(base_url)
            links = [
                link for link in links if link.startswith(base_url) and len(link) > base_url_len
            ]

        # remove duplicates while maintaining order
        links = list(OrderedDict.fromkeys(links))

        return links

    def _write_policy(self, domain: str, text: str):
        """Doc."""

        save_fpath = self.raw_data_fpath / f"{domain}.txt"
        with open(save_fpath, "w", encoding="utf-8") as text_file:
            text_file.write(text)

    def _filter_links(self, links):
        """Determine the correct privacy-policy link to use given a list of links, based on several heuristics"""

        GOOGLE_INT_PP_PATTERN = r".*google\.com/intl/.*/policies/privacy.*"

        # Filter only links containing the word "privacy" and not containing a '@'
        privacy_policy_links = [
            link for link in links if link.find("privacy") != -1 and link.find("@") == -1
        ]
        # Sort such that google privacy policy-like links appear last, and "https://policies.google.com/privacy" the first among them
        privacy_policy_links = sorted(
            privacy_policy_links,
            key=lambda x: ("google" in x, x != "https://policies.google.com/privacy"),
        )
        # filtering in case more that one 'privacy' link is found
        if len(privacy_policy_links) > 1:
            # filter for english versions
            filtered_policy_links = [
                link
                for link in privacy_policy_links
                if self._contains_any(link, ("/en/", "/en-", "hl=en"))
            ]
            if filtered_policy_links:
                privacy_policy_links = filtered_policy_links
        # replace all links beginning with "https://policies.google.com/privacy" with the English version
        privacy_policy_links = [
            "https://policies.google.com/privacy?hl=en"
            if ("https://policies.google.com/privacy" in link)
            or re.match(GOOGLE_INT_PP_PATTERN, link)
            else link
            for link in privacy_policy_links
        ]

        # return the first link
        return privacy_policy_links[0]

    def _contains_any(self, target_string, strings_to_check):
        """Doc."""

        return any(check_string in target_string for check_string in strings_to_check)


@dataclass
class PrivacyPolicyExtractor:
    """Doc."""

    raw_data_fpath: Path = Path("./policies")

    @timer(1000, beep=True)
    async def extract_domains(  # NOQA # C901
        self,
        domains: List[str],
        testing=False,
        update: bool = False,
        retry_issues: List[str] = [
            "[ConnectError] [Errno 11001] getaddrinfo failed",
            "Timeout",
        ],
        max_connections=None,
        max_keepalive_connections=None,
        keepalive_expiry=20,
        n_workers=100,
        **httpx_kwargs,
    ):
        """Doc."""

        # filter already extracted domains (if any)
        original_n_domains = len(domains)
        if not update:
            extracted_domains_set = [fpath.stem for fpath in self.raw_data_fpath.rglob("*.txt")]
            if len(extracted_domains_set):
                logging.info(
                    f"Ignoring {(n_existing := len(extracted_domains_set)):,} domains for which a policy is already extracted ({n_existing/original_n_domains:.1%})."
                )
                domains = list(set(domains) - set(extracted_domains_set))

        try:
            # Compactify the failed domains file, leaving only the latest issues and
            domain2issue = {}
            with open(
                self.raw_data_fpath / "failed_domains.txt", mode="r", encoding="utf-8"
            ) as infile:
                for line in infile:
                    try:
                        domain, issue = line.strip().split(", ")
                    except ValueError:
                        issue = ""
                    domain2issue[domain] = issue

            # removing already found files
            for key in extracted_domains_set:
                domain2issue.pop(key, None)

            # rewrite the failed domains file with only the latest issue
            with open(
                self.raw_data_fpath / "failed_domains.txt", mode="w", encoding="utf-8"
            ) as outfile:
                outfile.writelines(
                    [f"{domain}, {issue}\n" for domain, issue in domain2issue.items()]
                )

            # define failed domains as all those in the file minus those with issues in 'retry_issues'
            failed_domains = (
                [
                    domain
                    for domain, issue in domain2issue.items()
                    if not any([issue_str in issue for issue_str in retry_issues])
                ]
                if retry_issues
                else list(domain2issue.values())
            )

            logging.info(
                f"Ignoring {(n_failed := len(failed_domains)):,} previously-failed domains for extraction ({n_failed/original_n_domains:.1%})."
            )
            if n_retried := len(domain2issue) - n_failed:
                logging.info(f"Retrying {n_retried:,} domains.")
            domains = list(set(domains) - set(failed_domains))

        except FileNotFoundError:
            # create the failed domains file if missing
            with open(self.raw_data_fpath / "failed_domains.txt", mode="w", encoding="utf-8"):
                pass

        logging.info(f"Data extraction started ({len(domains):,} domains).")

        # Make requests asynchronously
        async with hishel.AsyncCacheClient(
            headers={"Accept-Language": "en-US,en;q=0.9"},
            limits=httpx.Limits(
                max_connections=max_connections,
                max_keepalive_connections=max_keepalive_connections,
                keepalive_expiry=keepalive_expiry,
            ),
            **httpx_kwargs,
        ) as async_client:

            # Enqueue domains
            domain_queue: asyncio.LifoQueue = asyncio.LifoQueue()
            for domain in domains:
                await domain_queue.put(domain)

            # Instantiate workers
            workers = [Worker(idx, self.raw_data_fpath) for idx in range(n_workers)]
            try:
                # create a task from each worker
                [
                    asyncio.create_task(worker.work(domain_queue, async_client, **httpx_kwargs))
                    for worker in workers
                ]
                # Wait until all tasks are done
                await domain_queue.join()
            except asyncio.CancelledError as exc:
                # Cancel all workers properly by 'None' poisoning
                logging.info(f"[extract_domains] {exc.__class__.__name__} - Killing all workers...")
                # empty the domain_queue
                with suppress(asyncio.QueueEmpty):
                    while True:
                        domain_queue.get_nowait()
                        domain_queue.task_done()
                    domain_queue.task_done()
                # poison all living workers
                for _ in range(len([worker for worker in workers if worker.is_alive])):
                    domain_queue.put_nowait(None)
                # ensure all dead
                with suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(domain_queue.join(), timeout=60.0)
            except ValueError as exc:
                logging.error(f"[extract_domains] [{exc.__class__.__name__}] {exc}. Quitting.")
            except Exception as exc:
                logging.error(
                    f"[extract_domains] Unhandled error: [{exc.__class__.__name__}] {exc}. Quitting."
                )
            else:
                # Went over all domains. Hooray!
                logging.info("[extract_domains] Data extraction finished.")
            finally:
                # groups "failed domain, issue" strings from all workers in a single list
                new_failed_domains = [str_ for worker in workers for str_ in worker.failed_domains]
                with open(
                    self.raw_data_fpath / "failed_domains.txt", mode="a", encoding="utf-8"
                ) as failed_domains_file:
                    failed_domains_file.writelines(new_failed_domains)
                # statistics
                print("# Policies found:")
                for worker in workers:
                    print(f"Worker {worker.idx}: {worker.policies_found:,}")
                print(f"Total: {sum([worker.policies_found for worker in workers])}")
