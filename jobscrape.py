# 
# Jurafsky NLP videos https://www.youtube.com/playlist?list=PL6397E4B26D00A269
# Jurafsky NLP notes: https://web.stanford.edu/~jurafsky/NLPCourseraSlides.html
# https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-1-for-beginners-bag-of-words
# https://web.stanford.edu/~jurafsky/NLPCourseraSlides.html

import re
import tqdm
import locale
import json
import sys
from bs4 import BeautifulSoup
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from concurrent import futures
from collections import namedtuple


BASE_URL = 'http://www.indeed.com'
SEARCH_URL = 'https://www.indeed.com/jobs?{0}'
NUM_THREADS = 50

Posting = namedtuple('Posting', ['url', 'text'])

def get_response(url):
    """ Opens the url. """
    req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    return urlopen(req)


def make_soup(url):
    """ Make BeautifulSoup from the url. """
    html = get_response(url).read()
    return BeautifulSoup(html, "lxml")


def make_query_url(query,location,start=0):
    """ Constructs an Indeed.com query url. """
    params = urlencode({ 'q': query, 'l': location, 'start': start})
    url = SEARCH_URL.format(params)
    return url


def test_query_url():
    qurl = "https://www.indeed.com/jobs?l=Tucson%2C+AZ&q=data+science"
    assert(make_query_url("data science", "Tucson, AZ") == qurl)


def get_search_info(url):
    """ Returns <initial>,<final>,<total> where
        these are given on the search page by the string
        'Jobs <initial> to <final> of <total>' """

    soup = make_soup(url)
    locale.setlocale( locale.LC_ALL, 'en_US.UTF-8' ) 
    fn = locale.atoi
    try:
        div = soup.find('div', id='searchCount')
        texts = div.text.split()
        # "Jobs 1 to 10 of 283"
        initial,final,total = texts[1],texts[3],texts[5]
        
        return fn(initial),fn(final),fn(total)
    except:
        return None,None,None

def get_job_links(soup):
    divs = soup.find_all('div', 'result')
    anchors = [div.find('a', 'turnstileLink') for div in divs]
    links = [BASE_URL + a['href'] for a in anchors]
    return links

def visible(element):
    if element.parent.name in ['style', 'script', '[document]', 'head', 'title']:
        return False
    elif re.match('<!--.*-->', str(element.encode('utf-8'))):
        return False
    return True
 

def get_text(url):
    """ Returns visible text, extracted from the location at url. """
    soup = make_soup(url)
    # https://www.quora.com/How-can-I-extract-only-text-data-from-HTML-pages
    data = soup.find_all(text=True)
    results = filter(visible, data)
    return ' '.join([str(s) for s in results])

def search_page_urls(query,location):
    """ Return a list of search urls for a multi-page search. """
    url = make_query_url(query,location)
    _,increment,total = get_search_info(url)
    num_iters = total // increment
    page_urls = [make_query_url(query,location,i*increment)
                 for i in range(num_iters)]
    return page_urls


def job_search(query,location):
    """ Return all Indeed.com job links associated
        with the given query and location."""
    links = []
    page_urls = search_page_urls(query,location)

    with futures.ThreadPoolExecutor(NUM_THREADS) as ex:
        to_do = [ex.submit(lambda url: get_job_links(make_soup(url)),url)
                 for url in page_urls]
        done = futures.as_completed(to_do)
        done = tqdm.tqdm(done, total=len(page_urls))
        for future in done:
            links += future.result()
    return links


def get_posting(url):
    """ Get the end job url and visible page text. """
    try:
        redirected = get_response(url).geturl()
        text = get_text(redirected)
        return Posting(redirected,text)
    except:
        return None


def get_job_postings(links):
    """ Returns all job postings associated with the list
        of Indeed.com job links. A posting is a pair
        <url>,<text>. """
    postings = []
    count = 0
    with futures.ThreadPoolExecutor(NUM_THREADS) as ex:
        to_do = [ex.submit(get_posting,link) for link in links]
        done = futures.as_completed(to_do)
        done = tqdm.tqdm(done, total=len(links))
        for future in done:
            posting = future.result()
            if posting is not None:
                postings.append(posting)
    return postings

def save_postings(postings, filename):
    with open(filename, "w") as f:
        json.dump(postings, f)

def load_postings(filename):
    with open(filename) as f:
        return [Posting(*data) for data in json.load(f)]

def main(query,location,outfile):
    print("Getting job links...")
    links = job_search(query,location)
    print("Getting postings...")
    postings = get_job_postings(links)
    print("{} postings found. Saving...".format(len(postings)))
    save_postings(postings,outfile)


if __name__=="__main__":
    try:
        query,location,outfile = sys.argv[1:]
    except:
        print("Error: Usage is python3 jobscrape.py <query> <location> <outfile>")
        sys.exit()
    main(query,location,outfile)
