from icrawler.builtin import BingImageCrawler

crawler = BingImageCrawler(storage={"root_dir": "images/dogs"})
crawler.crawl(keyword="dog", max_num=1000)