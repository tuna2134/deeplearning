from icrawler.builtin import BingImageCrawler

crawler = BingImageCrawler(storage={"root_dir": "images/whale"},
                           downloader_threads=4)
crawler.crawl(keyword="クジラ", min_size=(50, 100), max_num=1000)