import scrapy
from genshin.items import MyItem

class genshinSpider(scrapy.spiders.Spider):
    name = "genshin" #爬虫的名字是genshin，可以根据需求自由调整
    allowed_domains = ["https://fate.5ch.net/"] #允许爬取的网站域名

    start_urls= ["https://fate.5ch.net/test/read.cgi/gamef/1605888051/"] #初始URL，即爬虫爬取的第一个URL

    def parse(self, response):
        for i in range(1000):
            if i + 1 not in range(1, 5):  # Skip if i+1 is 1, 2, or 3
                item = MyItem()

                item['PID'] = response.xpath(
                    f"/html/body/div[2]/div/div[2]/article[{i + 1}]/details/summary/span[1]/text()").extract()
                item['UID'] = response.xpath(
                    f"/html/body/div[2]/div/div[2]/article[{i + 1}]/details/span[2]/text()").extract()
                item['name'] = response.xpath(
                    f"/html/body/div[2]/div/div[2]/article[{i + 1}]/details/summary/span[2]/b/a/text()").extract()
                item['time'] = response.xpath(
                    f"/html/body/div[2]/div/div[2]/article[{i + 1}]/details/span[1]/text()").extract()
                item['post'] = response.xpath(
                    f"/html/body/div[2]/div/div[2]/article[{i + 1}]/section/text()").extract()

                if item['PID'] or item['UID'] or item['name'] or item['time'] or item['post']:
                    yield item

                else:
                    print("All values are empty for item at index:", i + 1)