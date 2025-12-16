import scrapy

class MyItem(scrapy.Item):

    PID = scrapy.Field()
    UID = scrapy.Field()
    like = scrapy.Field()
    time = scrapy.Field()
    post = scrapy.Field()

