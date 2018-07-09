import scrapy
import json
import sys
import pandas as pd
import csv

from footballData.items import Match

class MatchSpider(scrapy.Spider):
    name = "match"
    detailedStats = True
    lis = []
    stages = []
    matches = []

    def start_requests(self):
      # cups = self.cups
      urls = ['https://sofifa.com/teams/national?v=07&e=154994&set=true',      
'https://sofifa.com/teams/national?v=08&e=155359&set=true',      
'https://sofifa.com/teams/national?v=09&e=155725&set=true',      
'https://sofifa.com/teams/national?v=10&e=156090&set=true',      
'https://sofifa.com/teams/national?v=11&e=156455&set=true',      
'https://sofifa.com/teams/national?v=12&e=156820&set=true',      
'https://sofifa.com/teams/national?v=13&e=157312&set=true',
'https://sofifa.com/teams/national?v=14&e=157662&set=true',
'https://sofifa.com/teams/national?v=15&e=158033&set=true',
'https://sofifa.com/teams/national?v=16&e=158410&set=true',
'https://sofifa.com/teams/national?v=17&e=158774&set=true',
'https://sofifa.com/teams/national?v=18&e=159065&set=true']
      # urls = ['https://sofifa.com/teams/national?offset=0']
      headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:48.0) Gecko/20100101 Firefox/48.0'}

      for url in urls:
        yield scrapy.Request(url=url, callback=self.parse, headers=headers)
    
    #COUNTRY
    def parse(self,response):



        countries = response.xpath('//*[@id="pjax-container"]/table/tbody/tr[*]/td[2]/div/a/text()').extract()
        overall = response.xpath('//td[contains(@id,"oa")]//div/span/text()').extract()
        attack = response.xpath('//td[contains(@id,"at")]//div/span/text()').extract()
        mid = response.xpath('//td[contains(@id,"md")]//div/span/text()').extract()
        defence = response.xpath('//td[contains(@id,"df")]//div/span/text()').extract()

        year = response.xpath('/html/body/div[1]/div/div/div/ul/li[1]/a/text()').extract()[0].split(',')[1].strip(' ')

        year_list = [year]*len(countries)

        list1 = list(zip(year_list,countries,overall,attack,mid,defence))

        self.lis.extend(list1)

            
        labels = ['year_list','countries','overall','attack','mid','defence']
        df = pd.DataFrame.from_records(self.lis,columns = labels)
        df.to_csv('sofifa_team_list.csv',index=False)
 
