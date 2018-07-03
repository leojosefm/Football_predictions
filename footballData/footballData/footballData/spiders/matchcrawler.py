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
    # allowed_domains = ["football-data.mx-api.enetscores.com",'json.mx-api.enetscores.com']
    # allowed_domains = ["football-lineups.com"]
    # cups = ['Africa_Cup_2017', 'AFC_Asian_Cup_2015', 'CONCACAF_Gold_Cup_2013', 'Copa_America_2016', 'Euro_2016', 'Friendlies_2018'
    		# , 'Confederations_Cup_2017', 'World_Cup_Russia_2018']

    # start_urls = ["http://football-data.mx-api.enetscores.com/page/xhr/standings/"]
    # urls = ["https://www.football-lineups.com/tourn/Friendlies_2007/fixture/"]
    # headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:48.0) Gecko/20100101 Firefox/48.0'}
    # countries = ['Scotland']

    # seasons = ['2015/2016','2014/2015','2013/2014']

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
      # fun()

      # print(self.lis)
      # with open("output.csv", "wb") as f:
      #       writer = csv.writer(f)
      #       writer.writerows(self.lis)
    
    #COUNTRY
    def parse(self,response):

        # base_url = "https://www.football-lineups.com"

        # print(response.url)
        # filename = response.url.split("/")[-1]

        # with open(filename, 'wb') as f:
        #     f.write(response.body)
        # self.log('Saved file %s' % filename)


        countries = response.xpath('//*[@id="pjax-container"]/table/tbody/tr[*]/td[2]/div/a/text()').extract()
        overall = response.xpath('//td[contains(@id,"oa")]//div/span/text()').extract()
        attack = response.xpath('//td[contains(@id,"at")]//div/span/text()').extract()
        mid = response.xpath('//td[contains(@id,"md")]//div/span/text()').extract()
        defence = response.xpath('//td[contains(@id,"df")]//div/span/text()').extract()

        year = response.xpath('/html/body/div[1]/div/div/div/ul/li[1]/a/text()').extract()[0].split(',')[1].strip(' ')

        year_list = [year]*len(countries)


        list1 = list(zip(year_list,countries,overall,attack,mid,defence))

        # print(countries)
        # print(list1)

        self.lis.extend(list1)

        print(len(self.lis))
        

        if len(self.lis) == 532:
            # print(len(sels))
            print(len(self.lis))
            print(self.lis)
            labels = ['year_list','countries','overall','attack','mid','defence']
            # with open("output.csv", "wb") as f:
            #     writer = csv.writer(f)
            #     writer.writerows(self.lis)
            # df = pd.DataFrame(columns = labels)
            df = pd.DataFrame.from_records(self.lis,columns = labels)
            # for i in len(self.lis):
            # 	print(self.lis[i])

            # 	df = df.concat(pd.DataFrame.from_records(self.lis[i]))

            # df = pd.DataFrame.from_records(self.lis)
            df.to_csv('sofifa_team_list.csv')
        # print(self.lis)
        # print(len(self.lis))

        # with open(filename, 'wb') as f:

        # for match in match_urls:
        # 	yield scrapy.Request(url=base_url+match, callback=self.parseLeague) #meta={'country':country}
    
    #LEAGUE
    def parseLeague(self,response):
        country = response.meta['country']
        selection = response.xpath('//div[@class="mx-dropdown mx-country-template-stage-selector"]/ul/li/text()').extract()
        # league =s election[322]
        
        href = response.xpath('//li[text()[contains(.,"'+league+'")]]/@data-snippetparams').re_first('"params":"(.+)"')
        url = 'http://football-data.mx-api.enetscores.com/page/xhr/standings/' + href
        yield scrapy.Request(url, callback=self.parseSeason,meta={'country':country,'league':league})
      
    #SEASON  
    def parseSeason(self,response):
        country = response.meta['country']
        league = response.meta['league']
        
        for season in self.seasons:
            href = response.xpath('//li[text()[contains(.,"'+season+'")]]/@data-snippetparams').re_first('"params":"(.+)"')
            url = 'http://football-data.mx-api.enetscores.com/page/xhr/standings/' + href
            yield scrapy.Request(url, callback=self.parseMatches,meta={'country':country,'league':league,'season':season})
    
    #OPEN SEASON
    def parseMatches(self,response):
        country = response.meta['country']
        league = response.meta['league']
        season = response.meta['season']
        href = response.xpath('//div[contains(@class,"mx-matches-finished-betting_extended")]/@data-params').re_first('params":"(.+)/')
        url = 'http://football-data.mx-api.enetscores.com/page/xhr/stage_results/' + href
        first_stage_url = url + '/1'
        yield scrapy.Request(first_stage_url, callback=self.parseStage, meta={'href':href,'country':country,'league':league,'season':season})
    
    #LOOP STAGES
    def parseStage(self,response):
        country = response.meta['country']
        league = response.meta['league']
        season = response.meta['season']
        href = response.meta['href']
        
        url = 'http://football-data.mx-api.enetscores.com/page/xhr/stage_results/' + href
        totalPages = response.xpath('//span[contains(@class,"mx-pager-next")]/@data-params').re_first('total_pages": "([0-9]+)"')
        
        if not self.stages:
            iterateStages = range(1,int(totalPages)+1)
        else:
            iterateStages = self.stages
            
        for stage in iterateStages:
            full_stage_url = url + '/' + str(stage)
            yield scrapy.Request(full_stage_url, callback=self.parseAllMatchesInStage,dont_filter = True, meta={'stage':stage,'country':country,'league':league,'season':season})
            
    #MATCHES IN STAGE  
    def parseAllMatchesInStage(self, response):
        country = response.meta['country']
        league = response.meta['league']
        season = response.meta['season']
        stage = response.meta['stage']
        matchesDataEventList = response.xpath('//a[@class="mx-link mx-hide"]/@data-event').extract()
        dateList = response.xpath('//span[@class="mx-time-startdatetime"]/text()').extract()
     
        matchList = list()
        if len(self.matches) >= 1:
            for match in self.matches:
                matchList.append(matchesDataEventList[match-1])
        else:
            matchList = list(matchesDataEventList)
            
        counter = 0
        for matchId in matchList:
            match = Match()
            match["matchId"] = matchId
            match["country"] = country
            match["league"] = league
            match["season"] = season
            date = dateList[counter]
            match["date"] = date
            
            url = 'http://football-data.mx-api.enetscores.com/page/xhr/match_center/' + matchId + '/'
            counter += 1
            yield scrapy.Request(url, callback=self.parseMatchGeneralStats,meta={'match':match})
            
    #MATCH GENERAL STATS
    def parseMatchGeneralStats(self, response):
        match = response.meta['match']
        
        stage = response.xpath('//span[@class="mx-stage-name"]/text()').re_first('\s([0-9]+)')
        match["stage"] = stage
        
        fullTeamName = response.xpath('//div[@class="mx-team-away-name mx-break-small"]/a/text()').re('\t+([^\n]+[^\t]+)\n+\t+')
        teamId = response.xpath('//div[@class="mx-team-away-name mx-break-small"]/a/@data-team').extract()
        teamAcronym = response.xpath('//div[@class="mx-team-away-name mx-show-small"]/a/text()').re('\t+([^\n]+[^\t]+)\n+\t+')
        homeTeamGoal = response.xpath('//div[@class="mx-res-home mx-js-res-home"]/@data-res').extract_first()
        awayTeamGoal = response.xpath('//div[@class="mx-res-away mx-js-res-away"]/@data-res').extract_first()
        
        match['homeTeamFullName'] = fullTeamName[0]
        match['awayTeamFullName'] = fullTeamName[1]
        match['homeTeamAcronym'] = teamAcronym[0]
        match['awayTeamAcronym'] = teamAcronym[1]
        match['homeTeamId'] = teamId[0]
        match['awayTeamId'] = teamId[1]
        match['homeTeamGoal'] = homeTeamGoal
        match['awayTeamGoal'] = awayTeamGoal
        matchId = match['matchId']
        
        url = 'http://football-data.mx-api.enetscores.com/page/xhr/event_gamecenter/' + matchId + '%2Fv2_lineup/'
        yield scrapy.Request(url, callback=self.parseSquad,meta={'match':match})

    #MATCH SQUADS
    def parseSquad(self, response):
        match = response.meta['match']
        players = response.xpath('//div[@class="mx-lineup-incident-name"]/text()').extract()
        playersId = response.xpath('//a/@data-player').extract()
        subsId = response.xpath('//div[@class="mx-lineup-container mx-float-left"]//div[@class="mx-collapsable-content"]//a/@data-player').extract()
        titularPlayerId = [x for x in playersId if x not in subsId]
        
        # player x y pitch position
        playersX = response.xpath('//div[contains(@class,"mx-lineup-pos")]/@class').re('mx-pos-row-([0-9]+)\s')
        playersY = response.xpath('//div[contains(@class,"mx-lineup-pos")]/@class').re('mx-pos-col-([0-9]+)\s')
        playersPos = response.xpath('//div[contains(@class,"mx-lineup-pos")]/@class').re('mx-pos-([0-9]+)\s')
        
        match['homePlayers'] = players[:11]
        match['homePlayersId'] = titularPlayerId[:11]
        match['homePlayersX'] = playersX[:11]
        match['homePlayersY'] = playersY[:11]       
       
        match['awayPlayers'] = players[11:]
        match['awayPlayersId'] = titularPlayerId[11:22]
        match['awayPlayersX'] = playersX[11:]
        match['awayPlayersY'] = playersY[11:]
        
        matchId = match['matchId']
        if self.detailedStats:
            url = 'http://json.mx-api.enetscores.com/live_data/actionzones/' + matchId + '/0?_=1'
            yield scrapy.Request(url, callback=self.parseMatchEvents,meta={'match':match})
        else:
            yield match
    
    #MATCH EVENTS
    def parseMatchEvents(self, response):
        #match = response.meta['match']
        #matchId = match['matchId']
        #url = 'http://json.mx-api.enetscores.com/live_data/actionzones/' + matchId + '/0?_=1'
        match = response.meta['match']
        jsonresponse = json.loads(response.body_as_unicode())
        
        try:
            goal = [s for s in jsonresponse["i"] if s['type']=='goal']
            shoton = [s for s in jsonresponse["i"] if s['type']=='shoton']
            shotoff = [s for s in jsonresponse["i"] if s['type']=='shotoff']
            foulcommit = [s for s in jsonresponse["i"] if s['type']=='foulcommit']
            card = [s for s in jsonresponse["i"] if s['type']=='card']
            corner = [s for s in jsonresponse["i"] if s['type']=='corner']
            subtypes = [s for s in jsonresponse["i"] if 'subtype' in s]
            cross = [s for s in subtypes if s['subtype']=='cross']
            possession = [s for s in subtypes if s['subtype']=='possession']
            
            match['goal'] = goal
            match['shoton'] = shoton
            match['shotoff'] = shotoff
            match['foulcommit'] = foulcommit
            match['card'] = card
            match['cross'] = cross
            match['corner'] = corner
            match['possession'] = possession
        
        except:
            e = sys.exc_info()[0]
            print('No Match Events: ' + str(e))
            
        yield match