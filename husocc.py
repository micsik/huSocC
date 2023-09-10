from configparser import SectionProxy
import sys
import re
from typing import Any
import pandas as pd
from nltk.corpus import stopwords
import huspacy
import textacy
import logging

logger = logging.getLogger('huSoc')
logger.setLevel(logging.INFO)

def addRank(kwDict, kw, rank, max=None):
    if kw in kwDict:
        if max is None or kwDict[kw] + rank <= max:
            kwDict[kw] = kwDict[kw] + rank
    else:
        kwDict[kw] = rank
    return kwDict

class HuSocClassifier():

    nlp = None
    config = {
        'topicWeight': 4,
        'freqWeight': 2,
        'rankLimit': 10,
        'wordScoreMax': 3,
        'max_kw' : 30,
        'max_top' : 20,
    }
    myStopWords = None
    keywordStopWords = None
    kwMap = None
    topicMap = None

    def __init__(self) -> None:
        logger.info("HuSocClassifier INIT")
        self.nlp = huspacy.load()

        stopwordset = stopwords.words('english')
        stopwordset.extend(stopwords.words('hungarian'))

        self.myStopWords = stopwordset
        myList=['érdekes','igazából','igazán','hát','erről','úgyhogy','tényleg','szóval','soha','miatt','főleg',
          'nagyon-nagyon','tudtam','tudom','akár','dolog','eléggé','hogyha','egyébként','azóta','ezeket','teljesen',
          'egyáltalán','engem','kevésbé','ember','kicsit','kicsi','mellett','furcsa','hiszem','tudom',
          'merthogy','attól','fontos',"előtt","mire","érdemes",'kellemes','következő','kérdés', 'érthető','csomó',
                "dolg","ról","nál",'megint','együtt',
                'különböző','bármi','mittudomén','nap',
                  'dehát', 'végülis', 'végüli', 'tulajdonképpen', 'hadd', 'sok', 'úgymond', 'azér', 'visszatérve',
              'máris', 'manapság', 'bizony',
                'sáv vége',
                'föl','be','ki','le','fel','szét',"össze","vissza","ide","oda",
                'ekkora','akkora','eleje','történet',"sáv",'mindenféle',"hely","probléma",
                'ilyesmi','kis','rossz','idő','szó','biztos',"téma",'mai','többi',"év",
                'ö','üm',"öö","öüm","ümö",'izé',"má","háát","hááát",
                'None',
                'mindegy','akkori',"belőle","rajta","róla","mennyi","hol","anonimizálva","ennyi","annyi"
                #TK listájából
                "idézőjel","sztori","szakdolgozat","jórész","rész","baromság","hónap",
                "példa", "fajta",
                "darab", "perc", "nevetés","satöbbi", "hűha", "jóóók", "egyértelmű","fél","rakás","köbméter", "jóóók",
                "kurva", "segg", "szereplő", "színű", "mink", "pici", "full",
                "ellenkező", "pont", "szóköz", "megjegyzés", "nyak", "prím", "ágazatár", "asztal", "bizonyos",
                "ásványvizes", "mélye",
                "előző", "mostani", "legalja", "adtak", "százalék", "mondat", 'időseb', "időpont", "hatos", "gramm",
                "szám", "interjúztató", "kuka", "bűnösö",
                "január","február","március","április","május","június","július","augusztus","szeptember","október","november","december",
                "kettő","ezer","száz","tíz", "ugye","négy","három","öt",
                "tőle","vele","néem","közte"
            ]
        self.myStopWords.extend(myList)
        self.myStopWords = list(dict.fromkeys(self.myStopWords))  #dedup
        self.myStopWords = sorted(self.myStopWords)

        # print(self.myStopWords)

        self.keywordStopWords = ['típus', 'szülő', 'munka', 'vonal', 'munkahely', 'kerület', 'tervezés', 'élet', 'időszak',
                            'fiatal', 'éves kor', "lényeg", 'példa',
                            "január","február","március","április","május","június","július","augusztus","szeptember","október","november","december",
                             'keret',"csomag","szint",
                            "None","helyzet","szerep","dolgozat","igény","név","gond","pénz","alap","irány",
                            "bemutatkozás", "fogalom","javaslat","kapcsolat","eszköz",'mondanivaló','szükség',
                            "osztály", "papír", "szempont", "jobb fizetés függvény"
                           ]
        self.init_topics()

    def init_topics(self):
        topicWord = pd.read_csv('tax-kw.tsv', sep='\t')

        self.kwMap = {}
        self.topicMap = {}

        for i, row in topicWord.iterrows():
            for kw in str(row['keywords']).split('; '):
                if kw=='nan': continue
                kw = kw.strip(' ,.;').lower()
                if ' ' in kw:
                    continue;
                count = 1  #row['count']
                topic = row['topic']
                #print(kw,' = ', topic)

                if topic not in self.topicMap:
                    self.topicMap[topic] = {}
                self.topicMap[topic][kw] = count

                #if kw in kwMap and kwMap[kw]['count'] >= count:
                #    continue
                if kw not in self.kwMap:
                    self.kwMap[kw] = {}
                self.kwMap[kw][topic] = 1;

        topicWords = []
        for kw in self.kwMap:
            tdict = self.kwMap[kw]
            for t in tdict:
                topicWords.append({'keyword': kw, 'topic': t, 'count': tdict[t]})

        for name, group in topicWord.groupby('topic'):
            t = name
            if t not in self.kwMap:
                self.kwMap[t] = {}
            if t not in self.kwMap[t]:
                self.kwMap[t][t] = 3
            if t not in self.topicMap:
                self.topicMap[t] = {}
            if t not in self.topicMap[t]:
                self.topicMap[t][t] = 3

        topics = pd.read_csv('taxonomy.tsv', sep='\t')
        topicWoKwCount = 0
        topicParents = {}
        for i, row in topics.iterrows():
            topic = row['level1']
            level = 1
            parent = None
            if not pd.isna(row['level2']):
                topic = row['level2']
                level = 2
                parent = row['level1']
            if not pd.isna(row['level3']):
                topic = row['level3']
                level = 3
                parent = row['level2']

            topicParents[topic]=parent
            #print(f"{row['level']}, {row['topic']}")
            topics.at[i,'topic'] = topic
            topics.at[i,'level'] = level
            if topic not in self.topicMap:
                # print(f"no keywords: {topic}")
                topicWoKwCount += 1
            else:
                # if len(self.topicMap[topic]) == 1 and topic in self.topicMap[topic]:
                    # print(f"no keywords: {topic}")
                topics.at[i,'keywords'] = "; ".join([kw for kw,c in sorted(self.topicMap[topic].items(), key=lambda w: w[1], reverse=True)])

        topics['level'] = topics['level'].astype('int32')

    def extractAllKeywords(self, doc):
        kwList = {}
        pos = {'NOUN', 'ADJ', 'PROPN' } #, 'PROPN', 'ADJ' 'VERB',
        for token in textacy.extract.basics.words(doc, filter_nums=True, include_pos=('NOUN', 'ADJ', 'PROPN')):
            #row = getMorph(word, emtsvDict)
            word = token.lemma_
            if word not in self.myStopWords and token.text not in self.myStopWords and len(word) > 2:
                addRank(kwList, word, 1, self.config['wordScoreMax'])
        finalKwList = [(kw, round(rank,2)) for (kw,rank) in sorted(kwList.items(),key=lambda w: w[1], reverse=True)]
        #finalKwList = [(kw,rank) for (kw,rank) in finalKwList if rank >= 0.4]
        return finalKwList

    def getKeywords(self, text):
        text = re.sub('-ö+-', '', text)
        text = re.sub('\*+-?', '', text)
        text = re.sub('<\w+>', '', text)
        # text = text.replace('-','')
        text = text.replace('- ', ' ')
        text = text.replace('/', ' ')
        text = text.replace(' khm ', ' ')
        text = text.replace(' kö ', ' ')
        text = text.replace('...', ' ')
        text = text.replace('û', 'ű')
        text = text.replace('õ', 'ő')
        text = text.replace(' ő ', ' ')
        text = text.replace(' ö ', ' ')
        text = text.replace('(…)', ' ')
        text = re.sub('\s+', ' ', text)

        doc = self.nlp(text)
        topicWords = self.extractAllKeywords(doc)
        keywords0 = textacy.extract.keyterms.yake(doc, window_size=2, include_pos=('NOUN'), normalize="lemma", topn=0.3)
        keywords = []
        for kw, rank in keywords0:
            kw = kw.strip('-.[]')
            rank = round(rank * 100)
            if not kw in self.myStopWords and '-' not in kw:
                keywords.append({'orig': kw, 'rank': rank})

        # print(f'ORIG KW {[w["orig"] for w in keywords]}')
        topKeywords = {}
        topNames = {}

        for row in keywords:
            kw = row['orig']
            kw = kw.strip()
            if len(kw) < 4 and kw != 'apa': continue
            if kw.lower() in self.keywordStopWords: continue
            if kw == 'None': continue
            row['kw'] = kw

        keywordsFiltered = [row for row in keywords if len(row) > 2]
        keywords = []
        for row in keywordsFiltered:
            kw = row['kw']
            if any(kw != row2['kw'] and kw in row2['kw'] for row2 in keywordsFiltered):
                # print('repeated ', kw)
                continue
            #         if row['taglist'][-1]['pos']=='ADJ':
            #             print('adjective ', kw)
            #             continue
            keywords.append(row)
        for row in keywords:
            kw = row['kw']
            hasPropn = False  # any('PROPN'==w['pos'] for w in row['taglist'])
            if hasPropn:
                topNames = addRank(topNames, kw, 100 - row['rank'])
            else:
                topKeywords = addRank(topKeywords, kw, 100 - row['rank'])
                # print(row['orig'], row['taglist'])
        return topKeywords, topNames, topicWords

    def findTopics(self, text):

        kwlist, namelist, topicWordList = self.getKeywords(text)
        # print(f'YAKE IN BLOCK: { kwlist}')
        # print(f'TOP NM IN BLOCK {namelist}')
        # print(f'topicWordList= {topicWordList}')

        # for kw, rank in sorted(kwlist.items(),key=lambda w: w[1], reverse=True) :
        #     topKeywords = addRank(topKeywords, kw, rank)
        # for kw, rank in sorted(namelist.items(),key=lambda w: w[1], reverse=True) :
        #     topNames = addRank(topNames, kw, rank)

        topicDict = {}
        for kw, rank in topicWordList:
            if kw in self.kwMap:
                tdict = self.kwMap[kw]
                for t in tdict:
                    topicDict = addRank(topicDict, t, tdict[t] * self.config['topicWeight'] + rank * self.config['freqWeight'])

        #blockTopics = sorted(topicDict.items(),key=lambda w: w[1], reverse=True)[:5]
        blockTopics = sorted(topicDict.items(),key=lambda w: w[1], reverse=True)
        #blockTopics = [(t,v) for (t,v) in blockTopics if t not in topicParents.values()] # ez nem jo, a csalad kikerul tole pl.
        logger.info("TOPICS unfilt: %s", blockTopics)

        blockTopics = [(t,v) for (t,v) in blockTopics if v >= self.config['rankLimit']]
        blockTopics = blockTopics[:15]
        maxRank = max([v for (t,v) in blockTopics])
        blockTopics = [(t, round(v/maxRank,2)) for (t,v) in blockTopics]
        # print("BLOCK TOPICS ", blockTopics)

        return blockTopics

if __name__ == '__main__':
    # huspacy.download()
    classifier = HuSocClassifier()
    for arg in sys.argv[1:]:
        f = open(arg, "r")
        text = ' '.join(f.readlines())
        print(arg, 'topics:', classifier.findTopics(text))