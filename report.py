import jobscrape
import numpy as np
import matplotlib.pyplot as plt
from corpus import Corpus
from gensim import models



languages = [['c','c++'], 'java', 'javascript','matlab',
             'python','r', 'sas', 'scala']

language_labels = ['C/C++', 'Java', 'Javascript', 'MATLAB',
                   'Python', 'R', 'SAS', 'Scala']

degrees = [['bachelor', 'bachelors', 'bs', 'b.s.'],
           ['masters', 'ms', 'm.s.'],
           ['doctorate', 'phd', 'ph.d', 'ph.d.']]
degree_labels = ['B.S.', 'M.S.', 'Ph.D.']

other_skills = ['excel', 'sql', 'nosql', 'redis', 'mongodb', 'hadoop',
                'spark', 'hbase', 'hive']



def texts(postings):
    # don't duplicate urls
    distinct_postings = {p.url:p.text for p in postings}
    return distinct_postings.values()


def load_datasets():
    seattle = jobscrape.load_postings("seattle_data_science_feb_20.txt")
    sanfran = jobscrape.load_postings("sanfran_data_science_feb_20.txt")
    denver = jobscrape.load_postings("denver_data_science_feb_20.txt")
    boston = jobscrape.load_postings("boston_data_science_feb_20.txt")
    newyork = jobscrape.load_postings("newyork_data_science_feb_20.txt")
    
    
    data = [Corpus(texts(boston)),
            Corpus(texts(denver)),
            Corpus(texts(newyork)),
            Corpus(texts(sanfran)),
            Corpus(texts(seattle))]
    names = ["Boston", "Denver", "NYC", "SF", "Seattle"]

    return data,names

            
def frequencies(corpus, queries):
    n = corpus.num()
    freqs = [(corpus.num(q)/n, q) for q in queries]
    return freqs


def abbrv(strings):
    if type(strings) == str:
        return strings
    else:
        return ",".join(strings)


def degree_plots(data,names,filename=None):
    n = len(data)
    f, axes = plt.subplots(1,n,sharey = True,figsize=(2*n,10/3))
    freqs = [frequencies(d,degrees) for d in data]
    ymax = max([p for ft in freqs for p,_ in ft])
    for i in range(n):
        barplot(axes[i],frequencies(data[i],degrees),
                ymax=ymax, labels=degree_labels,title=names[i])
    if filename is not None:
        plt.savefig(filename,bbox_inches="tight")
    plt.show()

def language_plots(data,name):
    f,axis = plt.subplots(1,1)
    freq = frequencies(data,languages)
    ymax = max([p for p,_ in freq])
    barplot(axis, freq, ymax=ymax,
            labels=language_labels,title=name)
    plt.show()

def make_keyword_plots(data,names):
    keywords = [("r","R","r.png"),
                ("python", "Python","python.png"),
                ("java", "Java", "java.png"),
                (["c","c++"], "C/C++", "c.png"),
                ("javascript", "Javascript", "js.png"),
                ("excel", "Excel", "excel.png"),
                ("sql", "SQL", "sql.png"),
                ("nosql", "NoSQL", "nosql.png"),
                ("hadoop", "Hadoop", "hadoop.png"),
                ("spark", "Spark", "spark.png"),
                ("mapreduce", "MapReduce", "mapreduce.png"),
                ("redis", "Redis", "redis.png")]
    for key,title,filename in keywords:
        keyword_comparison_plot(data,names,key,title,filename)
                

def keyword_comparison_plot(data,names,key,title,filename=None):
    n = len(data)
    freqs = [frequencies(d,[key])[0] for d in data]
    f,ax = plt.subplots(1,1,figsize=(5,5))
    barplot(ax,freqs,labels=names,
            title=title,ymax=max([p for p,_ in freqs]))
    if filename is not None:
        plt.savefig(filename)
    plt.show()
    
    
def barplot(ax,freqs,ystep=0.05,ymax=None,labels=None,title=None):
    plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    plt.rc('text', usetex=True)

    ax.set_xlim([0,len(freqs)])
    if labels is None:
        labels = [abbrv(item) for f,item in freqs]
    x_pos = np.arange(len(freqs))+0.5
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)

    values = [f for f,item in freqs]
    ax.bar(x_pos, values, facecolor='#a8a8a8',
            edgecolor='white',width=0.5, align='center')
    sides = ['left','right','top','bottom']
    ax.tick_params(**{side:'off' for side in sides})
    for side in sides:
        ax.spines[side].set_color('none')
    if ymax is None:
        ymax = max(values)
    ys = np.arange(0.05,ymax,ystep)
    ylabels = ['${:d}\%$'.format(int(y*100)) for y in ys]
    ax.set_yticks(ys)
    ax.set_yticklabels(ylabels)
    for y in ys:
        ax.axhline(y=y,color='w',linestyle='-',linewidth='2')


    if title is not None:
        ax.set_title(title)

def lsi(corp,num_topics=5):
    tfidf = models.TfidfModel(corp.corpus, normalize=True)
    tfidf_corpus = tfidf[corp.corpus]
    lsi = models.LsiModel(tfidf_corpus, id2word=corp.dictionary, num_topics=5)
    lsi_corpus = lsi[tfidf_corpus]
    return lsi,lsi_corpus
