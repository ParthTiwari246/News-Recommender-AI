from crypt import methods
from enum import unique
from tkinter import CENTER
from unicodedata import category
from flask import Flask, redirect, render_template, request
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict, Counter

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI']="sqlite:///data.db"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class Articles(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    category = db.Column(db.String,nullable=False)
    title = db.Column(db.String, nullable=False, unique=True)
    author = db.Column(db.String)
    content = db.Column(db.String, nullable=False)
        
    def __repr__(self):   
        return f"{self.title} - {self.author}"

@app.route("/")
def index():
    return render_template("home.html")

@app.route("/submission_validation", methods=["POST"])
def get_articles():
    category = str(request.form.get("cat"))
    title = request.form.get("title")
    author = request.form.get("author")
    content = request.form.get("content")

    article = Articles(category=category, title=title, author=author, content=content)
    db.session.add(article)
    db.session.commit()

    return render_template("home.html")

@app.route("/database")
def view_database():
    articles = Articles.query.all()
    return render_template("index.html", articles=articles)

@app.route("/recommend")
def recommend():
    return render_template("recommend.html")



###
business_df = pd.read_csv('CSVs/business.csv')
Business_articles = defaultdict(dict)
for i in range(len(business_df)):
    title_i = business_df['title'][i]
    content_i = business_df['news'][i]
    Business_articles[title_i]={"content":content_i}
Business_titles = list(Business_articles.keys())
Business_contents =[]
for title in Business_articles.keys():
    Business_contents.append(str(Business_articles[title]['content']))

Business_news = [Business_titles,Business_contents]    


entertainment_df = pd.read_csv("CSVs/entertainment.csv")
Entertainment_articles = defaultdict(dict)
for i in range(len(entertainment_df)):
    title_i = entertainment_df['title'][i]
    content_i = entertainment_df['news'][i]
    Entertainment_articles[title_i]={"content":content_i}    
Entertainment_titles = list(Entertainment_articles.keys())
Entertainment_contents =[]
for title in Entertainment_articles.keys():
    Entertainment_contents.append(str(Entertainment_articles[title]['content']))

Entertainment_news = [Entertainment_titles,Entertainment_contents]



sports_df = pd.read_csv("CSVs/sport.csv")
Sports_articles = defaultdict(dict)
for i in range(len(sports_df)):
    title_i = sports_df['title'][i]
    content_i = sports_df['news'][i]
    Sports_articles[title_i]={"content":content_i}    
Sports_titles = list(Sports_articles.keys())
Sports_contents =[]
for title in Sports_articles.keys():
    Sports_contents.append(str(Sports_articles[title]['content']))

Sports_news = [Sports_titles,Sports_contents]


tech_df = pd.read_csv("CSVs/tech.csv")
Technology_articles = defaultdict(dict)
for i in range(len(tech_df)):
    title_i = tech_df['title'][i]
    content_i = tech_df['news'][i]
    Technology_articles[title_i]={"content":content_i}    
Technology_titles = list(Technology_articles.keys())
Technology_contents =[]
for title in Technology_articles.keys():
    Technology_contents.append(str(Technology_articles[title]['content']))

Technology_news = [Technology_titles,Technology_contents]

politics_df = pd.read_csv("CSVs/politics.csv")
Politics_articles = defaultdict(dict)
for i in range(len(politics_df)):
    title_i = politics_df['title'][i]
    content_i = politics_df['news'][i]
    Politics_articles[title_i]={"content":content_i}    
Politics_contents =[]
Politics_titles = list(Politics_articles.keys())
for title in Politics_articles.keys():
    Politics_contents.append(str(Politics_articles[title]['content']))

Politics_news = [Politics_titles,Politics_contents]
###

from nltk.corpus import stopwords
stopWords = stopwords.words('english')

@app.route("/recommend_valid",methods=["POST"])
def paste_article():
    article = request.form.get("content")
    from sklearn.naive_bayes import MultinomialNB
    nb = MultinomialNB()

    from sklearn.svm import SVC
    svm = SVC(C= 10000000.0, gamma='auto', kernel='rbf')

    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=5)

    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression()

    from sklearn.tree import DecisionTreeClassifier
    dt = DecisionTreeClassifier(random_state=17)

    from sklearn.ensemble import RandomForestClassifier
    rfc = RandomForestClassifier()
    from collections import defaultdict, Counter
    from sklearn.model_selection import train_test_split
    import nltk

    t_path = "datasets/"

    all_docs = defaultdict(lambda: list())

    topic_list = list()
    text_list = list()

    for topic in os.listdir(t_path):
        d_path = t_path + topic + '/'

        for f in os.listdir(d_path):
            f_path = d_path + f
            file = open(f_path,'r',encoding='unicode_escape')
            text_list.append(file.read())
            file.close()
            topic_list.append(topic)

    title_train, title_test, category_train, category_test = train_test_split(text_list,topic_list)
    title_train, title_dev, category_train, category_dev = train_test_split(title_train,category_train)

    tokenizer = nltk.tokenize.RegexpTokenizer(r"\w+")
    stop_words = nltk.corpus.stopwords.words("english")

    vectorizer = TfidfVectorizer(tokenizer=tokenizer.tokenize, stop_words=stop_words)
    vectorizer.fit(iter(title_train))
    Xtr = vectorizer.transform(iter(title_train))
    Xde = vectorizer.transform(iter(title_dev))
    Xte = vectorizer.transform(iter(title_test))

    from sklearn.preprocessing import LabelEncoder
    
    encoder = LabelEncoder()
    encoder.fit(category_train)
    Ytr = encoder.transform(category_train)

    Yde = encoder.transform(category_dev)
    Yte = encoder.transform(category_test)


    classifier_names = ['naive bayes','support vector machine','k-nearest neighbour','logistic regression','decision tree','randomforestclassifier']
    classifier_list = [nb,svm,knn,lr,dt,rfc]
    contents = str(article)
    prediction_list = list()
    for i in range(len(classifier_list)):
        #print('\nClassifier name:',classifier_names[i])
        classifier = classifier_list[i]
        classifier.fit(Xtr,Ytr)
        encoded_contents = vectorizer.transform([contents])
        pred = classifier.predict(encoded_contents)
        prediction_list.append(pred[0]);
        print('predicted category: ',pred)
    prediction_list = Counter(prediction_list)
    category_index = prediction_list.most_common()[0][0]
    #print('final predicted category: ',category_index)
    category = {0:'Business',1:'Entertainment',2:'Politics',3:'Sports',4:'Technology'}
    #print('category of selected article: ',category[category_index],'\n')
    category = str(category[category_index])


    if(category=='Politics'):
        content = Politics_news[1]
        Recom_articles = Politics_articles
        Recom_content = Recom_articles.items()

        df = pd.DataFrame(Recom_content)
        df.columns = ["title","content"]
            

        import gensim
        import spacy
        import nltk
        nlp = spacy.load('en_core_web_lg')
        input_vec = nlp(contents)
        all_docs = [nlp(str(row)) for row in df['content']]

        sims = []
        doc_id = []
        for i in range(len(all_docs)):
            sim = all_docs[i].similarity(input_vec)
            sims.append(sim)
            doc_id.append(i)
            sims_docs = pd.DataFrame(list(zip(doc_id, sims)), columns = ['doc_id', 'sims'])
            
        simdocsort = sims_docs.sort_values(by = 'sims', ascending=False)
        top5simsdocs = df.iloc[simdocsort['doc_id'][1:6]]

        topsimscores = pd.concat([top5simsdocs, simdocsort['sims'][1:6]], axis=1)
        df_final = pd.DataFrame(list(zip(topsimscores['title'],topsimscores['content'],topsimscores['sims'])), columns=['title', 'news' , 'similarity score'])
        df.style.set_table_styles([{'selector': 'th', 'props': [('font-size', '12pt'),('border-style','solid'),('border-width','1px')]}])
        
        return render_template("Politics.html", data=df_final.to_html(justify="center", bold_rows=True, render_links=True))
    else:
        if(category=='Technology'):
            content = Technology_news[1]
            Recom_articles = Technology_articles
            Recom_content = Recom_articles.items()

            df = pd.DataFrame(Recom_content)
            df.columns = ["title","content"]
            

            import gensim
            import spacy
            import nltk
            nlp = spacy.load('en_core_web_lg')
            input_vec = nlp(contents)
            all_docs = [nlp(str(row)) for row in df['content']]

            sims = []
            doc_id = []
            for i in range(len(all_docs)):
                sim = all_docs[i].similarity(input_vec)
                sims.append(sim)
                doc_id.append(i)
                sims_docs = pd.DataFrame(list(zip(doc_id, sims)), columns = ['doc_id', 'sims'])
            
            simdocsort = sims_docs.sort_values(by = 'sims', ascending=False)
            top5simsdocs = df.iloc[simdocsort['doc_id'][1:6]]

            topsimscores = pd.concat([top5simsdocs, simdocsort['sims'][1:6]], axis=1)
            df_final = pd.DataFrame(list(zip(topsimscores['title'],topsimscores['content'],topsimscores['sims'])), columns=['title', 'news' , 'similarity score'])
            df.style.set_table_styles([{'selector': 'th', 'props': [('font-size', '12pt'),('border-style','solid'),('border-width','1px')]}])

            return render_template("Technology.html", data=df_final.to_html(justify="center", bold_rows=True, render_links=True))
        elif(category=='Entertainment'):
            content = Entertainment_news[1]            
            Recom_articles = Entertainment_articles
            Recom_content = Recom_articles.items()

            df = pd.DataFrame(Recom_content)
            df.columns = ["title","content"]
            

            import gensim
            import spacy
            import nltk
            nlp = spacy.load('en_core_web_lg')
            input_vec = nlp(contents)
            all_docs = [nlp(str(row)) for row in df['content']]

            sims = []
            doc_id = []
            for i in range(len(all_docs)):
                sim = all_docs[i].similarity(input_vec)
                sims.append(sim)
                doc_id.append(i)
                sims_docs = pd.DataFrame(list(zip(doc_id, sims)), columns = ['doc_id', 'sims'])
            
            simdocsort = sims_docs.sort_values(by = 'sims', ascending=False)
            top5simsdocs = df.iloc[simdocsort['doc_id'][1:6]]

            topsimscores = pd.concat([top5simsdocs, simdocsort['sims'][1:6]], axis=1)
            df_final = pd.DataFrame(list(zip(topsimscores['title'],topsimscores['content'],topsimscores['sims'])), columns=['title', 'news' , 'similarity score'])
            df.style.set_table_styles([{'selector': 'th', 'props': [('font-size', '12pt'),('border-style','solid'),('border-width','1px')]}])
            
            return render_template("Entertainment.html", data=df_final.to_html(justify="center", bold_rows=True, render_links=True))
        elif(category=='Sports'):
            content = Sports_news[1]
            Recom_articles = Sports_articles
            Recom_content = Recom_articles.items()

            df = pd.DataFrame(Recom_content)
            df.columns = ["title","content"]
            

            import gensim
            import spacy
            import nltk
            nlp = spacy.load('en_core_web_lg')
            input_vec = nlp(contents)
            all_docs = [nlp(str(row)) for row in df['content']]

            sims = []
            doc_id = []
            for i in range(len(all_docs)):
                sim = all_docs[i].similarity(input_vec)
                sims.append(sim)
                doc_id.append(i)
                sims_docs = pd.DataFrame(list(zip(doc_id, sims)), columns = ['doc_id', 'sims'])
            
            simdocsort = sims_docs.sort_values(by = 'sims', ascending=False)
            top5simsdocs = df.iloc[simdocsort['doc_id'][1:6]]

            topsimscores = pd.concat([top5simsdocs, simdocsort['sims'][1:6]], axis=1)
            df_final = pd.DataFrame(list(zip(topsimscores['title'],topsimscores['content'],topsimscores['sims'])), columns=['title', 'news' , 'similarity score'])
            df.style.set_table_styles([{'selector': 'th', 'props': [('font-size', '12pt'),('border-style','solid'),('border-width','1px')]}])
            
            return render_template("Sports.html", data=df_final.to_html(justify="center", bold_rows=True, render_links=True))
        elif(category=='Business'):
            content = Business_news[1]
            Recom_articles = Business_articles;
            Recom_content = Recom_articles.items()

            df = pd.DataFrame(Recom_content)
            df.columns = ["title","content"]
            

            import gensim
            import spacy
            import nltk
            nlp = spacy.load('en_core_web_lg')
            input_vec = nlp(contents)
            all_docs = [nlp(str(row)) for row in df['content']]

            sims = []
            doc_id = []
            for i in range(len(all_docs)):
                sim = all_docs[i].similarity(input_vec)
                sims.append(sim)
                doc_id.append(i)
                sims_docs = pd.DataFrame(list(zip(doc_id, sims)), columns = ['doc_id', 'sims'])
            
            simdocsort = sims_docs.sort_values(by = 'sims', ascending=False)
            top5simsdocs = df.iloc[simdocsort['doc_id'][1:6]]

            topsimscores = pd.concat([top5simsdocs, simdocsort['sims'][1:6]], axis=1)
            df_final = pd.DataFrame(list(zip(topsimscores['title'],topsimscores['content'],topsimscores['sims'])), columns=['title', 'news' , 'similarity score'])
            #df_final.to_html(justify="center", bold_rows=True, render_links=True"templates/{}.html".format(category))
            df.style.set_table_styles([{'selector': 'th', 'props': [('font-size', '12pt'),('border-style','solid'),('border-width','1px')]}])
            return render_template("Business.html", data=df_final.to_html(justify="center", bold_rows=True, render_links=True))

import database,files, makecsv, newsclass
if __name__ == "__main__":
    database()
    files()
    makecsv()
    newsclass()