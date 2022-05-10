from main import Articles
import pandas as pd

articles = Articles.query.all()
category = []
title = []
author = []
content = []
for article in articles:
    categories = article.category
    category.append(categories)
    titles = article.title
    authors = article.author
    contents = article.content
    title.append(titles)
    author.append(authors)
    content.append(contents)
zipped = list(zip(category, title, author, content))
database_df = pd.DataFrame(zipped, columns=["category","title","author","content"])
database_df.drop_duplicates()