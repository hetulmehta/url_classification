import pandas as pd
import numpy as np
from scipy.stats import randint
import seaborn as sns 
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
import streamlit as st
import joblib
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.metrics import precision_score, recall_score
from sklearn.calibration import CalibratedClassifierCV

from bs4 import BeautifulSoup
import bs4 as bs4
from urllib.parse import urlparse
import requests
import pandas as pd
import spacy as sp
from collections import Counter
sp.prefer_gpu()
import en_core_web_sm

dataset=pd.read_csv("FINAL_DATASET.csv")

df = dataset[['website_url','cleaned_website_text','Category1']].copy()
df.columns = ['website_url','cleaned_website_text','Category'] 

df['category_id'] = df['Category'].factorize()[0]
category_id_df = df[['Category', 'category_id']].drop_duplicates()


# Dictionaries for future use
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'Category']].values)
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5,
                        ngram_range=(1, 2), 
                        stop_words='english')

# We transform each cleaned_text into a vector
features = tfidf.fit_transform(df.cleaned_website_text).toarray()

labels = df.category_id


X = df['cleaned_website_text'] # Collection of text
y = df['Category']
X_train, X_test, y_train, y_test,indices_train,indices_test = train_test_split(features, 
                                                               labels, 
                                                               df.index, test_size=0.25, 
                                                               random_state=1)
model = LinearSVC()
model.fit(X_train, y_train)
y_p = model.predict(X_test)
calibrated_svc = CalibratedClassifierCV(estimator=model,cv="prefit")
calibrated_svc.fit(X_train,y_train)
y_pred= calibrated_svc.predict(X_test)
st.sidebar.title("URL Classification Web App")
status = st.sidebar.radio("Correlated terms with each of the categories: ", ('Dataset', 'Model')) 
cat = st.sidebar.multiselect("Select the categories : ",('Travel','Social Networking and Messaging','News','Streaming Services',
                                                       'Sports','Photography','Law and Government','Health and Fitness','Games',
                                                       'E-Commerce','Forums','Food','Education','Computers and Technology',
                                                       'Business/Corporate','Adult'))

if (status == 'Dataset'): 
    N= st.sidebar.slider("Select the number of terms needed", 1, 5)
    if st.sidebar.button("Enter",key='enter'):
        for Category, category_id in sorted(category_to_id.items()):
            for i in cat:
                if Category==i:
                    features_chi2 = chi2(features, labels == category_id)
                    indices = np.argsort(features_chi2[0])
                    feature_names = np.array(tfidf.get_feature_names())[indices]
                    unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
                    bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
                    st.write("\n==> %s:" %(Category))
                    st.write("  * Most Correlated Unigrams are: %s" %(', '.join(unigrams[-N:])))
                    st.write("  * Most Correlated Bigrams are: %s" %(', '.join(bigrams[-N:])))
else: 
    N= st.sidebar.slider("Select the number of terms needed", 1, 5)
    LinearSVC().fit(features, labels)
    if st.sidebar.button("Enter",key='enter'):
        for Category, category_id in sorted(category_to_id.items()):
            for i in cat:
                if Category==i:
                    indices = np.argsort(model.coef_[category_id])
                    feature_names = np.array(tfidf.get_feature_names())[indices]
                    unigrams = [v for v in reversed(feature_names) if len(v.split(' ')) == 1][:N]
                    bigrams = [v for v in reversed(feature_names) if len(v.split(' ')) == 2][:N]
                    st.write("\n==> '{}':".format(Category))
                    st.write("  * Top unigrams: %s" %(', '.join(unigrams)))
                    st.write("  * Top bigrams: %s" %(', '.join(bigrams))) 
X_train1, X_test1, y_train1, y_test1 = train_test_split(X, df['category_id'], 
                                                    test_size=0.25,
                                                    random_state = 0)

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5,
                        ngram_range=(1, 2), 
                        stop_words='english')

fitted_vectorizer = tfidf.fit(X_train1)
tfidf_vectorizer_vectors = fitted_vectorizer.transform(X_train1)

m = LinearSVC().fit(tfidf_vectorizer_vectors, y_train1)
m1=CalibratedClassifierCV(estimator=m,cv="prefit").fit(tfidf_vectorizer_vectors, y_train1)



class ScrapTool:
    def visit_url(self, website_url):
        '''
        Visit URL. Download the Content. Initialize the beautifulsoup object. Call parsing methods. Return Series object.
        '''
       # headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.71 Safari/537.36'}
        content = requests.get(website_url,timeout=20).content
        
        #lxml is apparently faster than other settings.
        soup = BeautifulSoup(content, "lxml")
        result = {
            "website_url": website_url,
            "website_name": self.get_website_name(website_url),
            "website_text": self.get_html_title_tag(soup)+self.get_html_meta_tags(soup)+self.get_html_heading_tags(soup)+
                                                               self.get_text_content(soup)
        }
        
        #Convert to Series object and return
        return pd.Series(result)
    
    def get_website_name(self,website_url):
        '''
        Example: returns "google" from "www.google.com"
        '''
        return "".join(urlparse(website_url).netloc.split(".")[-2])
    
    def get_html_title_tag(self,soup):
        '''Return the text content of <title> tag from a webpage'''
        return '. '.join(soup.title.contents)
    
    def get_html_meta_tags(self,soup):
        '''Returns the text content of <meta> tags related to keywords and description from a webpage'''
        tags = soup.find_all(lambda tag: (tag.name=="meta") & (tag.has_attr('name') & (tag.has_attr('content'))))
        content = [str(tag["content"]) for tag in tags if tag["name"] in ['keywords','description']]
        return ' '.join(content)
    
    def get_html_heading_tags(self,soup):
        '''returns the text content of heading tags. The assumption is that headings might contain relatively important text.'''
        tags = soup.find_all(["h1","h2","h3","h4","h5","h6"])
        content = [" ".join(tag.stripped_strings) for tag in tags]
        return ' '.join(content)
    
    def get_text_content(self,soup):
        '''returns the text content of the whole page with some exception to tags. See tags_to_ignore.'''
        tags_to_ignore = ['style', 'script', 'head', 'title', 'meta', '[document]',"h1","h2","h3","h4","h5","h6","noscript"]
        tags = soup.find_all(text=True)
        result = []
        for tag in tags:
            stripped_tag = tag.strip()
            if tag.parent.name not in tags_to_ignore\
                and isinstance(tag, bs4.element.Comment)==False\
                and not stripped_tag.isnumeric()\
                and len(stripped_tag)>0:
                result.append(stripped_tag)
        return ' '.join(result)


#anconda prompt ko run as adminstrator and copy paste this:python -m spacy download en
nlp = en_core_web_sm.load()
def clean_text(doc):
    '''
    Clean the document. Remove pronouns, stopwords, lemmatize the words and lowercase them
    '''
    doc = nlp(doc)
    tokens = []
    exclusion_list = ["nan"]
    for token in doc:
        if token.is_stop or token.is_punct or token.text.isnumeric() or (token.text.isalnum()==False) or token.text in exclusion_list :
            continue
        token = str(token.lemma_.lower().strip())
        tokens.append(token)
    return " ".join(tokens) 
st.title("URL CLASSIFICATION")
st.subheader("What is the category of the website ? ")
website=st.text_input("ENTER URL","Type here")
if(st.button('Submit')): 
    scrapTool = ScrapTool()
    try:
        web=dict(scrapTool.visit_url(website))
        text=(clean_text(web['website_text']))
        t=fitted_vectorizer.transform([text])
        ans=id_to_category[m1.predict(t)[0]]
        st.success(ans)
        st.subheader('Probability Prediction for each Category of the URL')
        data=pd.DataFrame(m1.predict_proba(t)*100,columns=df['Category'].unique())
        data=data.T
        data.columns=['Probability']
        data.index.name='Category'
        a=data.sort_values(['Probability'],ascending=False)
        a['Probability']=a['Probability'].apply(lambda x:round(x,2))
        st.write(a)
        sns.set(font_scale = 1.5)
        fig, ax = plt.subplots(figsize=(10,5))
        i=list(a.index)
        sns.barplot(i,a['Probability'])
        plt.title("Probability Prediction for each Category of the URL", fontsize=18)
        plt.ylabel('Probability', fontsize=16)
        plt.xlabel('Category ', fontsize=16)
        rects = ax.patches
        labels = a['Probability']
        for rect, label in zip(rects, labels):
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom', fontsize=14)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, horizontalalignment='center')
        st.pyplot(fig)
    except:
        st.error("Connection Timeout! The website denied us access") 
    
if st.sidebar.checkbox("Show raw data",False):
    st.subheader("URL Classification Raw Dataset")
    df1=pd.read_csv("FINAL_DATASET.csv")
    st.write(df1)
 
def plot_metrics(metrics_list):
        if 'Confusion Matrix' in metrics_list:
            st.subheader("Confusion Matrix")
            conf_mat = confusion_matrix(y_test, y1_pred,labels=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
            fig, ax = plt.subplots(figsize=(8,8))
            sns.heatmap(conf_mat, annot=True, cmap="OrRd", fmt='d',
            xticklabels=category_id_df.Category.values, 
            yticklabels=category_id_df.Category.values)
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            plt.title("CONFUSION MATRIX \n", size=16);
            st.pyplot(fig)
        if 'Classification Report' in metrics_list:
             st.subheader("Classification Report")
             report=classification_report(y_test, y1_pred,labels=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],target_names= list(df['Category'].unique()),output_dict=True)
             cr = pd.DataFrame(report).transpose()
             st.write(cr)
st.sidebar.subheader("Select your Classifier")
classifier = st.sidebar.selectbox("Classifier", ("Linear SVC","Random Forest Classifier","MultinomialNB","GaussianNB"))
if classifier =="Linear SVC":
     metrics = st.sidebar.multiselect("Select your metrics : ",("Confusion Matrix",'Classification Report'))
     if st.sidebar.button("Classify", key='classify'):
            st.subheader("Linear SVC Results")
            m = LinearSVC()
            m.fit(X_train, y_train)
            y1_pred = m.predict(X_test)
            accuracy = m.score(X_test, y_test)
            st.write("Accuracy: ", round(accuracy*100,2),"%")
            st.write("Precision: ",round(precision_score(y_test, y1_pred,labels=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],average='weighted')*100,2),"%")
            st.write("Recall: ", round(recall_score(y_test, y1_pred, labels=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],average='weighted')*100,2),"%")
            plot_metrics(metrics)
if classifier =="Random Forest Classifier":
        st.sidebar.subheader("Model Hyperparameters")
        n_estimators = st.sidebar.number_input("The number of trees in the forest", 100, 500, step=10, key='n_estimators')
        max_depth = st.sidebar.number_input("The maximum depth of the tree", 1, 15, step=1, key='n_estimators')
        metrics = st.sidebar.multiselect("Select your metrics : ",("Confusion Matrix",'Classification Report'))
        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Random Forest Results")
            m = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,random_state=0)
            m.fit(X_train, y_train)
            y1_pred = m.predict(X_test)
            accuracy = m.score(X_test, y_test)
            st.write("Accuracy: ", round(accuracy*100,2),"%")
            st.write("Precision: ",round(precision_score(y_test, y1_pred,labels=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],average='weighted')*100,2),"%")
            st.write("Recall: ", round(recall_score(y_test, y1_pred, labels=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],average='weighted')*100,2),"%")
            plot_metrics(metrics)
if classifier =="MultinomialNB":
         metrics = st.sidebar.multiselect("Select your metrics : ",("Confusion Matrix",'Classification Report'))
         if st.sidebar.button("Classify", key='classify'):
            st.subheader("MultinomialNB Results")
            m = MultinomialNB()
            m.fit(X_train, y_train)
            y1_pred = m.predict(X_test)
            accuracy = m.score(X_test, y_test)
            st.write("Accuracy: ", round(accuracy*100,2),"%")
            st.write("Precision: ",round(precision_score(y_test, y1_pred,labels=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],average='weighted')*100,2),"%")
            st.write("Recall: ", round(recall_score(y_test, y1_pred, labels=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],average='weighted')*100,2),"%")
            plot_metrics(metrics)
if classifier =="GaussianNB":
         metrics = st.sidebar.multiselect("Select your metrics : ",("Confusion Matrix",'Classification Report'))
         if st.sidebar.button("Classify", key='classify'):
            st.subheader("GaussianNB Results")
            m =GaussianNB()
            m.fit(X_train, y_train)
            y1_pred = m.predict(X_test)
            accuracy = m.score(X_test, y_test)
            st.write("Accuracy: ", round(accuracy*100,2),"%")
            st.write("Precision: ",round(precision_score(y_test, y1_pred,labels=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],average='weighted')*100,2),"%")
            st.write("Recall: ", round(recall_score(y_test, y1_pred, labels=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],average='weighted')*100,2),"%")
            plot_metrics(metrics)