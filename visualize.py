from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns

def plot_wordcloud(text, title):
    wordcloud = WordCloud(width = 800, height = 400, background_color = 'white').generate(text)
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation = 'bilinear')
    plt.axis('off')
    plt.title(title)
    plt.savefig('{title}.png', dpi = 300, bbox_inches = 'tight')
    plt.show()

def countplot(title, df, y):
    sns.countplot(y = 'subreddit', data = df)
    plt.title('Subreddit distribution')