### *not related to the project, I find this is a good opportunity to update my resume. Please scroll down for my project showcase, Thank you!.
![png](upwork.png)
https://www.upwork.com/freelancers/~010513853d76c89bea

![png](1.png)
![png](2.png)
![png](3.png)
![png](4.png)
![png](5.png)

# Correspondence of Wine Taster's Given Point and Comment In Their Reviews Using Sentiment Analysis (Hugging Face Transformers)


```python
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # for simple EDA
import seaborn as sns # for plotting
from transformers import AutoTokenizer # tokenizer
from transformers import AutoModelForSequenceClassification # Sequence Classification
from scipy.special import softmax # Softmax

```

## Read in data


```python
dataframe = pd.read_csv("./dataset/winemag-data-130k-v2.csv") #https://www.kaggle.com/datasets/zynicide/wine-reviews
dataframe = dataframe.head(500) # 130k is too big and slow for my learning purpose
dataframe.head() # a glance of our dataset
```




<div>
    
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>country</th>
      <th>description</th>
      <th>designation</th>
      <th>points</th>
      <th>price</th>
      <th>province</th>
      <th>region_1</th>
      <th>region_2</th>
      <th>taster_name</th>
      <th>taster_twitter_handle</th>
      <th>title</th>
      <th>variety</th>
      <th>winery</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Italy</td>
      <td>Aromas include tropical fruit, broom, brimston...</td>
      <td>Vulkà Bianco</td>
      <td>87</td>
      <td>NaN</td>
      <td>Sicily &amp; Sardinia</td>
      <td>Etna</td>
      <td>NaN</td>
      <td>Kerin O’Keefe</td>
      <td>@kerinokeefe</td>
      <td>Nicosia 2013 Vulkà Bianco  (Etna)</td>
      <td>White Blend</td>
      <td>Nicosia</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Portugal</td>
      <td>This is ripe and fruity, a wine that is smooth...</td>
      <td>Avidagos</td>
      <td>87</td>
      <td>15.0</td>
      <td>Douro</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Roger Voss</td>
      <td>@vossroger</td>
      <td>Quinta dos Avidagos 2011 Avidagos Red (Douro)</td>
      <td>Portuguese Red</td>
      <td>Quinta dos Avidagos</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>US</td>
      <td>Tart and snappy, the flavors of lime flesh and...</td>
      <td>NaN</td>
      <td>87</td>
      <td>14.0</td>
      <td>Oregon</td>
      <td>Willamette Valley</td>
      <td>Willamette Valley</td>
      <td>Paul Gregutt</td>
      <td>@paulgwine</td>
      <td>Rainstorm 2013 Pinot Gris (Willamette Valley)</td>
      <td>Pinot Gris</td>
      <td>Rainstorm</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>US</td>
      <td>Pineapple rind, lemon pith and orange blossom ...</td>
      <td>Reserve Late Harvest</td>
      <td>87</td>
      <td>13.0</td>
      <td>Michigan</td>
      <td>Lake Michigan Shore</td>
      <td>NaN</td>
      <td>Alexander Peartree</td>
      <td>NaN</td>
      <td>St. Julian 2013 Reserve Late Harvest Riesling ...</td>
      <td>Riesling</td>
      <td>St. Julian</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>US</td>
      <td>Much like the regular bottling from 2012, this...</td>
      <td>Vintner's Reserve Wild Child Block</td>
      <td>87</td>
      <td>65.0</td>
      <td>Oregon</td>
      <td>Willamette Valley</td>
      <td>Willamette Valley</td>
      <td>Paul Gregutt</td>
      <td>@paulgwine</td>
      <td>Sweet Cheeks 2012 Vintner's Reserve Wild Child...</td>
      <td>Pinot Noir</td>
      <td>Sweet Cheeks</td>
    </tr>
  </tbody>
</table>
</div>



## Quick EDA


```python
ax = dataframe['points'].value_counts().sort_index() \
    .plot(kind = 'bar',
          title = 'Count of Wine Reviews by Points')
ax.set_xlabel('Review Points')
plt.show()
```


    
![png](output_5_0.png)
    


## Getting an example


```python
example = dataframe['description'][1] # get the first row for an example
print(example) # so we don't forget what our example is
```

    This is ripe and fruity, a wine that is smooth while still structured. Firm tannins are filled out with juicy red berry fruits and freshened with acidity. It's  already drinkable, although it will certainly be better from 2016.
    

## Roberta Pretrained Model


```python
MODEL = "cardiffnlp/twitter-roberta-base-sentiment" # a pretrained model for sentiment analysis
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
```


```python
# Run for Roberta model
encoded_text = tokenizer(example, return_tensors='pt')
output = model(**encoded_text)
scores = output[0][0].detach().numpy() #tensor to numpy, to store locally in array
scores = softmax(scores) #apply softmax
scores_dict = {
    'negative' : scores[0],
    'neutral' : scores[1],
    'positive' : scores[2]
}
print(scores_dict)

```

    {'negative': 0.0039271605, 'neutral': 0.053314235, 'positive': 0.9427586}
    


```python
#make it a function for iterration
def polarity_scores(example):
    encoded_text = tokenizer(example, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
        'negative' : scores[0],
        'neutral' : scores[1],
        'positive' : scores[2]
    }
    return scores_dict
```


```python
# iterration to do polarity_scores for the entire dataframe
res = {} #result dict
for i, row in dataframe.iterrows():
    desc = row['description']
    myid = row['id']
    res[myid] = polarity_scores(desc) #dict index by id
```


```python
# result of the entire dataframe polarity_scores
res
```




    {0: {'negative': 0.08982867, 'neutral': 0.7339461, 'positive': 0.1762253},
     1: {'negative': 0.0039271605, 'neutral': 0.053314235, 'positive': 0.9427586},
     2: {'negative': 0.0080102775, 'neutral': 0.6061275, 'positive': 0.38586217},
     3: {'negative': 0.00287752, 'neutral': 0.19636476, 'positive': 0.8007577},
     4: {'negative': 0.019464169, 'neutral': 0.2584998, 'positive': 0.722036},
     5: {'negative': 0.008615631, 'neutral': 0.4848579, 'positive': 0.50652647},
     6: {'negative': 0.003338339, 'neutral': 0.4090565, 'positive': 0.5876051},
     7: {'negative': 0.008765581, 'neutral': 0.26616234, 'positive': 0.725072},
     8: {'negative': 0.0016773993, 'neutral': 0.12044009, 'positive': 0.8778825},
     9: {'negative': 0.0014241356, 'neutral': 0.035167314, 'positive': 0.9634085},
     10: {'negative': 0.0019016173, 'neutral': 0.12926817, 'positive': 0.8688302},
     11: {'negative': 0.018896688, 'neutral': 0.72501206, 'positive': 0.25609127},
     12: {'negative': 0.0061704363, 'neutral': 0.3393405, 'positive': 0.6544891},
     13: {'negative': 0.030464035, 'neutral': 0.7050586, 'positive': 0.26447743},
     14: {'negative': 0.005637112, 'neutral': 0.23728883, 'positive': 0.75707406},
     15: {'negative': 0.0014542881, 'neutral': 0.07371222, 'positive': 0.92483354},
     16: {'negative': 0.009275206, 'neutral': 0.5621112, 'positive': 0.42861354},
     17: {'negative': 0.0027433883, 'neutral': 0.10138971, 'positive': 0.8958669},
     18: {'negative': 0.020623691, 'neutral': 0.74485177, 'positive': 0.23452458},
     19: {'negative': 0.0027772824, 'neutral': 0.15396862, 'positive': 0.84325415},
     20: {'negative': 0.0063348357, 'neutral': 0.46616554, 'positive': 0.5274996},
     21: {'negative': 0.018708065, 'neutral': 0.5105875, 'positive': 0.47070444},
     22: {'negative': 0.0037590575, 'neutral': 0.44020405, 'positive': 0.5560369},
     23: {'negative': 0.03940022, 'neutral': 0.60787964, 'positive': 0.35272017},
     24: {'negative': 0.011335793, 'neutral': 0.79593897, 'positive': 0.19272524},
     25: {'negative': 0.00332907, 'neutral': 0.35890302, 'positive': 0.6377679},
     26: {'negative': 0.002377074, 'neutral': 0.17388195, 'positive': 0.823741},
     27: {'negative': 0.008211714, 'neutral': 0.56136453, 'positive': 0.4304237},
     28: {'negative': 0.01043023, 'neutral': 0.6294873, 'positive': 0.3600825},
     29: {'negative': 0.0017046696, 'neutral': 0.09723644, 'positive': 0.9010589},
     30: {'negative': 0.0019417674, 'neutral': 0.17647704, 'positive': 0.82158124},
     31: {'negative': 0.0013935586, 'neutral': 0.11257471, 'positive': 0.8860317},
     32: {'negative': 0.0035206552, 'neutral': 0.5333703, 'positive': 0.46310905},
     33: {'negative': 0.009215789, 'neutral': 0.63443965, 'positive': 0.35634452},
     34: {'negative': 0.0061440924, 'neutral': 0.2659615, 'positive': 0.72789437},
     35: {'negative': 0.10889749, 'neutral': 0.7313807, 'positive': 0.15972185},
     36: {'negative': 0.007529578, 'neutral': 0.42986184, 'positive': 0.56260854},
     37: {'negative': 0.005071234, 'neutral': 0.4269, 'positive': 0.56802875},
     38: {'negative': 0.0027572673, 'neutral': 0.1818679, 'positive': 0.81537485},
     39: {'negative': 0.0011291009, 'neutral': 0.024132067, 'positive': 0.9747388},
     40: {'negative': 0.0044908416, 'neutral': 0.56302434, 'positive': 0.43248484},
     41: {'negative': 0.03487843, 'neutral': 0.7995966, 'positive': 0.16552496},
     42: {'negative': 0.0021897985, 'neutral': 0.23168041, 'positive': 0.7661298},
     43: {'negative': 0.0039591608, 'neutral': 0.30577147, 'positive': 0.6902694},
     44: {'negative': 0.0036890102, 'neutral': 0.3114208, 'positive': 0.6848902},
     45: {'negative': 0.010818684, 'neutral': 0.62075156, 'positive': 0.36842972},
     46: {'negative': 0.0014755109, 'neutral': 0.10777923, 'positive': 0.8907453},
     47: {'negative': 0.0011735978,
      'neutral': 0.121810265,
      'positive': 0.87701607},
     48: {'negative': 0.0034327102, 'neutral': 0.25689366, 'positive': 0.7396737},
     49: {'negative': 0.0014547639, 'neutral': 0.06363338, 'positive': 0.93491185},
     50: {'negative': 0.068753116, 'neutral': 0.8397595, 'positive': 0.09148741},
     51: {'negative': 0.19859733, 'neutral': 0.6427678, 'positive': 0.15863493},
     52: {'negative': 0.01128818, 'neutral': 0.54323375, 'positive': 0.44547802},
     53: {'negative': 0.0011407422, 'neutral': 0.07737268, 'positive': 0.9214866},
     54: {'negative': 0.0021195363, 'neutral': 0.16981688, 'positive': 0.8280636},
     55: {'negative': 0.002965415, 'neutral': 0.28231266, 'positive': 0.7147219},
     56: {'negative': 0.0017647254, 'neutral': 0.20106101, 'positive': 0.7971743},
     57: {'negative': 0.0036447651, 'neutral': 0.32967144, 'positive': 0.66668385},
     58: {'negative': 0.017688503, 'neutral': 0.3377876, 'positive': 0.6445239},
     59: {'negative': 0.33189198, 'neutral': 0.5901469, 'positive': 0.07796111},
     60: {'negative': 0.0066454834, 'neutral': 0.56745046, 'positive': 0.42590407},
     61: {'negative': 0.001965703, 'neutral': 0.23628078, 'positive': 0.7617535},
     62: {'negative': 0.002262602, 'neutral': 0.080133885, 'positive': 0.9176035},
     63: {'negative': 0.00447794, 'neutral': 0.25838897, 'positive': 0.7371331},
     64: {'negative': 0.0015246223, 'neutral': 0.05960588, 'positive': 0.9388695},
     65: {'negative': 0.0012909997, 'neutral': 0.097090326, 'positive': 0.9016187},
     66: {'negative': 0.0012011416, 'neutral': 0.050034586, 'positive': 0.9487642},
     67: {'negative': 0.007910573, 'neutral': 0.512003, 'positive': 0.4800864},
     68: {'negative': 0.002604502, 'neutral': 0.15371056, 'positive': 0.8436849},
     69: {'negative': 0.001618825, 'neutral': 0.0903419, 'positive': 0.9080392},
     70: {'negative': 0.044573672, 'neutral': 0.594323, 'positive': 0.36110333},
     71: {'negative': 0.004954654, 'neutral': 0.15394619, 'positive': 0.84109914},
     72: {'negative': 0.092126474, 'neutral': 0.83543134, 'positive': 0.07244218},
     73: {'negative': 0.012033545, 'neutral': 0.50569767, 'positive': 0.4822688},
     74: {'negative': 0.0015161174, 'neutral': 0.2119441, 'positive': 0.7865398},
     75: {'negative': 0.0070597013, 'neutral': 0.3448501, 'positive': 0.6480901},
     76: {'negative': 0.0024242962, 'neutral': 0.36958277, 'positive': 0.6279929},
     77: {'negative': 0.0035096447, 'neutral': 0.23582676, 'positive': 0.76066357},
     78: {'negative': 0.01077981, 'neutral': 0.33233988, 'positive': 0.6568804},
     79: {'negative': 0.0014879525, 'neutral': 0.13996083, 'positive': 0.85855126},
     80: {'negative': 0.19847913, 'neutral': 0.6736862, 'positive': 0.12783466},
     81: {'negative': 0.6416645, 'neutral': 0.3257509, 'positive': 0.032584623},
     82: {'negative': 0.0010276805,
      'neutral': 0.014609447,
      'positive': 0.98436284},
     83: {'negative': 0.0013720685, 'neutral': 0.053020373, 'positive': 0.9456076},
     84: {'negative': 0.0025109851, 'neutral': 0.24585764, 'positive': 0.75163144},
     85: {'negative': 0.0055727856, 'neutral': 0.39780164, 'positive': 0.59662557},
     86: {'negative': 0.0025281182, 'neutral': 0.28594884, 'positive': 0.71152306},
     87: {'negative': 0.004924638, 'neutral': 0.6901189, 'positive': 0.30495644},
     88: {'negative': 0.042935736, 'neutral': 0.7276907, 'positive': 0.22937353},
     89: {'negative': 0.0019143622, 'neutral': 0.094014, 'positive': 0.9040717},
     90: {'negative': 0.0018846635, 'neutral': 0.07902791, 'positive': 0.9190874},
     91: {'negative': 0.01237007, 'neutral': 0.42432454, 'positive': 0.5633054},
     92: {'negative': 0.004126905, 'neutral': 0.29985482, 'positive': 0.6960183},
     93: {'negative': 0.0013414435, 'neutral': 0.03638912, 'positive': 0.9622694},
     94: {'negative': 0.105117, 'neutral': 0.6923511, 'positive': 0.20253193},
     95: {'negative': 0.0032257747, 'neutral': 0.19563012, 'positive': 0.8011441},
     96: {'negative': 0.0011920092, 'neutral': 0.07780086, 'positive': 0.9210071},
     97: {'negative': 0.004056144, 'neutral': 0.26097146, 'positive': 0.7349724},
     98: {'negative': 0.033056792, 'neutral': 0.72512305, 'positive': 0.24182014},
     99: {'negative': 0.005019879, 'neutral': 0.15861166, 'positive': 0.8363685},
     100: {'negative': 0.0011527453,
      'neutral': 0.060669817,
      'positive': 0.9381774}}




```python
#make a dataframe of the polarity_scores result and merge with the original dataframe
res_dataframe = pd.DataFrame(res).T
res_dataframe = res_dataframe.reset_index().rename(columns={'index': 'id'})
res_mergerd_dataframe = res_dataframe.merge(dataframe, how='left')
```


```python
res_mergerd_dataframe 
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>negative</th>
      <th>neutral</th>
      <th>positive</th>
      <th>country</th>
      <th>description</th>
      <th>designation</th>
      <th>points</th>
      <th>price</th>
      <th>province</th>
      <th>region_1</th>
      <th>region_2</th>
      <th>taster_name</th>
      <th>taster_twitter_handle</th>
      <th>title</th>
      <th>variety</th>
      <th>winery</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0.089829</td>
      <td>0.733946</td>
      <td>0.176225</td>
      <td>Italy</td>
      <td>Aromas include tropical fruit, broom, brimston...</td>
      <td>Vulkà Bianco</td>
      <td>87</td>
      <td>NaN</td>
      <td>Sicily &amp; Sardinia</td>
      <td>Etna</td>
      <td>NaN</td>
      <td>Kerin O’Keefe</td>
      <td>@kerinokeefe</td>
      <td>Nicosia 2013 Vulkà Bianco  (Etna)</td>
      <td>White Blend</td>
      <td>Nicosia</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0.003927</td>
      <td>0.053314</td>
      <td>0.942759</td>
      <td>Portugal</td>
      <td>This is ripe and fruity, a wine that is smooth...</td>
      <td>Avidagos</td>
      <td>87</td>
      <td>15.0</td>
      <td>Douro</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Roger Voss</td>
      <td>@vossroger</td>
      <td>Quinta dos Avidagos 2011 Avidagos Red (Douro)</td>
      <td>Portuguese Red</td>
      <td>Quinta dos Avidagos</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0.008010</td>
      <td>0.606128</td>
      <td>0.385862</td>
      <td>US</td>
      <td>Tart and snappy, the flavors of lime flesh and...</td>
      <td>NaN</td>
      <td>87</td>
      <td>14.0</td>
      <td>Oregon</td>
      <td>Willamette Valley</td>
      <td>Willamette Valley</td>
      <td>Paul Gregutt</td>
      <td>@paulgwine</td>
      <td>Rainstorm 2013 Pinot Gris (Willamette Valley)</td>
      <td>Pinot Gris</td>
      <td>Rainstorm</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>0.002878</td>
      <td>0.196365</td>
      <td>0.800758</td>
      <td>US</td>
      <td>Pineapple rind, lemon pith and orange blossom ...</td>
      <td>Reserve Late Harvest</td>
      <td>87</td>
      <td>13.0</td>
      <td>Michigan</td>
      <td>Lake Michigan Shore</td>
      <td>NaN</td>
      <td>Alexander Peartree</td>
      <td>NaN</td>
      <td>St. Julian 2013 Reserve Late Harvest Riesling ...</td>
      <td>Riesling</td>
      <td>St. Julian</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>0.019464</td>
      <td>0.258500</td>
      <td>0.722036</td>
      <td>US</td>
      <td>Much like the regular bottling from 2012, this...</td>
      <td>Vintner's Reserve Wild Child Block</td>
      <td>87</td>
      <td>65.0</td>
      <td>Oregon</td>
      <td>Willamette Valley</td>
      <td>Willamette Valley</td>
      <td>Paul Gregutt</td>
      <td>@paulgwine</td>
      <td>Sweet Cheeks 2012 Vintner's Reserve Wild Child...</td>
      <td>Pinot Noir</td>
      <td>Sweet Cheeks</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>495</th>
      <td>495</td>
      <td>0.001134</td>
      <td>0.028352</td>
      <td>0.970514</td>
      <td>US</td>
      <td>This El Dorado wine opens with aromas of sweet...</td>
      <td>Battonage</td>
      <td>87</td>
      <td>18.0</td>
      <td>California</td>
      <td>El Dorado</td>
      <td>Sierra Foothills</td>
      <td>Virginie Boone</td>
      <td>@vboone</td>
      <td>Lava Cap 2010 Battonage Chardonnay (El Dorado)</td>
      <td>Chardonnay</td>
      <td>Lava Cap</td>
    </tr>
    <tr>
      <th>496</th>
      <td>496</td>
      <td>0.002118</td>
      <td>0.051176</td>
      <td>0.946706</td>
      <td>Spain</td>
      <td>This barrel-fermented Verdejo is interesting, ...</td>
      <td>Collection Blanco</td>
      <td>87</td>
      <td>25.0</td>
      <td>Northern Spain</td>
      <td>Rueda</td>
      <td>NaN</td>
      <td>Michael Schachner</td>
      <td>@wineschach</td>
      <td>Loess 2009 Collection Blanco Verdejo (Rueda)</td>
      <td>Verdejo</td>
      <td>Loess</td>
    </tr>
    <tr>
      <th>497</th>
      <td>497</td>
      <td>0.004960</td>
      <td>0.381865</td>
      <td>0.613175</td>
      <td>US</td>
      <td>This wine has the variety's trademark notes of...</td>
      <td>Babcock Vineyard</td>
      <td>87</td>
      <td>30.0</td>
      <td>California</td>
      <td>Suisun Valley</td>
      <td>North Coast</td>
      <td>Virginie Boone</td>
      <td>@vboone</td>
      <td>MICA Cellars 2009 Babcock Vineyard Cabernet Fr...</td>
      <td>Cabernet Franc</td>
      <td>MICA Cellars</td>
    </tr>
    <tr>
      <th>498</th>
      <td>498</td>
      <td>0.003725</td>
      <td>0.146415</td>
      <td>0.849860</td>
      <td>US</td>
      <td>There are lot's of cherry, cola, sandalwood an...</td>
      <td>Annabella</td>
      <td>87</td>
      <td>17.0</td>
      <td>California</td>
      <td>Carneros</td>
      <td>Napa-Sonoma</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Michael Pozzan 2010 Annabella Pinot Noir (Carn...</td>
      <td>Pinot Noir</td>
      <td>Michael Pozzan</td>
    </tr>
    <tr>
      <th>499</th>
      <td>499</td>
      <td>0.003745</td>
      <td>0.247605</td>
      <td>0.748650</td>
      <td>France</td>
      <td>This is a big, spicy wine, with very ripe flav...</td>
      <td>L'Esprit de Provence</td>
      <td>87</td>
      <td>20.0</td>
      <td>Provence</td>
      <td>Côtes de Provence</td>
      <td>NaN</td>
      <td>Roger Voss</td>
      <td>@vossroger</td>
      <td>Domaine du Grand Cros 2011 L'Esprit de Provenc...</td>
      <td>Rosé</td>
      <td>Domaine du Grand Cros</td>
    </tr>
  </tbody>
</table>
<p>500 rows × 17 columns</p>
</div>



## Plot The Result


```python
fig, axs = plt.subplots(1, 3, figsize=(15,5))
sns.barplot(data=res_mergerd_dataframe, x= 'points', y='positive', ax=axs[0])
sns.barplot(data=res_mergerd_dataframe, x= 'points', y='neutral', ax=axs[1])
sns.barplot(data=res_mergerd_dataframe, x= 'points', y='negative', ax=axs[2])
axs[0].set_title('positive')
axs[1].set_title('neutral')
axs[2].set_title('negative')
plt.show()
```


    
![png](output_17_0.png)
    


# Conclusion
## Taster's given description of the wine is closely corresponds to their given points of the wine
