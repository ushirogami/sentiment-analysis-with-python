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
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
      'positive': 0.9381774},
     101: {'negative': 0.0024925866, 'neutral': 0.25386858, 'positive': 0.7436388},
     102: {'negative': 0.0017172545,
      'neutral': 0.16736011,
      'positive': 0.83092266},
     103: {'negative': 0.022565316, 'neutral': 0.7368059, 'positive': 0.24062882},
     104: {'negative': 0.0038721825, 'neutral': 0.35107997, 'positive': 0.6450478},
     105: {'negative': 0.002523806, 'neutral': 0.23333022, 'positive': 0.764146},
     106: {'negative': 0.0024020178, 'neutral': 0.1685946, 'positive': 0.8290034},
     107: {'negative': 0.0056453384, 'neutral': 0.48628262, 'positive': 0.5080721},
     108: {'negative': 0.0048399046, 'neutral': 0.4318815, 'positive': 0.5632786},
     109: {'negative': 0.0028820804, 'neutral': 0.24386051, 'positive': 0.7532574},
     110: {'negative': 0.0028323284,
      'neutral': 0.16213778,
      'positive': 0.83502996},
     111: {'negative': 0.012362829, 'neutral': 0.45951802, 'positive': 0.52811915},
     112: {'negative': 0.0013607455,
      'neutral': 0.061879776,
      'positive': 0.9367595},
     113: {'negative': 0.004541226, 'neutral': 0.341233, 'positive': 0.6542257},
     114: {'negative': 0.013485117, 'neutral': 0.8467753, 'positive': 0.13973956},
     115: {'negative': 0.0032288374,
      'neutral': 0.22672628,
      'positive': 0.77004486},
     116: {'negative': 0.0064192577, 'neutral': 0.3999412, 'positive': 0.59363955},
     117: {'negative': 0.0026155734,
      'neutral': 0.084119864,
      'positive': 0.9132646},
     118: {'negative': 0.009478568, 'neutral': 0.40929708, 'positive': 0.5812244},
     119: {'negative': 0.0027377687, 'neutral': 0.1208561, 'positive': 0.87640613},
     120: {'negative': 0.024523484, 'neutral': 0.40148035, 'positive': 0.5739962},
     121: {'negative': 0.001376702, 'neutral': 0.03710073, 'positive': 0.9615225},
     122: {'negative': 0.029345727, 'neutral': 0.4868004, 'positive': 0.48385388},
     123: {'negative': 0.010866446, 'neutral': 0.5651359, 'positive': 0.42399767},
     124: {'negative': 0.001869778, 'neutral': 0.06353256, 'positive': 0.9345976},
     125: {'negative': 0.0050928174, 'neutral': 0.16835466, 'positive': 0.8265525},
     126: {'negative': 0.0036424212, 'neutral': 0.22683272, 'positive': 0.7695248},
     127: {'negative': 0.006961738, 'neutral': 0.3718857, 'positive': 0.6211526},
     128: {'negative': 0.0075293086, 'neutral': 0.20452672, 'positive': 0.787944},
     129: {'negative': 0.046104055, 'neutral': 0.39252526, 'positive': 0.5613707},
     130: {'negative': 0.0027835767,
      'neutral': 0.03780912,
      'positive': 0.95940727},
     131: {'negative': 0.0029189656, 'neutral': 0.1007511, 'positive': 0.89633},
     132: {'negative': 0.0012998419,
      'neutral': 0.04109098,
      'positive': 0.95760924},
     133: {'negative': 0.0016291813,
      'neutral': 0.023926944,
      'positive': 0.97444385},
     134: {'negative': 0.019247157, 'neutral': 0.5157879, 'positive': 0.4649649},
     135: {'negative': 0.0061774743,
      'neutral': 0.115931444,
      'positive': 0.8778911},
     136: {'negative': 0.0038680532,
      'neutral': 0.11662601,
      'positive': 0.87950593},
     137: {'negative': 0.0021828858,
      'neutral': 0.041251488,
      'positive': 0.9565656},
     138: {'negative': 0.008994783, 'neutral': 0.58115405, 'positive': 0.40985116},
     139: {'negative': 0.0029752443,
      'neutral': 0.059896678,
      'positive': 0.93712807},
     140: {'negative': 0.0050178827,
      'neutral': 0.104233414,
      'positive': 0.8907487},
     141: {'negative': 0.005033133, 'neutral': 0.20376764, 'positive': 0.7911992},
     142: {'negative': 0.003972445, 'neutral': 0.08021089, 'positive': 0.91581666},
     143: {'negative': 0.007340236, 'neutral': 0.5234167, 'positive': 0.4692431},
     144: {'negative': 0.0027970094, 'neutral': 0.30563524, 'positive': 0.6915677},
     145: {'negative': 0.012846455, 'neutral': 0.843053, 'positive': 0.1441005},
     146: {'negative': 0.0026079868,
      'neutral': 0.24821441,
      'positive': 0.74917763},
     147: {'negative': 0.0011719131,
      'neutral': 0.030446637,
      'positive': 0.9683814},
     148: {'negative': 0.0019565604, 'neutral': 0.17089851, 'positive': 0.827145},
     149: {'negative': 0.0017560892, 'neutral': 0.06778136, 'positive': 0.9304625},
     150: {'negative': 0.0016066941,
      'neutral': 0.09805329,
      'positive': 0.90033996},
     151: {'negative': 0.0008955602,
      'neutral': 0.05268035,
      'positive': 0.94642407},
     152: {'negative': 0.0013462964,
      'neutral': 0.07186551,
      'positive': 0.92678815},
     153: {'negative': 0.0077148997, 'neutral': 0.2849065, 'positive': 0.7073786},
     154: {'negative': 0.0070432695,
      'neutral': 0.40539646,
      'positive': 0.58756024},
     155: {'negative': 0.001213565, 'neutral': 0.05228861, 'positive': 0.94649786},
     156: {'negative': 0.0030738716, 'neutral': 0.11764059, 'positive': 0.8792855},
     157: {'negative': 0.0060958327,
      'neutral': 0.097546846,
      'positive': 0.8963573},
     158: {'negative': 0.0052645016, 'neutral': 0.2805701, 'positive': 0.7141654},
     159: {'negative': 0.0060953796,
      'neutral': 0.48158798,
      'positive': 0.51231664},
     160: {'negative': 0.0021090826, 'neutral': 0.03217607, 'positive': 0.9657149},
     161: {'negative': 0.0053231246,
      'neutral': 0.59113663,
      'positive': 0.40354028},
     162: {'negative': 0.005780781, 'neutral': 0.3743985, 'positive': 0.6198207},
     163: {'negative': 0.003686852, 'neutral': 0.08449391, 'positive': 0.9118192},
     164: {'negative': 0.0057116398,
      'neutral': 0.39895952,
      'positive': 0.59532887},
     165: {'negative': 0.0037737144, 'neutral': 0.3568338, 'positive': 0.6393925},
     166: {'negative': 0.0019618552,
      'neutral': 0.039787795,
      'positive': 0.95825034},
     167: {'negative': 0.0062835873, 'neutral': 0.18252255, 'positive': 0.8111939},
     168: {'negative': 0.0011222762, 'neutral': 0.06427835, 'positive': 0.9345994},
     169: {'negative': 0.002349821, 'neutral': 0.13500491, 'positive': 0.8626453},
     170: {'negative': 0.0028257873,
      'neutral': 0.08770158,
      'positive': 0.90947264},
     171: {'negative': 0.0015124548,
      'neutral': 0.029767891,
      'positive': 0.9687196},
     172: {'negative': 0.0022122555,
      'neutral': 0.069147326,
      'positive': 0.9286405},
     173: {'negative': 0.0013265104, 'neutral': 0.02147174, 'positive': 0.9772018},
     174: {'negative': 0.0027719208,
      'neutral': 0.12684084,
      'positive': 0.87038726},
     175: {'negative': 0.0039421665, 'neutral': 0.52313834, 'positive': 0.4729195},
     176: {'negative': 0.010883744, 'neutral': 0.06785028, 'positive': 0.921266},
     177: {'negative': 0.0010073761,
      'neutral': 0.030316459,
      'positive': 0.9686762},
     178: {'negative': 0.45844132, 'neutral': 0.44926128, 'positive': 0.092297375},
     179: {'negative': 0.005691415, 'neutral': 0.191835, 'positive': 0.8024736},
     180: {'negative': 0.002091936, 'neutral': 0.10456484, 'positive': 0.8933432},
     181: {'negative': 0.0047992067,
      'neutral': 0.05741748,
      'positive': 0.93778336},
     182: {'negative': 0.06083825, 'neutral': 0.5677428, 'positive': 0.37141898},
     183: {'negative': 0.0017244342,
      'neutral': 0.027186733,
      'positive': 0.9710888},
     184: {'negative': 0.0050604343, 'neutral': 0.41268423, 'positive': 0.5822553},
     185: {'negative': 0.0033494101,
      'neutral': 0.067101784,
      'positive': 0.9295488},
     186: {'negative': 0.0014244849,
      'neutral': 0.017244112,
      'positive': 0.98133135},
     187: {'negative': 0.003637854, 'neutral': 0.21080144, 'positive': 0.7855607},
     188: {'negative': 0.0014376342,
      'neutral': 0.03241841,
      'positive': 0.96614397},
     189: {'negative': 0.0032310325,
      'neutral': 0.07448434,
      'positive': 0.92228466},
     190: {'negative': 0.005539269, 'neutral': 0.48426616, 'positive': 0.5101946},
     191: {'negative': 0.0012704349,
      'neutral': 0.050555296,
      'positive': 0.9481743},
     192: {'negative': 0.0009470056,
      'neutral': 0.031617418,
      'positive': 0.9674356},
     193: {'negative': 0.21337388, 'neutral': 0.6486304, 'positive': 0.13799568},
     194: {'negative': 0.0013280765,
      'neutral': 0.075189374,
      'positive': 0.92348254},
     195: {'negative': 0.0856367, 'neutral': 0.75329435, 'positive': 0.16106902},
     196: {'negative': 0.0016423851, 'neutral': 0.13431598, 'positive': 0.8640416},
     197: {'negative': 0.002324289, 'neutral': 0.08137652, 'positive': 0.91629916},
     198: {'negative': 0.0012254287,
      'neutral': 0.07185217,
      'positive': 0.92692244},
     199: {'negative': 0.0017225702,
      'neutral': 0.12491649,
      'positive': 0.87336093},
     200: {'negative': 0.00384611, 'neutral': 0.14829993, 'positive': 0.84785396},
     201: {'negative': 0.0018181278,
      'neutral': 0.07950328,
      'positive': 0.91867864},
     202: {'negative': 0.0011297371, 'neutral': 0.04363369, 'positive': 0.9552366},
     203: {'negative': 0.0020248569, 'neutral': 0.06949215, 'positive': 0.928483},
     204: {'negative': 0.0020315812,
      'neutral': 0.30123043,
      'positive': 0.69673795},
     205: {'negative': 0.0060017626, 'neutral': 0.3858684, 'positive': 0.60812986},
     206: {'negative': 0.0019564328,
      'neutral': 0.115768895,
      'positive': 0.8822747},
     207: {'negative': 0.006877592, 'neutral': 0.28039157, 'positive': 0.7127308},
     208: {'negative': 0.004408313, 'neutral': 0.36947536, 'positive': 0.62611634},
     209: {'negative': 0.0037335893, 'neutral': 0.3026584, 'positive': 0.693608},
     210: {'negative': 0.0017160592,
      'neutral': 0.10617888,
      'positive': 0.89210504},
     211: {'negative': 0.0020262005,
      'neutral': 0.14982074,
      'positive': 0.84815305},
     212: {'negative': 0.03488444, 'neutral': 0.4755438, 'positive': 0.48957178},
     213: {'negative': 0.004432919, 'neutral': 0.53237, 'positive': 0.46319714},
     214: {'negative': 0.016294802, 'neutral': 0.5961417, 'positive': 0.38756353},
     215: {'negative': 0.02258977, 'neutral': 0.69072515, 'positive': 0.28668505},
     216: {'negative': 0.011653971, 'neutral': 0.7860244, 'positive': 0.20232165},
     217: {'negative': 0.012663594, 'neutral': 0.34715036, 'positive': 0.640186},
     218: {'negative': 0.0014016726, 'neutral': 0.1233537, 'positive': 0.8752446},
     219: {'negative': 0.003242523, 'neutral': 0.119139545, 'positive': 0.8776179},
     220: {'negative': 0.00811445, 'neutral': 0.65520906, 'positive': 0.3366765},
     221: {'negative': 0.0020747206,
      'neutral': 0.14379358,
      'positive': 0.85413164},
     222: {'negative': 0.012018314, 'neutral': 0.5368189, 'positive': 0.4511628},
     223: {'negative': 0.0023463431,
      'neutral': 0.18563329,
      'positive': 0.81202036},
     224: {'negative': 0.031039177, 'neutral': 0.6174302, 'positive': 0.35153064},
     225: {'negative': 0.0021517167,
      'neutral': 0.080658026,
      'positive': 0.9171902},
     226: {'negative': 0.0047382563,
      'neutral': 0.22314338,
      'positive': 0.77211845},
     227: {'negative': 0.0059789787,
      'neutral': 0.21734378,
      'positive': 0.77667725},
     228: {'negative': 0.0026186244,
      'neutral': 0.083385795,
      'positive': 0.9139955},
     229: {'negative': 0.0020649522, 'neutral': 0.0941752, 'positive': 0.9037599},
     230: {'negative': 0.0019903842,
      'neutral': 0.049265098,
      'positive': 0.9487445},
     231: {'negative': 0.0038030008,
      'neutral': 0.118543364,
      'positive': 0.8776536},
     232: {'negative': 0.24644384, 'neutral': 0.46873862, 'positive': 0.28481755},
     233: {'negative': 0.024229791, 'neutral': 0.20582795, 'positive': 0.7699423},
     234: {'negative': 0.0039014227, 'neutral': 0.23093265, 'positive': 0.7651659},
     235: {'negative': 0.7192506, 'neutral': 0.2312181, 'positive': 0.049531266},
     236: {'negative': 0.13290559, 'neutral': 0.6367826, 'positive': 0.2303118},
     237: {'negative': 0.0015149169,
      'neutral': 0.035022967,
      'positive': 0.96346205},
     238: {'negative': 0.10806702, 'neutral': 0.73474395, 'positive': 0.15718906},
     239: {'negative': 0.0571674, 'neutral': 0.42766866, 'positive': 0.51516396},
     240: {'negative': 0.010902412, 'neutral': 0.19877718, 'positive': 0.7903204},
     241: {'negative': 0.0028615147,
      'neutral': 0.115608044,
      'positive': 0.88153046},
     242: {'negative': 0.0023361, 'neutral': 0.2062517, 'positive': 0.7914122},
     243: {'negative': 0.001497907, 'neutral': 0.06413239, 'positive': 0.93436974},
     244: {'negative': 0.0029715025, 'neutral': 0.06730191, 'positive': 0.9297266},
     245: {'negative': 0.028571548, 'neutral': 0.34301347, 'positive': 0.628415},
     246: {'negative': 0.002309908,
      'neutral': 0.053249765,
      'positive': 0.94444036},
     247: {'negative': 0.0026842707, 'neutral': 0.18890613, 'positive': 0.8084096},
     248: {'negative': 0.0028353294,
      'neutral': 0.062128883,
      'positive': 0.93503577},
     249: {'negative': 0.0023801262,
      'neutral': 0.07749535,
      'positive': 0.92012453},
     250: {'negative': 0.0026295832, 'neutral': 0.2888317, 'positive': 0.7085387},
     251: {'negative': 0.015932377, 'neutral': 0.436142, 'positive': 0.54792565},
     252: {'negative': 0.015469458, 'neutral': 0.30151492, 'positive': 0.68301564},
     253: {'negative': 0.27772048, 'neutral': 0.583116, 'positive': 0.13916348},
     254: {'negative': 0.0017445283,
      'neutral': 0.08086765,
      'positive': 0.91738784},
     255: {'negative': 0.001360959, 'neutral': 0.07295731, 'positive': 0.9256817},
     256: {'negative': 0.0021852485,
      'neutral': 0.06491153,
      'positive': 0.93290323},
     257: {'negative': 0.010838068, 'neutral': 0.21043457, 'positive': 0.7787273},
     258: {'negative': 0.0018248602,
      'neutral': 0.065494984,
      'positive': 0.93268013},
     259: {'negative': 0.002754713, 'neutral': 0.22114287, 'positive': 0.77610236},
     260: {'negative': 0.013027994, 'neutral': 0.40686697, 'positive': 0.580105},
     261: {'negative': 0.009672806, 'neutral': 0.6810737, 'positive': 0.30925345},
     262: {'negative': 0.0014815539,
      'neutral': 0.046770774,
      'positive': 0.9517477},
     263: {'negative': 0.0015055342,
      'neutral': 0.030853355,
      'positive': 0.9676412},
     264: {'negative': 0.0055537676,
      'neutral': 0.31222606,
      'positive': 0.68222016},
     265: {'negative': 0.0023482826, 'neutral': 0.19568978, 'positive': 0.8019619},
     266: {'negative': 0.008299519, 'neutral': 0.3105088, 'positive': 0.6811917},
     267: {'negative': 0.001702535, 'neutral': 0.1385932, 'positive': 0.85970426},
     268: {'negative': 0.0017616523, 'neutral': 0.15228863, 'positive': 0.8459497},
     269: {'negative': 0.005821612, 'neutral': 0.21177362, 'positive': 0.78240484},
     270: {'negative': 0.0012231815,
      'neutral': 0.038051233,
      'positive': 0.96072555},
     271: {'negative': 0.001592068, 'neutral': 0.07460285, 'positive': 0.92380506},
     272: {'negative': 0.0011484188,
      'neutral': 0.05037169,
      'positive': 0.94847983},
     273: {'negative': 0.0017085823, 'neutral': 0.047552, 'positive': 0.95073944},
     274: {'negative': 0.0013068821, 'neutral': 0.028225122, 'positive': 0.970468},
     275: {'negative': 0.00856624, 'neutral': 0.6979876, 'positive': 0.29344615},
     276: {'negative': 0.0028790692, 'neutral': 0.04936687, 'positive': 0.947754},
     277: {'negative': 0.0018624181, 'neutral': 0.04853035, 'positive': 0.9496072},
     278: {'negative': 0.0021320926,
      'neutral': 0.07567793,
      'positive': 0.92218995},
     279: {'negative': 0.0014195892,
      'neutral': 0.012365564,
      'positive': 0.9862148},
     280: {'negative': 0.05394419, 'neutral': 0.28135568, 'positive': 0.66470015},
     281: {'negative': 0.0019296188,
      'neutral': 0.030007934,
      'positive': 0.96806246},
     282: {'negative': 0.0020161946,
      'neutral': 0.06583988,
      'positive': 0.93214387},
     283: {'negative': 0.0022639357,
      'neutral': 0.058311243,
      'positive': 0.9394248},
     284: {'negative': 0.17240302, 'neutral': 0.5207404, 'positive': 0.30685657},
     285: {'negative': 0.0031235667,
      'neutral': 0.06081864,
      'positive': 0.93605775},
     286: {'negative': 0.0013416482,
      'neutral': 0.021944651,
      'positive': 0.97671366},
     287: {'negative': 0.0022681016,
      'neutral': 0.19175361,
      'positive': 0.80597824},
     288: {'negative': 0.094363935, 'neutral': 0.46292743, 'positive': 0.44270867},
     289: {'negative': 0.0014365765,
      'neutral': 0.019784912,
      'positive': 0.97877854},
     290: {'negative': 0.0015266528,
      'neutral': 0.031006997,
      'positive': 0.9674664},
     291: {'negative': 0.0039715623,
      'neutral': 0.37364545,
      'positive': 0.62238306},
     292: {'negative': 0.001834506, 'neutral': 0.033834476, 'positive': 0.9643311},
     293: {'negative': 0.009380765, 'neutral': 0.2841121, 'positive': 0.70650715},
     294: {'negative': 0.0018270368,
      'neutral': 0.06045375,
      'positive': 0.93771917},
     295: {'negative': 0.00095278054,
      'neutral': 0.040684517,
      'positive': 0.9583627},
     296: {'negative': 0.0025127204,
      'neutral': 0.07276558,
      'positive': 0.92472166},
     297: {'negative': 0.0011847654, 'neutral': 0.033821236, 'positive': 0.964994},
     298: {'negative': 0.005016166, 'neutral': 0.42440578, 'positive': 0.57057804},
     299: {'negative': 0.008146392, 'neutral': 0.6026408, 'positive': 0.38921282},
     300: {'negative': 0.0012389987,
      'neutral': 0.023602132,
      'positive': 0.9751589},
     301: {'negative': 0.0018194031,
      'neutral': 0.073726244,
      'positive': 0.9244544},
     302: {'negative': 0.0025697863, 'neutral': 0.11746756, 'positive': 0.8799626},
     303: {'negative': 0.045489848, 'neutral': 0.6890896, 'positive': 0.2654206},
     304: {'negative': 0.071305595, 'neutral': 0.80704063, 'positive': 0.12165376},
     305: {'negative': 0.0015832573, 'neutral': 0.07262124, 'positive': 0.9257955},
     306: {'negative': 0.0019685281,
      'neutral': 0.033256173,
      'positive': 0.96477526},
     307: {'negative': 0.0016293633, 'neutral': 0.09318213, 'positive': 0.9051885},
     308: {'negative': 0.0016044659, 'neutral': 0.07027328, 'positive': 0.9281222},
     309: {'negative': 0.003938785, 'neutral': 0.13899478, 'positive': 0.85706645},
     310: {'negative': 0.0025943692, 'neutral': 0.1522268, 'positive': 0.84517884},
     311: {'negative': 0.034186225, 'neutral': 0.3234295, 'positive': 0.6423843},
     312: {'negative': 0.0016076438,
      'neutral': 0.054022044,
      'positive': 0.94437027},
     313: {'negative': 0.0019671544,
      'neutral': 0.044288754,
      'positive': 0.95374405},
     314: {'negative': 0.0013547322,
      'neutral': 0.030612564,
      'positive': 0.96803266},
     315: {'negative': 0.0013887509, 'neutral': 0.15551926, 'positive': 0.843092},
     316: {'negative': 0.0010131297,
      'neutral': 0.036401164,
      'positive': 0.9625857},
     317: {'negative': 0.009894588, 'neutral': 0.36160153, 'positive': 0.6285039},
     318: {'negative': 0.017265951, 'neutral': 0.26447648, 'positive': 0.71825755},
     319: {'negative': 0.0012493661, 'neutral': 0.0844193, 'positive': 0.9143314},
     320: {'negative': 0.0043509863, 'neutral': 0.3715219, 'positive': 0.6241271},
     321: {'negative': 0.048108347, 'neutral': 0.6486238, 'positive': 0.30326784},
     322: {'negative': 0.0023553504,
      'neutral': 0.21048935,
      'positive': 0.78715533},
     323: {'negative': 0.0042478554, 'neutral': 0.35615626, 'positive': 0.6395959},
     324: {'negative': 0.0062796143, 'neutral': 0.30372572, 'positive': 0.6899946},
     325: {'negative': 0.0030938026, 'neutral': 0.3007524, 'positive': 0.6961538},
     326: {'negative': 0.0012501491, 'neutral': 0.08519993, 'positive': 0.9135499},
     327: {'negative': 0.0036712876,
      'neutral': 0.098305196,
      'positive': 0.8980235},
     328: {'negative': 0.0013711074,
      'neutral': 0.062498976,
      'positive': 0.9361299},
     329: {'negative': 0.00154826, 'neutral': 0.06272747, 'positive': 0.93572426},
     330: {'negative': 0.004836442, 'neutral': 0.46239173, 'positive': 0.5327718},
     331: {'negative': 0.00114512, 'neutral': 0.048075937, 'positive': 0.950779},
     332: {'negative': 0.001235692,
      'neutral': 0.044645093,
      'positive': 0.95411915},
     333: {'negative': 0.001574734, 'neutral': 0.052743383, 'positive': 0.9456819},
     334: {'negative': 0.0052342876, 'neutral': 0.3705142, 'positive': 0.6242515},
     335: {'negative': 0.012751814, 'neutral': 0.47341654, 'positive': 0.5138317},
     336: {'negative': 0.15909953, 'neutral': 0.3983484, 'positive': 0.44255206},
     337: {'negative': 0.0022559839,
      'neutral': 0.11510693,
      'positive': 0.88263714},
     338: {'negative': 0.40132573, 'neutral': 0.5248681, 'positive': 0.07380616},
     339: {'negative': 0.0013167958,
      'neutral': 0.06850447,
      'positive': 0.93017876},
     340: {'negative': 0.14856566, 'neutral': 0.65628934, 'positive': 0.19514495},
     341: {'negative': 0.4765272, 'neutral': 0.44982073, 'positive': 0.073652096},
     342: {'negative': 0.0056909746, 'neutral': 0.240432, 'positive': 0.75387704},
     343: {'negative': 0.22216924, 'neutral': 0.6154759, 'positive': 0.16235493},
     344: {'negative': 0.56549627, 'neutral': 0.3617989, 'positive': 0.07270489},
     345: {'negative': 0.0077996897,
      'neutral': 0.10115976,
      'positive': 0.89104056},
     346: {'negative': 0.0048083547,
      'neutral': 0.084481895,
      'positive': 0.9107098},
     347: {'negative': 0.015954627, 'neutral': 0.5199003, 'positive': 0.46414506},
     348: {'negative': 0.0021121064,
      'neutral': 0.074302696,
      'positive': 0.92358524},
     349: {'negative': 0.016886586, 'neutral': 0.19527605, 'positive': 0.7878373},
     350: {'negative': 0.0024372495,
      'neutral': 0.09919617,
      'positive': 0.89836663},
     351: {'negative': 0.0026429505,
      'neutral': 0.06854445,
      'positive': 0.92881256},
     352: {'negative': 0.0012814873,
      'neutral': 0.024386361,
      'positive': 0.97433215},
     353: {'negative': 0.0020180142,
      'neutral': 0.044736713,
      'positive': 0.9532453},
     354: {'negative': 0.0051784096, 'neutral': 0.15547228, 'positive': 0.8393493},
     355: {'negative': 0.0036896581,
      'neutral': 0.17274608,
      'positive': 0.82356423},
     356: {'negative': 0.002647591, 'neutral': 0.0743338, 'positive': 0.9230186},
     357: {'negative': 0.0021210164,
      'neutral': 0.07119284,
      'positive': 0.92668617},
     358: {'negative': 0.002529938, 'neutral': 0.05866926, 'positive': 0.9388009},
     359: {'negative': 0.0029396848,
      'neutral': 0.17125896,
      'positive': 0.82580143},
     360: {'negative': 0.0059127305, 'neutral': 0.30097523, 'positive': 0.693112},
     361: {'negative': 0.0024834375, 'neutral': 0.16214208, 'positive': 0.8353745},
     362: {'negative': 0.0010809866,
      'neutral': 0.054774866,
      'positive': 0.9441441},
     363: {'negative': 0.001818759, 'neutral': 0.12930326, 'positive': 0.868878},
     364: {'negative': 0.0022632722,
      'neutral': 0.053961597,
      'positive': 0.94377506},
     365: {'negative': 0.0031029168,
      'neutral': 0.08273159,
      'positive': 0.91416544},
     366: {'negative': 0.015984671, 'neutral': 0.5292373, 'positive': 0.45477808},
     367: {'negative': 0.003490952, 'neutral': 0.1502532, 'positive': 0.8462559},
     368: {'negative': 0.0023228452, 'neutral': 0.0893005, 'positive': 0.9083767},
     369: {'negative': 0.0032174275, 'neutral': 0.12690604, 'positive': 0.8698765},
     370: {'negative': 0.008899839, 'neutral': 0.5771233, 'positive': 0.41397685},
     371: {'negative': 0.0014974362, 'neutral': 0.08724618, 'positive': 0.9112564},
     372: {'negative': 0.0036869806,
      'neutral': 0.35299614,
      'positive': 0.64331686},
     373: {'negative': 0.0020602425,
      'neutral': 0.13727231,
      'positive': 0.86066747},
     374: {'negative': 0.0018718223, 'neutral': 0.0555087, 'positive': 0.94261944},
     375: {'negative': 0.0027474524,
      'neutral': 0.20961918,
      'positive': 0.78763336},
     376: {'negative': 0.0065935776, 'neutral': 0.11939845, 'positive': 0.874008},
     377: {'negative': 0.0014752274, 'neutral': 0.19011122, 'positive': 0.8084135},
     378: {'negative': 0.0031550718,
      'neutral': 0.09466671,
      'positive': 0.90217817},
     379: {'negative': 0.0013786121, 'neutral': 0.04533287, 'positive': 0.9532885},
     380: {'negative': 0.007301357, 'neutral': 0.22926682, 'positive': 0.76343185},
     381: {'negative': 0.0050553638, 'neutral': 0.10114224, 'positive': 0.8938024},
     382: {'negative': 0.048618745, 'neutral': 0.66470015, 'positive': 0.28668115},
     383: {'negative': 0.0014022836,
      'neutral': 0.014924065,
      'positive': 0.9836737},
     384: {'negative': 0.0018593633, 'neutral': 0.06503475, 'positive': 0.9331059},
     385: {'negative': 0.0015992444,
      'neutral': 0.108052425,
      'positive': 0.8903483},
     386: {'negative': 0.0022740988,
      'neutral': 0.06376852,
      'positive': 0.93395734},
     387: {'negative': 0.0013348178,
      'neutral': 0.033813477,
      'positive': 0.96485174},
     388: {'negative': 0.0029715844,
      'neutral': 0.11914762,
      'positive': 0.87788075},
     389: {'negative': 0.0077690603, 'neutral': 0.14138882, 'positive': 0.8508421},
     390: {'negative': 0.0024105487,
      'neutral': 0.034404695,
      'positive': 0.9631847},
     391: {'negative': 0.012755574, 'neutral': 0.11117341, 'positive': 0.87607104},
     392: {'negative': 0.0015548973, 'neutral': 0.016998094, 'positive': 0.981447},
     393: {'negative': 0.0015555794,
      'neutral': 0.07374388,
      'positive': 0.92470056},
     394: {'negative': 0.0035041797, 'neutral': 0.21828802, 'positive': 0.7782078},
     395: {'negative': 0.005017797, 'neutral': 0.5956466, 'positive': 0.39933556},
     396: {'negative': 0.0066129286, 'neutral': 0.23016112, 'positive': 0.763226},
     397: {'negative': 0.009927992, 'neutral': 0.42798594, 'positive': 0.56208605},
     398: {'negative': 0.003811647, 'neutral': 0.12851916, 'positive': 0.86766917},
     399: {'negative': 0.004169381, 'neutral': 0.21303159, 'positive': 0.782799},
     400: {'negative': 0.0011210016,
      'neutral': 0.023390967,
      'positive': 0.97548807},
     401: {'negative': 0.0043874816, 'neutral': 0.280505, 'positive': 0.71510756},
     402: {'negative': 0.005357187, 'neutral': 0.40910783, 'positive': 0.58553505},
     403: {'negative': 0.0018143464,
      'neutral': 0.09233156,
      'positive': 0.90585417},
     404: {'negative': 0.0046397694, 'neutral': 0.1449965, 'positive': 0.85036373},
     405: {'negative': 0.0011862254, 'neutral': 0.05683042, 'positive': 0.9419834},
     406: {'negative': 0.0025873897, 'neutral': 0.30007407, 'positive': 0.6973386},
     407: {'negative': 0.015041488, 'neutral': 0.27847758, 'positive': 0.7064809},
     408: {'negative': 0.0049271076, 'neutral': 0.16406767, 'positive': 0.8310052},
     409: {'negative': 0.0020594571, 'neutral': 0.13367853, 'positive': 0.864262},
     410: {'negative': 0.001391416,
      'neutral': 0.048258673,
      'positive': 0.95034987},
     411: {'negative': 0.0012066521, 'neutral': 0.0376159, 'positive': 0.96117747},
     412: {'negative': 0.002216506,
      'neutral': 0.083361775,
      'positive': 0.91442174},
     413: {'negative': 0.001824847, 'neutral': 0.13421515, 'positive': 0.86396},
     414: {'negative': 0.004987544, 'neutral': 0.49092337, 'positive': 0.5040891},
     415: {'negative': 0.002696634, 'neutral': 0.19737391, 'positive': 0.79992944},
     416: {'negative': 0.0009968516,
      'neutral': 0.038967412,
      'positive': 0.96003574},
     417: {'negative': 0.0012795964,
      'neutral': 0.11396129,
      'positive': 0.88475907},
     418: {'negative': 0.0058776205, 'neutral': 0.17508414, 'positive': 0.8190383},
     419: {'negative': 0.0025958242,
      'neutral': 0.17641523,
      'positive': 0.82098895},
     420: {'negative': 0.0034790335,
      'neutral': 0.104109734,
      'positive': 0.8924113},
     421: {'negative': 0.0084656915, 'neutral': 0.20143071, 'positive': 0.7901036},
     422: {'negative': 0.0030048816, 'neutral': 0.4733383, 'positive': 0.5236568},
     423: {'negative': 0.0036702163,
      'neutral': 0.23806214,
      'positive': 0.75826764},
     424: {'negative': 0.09678982, 'neutral': 0.5230781, 'positive': 0.38013217},
     425: {'negative': 0.0015710591, 'neutral': 0.07686252, 'positive': 0.9215664},
     426: {'negative': 0.0026461782,
      'neutral': 0.081545025,
      'positive': 0.9158088},
     427: {'negative': 0.0017371409, 'neutral': 0.05696298, 'positive': 0.9412998},
     428: {'negative': 0.0016547128, 'neutral': 0.10712444, 'positive': 0.8912208},
     429: {'negative': 0.0024851866,
      'neutral': 0.059007615,
      'positive': 0.9385072},
     430: {'negative': 0.0028813474,
      'neutral': 0.15726806,
      'positive': 0.83985054},
     431: {'negative': 0.0010568601,
      'neutral': 0.021647511,
      'positive': 0.9772956},
     432: {'negative': 0.0042996383,
      'neutral': 0.37815323,
      'positive': 0.61754715},
     433: {'negative': 0.0031471765, 'neutral': 0.08201102, 'positive': 0.9148418},
     434: {'negative': 0.0047815866,
      'neutral': 0.35895547,
      'positive': 0.63626295},
     435: {'negative': 0.0018545169,
      'neutral': 0.048635915,
      'positive': 0.9495095},
     436: {'negative': 0.00789708, 'neutral': 0.717762, 'positive': 0.27434084},
     437: {'negative': 0.001558407, 'neutral': 0.04185217, 'positive': 0.9565894},
     438: {'negative': 0.006262747, 'neutral': 0.22734953, 'positive': 0.7663877},
     439: {'negative': 0.008382979, 'neutral': 0.6330705, 'positive': 0.3585465},
     440: {'negative': 0.0062794825, 'neutral': 0.31155592, 'positive': 0.6821646},
     441: {'negative': 0.0022196525,
      'neutral': 0.11185996,
      'positive': 0.88592035},
     442: {'negative': 0.0046732775, 'neutral': 0.12527563, 'positive': 0.8700511},
     443: {'negative': 0.0022289036, 'neutral': 0.25648683, 'positive': 0.7412843},
     444: {'negative': 0.0061261663,
      'neutral': 0.74606687,
      'positive': 0.24780701},
     445: {'negative': 0.0014872907,
      'neutral': 0.105518416,
      'positive': 0.8929943},
     446: {'negative': 0.0035550806, 'neutral': 0.24355514, 'positive': 0.7528898},
     447: {'negative': 0.0049335794, 'neutral': 0.45379922, 'positive': 0.5412672},
     448: {'negative': 0.0013625112,
      'neutral': 0.08694405,
      'positive': 0.91169345},
     449: {'negative': 0.001731194, 'neutral': 0.03905671, 'positive': 0.95921206},
     450: {'negative': 0.0034502929, 'neutral': 0.2562803, 'positive': 0.7402694},
     451: {'negative': 0.0028188315,
      'neutral': 0.082654275,
      'positive': 0.91452694},
     452: {'negative': 0.009511895, 'neutral': 0.26096877, 'positive': 0.72951937},
     453: {'negative': 0.0010845285,
      'neutral': 0.035796966,
      'positive': 0.9631185},
     454: {'negative': 0.004516475, 'neutral': 0.38831887, 'positive': 0.6071646},
     455: {'negative': 0.0010332753,
      'neutral': 0.020004177,
      'positive': 0.9789626},
     456: {'negative': 0.001906644, 'neutral': 0.04142473, 'positive': 0.9566686},
     457: {'negative': 0.0028965075, 'neutral': 0.028739499, 'positive': 0.968364},
     458: {'negative': 0.001720425, 'neutral': 0.0787144, 'positive': 0.91956514},
     459: {'negative': 0.0011978162,
      'neutral': 0.03915303,
      'positive': 0.95964915},
     460: {'negative': 0.0018141844,
      'neutral': 0.052481323,
      'positive': 0.9457045},
     461: {'negative': 0.0019314928, 'neutral': 0.06591533, 'positive': 0.9321532},
     462: {'negative': 0.002179067, 'neutral': 0.11408832, 'positive': 0.8837326},
     463: {'negative': 0.0015964255,
      'neutral': 0.026767485,
      'positive': 0.9716361},
     464: {'negative': 0.0041924245,
      'neutral': 0.18988724,
      'positive': 0.80592036},
     465: {'negative': 0.0016311737,
      'neutral': 0.053753685,
      'positive': 0.9446152},
     466: {'negative': 0.0031208263, 'neutral': 0.0783199, 'positive': 0.91855925},
     467: {'negative': 0.0056310883, 'neutral': 0.21730873, 'positive': 0.7770602},
     468: {'negative': 0.0024045464,
      'neutral': 0.092373855,
      'positive': 0.9052216},
     469: {'negative': 0.0034826742,
      'neutral': 0.16958606,
      'positive': 0.82693124},
     470: {'negative': 0.0034478384,
      'neutral': 0.055728897,
      'positive': 0.94082326},
     471: {'negative': 0.0020997038, 'neutral': 0.06391273, 'positive': 0.9339875},
     472: {'negative': 0.0016759019,
      'neutral': 0.042151153,
      'positive': 0.9561729},
     473: {'negative': 0.008807272, 'neutral': 0.15853651, 'positive': 0.83265626},
     474: {'negative': 0.0017210548,
      'neutral': 0.037461326,
      'positive': 0.9608176},
     475: {'negative': 0.006389031, 'neutral': 0.107042745, 'positive': 0.8865682},
     476: {'negative': 0.006301088, 'neutral': 0.15741809, 'positive': 0.8362809},
     477: {'negative': 0.0029417975, 'neutral': 0.09015876, 'positive': 0.9068995},
     478: {'negative': 0.016737795, 'neutral': 0.18773311, 'positive': 0.7955291},
     479: {'negative': 0.0030015865, 'neutral': 0.29985493, 'positive': 0.6971435},
     480: {'negative': 0.008431679, 'neutral': 0.6083774, 'positive': 0.38319093},
     481: {'negative': 0.0022144415, 'neutral': 0.11094296, 'positive': 0.8868426},
     482: {'negative': 0.015803317, 'neutral': 0.17271115, 'positive': 0.81148547},
     483: {'negative': 0.0017885679, 'neutral': 0.14292507, 'positive': 0.8552863},
     484: {'negative': 0.0012183307,
      'neutral': 0.055165984,
      'positive': 0.94361573},
     485: {'negative': 0.006093549, 'neutral': 0.16327295, 'positive': 0.8306335},
     486: {'negative': 0.0063241334, 'neutral': 0.20865841, 'positive': 0.7850175},
     487: {'negative': 0.0026781927, 'neutral': 0.2519374, 'positive': 0.74538445},
     488: {'negative': 0.0023121207, 'neutral': 0.13433784, 'positive': 0.8633501},
     489: {'negative': 0.0017313893,
      'neutral': 0.076472476,
      'positive': 0.9217962},
     490: {'negative': 0.0023846764, 'neutral': 0.08974115, 'positive': 0.9078741},
     491: {'negative': 0.0057533323, 'neutral': 0.21122837, 'positive': 0.7830183},
     492: {'negative': 0.0019216695,
      'neutral': 0.15942773,
      'positive': 0.83865064},
     493: {'negative': 0.011044454, 'neutral': 0.3625135, 'positive': 0.6264421},
     494: {'negative': 0.004812324, 'neutral': 0.16414443, 'positive': 0.83104324},
     495: {'negative': 0.0011342847,
      'neutral': 0.028352011,
      'positive': 0.9705137},
     496: {'negative': 0.002117641, 'neutral': 0.051176164, 'positive': 0.9467062},
     497: {'negative': 0.004959728, 'neutral': 0.38186505, 'positive': 0.6131752},
     498: {'negative': 0.003724978, 'neutral': 0.14641494, 'positive': 0.84986013},
     499: {'negative': 0.0037449745, 'neutral': 0.24760497, 'positive': 0.74865}}




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
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
