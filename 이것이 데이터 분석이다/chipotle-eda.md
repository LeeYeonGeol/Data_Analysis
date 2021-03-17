# Step 1 탐색: 데이터의 기초 정보 살펴보기

## chipotle 데이터셋의 기초 정보 출력하기


```python
import pandas as pd
file_path = './python-data-analysis-master/data/chipotle.tsv'
# read_csv() 함수로 데이터를 데이터 프레임 형태로 불러온다.
chipo = pd.read_csv(file_path, sep = '\t')

print(chipo.shape)
print("------------------------")
print(chipo.info())
```

    (4622, 5)
    ------------------------
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 4622 entries, 0 to 4621
    Data columns (total 5 columns):
     #   Column              Non-Null Count  Dtype 
    ---  ------              --------------  ----- 
     0   order_id            4622 non-null   int64 
     1   quantity            4622 non-null   int64 
     2   item_name           4622 non-null   object
     3   choice_description  3376 non-null   object
     4   item_price          4622 non-null   object
    dtypes: int64(2), object(3)
    memory usage: 180.7+ KB
    None
    


```python
print(chipo.shape)
```

    (4622, 5)
    

## chipotle 데이터셋의 행과 열, 데이터 확인하기


```python
# chipo라는 데이터 프레임에서 순서대로 10개의 데이터를 보여준다.
chipo.head(10)
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
      <th>order_id</th>
      <th>quantity</th>
      <th>item_name</th>
      <th>choice_description</th>
      <th>item_price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>Chips and Fresh Tomato Salsa</td>
      <td>NaN</td>
      <td>$2.39</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>Izze</td>
      <td>[Clementine]</td>
      <td>$3.39</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1</td>
      <td>Nantucket Nectar</td>
      <td>[Apple]</td>
      <td>$3.39</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>Chips and Tomatillo-Green Chili Salsa</td>
      <td>NaN</td>
      <td>$2.39</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>2</td>
      <td>Chicken Bowl</td>
      <td>[Tomatillo-Red Chili Salsa (Hot), [Black Beans...</td>
      <td>$16.98</td>
    </tr>
    <tr>
      <th>5</th>
      <td>3</td>
      <td>1</td>
      <td>Chicken Bowl</td>
      <td>[Fresh Tomato Salsa (Mild), [Rice, Cheese, Sou...</td>
      <td>$10.98</td>
    </tr>
    <tr>
      <th>6</th>
      <td>3</td>
      <td>1</td>
      <td>Side of Chips</td>
      <td>NaN</td>
      <td>$1.69</td>
    </tr>
    <tr>
      <th>7</th>
      <td>4</td>
      <td>1</td>
      <td>Steak Burrito</td>
      <td>[Tomatillo Red Chili Salsa, [Fajita Vegetables...</td>
      <td>$11.75</td>
    </tr>
    <tr>
      <th>8</th>
      <td>4</td>
      <td>1</td>
      <td>Steak Soft Tacos</td>
      <td>[Tomatillo Green Chili Salsa, [Pinto Beans, Ch...</td>
      <td>$9.25</td>
    </tr>
    <tr>
      <th>9</th>
      <td>5</td>
      <td>1</td>
      <td>Steak Burrito</td>
      <td>[Fresh Tomato Salsa, [Rice, Black Beans, Pinto...</td>
      <td>$9.25</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(chipo.columns)
print("-----------------------------")
print(chipo.index)
```

    Index(['order_id', 'quantity', 'item_name', 'choice_description',
           'item_price'],
          dtype='object')
    -----------------------------
    RangeIndex(start=0, stop=4622, step=1)
    

# quantity와 item_price의 수치적 특징
우선 quantity와 item_price의 수치적 특징을 살펴본다. 이 두 피처는 **연속형 피처**이다. 연속형 피처는 키와 몸무게처럼 어떠한 값도 가질 수 있는 연속적인 숫자 형태를 의미한다.

## describe() 함수로 기초 통계량 출력하기


```python
# order_id는 숫자의 의미를 가지지 않기 때문에 str으로 변환합니다.
chipo['order_id'] = chipo['order_id'].astype(str)
print(chipo.describe()) # chipo 데이터 프레임에서 수치형 피처들의 기초 통계량을 확인합니다.
```

              quantity
    count  4622.000000
    mean      1.075725
    std       0.410186
    min       1.000000
    25%       1.000000
    50%       1.000000
    75%       1.000000
    max      15.000000
    

## unique() 함수로 범주형 피처의 개수 출력하기


```python
print(len(chipo['order_id'].unique())) # order_id의 개수를 출력합니다.
print(len(chipo['item_name'].unique())) # item_name의 개수를 출력합니다.
```

    1834
    50
    

# Step 2 인사이트의 발견: 탐색과 시각화하기

## 가장 많이 주문한 아이템 Top 10 출력하기


```python
# 가장 많이 주문한 아이템 Top 10을 출력한다.
item_count = chipo['item_name'].value_counts()[:10]
for idx, (val, cnt) in enumerate(item_count.iteritems(), 1):
    print("Top", idx, ":", val, cnt)
```

    Top 1 : Chicken Bowl 726
    Top 2 : Chicken Burrito 553
    Top 3 : Chips and Guacamole 479
    Top 4 : Steak Burrito 368
    Top 5 : Canned Soft Drink 301
    Top 6 : Chips 211
    Top 7 : Steak Bowl 211
    Top 8 : Bottled Water 162
    Top 9 : Chicken Soft Tacos 115
    Top 10 : Chicken Salad Bowl 110
    

# 아이템별 주문 개수와 총량

## 아이템별 주문 개수와 총량 구하기


```python
# 아이템별 주문 개수를 출력합니다.
order_count = chipo.groupby('item_name')['order_id'].count()
order_count[:10] # 아이템별 주문 개수를 출력합니다.
```




    item_name
    6 Pack Soft Drink         54
    Barbacoa Bowl             66
    Barbacoa Burrito          91
    Barbacoa Crispy Tacos     11
    Barbacoa Salad Bowl       10
    Barbacoa Soft Tacos       25
    Bottled Water            162
    Bowl                       2
    Burrito                    6
    Canned Soda              104
    Name: order_id, dtype: int64




```python
# 아이템별 주문 총량을 계산합니다.
item_quantity = chipo.groupby('item_name')['quantity'].sum()
item_quantity[:10] # 아이템별 주문 총량을 출력합니다.
```




    item_name
    6 Pack Soft Drink         55
    Barbacoa Bowl             66
    Barbacoa Burrito          91
    Barbacoa Crispy Tacos     12
    Barbacoa Salad Bowl       10
    Barbacoa Soft Tacos       25
    Bottled Water            211
    Bowl                       4
    Burrito                    6
    Canned Soda              126
    Name: quantity, dtype: int64



# 시각화

## 시각화로 분석 결과 살펴보기


```python
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt

item_name_list = item_quantity.index.tolist()
x_pos = np.arange(len(item_name_list))
order_cnt = item_quantity.values.tolist()

plt.bar(x_pos, order_cnt, align='center')
plt.ylabel('ordered_item_count')
plt.title('Distribution of all orderd item')

plt.show()
```


![png](output_21_0.png)


# Step 3 데이터 전처리: 전처리 함수 사용하기

## item_price 피처 살펴보기


```python
print(chipo.info())
print('-----------')
chipo['item_price'].head()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 4622 entries, 0 to 4621
    Data columns (total 5 columns):
     #   Column              Non-Null Count  Dtype 
    ---  ------              --------------  ----- 
     0   order_id            4622 non-null   object
     1   quantity            4622 non-null   int64 
     2   item_name           4622 non-null   object
     3   choice_description  3376 non-null   object
     4   item_price          4622 non-null   object
    dtypes: int64(1), object(4)
    memory usage: 180.7+ KB
    None
    -----------
    




    0     $2.39 
    1     $3.39 
    2     $3.39 
    3     $2.39 
    4    $16.98 
    Name: item_price, dtype: object



## apply()와 lambda를 이용해 데이터 전처리하기


```python
# column 단위 데이터에 apply() 함수로 전처리를 적용한다.
chipo['item_price'] = chipo['item_price'].apply(lambda x: float(x[1:]))
chipo.describe()
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
      <th>quantity</th>
      <th>item_price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>4622.000000</td>
      <td>4622.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1.075725</td>
      <td>7.464336</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.410186</td>
      <td>4.245557</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>1.090000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.000000</td>
      <td>3.390000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.000000</td>
      <td>8.750000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.000000</td>
      <td>9.250000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>15.000000</td>
      <td>44.250000</td>
    </tr>
  </tbody>
</table>
</div>



# Step 4 탐색적 분석: 스무고개로 개념적 탐색 분석하기

## 주문당 평균 계산금액 출력하기


```python
# 주문당 평균 계산금액을 출력한다.
chipo.groupby('order_id')['item_price'].sum().mean()
```




    18.811428571428717



## 한 주문에 10달러 이상 지불한 주문 번호(id) 출력하기


```python
# 한 주문에 10달러 이상 지불한 id를 출력한다.
chipo_orderid_group = chipo.groupby('order_id').sum()
results = chipo_orderid_group[chipo_orderid_group.item_price >= 10]
print(results[:10])
print(results.index.values)
```

              quantity  item_price
    order_id                      
    1                4       11.56
    10               2       13.20
    100              2       10.08
    1000             2       20.50
    1001             2       10.08
    1002             2       10.68
    1003             2       13.00
    1004             2       21.96
    1005             3       12.15
    1006             8       71.40
    ['1' '10' '100' ... '997' '998' '999']
    

## 각 아이템의 가격 구하기


```python
# 각 아이템의 가격을 계산한다.
chipo_one_item = chipo[chipo.quantity == 1]
price_per_item = chipo_one_item.groupby('item_name').min()
price_per_item.sort_values(by = "item_price", ascending = False)[:10]
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
      <th>order_id</th>
      <th>quantity</th>
      <th>choice_description</th>
      <th>item_price</th>
    </tr>
    <tr>
      <th>item_name</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Steak Salad Bowl</th>
      <td>1032</td>
      <td>1</td>
      <td>[Fresh Tomato Salsa, Lettuce]</td>
      <td>9.39</td>
    </tr>
    <tr>
      <th>Barbacoa Salad Bowl</th>
      <td>1283</td>
      <td>1</td>
      <td>[Fresh Tomato Salsa, Guacamole]</td>
      <td>9.39</td>
    </tr>
    <tr>
      <th>Carnitas Salad Bowl</th>
      <td>1035</td>
      <td>1</td>
      <td>[Fresh Tomato Salsa, [Rice, Black Beans, Chees...</td>
      <td>9.39</td>
    </tr>
    <tr>
      <th>Carnitas Soft Tacos</th>
      <td>1011</td>
      <td>1</td>
      <td>[Fresh Tomato Salsa (Mild), [Black Beans, Rice...</td>
      <td>8.99</td>
    </tr>
    <tr>
      <th>Carnitas Crispy Tacos</th>
      <td>1774</td>
      <td>1</td>
      <td>[Fresh Tomato Salsa, [Fajita Vegetables, Rice,...</td>
      <td>8.99</td>
    </tr>
    <tr>
      <th>Steak Soft Tacos</th>
      <td>1054</td>
      <td>1</td>
      <td>[Fresh Tomato Salsa (Mild), [Cheese, Sour Cream]]</td>
      <td>8.99</td>
    </tr>
    <tr>
      <th>Carnitas Salad</th>
      <td>1500</td>
      <td>1</td>
      <td>[[Fresh Tomato Salsa (Mild), Roasted Chili Cor...</td>
      <td>8.99</td>
    </tr>
    <tr>
      <th>Carnitas Bowl</th>
      <td>1007</td>
      <td>1</td>
      <td>[Fresh Tomato (Mild), [Guacamole, Lettuce, Ric...</td>
      <td>8.99</td>
    </tr>
    <tr>
      <th>Barbacoa Soft Tacos</th>
      <td>1103</td>
      <td>1</td>
      <td>[Fresh Tomato Salsa, [Black Beans, Cheese, Let...</td>
      <td>8.99</td>
    </tr>
    <tr>
      <th>Barbacoa Crispy Tacos</th>
      <td>110</td>
      <td>1</td>
      <td>[Fresh Tomato Salsa, Guacamole]</td>
      <td>8.99</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 아이템 가격 분포 그래프를 출력한다.
item_name_list = price_per_item.index.tolist()
x_pos = np.arange(len(item_name_list))
item_price = price_per_item['item_price'].tolist()

plt.bar(x_pos, item_price, align='center')
plt.ylabel('item price($)')
plt.title('Distribution of item price')
plt.show()

# 아이템 가격 히스토그램을 출력한다.
plt.hist(item_price)
plt.ylabel('counts')
plt.title('Histogram of item price')
plt.show()
```


![png](output_34_0.png)



![png](output_34_1.png)


## 가장 비싼 주문에서 아이템이 총 몇 개 팔렸는지 구하기


```python
# 가장 비싼 주문에서 아이템이 총 몇 개 팔렸는지를 계산한다.
chipo.groupby('order_id').sum().sort_values(by='item_price', ascending=False)[:5]
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
      <th>quantity</th>
      <th>item_price</th>
    </tr>
    <tr>
      <th>order_id</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>926</th>
      <td>23</td>
      <td>205.25</td>
    </tr>
    <tr>
      <th>1443</th>
      <td>35</td>
      <td>160.74</td>
    </tr>
    <tr>
      <th>1483</th>
      <td>14</td>
      <td>139.00</td>
    </tr>
    <tr>
      <th>691</th>
      <td>11</td>
      <td>118.25</td>
    </tr>
    <tr>
      <th>1786</th>
      <td>20</td>
      <td>114.30</td>
    </tr>
  </tbody>
</table>
</div>



## 'Veggie Salad Bowl'이 몇 번 주문되었는지 구하기


```python
# 'Veggi Sald Bowl'이 몇 번 주문되었는지를 계산한다.
chipo_salad = chipo[chipo['item_name'] == "Veggie Salad Bowl"]
# 한 주문 내에서 중복 집계된 item_name을 제거한다.
chipo_salad = chipo_salad.drop_duplicates(['item_name', 'order_id'])

print(len(chipo_salad))
chipo_salad.head(5)
```

    18
    




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
      <th>order_id</th>
      <th>quantity</th>
      <th>item_name</th>
      <th>choice_description</th>
      <th>item_price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>186</th>
      <td>83</td>
      <td>1</td>
      <td>Veggie Salad Bowl</td>
      <td>[Fresh Tomato Salsa, [Fajita Vegetables, Rice,...</td>
      <td>11.25</td>
    </tr>
    <tr>
      <th>295</th>
      <td>128</td>
      <td>1</td>
      <td>Veggie Salad Bowl</td>
      <td>[Fresh Tomato Salsa, [Fajita Vegetables, Lettu...</td>
      <td>11.25</td>
    </tr>
    <tr>
      <th>455</th>
      <td>195</td>
      <td>1</td>
      <td>Veggie Salad Bowl</td>
      <td>[Fresh Tomato Salsa, [Fajita Vegetables, Rice,...</td>
      <td>11.25</td>
    </tr>
    <tr>
      <th>496</th>
      <td>207</td>
      <td>1</td>
      <td>Veggie Salad Bowl</td>
      <td>[Fresh Tomato Salsa, [Rice, Lettuce, Guacamole...</td>
      <td>11.25</td>
    </tr>
    <tr>
      <th>960</th>
      <td>394</td>
      <td>1</td>
      <td>Veggie Salad Bowl</td>
      <td>[Fresh Tomato Salsa, [Fajita Vegetables, Lettu...</td>
      <td>8.75</td>
    </tr>
  </tbody>
</table>
</div>



## 'Chicken Bowl'을 2개 이상 주문한 주문 횟수 구하기


```python
# 'Chicken Bowl'을 2개 이상 주문한 주문 횟수를 구한다.
chipo_chicken = chipo[chipo['item_name'] == "Chicken Bowl"]
chipo_chicken_result = chipo_chicken[chipo_chicken['quantity'] >= 2]
print(chipo_chicken_result.shape[0])
```

    33
    
