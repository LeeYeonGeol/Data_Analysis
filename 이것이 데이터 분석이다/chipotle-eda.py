#!/usr/bin/env python
# coding: utf-8

# # Step 1 탐색: 데이터의 기초 정보 살펴보기

# ## chipotle 데이터셋의 기초 정보 출력하기

# In[1]:


import pandas as pd
file_path = './python-data-analysis-master/data/chipotle.tsv'
# read_csv() 함수로 데이터를 데이터 프레임 형태로 불러온다.
chipo = pd.read_csv(file_path, sep = '\t')

print(chipo.shape)
print("------------------------")
print(chipo.info())


# In[2]:


print(chipo.shape)


# ## chipotle 데이터셋의 행과 열, 데이터 확인하기

# In[3]:


# chipo라는 데이터 프레임에서 순서대로 10개의 데이터를 보여준다.
chipo.head(10)


# In[4]:


print(chipo.columns)
print("-----------------------------")
print(chipo.index)


# # quantity와 item_price의 수치적 특징
# 우선 quantity와 item_price의 수치적 특징을 살펴본다. 이 두 피처는 **연속형 피처**이다. 연속형 피처는 키와 몸무게처럼 어떠한 값도 가질 수 있는 연속적인 숫자 형태를 의미한다.

# ## describe() 함수로 기초 통계량 출력하기

# In[5]:


# order_id는 숫자의 의미를 가지지 않기 때문에 str으로 변환합니다.
chipo['order_id'] = chipo['order_id'].astype(str)
print(chipo.describe()) # chipo 데이터 프레임에서 수치형 피처들의 기초 통계량을 확인합니다.


# ## unique() 함수로 범주형 피처의 개수 출력하기

# In[6]:


print(len(chipo['order_id'].unique())) # order_id의 개수를 출력합니다.
print(len(chipo['item_name'].unique())) # item_name의 개수를 출력합니다.


# # Step 2 인사이트의 발견: 탐색과 시각화하기

# ## 가장 많이 주문한 아이템 Top 10 출력하기

# In[7]:


# 가장 많이 주문한 아이템 Top 10을 출력한다.
item_count = chipo['item_name'].value_counts()[:10]
for idx, (val, cnt) in enumerate(item_count.iteritems(), 1):
    print("Top", idx, ":", val, cnt)


# # 아이템별 주문 개수와 총량

# ## 아이템별 주문 개수와 총량 구하기

# In[8]:


# 아이템별 주문 개수를 출력합니다.
order_count = chipo.groupby('item_name')['order_id'].count()
order_count[:10] # 아이템별 주문 개수를 출력합니다.


# In[9]:


# 아이템별 주문 총량을 계산합니다.
item_quantity = chipo.groupby('item_name')['quantity'].sum()
item_quantity[:10] # 아이템별 주문 총량을 출력합니다.


# # 시각화

# ## 시각화로 분석 결과 살펴보기

# In[9]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt

item_name_list = item_quantity.index.tolist()
x_pos = np.arange(len(item_name_list))
order_cnt = item_quantity.values.tolist()

plt.bar(x_pos, order_cnt, align='center')
plt.ylabel('ordered_item_count')
plt.title('Distribution of all orderd item')

plt.show()


# # Step 3 데이터 전처리: 전처리 함수 사용하기

# ## item_price 피처 살펴보기

# In[16]:


print(chipo.info())
print('-----------')
chipo['item_price'].head()


# ## apply()와 lambda를 이용해 데이터 전처리하기

# In[17]:


# column 단위 데이터에 apply() 함수로 전처리를 적용한다.
chipo['item_price'] = chipo['item_price'].apply(lambda x: float(x[1:]))
chipo.describe()


# # Step 4 탐색적 분석: 스무고개로 개념적 탐색 분석하기

# ## 주문당 평균 계산금액 출력하기

# In[20]:


# 주문당 평균 계산금액을 출력한다.
chipo.groupby('order_id')['item_price'].sum().mean()


# ## 한 주문에 10달러 이상 지불한 주문 번호(id) 출력하기

# In[17]:


# 한 주문에 10달러 이상 지불한 id를 출력한다.
chipo_orderid_group = chipo.groupby('order_id').sum()
results = chipo_orderid_group[chipo_orderid_group.item_price >= 10]
print(results[:10])
print(results.index.values)


# ## 각 아이템의 가격 구하기

# In[26]:


# 각 아이템의 가격을 계산한다.
chipo_one_item = chipo[chipo.quantity == 1]
price_per_item = chipo_one_item.groupby('item_name').min()
price_per_item.sort_values(by = "item_price", ascending = False)[:10]


# In[19]:


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


# ## 가장 비싼 주문에서 아이템이 총 몇 개 팔렸는지 구하기

# In[20]:


# 가장 비싼 주문에서 아이템이 총 몇 개 팔렸는지를 계산한다.
chipo.groupby('order_id').sum().sort_values(by='item_price', ascending=False)[:5]


# ## 'Veggie Salad Bowl'이 몇 번 주문되었는지 구하기

# In[30]:


# 'Veggi Sald Bowl'이 몇 번 주문되었는지를 계산한다.
chipo_salad = chipo[chipo['item_name'] == "Veggie Salad Bowl"]
# 한 주문 내에서 중복 집계된 item_name을 제거한다.
chipo_salad = chipo_salad.drop_duplicates(['item_name', 'order_id'])

print(len(chipo_salad))
chipo_salad.head(5)


# ## 'Chicken Bowl'을 2개 이상 주문한 주문 횟수 구하기

# In[32]:


# 'Chicken Bowl'을 2개 이상 주문한 주문 횟수를 구한다.
chipo_chicken = chipo[chipo['item_name'] == "Chicken Bowl"]
chipo_chicken_result = chipo_chicken[chipo_chicken['quantity'] >= 2]
print(chipo_chicken_result.shape[0])

