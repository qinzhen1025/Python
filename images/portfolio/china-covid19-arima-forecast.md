---

title: "中国 COVID-19 疫情指标统计分析与 ARIMA 滚动预测（2020–2022）"

collection: portfolio

type: "Machine Learning"

permalink: /portfolio/china-covid19-arima-forecast

date: 2026-01-18

excerpt: "基于公开疫情数据完成中国疫情指标洞察，并用 ARIMA 实现 7 天滚动预测与误差评估，为趋势研判提供可量化基线模型。"

header:

&nbsp; teaser: /images/portfolio/china-covid19-arima-forecast/teaser.png

tags:

\- 时间序列

\- ARIMA

\- 疫情分析

\- 统计分析

\- 滚动预测

\- 模型评估

tech\_stack:

\- name: Python

\- name: Pandas

\- name: Statsmodels

\- name: Scikit-learn

\- name: Matplotlib

\- name: Seaborn

---

背景（Background）



本项目基于公开 COVID-19 数据（以国家维度统计），聚焦中国在 2020-01-23 至 2022-12-31 时间段内的疫情关键指标变化，包括新增确诊、死亡、政策严格度指数（stringency index）与再生数（reproduction rate）。



目标分两部分：



统计分析与可视化：用描述性统计、相关性与滞后相关分析，刻画指标间关系及其随时间的变化。



时间序列预测基线：以 new\_cases\_smoothed 为目标序列，构建 ARIMA 7 天滚动预测流程，并用 RMSE/MAE/MAPE 等指标评估模型效果与稳定性。



核心实现（Implementation）

1）数据筛选与预处理（中国 + 时间范围 + 关键字段）



只保留项目核心逻辑：国家筛选、字段筛选、日期处理、时间窗口裁剪。



\# 筛选中国 \& 保留关键字段

china\_data = df\[df\["country"] == "China"]\[\[

&nbsp;   "date",

&nbsp;   "total\_cases", "new\_cases", "new\_cases\_smoothed",

&nbsp;   "total\_deaths", "new\_deaths", "new\_deaths\_smoothed",

&nbsp;   "stringency\_index", "reproduction\_rate"

]].copy()



china\_data\["date"] = pd.to\_datetime(china\_data\["date"])

china\_data = china\_data\[

&nbsp;   (china\_data\["date"] >= "2020-01-23") \&

&nbsp;   (china\_data\["date"] <= "2022-12-31")

]



2）描述性统计：对关键指标形成“可复用结论表”



用均值/分位数/极值及其发生日期，帮助快速定位关键阶段。



analysis\_cols = \["new\_cases", "new\_deaths", "stringency\_index", "reproduction\_rate"]



stats = {}

for col in analysis\_cols:

&nbsp;   s = china\_data\[col]

&nbsp;   stats\[col] = {

&nbsp;       "mean": s.mean(),

&nbsp;       "median": s.median(),

&nbsp;       "p25": s.quantile(0.25),

&nbsp;       "p75": s.quantile(0.75),

&nbsp;       "min": s.min(),

&nbsp;       "max": s.max(),

&nbsp;       "max\_date": china\_data.loc\[s.idxmax(), "date"] if s.notna().any() else None

&nbsp;   }



stats\_df = pd.DataFrame(stats).T

stats\_df



3）统计可视化与滞后效应分析（Correlation + Lag）



Notebook 中这里输出了多张图（见“图片资源清单”），建议在作品集中按“总览 → 相关性 → 滚动相关 → 滞后相关”组织，体现分析深度与逻辑闭环。



你在 Notebook 里主要做了：



多指标时间序列总览（2×2）



相关性热图与对比分析（整体/分期）



滚动相关性（展示结构变化）



滞后相关（不同滞后天数下的相关系数曲线 + 热图）



在作品集中建议直接展示图片 + 简洁结论（见下文 Results）。



4）ARIMA：7 天滚动预测（Rolling Forecast）



核心思想：每次用当前训练集拟合 ARIMA，预测未来 7 天；然后把真实值并入训练集，继续下一轮，形成可解释的“逐步推演式预测”。



from statsmodels.tsa.stattools import adfuller

from statsmodels.tsa.arima.model import ARIMA

from sklearn.metrics import mean\_squared\_error, mean\_absolute\_error, mean\_absolute\_percentage\_error

import numpy as np



\# 目标序列：平滑后的新增确诊

ts = china\_data\[\["date", "new\_cases\_smoothed"]].dropna().copy()

ts = ts.set\_index("date")\["new\_cases\_smoothed"]



def is\_stationary(series):

&nbsp;   p = adfuller(series.dropna(), autolag="AIC")\[1]

&nbsp;   return p < 0.05



\# 差分直到平稳（Notebook 中采用类似思路）

d = 0

diff = ts.copy()

while (not is\_stationary(diff)) and d < 3:

&nbsp;   diff = diff.diff().dropna()

&nbsp;   d += 1



\# 7天滚动预测（示意：p,q 可在 Notebook 中通过 ACF/PACF 或经验设定）

p, q = 2, 2

horizon = 7



y\_true\_all, y\_pred\_all = \[], \[]

history = ts.copy()



\# 滚动：每轮预测 horizon 天，然后把真实值加入 history

start\_idx = 0

while start\_idx + horizon < len(ts):

&nbsp;   train = history.iloc\[: start\_idx + (len(ts) - len(history))] if len(history) != len(ts) else ts.iloc\[:start\_idx]

&nbsp;   if len(train) < 30:  # 保证足够长度（可按需要调整）

&nbsp;       start\_idx += horizon

&nbsp;       continue



&nbsp;   model = ARIMA(train, order=(p, d, q)).fit()

&nbsp;   forecast = model.forecast(steps=horizon)



&nbsp;   test = ts.iloc\[start\_idx : start\_idx + horizon]

&nbsp;   y\_true\_all.extend(test.values.tolist())

&nbsp;   y\_pred\_all.extend(forecast.values.tolist())



&nbsp;   start\_idx += horizon



\# 误差指标

rmse = mean\_squared\_error(y\_true\_all, y\_pred\_all, squared=False)

mae  = mean\_absolute\_error(y\_true\_all, y\_pred\_all)

mape = mean\_absolute\_percentage\_error(y\_true\_all, y\_pred\_all)



rmse, mae, mape





注：你原 Notebook 里包含更完整的打印信息、窗口误差统计与可视化，这里在作品集只保留“能讲清楚方法”的核心骨架。



结果与分析（Results \& Analysis）

1）疫情关键指标总览



建议在图下用 3~5 句话总结：新增、死亡、政策严格度与再生数在 2020–2022 的主要波段特征，以及峰值发生的阶段性含义。



2）相关性：指标之间的联动关系



说明整体相关性与分阶段差异：例如政策严格度与新增之间是否存在同步/反向关系，以及是否随阶段变化而变化。



3）滚动相关性：结构是否稳定



强调“相关关系随时间漂移”的证据，这也是后续预测建模可能失效/退化的原因之一（数据非平稳、机制变化）。



4）滞后效应：谁领先、谁滞后



用一句话点出你在 Notebook 里输出的“最优滞后天数”结论：例如某指标对新增/死亡存在 X 天游程的领先或滞后（正相关/负相关）。



5）ARIMA 7 天滚动预测效果



强调这是一种可解释的统计基线：优点是实现简单、可解释、适合做 baseline；缺点是在突发波动/结构变化阶段误差增大。



如果误差曲线出现“阶段性抬升”，建议用 1~2 句把原因归因到：政策调整、统计口径变化、疫情阶段切换导致序列机制变化。



个人贡献（What I did）



完成中国区间数据筛选、字段清洗与时间窗裁剪，建立可复用的分析数据集；



对新增、死亡、政策严格度、再生数进行描述性统计与多视角可视化；



构建相关性、滚动相关与滞后相关分析框架，输出可解释的“领先/滞后”洞察；



以 new\_cases\_smoothed 为目标，构建 ARIMA 7 天滚动预测流程，并用 RMSE/MAE/MAPE 做评估与稳定性分析。



可优化方向（Next Steps）



用 AIC/BIC 做 (p,d,q) 网格搜索或自动寻优，提高基线公平性；



尝试 SARIMA/Prophet 等季节性与结构变化更强的模型；



加入外生变量（stringency\_index、reproduction\_rate）构建 ARIMAX/SARIMAX；



对突发波动阶段单独建模（分段建模或变点检测）。

