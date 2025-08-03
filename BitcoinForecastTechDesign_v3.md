Bitcoin Medium‑Term Forecast &

Recommendation Platform — Technical Design v3

1. Objective & Scope

This system provides an automated, explainable forecasting engine for Bitcoin’s medium-term outlook. It

publishes a weekly (every Sunday 00:00 UTC) and monthly (1st of each month) report with two key outputs


:

- Directional outlook (Bullish / Neutral / Bearish) for the next 4–12 weeks, and

- Level forecasts – 20th, 50th (median), and 80th percentile price targets at 4-week and 12-week horizons. 

Primary KPIs (targets):

- 4-week Mean Absolute Percentage Error (MAPE) < 17%


- Directional accuracy (hit-rate) > 65%


- End-to-end latency from data cutoff to report publication < 10 minutes


Scope: The focus is on medium-term trends (multi-week). High-frequency or intraday signals are excluded

as per requirements


. The system emphasizes  explainability, using interpretable features (e.g. trend

indicators, on-chain metrics, macro indexes) so that each forecast can be rationalized in terms of known

market factors. 

Revision Overview:  This v3 design builds upon the solid foundation of v2, which was comprehensive in

covering data pipelines, feature engineering, modeling, and deployment. We retain all core components –

including the weekly cadence, forecast intervals (60% and 90% confidence bands), and MLflow experiment

tracking – and enhance the model by integrating proven analytical methods from top Bitcoin analysts.

These   include   long-horizon  logarithmic   growth   curves   (LGC)


,  global   liquidity   flows



,   key

macro indicators  (e.g. USD strength, gold/BTC trends


),  sentiment analysis  (on-chain and technical

sentiment signals


), and Elliott Wave pattern heuristics


. By blending these approaches in a unified

model, the system aims to improve predictive accuracy and robustness – an approach validated by analysts

who anticipated market turns using such multi-framework insights


.

2. High‑Level Architecture (Weekly Cadence)

The   platform   follows   a  modular   pipeline  architecture,   with   distinct   stages   orchestrated   on   a   weekly

schedule. Each stage is implemented as an isolated module or Docker container, enabling clear interfaces

and ease of maintenance. A high-level data flow is illustrated below:

flowchart TD

    A[Data Feeds<br/>(CoinGecko, FRED, etc.)] --> B[ETL Jobs<br/>(Ingestion 

Layer)]

    B --> C[TimescaleDB<br/>(Time-series DB)]


    C --> D[Feature Builder<br/>(compute indicators)]

    D --> E[Model Hub<br/>(ML & DL Models)]

    E --> F[Forecaster Service<br/>(Ensemble Logic)]

    F --> G[API & UI Layer<br/>(FastAPI & Streamlit)]

    %% Weekly scheduler trigger

    B -.->|Celery Beat (Sun 03:00 UTC)| B

Each   node   represents   a   service   module   (with   an   internal   schedule   or   triggered   task),   and   solid   arrows

denote   primary   data   flow.   The   dotted   arrow   indicates   the   scheduled   trigger   for   the   weekly   run.   Inter-

module communication occurs via shared storage (database, artifact store) and a message queue for task

orchestration.   All   containers   share   a   Redis   instance   for   job   queuing,   and   the   internal   Docker   network

facilitates API calls between services.

Component Responsibilities & Interfaces:

## Id

Module /

Service

Tech Stack

Purpose / Interface

## A

## B

## C

## D

Data Feeds

(ingestion)

Python  httpx , 
AsyncIO

ETL Jobs

(ingestion)

Python  pandas , 
psycopg2

Database

(Time-

series)

TimescaleDB

(PostgreSQL 15)

Fetch raw data from public APIs (REST endpoints or CSV

feeds). Runs as scheduled tasks that call external data

providers. Output: Raw JSON/CSV data (in-memory or on

disk) for the ETL layer


.

Transform & store data. Normalizes raw feed data into a

canonical schema and upserts into the time-series DB

(TimescaleDB). Includes data cleaning (e.g., aggregating

daily to weekly) and error retries
A. Output: updated  btc_weekly  table in the database.

. Inputs: raw data from


Store all weekly metrics. Central repository for historical

features and market data. Provides SQL interface for

feature builder and for API queries
hypertable  btc_weekly  indexed by week.


. The main table is a

Feature

Builder

Python  pandas , 
numpy , technical

Compute engineered features from DB data each week.
Implements functions that read from  btc_weekly  and

produce new feature series (pandas Series) aligned by week

analysis libs


. Output: a feature matrix (DataFrame) with latest

values for model input, persisted or passed to Model Hub.

## E

Model Hub

(training)

scikit-learn , 
XGBoost , 
TensorFlow/

Train and version predictive models. Runs weekly (or as

needed) to update the Direction Classifier and Level

Forecast models. Utilizes MLflow to log parameters,

performance metrics, and to version model artifacts



. Inputs: feature matrix (from D) and target labels

Keras ,  MLflow

derived from DB. Output: serialized model objects (pickled

sklearn models, saved neural network weights) stored

locally or in S3, and updated model registry in MLflow.


## Id

Module /

Service

Tech Stack

Purpose / Interface

## F

Forecaster

Service

Python (custom
logic),  XGBoost

runtime, 
TensorFlow

runtime

Generate forecasts using the latest models and features.

On schedule, it loads the model artifacts from E, retrieves

the most recent feature values, and produces the 4-week

and 12-week ahead forecasts (direction probabilities and

price intervals)


. Implements ensemble logic (combining

models and applying post-process adjustments). Output: a

forecast recommendation object written to a 
btc_recommendations  table and cached for the API.

## G

FastAPI (REST

## Api & Ui

backend),

Disseminate results. Exposes REST endpoints for
programmatic access to forecasts (e.g.  GET /forecast/
current ) and feature data queries

. Also hosts an


Layer

Streamlit (front-
end)

interactive dashboard for visualization of trends, model

outputs, and historical performance. Inputs: reads from

the database and forecast table.

All   modules   are   containerized   and   communicate   via   well-defined   interfaces   (database   tables,   REST

endpoints,   or   shared   volumes   for   artifacts).   This   modular   design   ensures   the   system   is   amenable   to

automated code generation and testing, as each component’s input/output is clearly specified. Scheduling

is managed by a Celery Beat service that triggers the ETL, training, and forecasting tasks in sequence each

week


. 

Note: The architecture remains 100% free-tier friendly – all data sources are public and no paid API keys

are required (any required API tokens for free services are stored in environment variables)


. This allows

the system to be run by anyone without subscription costs.

3. Data Sources (Public Free-Tier)

All data inputs are from free public resources, aggregated to a weekly frequency. The system draws on a

blend of market, on-chain, and macroeconomic data to reflect the diverse drivers of Bitcoin price. Each

source is ingested via module A and normalized in module B. The table below lists the key data feeds:

Domain

Provider / API

Freq.

Notes

BTC Price

CoinGecko REST

## (Ohlcv)

## Api

Daily

→

Historical Bitcoin price and volume. Endpoint:  /coins/
bitcoin/market_chart?

vs_currency=usd&days=max&interval=daily


.

weekly

No API key required. Daily data is aggregated to weekly

(e.g., last close of week).

Realized

Price (on-

chain)

LookIntoBitcoin

(public CSV)

Weekly

Bitcoin realized price (average on-chain cost basis).
Source CSV:  realized-price.csv

. No key


needed.


Domain

Provider / API

Freq.

Notes

Investor

Sentiment

## – Nupl

Fed

Balance

Sheet

ECB Total

Assets

U.S. Dollar

Index

## (Dxy)

## Fred

10-Year

UST Yield

## Fred

LookIntoBitcoin

(public CSV)

Weekly

Net Unrealized Profit/Loss (NUPL), a measure of
aggregate profit sentiment. Source:  net-unrealized-
profit-loss.csv


.

FRED (Federal

Reserve

Weekly

Federal Reserve total assets (liquidity indicator). Series
code:  WALCL  (aggregated weekly). Reflects USD

Economic Data)

liquidity.

## Ecb Sdw

(Statistical Data

Weekly

European Central Bank total assets. Series: 
## Bsi.M.U2.N.A.Tl01.A.1.Z5.0000.Z01.E


.

Warehouse)

Reflects Euro liquidity.

Daily

→

weekly

Daily

→

U.S. Dollar Index (DTWEXBGS)


. Inverse proxy for

global risk appetite; down-sampled to weekly average.

10-year U.S. Treasury Yield (DGS10)


. Optional macro

input (interest rate environment). Down-sampled to

weekly

weekly.

Gold Price

FRED or Yahoo

Finance

Equity

via Yahoo Finance (GLD ETF ticker).
500 or

Nasdaq)

Yahoo Finance
(via  yfinance

Python)

Daily

→

weekly

Daily

→

weekly

Price of gold (e.g. London Bullion USD/oz). Used to

gauge BTC vs gold trends. Down-sampled to weekly. Free

via FRED (series GOLDAMGBD228NLBM) or Yahoo
Finance (ticker  GC=F ).

Global risk asset proxy. E.g., S&P 500 index ( ^GSPC ) or
Nasdaq Composite ( ^IXIC ). Used for correlation

features. Daily closes averaged to weekly. Free (no key,

via public Yahoo API).

All fetched data is aggregated to ISO week periods (week starting Monday 00:00 UTC). Module B performs

this aggregation – for example, summing daily volumes or taking the last available price of the week as the
weekly close. The unified table   btc_weekly   (stored in TimescaleDB) is the “single source of truth” for

modeling. It has the following schema (all fields per week)


:

CREATE TABLE btc_weekly (

week_start

## Timestamptz Primary Key, -- E.G. '2025-07-07 00:00:00+00'

close_usd

## Double Precision,

-- Bitcoin price close of week (USD)

realised_price DOUBLE PRECISION,

-- Realized price (on-chain cost 

basis)

nupl

fed_liq

ecb_liq

dxy

ust10

## Double Precision,

-- Net Unrealized Profit/Loss

## Double Precision,

-- Fed balance sheet total ($)

## Double Precision,

-- ECB balance sheet total (€ equiv)

## Double Precision,

-- Dollar index value

## Double Precision,

-- 10Y Treasury yield (%)

gold_price

## Double Precision,

-- Gold price (USD)


spx_index

## Double Precision

-- S&P500 or Nasdaq index value

);

(Fields like   gold_price   and   spx_index   are new in v3 to support additional macro features. They can be

backfilled using historical data from FRED/Yahoo to extend the series.)

An ingestion job runs at the end of each week (Sunday 03:00 UTC) to pull the latest data and update this

table


. If any daily source is missing recent data (e.g., API delay), the system uses the last available values

or interpolates for that week to ensure feature continuity. All data sources are  public/free, ensuring the

system can be deployed without subscription credentials


.

4. Feature Engineering (Weekly Indicators)

The Feature Builder module (D) computes a rich set of  derived features  from the raw data, capturing

technical trends, macro regimes, and investor sentiment. Each feature is implemented as a function that
takes   the   appropriate   series   (or   multiple   series)   from   btc_weekly   and   returns   a   weekly   time   series
aligned with  week_start

. These features form the input vector for the predictive models. Below is a


summary of features:

Feature Name

Definition ( function )

Description & Rationale

SMA_ratio_52w

ma_ratio(close_usd, 52)

may signal overextension, while <<1 could

52-week moving average ratio – The price

relative to its 1-year simple moving average

(close / SMA_52w)


. Indicates trend position

(above or below long-term trend). A value >>1

indicate oversold levels. Helps detect mean-

reversion opportunities (as used by analysts

monitoring long-term MAs as reversion zones


).

Log Growth Curve deviation – Percent

deviation of the current price from the fitted

long-term logarithmic growth curve


. The

LGC is a curve fit to Bitcoin’s historical growth

(updated annually; see §5.3) to model its

adoption-driven uptrend. This feature gauges

how far price has stretched above or below

“fair value” on the log curve. It anchors the

forecast to realistic bounds, preventing

extreme overshoot. (Analyst Dave the Wave

famously uses log curves to identify cycle

peaks/troughs


).

LGC_distance

lgc_distance(close_usd)


Feature Name

Definition ( function )

Description & Rationale

Liquidity_Z

Liquidity change Z-score – Z-score of the 52-

week change in global central bank liquidity

(Fed + ECB balance sheets)


. Essentially,

how unusual the recent liquidity injection/

liquidity_z(Δ(Fed+ECB), 

drain is, compared to the past year. A positive

52)

Liquidity_Z means liquidity is expanding faster

than normal (risk-on), negative means

tightening (risk-off). This feature captures

macro liquidity flows, which have proven to

lead crypto market moves



.

Net Unrealized P/L Z-score – Z-score of the

Net Unrealized Profit/Loss (investor aggregate

sentiment) over a 1-year window


. High

NUPL_Z indicates a euphoric, profit-heavy

market (potential overvaluation), whereas

## Nupl_Z

zscore(nupl, 52)

low or negative indicates fear/ capitulation

Realised_to_Spot

ratio(close_usd, 

realised_price)

(undervaluation). This is a contrarian

sentiment indicator; extreme values often

precede trend reversals. (Analogous to the

MVRV Z-score used to flag overvaluation peaks


).

Market Value to Realized Value – Price

divided by realized price


. It measures

relative valuation: how far above the on-chain

cost basis the market is trading. Values much

above 1 suggest the market is pricing in high

profit (greed), while near or below 1 suggests

undervaluation (major bottoms often occur

near 1). This complements NUPL in assessing

if the market is over-heated or not.

Relative Strength Index (14-week) –

Momentum oscillator computed from weekly

price changes over ~3 months. Scaled 0-100.

RSI highlights overbought/oversold

conditions; e.g., RSI > 70 often precedes

RSI_14w

rsi(close_usd, 14)

corrections, RSI < 30 signals strong oversold

conditions. By including RSI, the model can

detect trend exhaustion or strength. Notably,

analysts pointed to bearish RSI divergences

(price rising but RSI falling) as a precursor to

the early-2025 drop


.


Feature Name

Definition ( function )

Description & Rationale

DXY_26w_trend

value_vs_SMA(dxy, 26)

Gold_corr_26w

corr(BTC_returns, 

Gold_returns, 26)

Dollar Index trend – The USD index value

relative to its 26-week average (DXY /

SMA_26w of DXY). This gauges the Dollar’s

momentum. A DXY above its half-year average

indicates dollar strength (risk-off), while below

indicates dollar weakness (risk-on). A strong

USD tends to put downward pressure on BTC


. This feature allows the model to account

for the prevailing currency macro regime.

(Also used in the regime filter, §5.4.)

BTC-Gold 6mo Correlation – The correlation

coefficient between weekly BTC % returns and

gold % returns over the past 26 weeks. This

serves as a proxy for risk regime: if BTC

behaves more like a safe-haven (correlating

with gold) or risk asset. A sudden drop in

correlation (or a slowing gold/BTC ratio

trend) can signal shifts in market narrative


. A low or negative correlation could

indicate unique BTC bullish drives or risk-off

divergences, aiding contextual forecast

adjustments.

BTC-Equity 6mo Correlation – Correlation of

BTC with stock market (S&P 500 or Nasdaq)

returns, 26-week window. High correlation

suggests BTC moving with risk assets, while

Equity_corr_26w

corr(BTC_returns, 

decoupling could indicate crypto-specific

(optional)

SPX_returns, 26)

drivers. This feature captures broader risk

sentiment in line with analyst frameworks

that track BTC’s correlation with Nasdaq


.

(This is optional and can be toggled in

experiments to avoid redundancy with DXY).


Feature Name

Definition ( function )

Description & Rationale

Wave_Stage

(experimental)

Pattern analysis on price series

Elliott Wave phase indicator – A categorical

or numeric feature indicating the current 

cycle wave position. Derived from peak/

trough analysis of the price series: e.g.,

whether the market is in an impulsive upswing

or a corrective wave. A simple implementation

uses recent major highs/lows and Fibonacci

retracement levels to classify the phase (e.g.,

“Wave 2 correction” vs “Wave 3 impulse”). For

instance, after a large impulse, if a drop

retraces ~61.8% (a common Wave 2

magnitude), we mark a corrective phase



. This feature injects pattern-recognition

logic from Elliott Wave theory, which was

instrumental for some analysts in predicting

the $109K→$74K crash and subsequent
bounce

. By knowing the probable wave


stage, the model can adjust expectations (e.g.,

anticipating a big Wave 3 rally after a Wave 2

correction, as BitQuant did


).

Implementation note: Most of the above are computed with standard libraries (Pandas, NumPy). RSI can be
calculated via a rolling window of gains/losses. Correlations use Pandas  .corr()  on rolling windows. The
Wave_Stage  feature may use a custom utility that finds local maxima/minima and matches patterns; this

is an area for iterative development and can be kept simple initially (e.g., binary flag for “corrective phase”

vs “impulsive phase”). All feature functions are unit-tested with known scenarios (see §11).

The   resulting   feature   set   is   stored   in   a   feature   matrix   X_t   for   each   weekly   timestamp   t .   Before
forecasting   each   week,   the   latest   X_t   is   assembled   (using   the   newest   data   point   for   each   feature).

Historical  feature  values  are  also  stored  or  can  be  re-derived  from  the  DB  to  train  models.  The  design

ensures these features directly map to meaningful market concepts, enhancing explainability: e.g., if the

model forecasts bearish, one can see if that’s due to an extremely high NUPL_Z (overbullish sentiment) or

liquidity turning negative, etc.

5. Model & Signal Modules

This platform employs a combination of machine learning models for prediction and domain-specific logic

for adjustments, forming a hybrid ensemble. The modeling is split into two tasks: a direction classifier to

predict   the   probability   of   bullish/neutral/bearish,   and   one   or   more  level   forecasters  to   predict   price

targets. By separating classification and regression, we can enforce consistency (e.g., direction sign aligns

with forecasted price change) and incorporate specialized algorithms for each sub-problem. All models are

trained on weekly historical data (most features spanning back to 2011–2012 where available).


5.1 Direction Classifier (Trend Predictor)

Algorithm: Gradient-Boosted Decision Trees (XGBoost  XGBClassifier ).

Objective: Classify the 4-week forward return sign into 3 classes: Bullish, Neutral, or Bearish


. Bullish

means the price is expected to be at least +5% higher in 4 weeks, Bearish means at least –5% lower, and

Neutral for movements in between (i.e., if 4-week change is within ±5%, it’s “Neutral”)


. This 5% threshold

avoids reacting to noise and was chosen based on historical volatility to ensure class balance.

Features Inputs: The classifier uses the full feature vector from §4 at time t (current week’s data) to predict

the return by t+4 weeks. Key features like Liquidity_Z, DXY_trend, and NUPL_Z give it insight into macro and

sentiment conditions that drive medium-term direction. For example, a strongly positive Liquidity_Z and

weak dollar (DXY down) environment is conducive to bullish moves


, while an extremely high NUPL_Z or

RSI could warn of a bearish correction


. By including these diverse features, the classifier effectively

emulates the multi-factor analysis a human analyst would do, but in a quantitative manner. 

Output:  The   model   produces   probabilities  P(Bullish),  P(Neutral),  P(Bearish)  for   the   4-week   outcome.

These  probabilities  are  later  adjusted  by  ensemble  logic  (risk  regime  filter)  and  used  to  make  the  final

directional   call   (see   §6).   We   log   the   classifier’s   Brier   score   (probability   calibration)   and   accuracy   during

backtests


 to ensure it meets the >65% directional hit-rate target. The choice of XGBoost provides a good

balance   of   interpretability   (feature   importance   can   be   extracted)   and   accuracy   for   tabular   data,   and   it

handles   the   small   dataset   (on   the   order   of   500   weekly   observations)   well   with   regularization.   We   also

consider its output (e.g., partial dependence of features) for explainability to end-users.

5.2 Level Forecasters (Price Target Regression)

For the numeric price prediction, we employ  two complementary models  and blend their outputs. This

ensemble of a linear model and a neural network leverages both statistical transparency and non-linear

pattern recognition:

5.2.1 Bayesian Ridge Regressor

Algorithm: Bayesian Ridge Regression (via scikit-learn).

Target: Log-price return over horizon (4-week and 12-week). We model the logarithm of price to stabilize

variance and then derive price forecasts. For each horizon (4w and 12w), a separate regression model is

trained.

Features:  Uses   the   same   feature   set   as   the   classifier  plus  the   classifier’s   own   output   probabilities   as

additional   features


.   Including   the   classifier’s   Bull/Neutral/Bear   probabilities   provides   a   summary   of

directional   outlook   that   the   regression   can   use   (effectively   a   feature   that   encapsulates   non-linear

interactions   learned   by   the   classifier).   All   numeric   features   from   §4   are   used,   including   LGC_distance,

Liquidity_Z, etc., as of time t to predict log(price at t+horizon). Additionally, long-term anchors like the LGC

fair value are included: e.g., the current LGC projected value at t+12w can be an input to the 12-week model

(ensuring long-term forecasts gravitate toward the growth curve). 

Output: Mean forecast of log-price at t+h, which is converted to a price forecast. The Bayesian Ridge model

also provides an  uncertainty estimate  (via the posterior variance of weights) that we use to compute a

predictive interval

. We generate an approximate  60% confidence interval  (≈20th–80th percentile)
and 90% interval for the price at horizon. These intervals are crucial for expressing forecast uncertainty to



users and are logged for interval coverage during backtesting


. Bayesian regression was chosen for its

ability   to  quantify   uncertainty  and   its   robust   performance   on   limited   data.   It   serves   as   a   baseline

“statistical” model that is less likely to overfit and thus provides a stable anchor prediction.

5.2.2 LSTM Neural Network Model

Algorithm: Long Short-Term Memory (LSTM) neural network (implemented in TensorFlow/Keras).

Target:  Price  time  series  over  the  horizon.   We   use   sequence-to-one   prediction:   the   model  looks   at  the

recent sequence of data and predicts the price h weeks ahead. Two instances are trained: one for 4-week

ahead price, one for 12-week ahead.

Input Sequence: Past 52 weeks (1 year) of data (can include multi-variate input). In practice, we feed in a

sequence of vectors [price, key features] for the past n weeks. Key features could include price returns,

momentum, and perhaps liquidity or trend indicators to help the network. (We keep input dimensionality

small   to   avoid   overfitting   –   e.g.,   use   3-5   most   important   features   as   determined   by   XGBoost   feature

importance, such as close_usd, SMA_ratio, Liquidity_Z, NUPL_Z). The LSTM thus captures temporal patterns,

seasonality, and lag effects that static regressors might miss. For example, it might learn the typical post-

halving year boom-bust shape or momentum continuation patterns.

Architecture:  A single hidden LSTM layer (e.g., 50 units) followed by a dense output layer. We keep the

network   relatively   small   given   the   limited   data.   We   train   using   early   stopping   on   validation   to   prevent

overfitting.

Output: Forecasted price (we train it on actual price, or log-price for stability). Unlike the Bayesian model,

the LSTM does not naturally provide confidence intervals. We address this by either: (a) using the Bayesian

model’s interval as the primary interval and treating the LSTM as a point estimate enhancer, or (b) using

techniques like Monte Carlo dropout to simulate an ensemble of networks for uncertainty. In this design,

we will use the simpler approach (a) – i.e., rely on Bayesian for interval, and use LSTM for improved point

accuracy and as a sanity check/alternative view. 

Role in Ensemble: The LSTM serves as a non-linear pattern detector – it might catch complex relationships

(like delayed effects of a liquidity change or the rhythm of Elliott wave-like cycles) that a linear model might

miss. For instance, if the market is in a corrective wave (per Wave_Stage feature), the linear model might

overshoot by assuming mean reversion, but the LSTM, seeing a similar pattern in history, might predict a

stronger rebound, aligning with how human wave analysts forecast a big Wave 3 rally after a Wave 2 dip


.   Combining   it   with   the   Bayesian   model   (which   might   be   more   conservative)   can   yield   a   balanced

forecast. We log the LSTM’s performance in backtests and use it primarily if it demonstrably improves error

metrics (MAPE). If not, the system can default to the Bayesian forecast alone. (This modular approach allows

easy activation/deactivation of the LSTM component.)

5.3 Log-Growth-Curve Anchor

The  Logarithmic Growth Curve (LGC)  of Bitcoin is used as an external model for long-term value. We fit

the LGC to historical price data using non-linear least squares (fit to a function of form log(price) = f(time),

e.g., a logistic or power-law curve)


. The parameters (slope, intercept, curvature) are re-estimated once

per year  using all data up to that point, ensuring the curve remains up-to-date with structural changes

(e.g., if adoption accelerates or decelerates over cycles). 

The LGC provides a “fair value” baseline and upper/lower bounds (often the curve is ± some multiples of

standard deviation in log terms). In the system, we use this in two ways: - As a  feature (LGC_distance)  –


already described, the percentage above/below the LGC at present. - As a regressor input – the projected

LGC price at 4 weeks and 12 weeks in the future can be given to the level forecasters. Essentially, “what does

the LGC say the price should be in 1–3 months?” This anchors the model’s expectations. For example, if all

features are bullish but the LGC projected value is much lower than current price, the model might temper

its forecast, reflecting that Bitcoin rarely stays far above its log growth trend for long


. Conversely, if

current price is below LGC fair value, that adds upside bias.

By incorporating LGC, the system mirrors the strategy of analysts who use long-term log curves to identify

when   the   market   is   overextended.   It   improves  stability  of   forecasts   and   guards   against   irrational

extrapolation, thereby enhancing accuracy over longer horizons.

5.4 Liquidity Regime Filter

Global liquidity and currency strength define high-level regimes for risk assets. We implement a simple

regime   classification:  Risk-On  vs  Risk-Off,   based   on   liquidity   trends   and   DXY:   -  Risk-On   =   TRUE  if
Liquidity_Z > 0 AND DXY_26w_trend < 1  (i.e., DXY below its 26-week average).

- Risk-On = FALSE otherwise (meaning liquidity growth is below average or negative, or the dollar is strong)


.

This rule is rooted in observed dynamics: positive central bank liquidity flow combined with a weakening

USD creates a favorable backdrop for Bitcoin (more liquidity chasing risk assets)


. If either condition fails

(liquidity drying up or dollar strengthening), the environment is less supportive or outright bearish for BTC.

We use this filter in the ensemble decision logic (§6) to adjust forecasts. Specifically, if  Risk-On  is FALSE

(risk-off regime), we will down-weight bullish signals. This mimics caution during adverse macro regimes.

The  threshold  for  liquidity  (0  =  above/below  52-week  average)  and  the  26-week  DXY  average  are  initial

heuristics; these can be tuned via Hyperparameter search (we allow a shift, e.g., maybe require Liquidity_Z >

+0.5 for Risk-On, etc.)


. 

Additionally, the regime flag can be provided as an input feature to models (so they can learn different

behaviors in each regime). In future iterations, a more nuanced  regime-switching model  could be used

(e.g.,   a   Hidden   Markov   Model   to   probabilistically   detect   regimes,   or   training   separate   models   for   each

regime).   For   now,   this   binary   filter   is   a   straightforward   and   explainable   proxy   for   regime   awareness,

aligning with analysts like Michael Howell who emphasize liquidity conditions as leading indicators


.

6. Ensemble Forecast & Decision Logic

After obtaining predictions from the models, the system applies ensemble logic to form the final forecast

recommendation. This step integrates the classifier’s probabilistic view with the regression models’ point

forecasts, and applies the regime-based adjustment for a balanced outcome.

6.1 Direction Determination: We adjust the raw classifier probabilities based on the Liquidity Regime filter:
- If  Risk-On = FALSE  (i.e., macro conditions are unfavorable), reduce the Bullish probability by a certain

factor (e.g., 30% relative reduction)


  and correspondingly increase Bearish/Neutral odds. This reflects

that  in  risk-off  climates,  even  optimistic  signals  should  be  viewed  with  caution  (several  analysts  flipped

bearish in early 2025 due to liquidity tightening despite bullish sentiment elsewhere


). The 30% down-


weight is a hyperparameter (subject to tuning), ensuring we don’t over-suppress or under-suppress bullish

calls.
- If   Risk-On   =   TRUE ,   leave   probabilities   as-is   (or   even   could   modestly   boost   bullish   probability,   but

currently we treat risk-on as the normal state).

After this adjustment, we take Direction = argmax(adjusted probabilities)


. Thus, the system chooses

the   class   with   highest   probability   as   the   directional   outlook.   For   example,   if   initially   P(Bullish)=0.5,

P(Neutral)=0.3, P(Bearish)=0.2, but regime is Risk-Off, Bullish P might be lowered to ~0.35, making Neutral

the highest – hence final direction becomes Neutral. This way, macro regime serves as a sanity check on the

ML model’s exuberance. (During backtesting, we verify that this filter improves directional accuracy beyond

the classifier alone, capturing those instances where macro overrides technicals.)

6.2 Price Target Blending:  We produce the 4-week and 12-week  price forecast ranges  by blending the

outputs of the Bayesian and LSTM regressors: - Let $P^{(B)}{4w}$ be the Bayesian predicted price for 4 weeks

ahead (median) and $P^{(L)}$ be the LSTM predicted price. We define  final 4-week target  = $\frac{P^{(B)}

{4w} + P^{(L)}$ (simple average). Similarly for 12-week. This }}{2blending of models  tends to reduce error

from   any   single-model   bias   or   mis-specification,   as   errors   may   cancel   out.   We   chose   an   equal   weight

initially, but these weights can be adjusted based on validation performance (e.g., if one model consistently

outperforms, it can be weighted higher, or using a meta-learner to optimize the combination).

- Prediction Interval: The Bayesian model yields a 60% confidence interval (20th–80th percentile) and 90%

interval   for   each   horizon


.   We   use   the   Bayesian   20th   and   80th   percentiles   as   the  forecast   range

displayed to users (this interval is relatively tight and indicates likely range if things go as expected). In a

risk-off regime or if the two models disagree widely, we may expand the interval for caution. For example,

we can take the min and max of the Bayesian and LSTM predictions for the 20th and 80th percentiles

respectively, ensuring the range covers both model views. The 90% interval (5th–95th) is used internally for

risk assessment and is shown on the fan chart (dashboard) for context but not in the headline numbers. 

Thus, the final Recommendation object for the week includes:
- week_start : the reference week of the forecast (e.g., "2025-W28").
- direction : "Bullish"/"Neutral"/"Bearish" (final categorical outlook).
- probabilities : e.g., {"Bullish": 0.xx, "Neutral": 0.yy, "Bearish": 0.zz} (after regime adjustment, summing

to 1).
- target_4w : point forecast for 4 weeks ahead (median expected price).
- range_4w : [low, high] bounds corresponding to 20th and 80th percentile outcomes.
- target_12w : point forecast for 12 weeks ahead.
- range_12w : [low, high] 20th–80th percentile interval for 12 weeks ahead.

This   object   is   then   written   to   the   btc_recommendations   table   and   served   via   the   API



.   For

example,
like:
"direction":   "Bullish",   "probabilities":   {"Bullish":0.68,"Neutral":0.22,"Bearish":

something

  might

outlook

output

Bullish

a

0.10},   "target_4w":   85,000,   "range_4w":   [75,000,   92,000] ,   meaning   a   ~68%   chance   of   a

bullish move with a median target of $85K and central 60% confidence that price will land between $75K–

$92K in 4 weeks.

Rationale: The ensemble logic ensures robustness. By blending models and integrating the regime filter,

we combine multiple predictive signals – much like seasoned traders do by looking at technical patterns,

macro trends, and sentiment together


. This reduces reliance on any single method. For instance, if pure


 
 
 
 
 
 
data-driven models lean bullish but we know liquidity is tightening and even experts turned cautious, the

system will pull back on the bullishness (lowering potential overshoot in forecasts). Conversely, in a high-

liquidity, strong momentum regime, the system won’t underplay a bullish signal. This dynamic adjustment

is aimed at improving real-world performance metrics (like avoiding false alarms in bad conditions, and not

missing opportunities in good conditions), contributing to hitting our KPI targets.

Finally, all these decisions are  logged  for traceability. The probabilities and key features for each forecast

are stored, so that we can later explain  why  a certain forecast was made (e.g., “Bullish because liquidity

expanded and sentiment was still moderate, etc.”). This satisfies the explainability objective – each forecast

can be broken down into its driving factors, akin to how analysts justify their calls with the frameworks

we’ve embedded


.

7. Back-Testing, Validation & Hyperparameter Tuning

To evaluate and tune the system, we perform extensive  back-testing  on historical data, using a  sliding

window approach


: - Start with a training set from ~2012 (or earliest available data) up to 2018, and test

on the year 2019. Then roll the window: train 2013–2019, test 2020, and so forth, up to training 2018–2024,

testing on early 2025. This yields multiple out-of-sample test segments, simulating how the model would

have performed if it had been used in past years sequentially. We ensure that each test year is completely

out-of-sample (all data up to the year before is used for training)


. - We compute metrics for each test

segment and average them: specifically, MAPE for 4-week and 12-week forecasts (to judge accuracy of level

predictions), Directional accuracy (the fraction of weeks where the direction classifier was correct, and also

the Brier score for probabilistic calibration), and Interval coverage (the percentage of time the actual price

fell within our predicted 60% and 90% intervals, aiming for ~60% and ~90% respectively)


. We also track

Hit-rate of target (whether the actual 4 or 12-week price landed between the forecast range endpoints) as

a user-facing metric of usefulness.

During these backtests, we also validate the contributions of new features and modules: - Feature ablation

tests: We selectively remove one group of features at a time (e.g., remove Liquidity_Z & macro features, or

remove   sentiment   features)   to   see   the   impact   on   performance.   For   instance,   we   expect   that   removing

Liquidity_Z and DXY features will reduce accuracy in periods where macro dominated (like 2018 bear or

2022 tightening). Similarly, removing LGC_distance might lead to larger errors during bubble peaks. This

helps justify the inclusion of each method by quantifying its effect. - Model comparison: We evaluate the

Bayesian model alone vs LSTM alone vs the blend. If the blended forecast yields lower MAPE consistently,

we keep it. If the LSTM does not add value (or is erratic), we might drop it or adjust its architecture. We also

ensure   the   classifier’s   influence   is   positive:   e.g.,   using   classifier   probabilities   in   regression   should   not

worsen MAPE; if it does, we revisit integration.

Hyperparameter   Tuning:  We   use   Hyperopt   (TPE   algorithm)   or   a   similar   framework   to   tune   key

hyperparameters


: - XGBoost classifier parameters: e.g., tree depth, learning rate, L2 regularization, etc.

We  tune  these  by  optimizing  a  weighted   objective  (e.g.,  Brier   score  +   directional   accuracy).   -   Ensemble

parameters: the down-weighting factor for bullish probability in risk-off (initially 0.3). We can treat that as a

continuous   parameter   and   find   an   optimal   value   that   maximizes   overall   hit-rate   or   F1   on   directional

predictions. - Thresholds: the 5% neutral band threshold for classifier, maybe fine-tune to 4% or 6% if that

yields better class balance. - LSTM architecture: number of units, training epochs, learning rate. We typically

use   a   separate   validation   set   (e.g.,   last   1   year   of   training   data   in   each   window   as   validation)   for   early

stopping, but we can also include some hyperparams in the search. - Feature transformations: e.g., the


period of RSI (14 weeks was chosen by convention; we might see if 10 or 20 gives better results), or whether

to use log scaling for certain features.

The tuning process logs every trial in  MLflow  with the parameter values and resulting metrics


. We

maintain a  model registry  such that we can easily promote a particular configuration to production if it

shows the best performance. Every weekly training run also logs that week’s new model performance on a

rolling validation (like last 52 weeks) so we can monitor if performance is drifting.

Cross-Validation:  In addition to yearly rolling, we can do k-fold time-series CV within the training set for

more   robust   hyperparam   search.   However,   given   the   serial   correlation,   rolling   window   is   the   primary

method.

Benchmarking:  We   always   compare   against   naive   benchmarks   –   e.g.,   “no-change”   forecast   (last   price

persists) and simple ARIMA or exponential smoothing forecasts – to ensure our models outperform them

significantly. We also track performance against a  human analyst simulation: e.g., using a simple rule-

based strategy derived from the same features (like if Liquidity_Z high and NUPL not extreme, then bullish,

else neutral, etc.) to ensure the ML is adding value beyond heuristic frameworks. This ties into the idea that

each   integrated   framework   (liquidity,   sentiment,   etc.)   is   beneficial   –   the   ML   should   combine   them   in   a

superior way to any single rule.

In backtesting results (which will be documented), we expect to see improvements in prediction accuracy

due to the newly integrated methods. For example, periods like early 2025 crash should be better handled:

our   model,   having   Liquidity_Z   dropping   and   RSI   divergence   flagged,   should   have   increased   Bearish

probabilities   ahead   of   time   –   much   like   the   analysts   in   the   article   did



.   The   success   criterion   is

meeting or exceeding the KPI targets (MAPE, hit-rate) on the test sets. If not met, we iterate on feature

engineering or model complexity.

Finally, these tests are automated in the CI pipeline (see §11): e.g., a backtest can run on each new code

push (on a subset of data for speed) to ensure that any change doesn’t regress performance.

8. Scheduling & Orchestration

All components are orchestrated to run on a  weekly schedule, with a predetermined order, using Celery

Beat (or a Cron job) as the scheduler. The schedule (UTC times) is as follows



:

- 
Sundays 03:00 UTC –  ingest_weekly : Trigger Data Feed (A) and ETL (B) jobs to fetch the prior

week’s data and update the database. This job ensures by shortly after week-close (Sunday

midnight), we have all needed data for the week. It aggregates any daily data to weekly (the

week_start timestamp is Monday 00:00 of last week). On success, it notifies or triggers the next job.
Sundays 03:15 UTC –  train_update : Trigger Model Hub (E) to retrain/update models. It reads all

- 
data up to the latest week from the DB and recomputes features (D is invoked internally or as part of
training pipeline) to get  X  and target  y . It then fits the XGBoost classifier, Bayesian regressor, and

LSTM (if enabled) on the full available training set (or updates them incrementally if we choose). New

model artifacts are saved (and old ones archived) and metrics logged. If retraining fails or metrics

are off, it can revert to last known good model (this is handled via checking validation scores). 


- 
Sundays 03:20 UTC –  publish_forecast : Trigger Forecaster Service (F). This loads the freshly

trained models (or keeps them in memory from the training step), computes the latest features for

the current week (using D), and generates the forecast object (as in §6). The result is written to the 
btc_recommendations  table in the DB and also cached in a JSON file or in-memory store for quick

API serving. After publishing, it may also automatically push a notification or email (not in scope, but

possible extension). 
Monthly Rollup (1st of month 03:30 UTC –  monthly_rollup ): This optional job generates a

- 
special monthly report focusing on the 12-week horizon. It can, for example, compile the last weekly

forecasts into a monthly summary or ensure the 12-week model is run with any monthly-specific

data. In our design, the 12-week forecast is anyway produced weekly; this job might simply collate

them or produce a nicely formatted long-term outlook (emphasizing the latest 12-week target and

intervals)


. 

Each of these tasks is implemented as a Celery task, and Celery Beat schedules them at the given cron

times. The tasks are also connected via dependencies (e.g., train waits for ingest to succeed). If any task fails

(exception   or   bad   data),   an   alert   is   sent   (to   a   monitoring   channel)   and   the   pipeline   can   retry   or   abort

gracefully.   The   10-minute   latency   KPI


  is   satisfied:   by   03:20   UTC   we   have   the   forecast,   which   is   <10

minutes from 03:10 (approx. when the week’s data is fully available from sources). 

Parallelization: The architecture is scalable – e.g., if ingest from multiple sources can occur in parallel, the

ETL container can spawn concurrent fetch jobs for CoinGecko, FRED, etc. using AsyncIO. The training job

typically is fast (XGBoost and Bayes are quick; LSTM is the slowest but on a small network it should train

within a minute or two on CPU). If needed, one could use a GPU for LSTM training, but it’s likely overkill for

weekly data length.

Orchestration in Deployment: In a cloud setup (see Deployment §13), these schedules can be managed by

a combination of ECS tasks or a workflow engine, but the simplest is to keep using Celery in a persistent
worker. We rely on  Docker Compose  locally (the   worker   service runs Celery and executes these tasks

when triggered)


. The scheduling times can be adjusted easily via config (to account for data availability;

e.g., if some data arrives later on Sundays).

9. API Endpoints (FastAPI)

The platform exposes a RESTful API for programmatic access to forecasts and data, using FastAPI (container

G). Key endpoints include:

- 
GET /forecast/current  – Returns the latest weekly forecast in JSON format


. The response

includes the fields outlined in §6 (week_start, direction, probabilities, target_4w, range_4w,

target_12w, range_12w). For example:

{

"week_start": "2025-W28",

"direction": "Bullish",

"probabilities": {"Bullish": 0.71, "Neutral": 0.19, "Bearish": 0.10},

"target_4w": 84500,

"range_4w": [72000, 91000],

"target_12w": 96500,


"range_12w": [68000, 115000]

}

This allows external applications or analysts to fetch the latest signal and integrate it into their tools
or reports. The data is retrieved from the  btc_recommendations  table (or cache) where the

Forecaster service stored it.
GET /forecast/<iso_week>  – (Planned) Retrieve a historical forecast for a given week (to allow

- 
- 
querying past forecasts, if stored). This would help in analyzing track record via the API.
GET /features/{feature_name}?from=YYYY-MM&to=YYYY-MM  – Provides a time series of a
given feature over the specified date range, in JSON or CSV form . For example,  /features/
SMA_ratio_52w?from=2020-01&to=2025-07  returns the weekly SMA_ratio values from Jan 2020


to Jul 2025. This is useful for users who want to perform their own analysis or verify what the model
is seeing. The data is fetched from the DB ( btc_weekly  or a derived features table). We include all

engineered features (SMA_ratio, LGC_distance, etc.) as accessible series. If a feature is computational

(not stored), the API will compute it on the fly from base data.
GET /data/btc_price?from=YYYY&to=YYYY  (and similar) – We can also provide raw data

- 
endpoints (price, volume, etc., as pass-through from our DB) if needed, though it overlaps with data

provider APIs. The focus is on forecast and feature transparency.
GET /explain/current  – (Future enhancement) This could return a simple explanation of the

- 
current forecast, e.g., a list of top 3 contributing factors. Implementation could use the classifier’s

feature importance or SHAP values to list, for instance: “Liquidity regime is Risk-On and NUPL

sentiment is neutral, contributing to a bullish outlook.” For now, this is not fully implemented, but

the data (like risk regime flag, extreme feature values) is available. 

All   endpoints   implement   input   validation   and   return   appropriate   HTTP   status   codes.   The   API   service   is

stateless and can be scaled horizontally if needed behind a load balancer. The responses are cached (the

forecast updates only weekly, so caching that response for a few minutes is fine to handle spikes in traffic).

Security: Since all data is public and no user-specific info, we can leave it open or use a simple API key if

needed. Deployment can restrict it to certain domain or require a token if necessary.

The API design ensures that any external dashboard or script can retrieve the up-to-date forecast quickly

(<100ms query). This also decouples the back-end from the front-end – the Streamlit app (below) actually

uses these API endpoints under the hood to get data, which is a clean separation of concerns.

10. Visualization Dashboard (Streamlit)

A  Streamlit  web   dashboard   provides   an   interactive   visualization   of   the   forecasts   and   underlying   data,

catering to users who prefer a UI. It runs as container G (alongside the API, or integrated into the same

FastAPI app via mount). The dashboard includes several components:

- 
Chart A – Price vs Trend: Weekly Bitcoin price history chart, overlaid with the 52-week SMA and the

LGC fair-value curve


. This shows the user where the current price stands relative to long-term

trends. For example, it highlights when price goes far above the SMA or the log-growth fair value

(which historically precedes corrections


). This contextualizes the forecast in terms of technical

position.


- 
Chart B – Liquidity & Macro: A dual-axis chart plotting Liquidity_Z (and optionally DXY or other

macro) against BTC price


. One axis (left) is BTC price; right axis is the liquidity Z-score (and

perhaps also shows when regime filter is on/off). This illustrates how liquidity injections or

tightening correlate with market moves (e.g., users can see in 2023–2025 how central bank assets

changes preceded BTC rallies or drops


). We use color shading to denote Risk-On vs Risk-Off

periods on this chart for clarity.

- 
Chart C – Forecast Fan Chart: A forward-looking chart showing the current forecast. It plots the

latest known price point and then the projected trajectory as a fan of possible outcomes.

Specifically, we plot the median 4-week and 12-week target, and shade the area between the 20th

and 80th percentile as a 60% confidence band, and a lighter shade between 5th and 95th percentile

for the 90% band. This gives a visual of the uncertainty – a narrower fan means higher confidence.

The fan chart updates each week after new forecast. Users can visually see if the forecast expects a

slight dip then rise (if median 12w is above current, but 4w is below, for example).

- 
Chart D – Sentiment & TA (optional): We can include a chart combining an indicator like NUPL_Z or

RSI with price. For instance, plot weekly RSI values (or NUPL values) as a separate line or subplot

along with price. Mark regions where RSI > 70 or < 30, or where NUPL_Z > 2 or < -2 (historically
extreme greed/fear). This would show at a glance when the model might be getting contrarian

signals. E.g., if RSI divergence happened, we could annotate it. This chart helps users understand the

technical and on-chain context (e.g., “we see RSI divergence here, which is why the model turned bearish”)

as those were key in analysts’ predictions


.

- 
Table – Historical Forecast vs Actual: A table listing, for each of the last N weeks (e.g., 52 weeks),

what the forecast was and what actually happened


. Columns: Week, Forecast Direction, Forecast

4w median & range, Actual 4w later price change (% and whether direction was correct), etc. This

gives transparency about performance – users can see how often it was right, and in which

conditions it errs. It’s essentially a rolling backtest on display. This table will highlight if, say, the

model missed a big move or correctly called a turning point (providing trust or informing needed

improvements). 

All charts are interactive (zoomable) and update automatically each week after new data. The UI will also

show the latest values of key features (perhaps in a sidebar): e.g., “Current Liquidity_Z = +1.5 (above average

liquidity influx), NUPL_Z = +2.1 (greed zone), RSI=75 (overbought)”. This gives an immediate sense of  why

the model might be forecasting bull or bear (if those numbers are extreme). It aligns with explainability –

showing the user the same evidence the model considered.

The Streamlit app layout is designed for clarity: clear section headings (e.g., “Current Outlook” with the

forecast info and fan chart, “Market Trends” with the historical charts, etc.) and tooltips that explain each

chart/metric.   We   ensure   the  color   scheme  matches   common   intuition   (e.g.,   green   for   bullish,   red   for

bearish, etc.). 

In terms of implementation, the dashboard calls the FastAPI endpoints to get JSON data (for forecasts and

features) and then uses Plotly or matplotlib to render charts. It refreshes on a weekly basis or when user

requests. Since Streamlit is live Python, it can also query the database directly if needed.

This visualization component is critical for user adoption, as it provides the “face” of our forecasting system

and communicates the insights in an intuitive way. By incorporating charts that mirror the frameworks

(SMA, LGC, liquidity, sentiment, patterns), we ensure the system’s inner workings are transparent to the

user, similar to how analysts would present their rationale in charts


.


11. Testing & Quality Assurance

To maintain reliability, we implement rigorous testing and validation at each module level, as well as end-to-

end:

- 
Unit Tests: Each feature calculation function (module D) has unit tests using known input
dataframes to verify correctness. For example, we test  ma_ratio  on a sample series to ensure
SMA_ratio_52w is computed correctly, and  lgc_distance  on a dummy exponential series to

ensure it returns near-zero for points on the curve. Similarly, data ingestion parsers (module B) are

tested with sample API responses (we save a few JSON/CSV samples from providers as fixtures) to

ensure the ETL correctly parses and stores the results, including handling missing data or API quirks.

The classifier and regressor training functions are tested on synthetic data to ensure they output

predictions of expected shape and type.

- 
Integration Tests (Pipeline): We run the entire pipeline on a small slice of data (e.g., use the year

2019 data as a test scenario). This integration test spins up a test DB, inserts known historical data

for a few weeks, runs the feature builder, then training, then forecasting, and checks that: 

- 
No exceptions occur, 

- 
The outputs make sense (e.g., probabilities sum to 1, forecast values are numeric and within

reasonable bounds), 

- 
The API endpoints return HTTP 200 and correct JSON structure for the test forecast. We also test the

scheduler in a simulated mode (or just call tasks in order) to ensure the ordering dependencies

(ingest before train, etc.) are respected.

- 
Backtest Validation: The back-testing procedure itself is validated by checking that when we

recombine all test segments, the model’s overall error aligns with the metrics computed

incrementally. Any discrepancy might indicate data leakage or misaligned splits. We ensure that

training sets never include future data. We also compare backtest results with a naive persistence

model to confirm our model is adding value (this is more an analysis, but it's automatically done and

if our model ever underperforms naive, that triggers a review).

- 
Performance Tests: We test that the weekly run indeed completes within the 10-minute latency

budget on a typical environment (e.g., an AWS t3.small or similar). This involves timing the ingestion

(should be <1min normally, mostly network-bound), training (XGBoost and Bayes are seconds; LSTM

might be ~30s with CPU and our small config), and forecasting (<1s). If needed, we optimize any slow

parts or pre-fetch data ahead of time.

- 
Robustness Tests: We simulate scenarios such as a data source failing or returning incomplete data

for a week. The ingestion code is designed with retries and fallback (e.g., if LookIntoBitcoin CSV for

this week is not updated yet, we reuse last week’s value or mark it as unchanged). We test these

fallbacks by pointing to an empty or older file. The system should still produce a forecast (with a

warning logged about using stale data). We also test extreme inputs: e.g., feeding an outlier price

jump to ensure features like z-scores handle it (they should, by definition, but ensuring no overflow

or errors).

- 
User Acceptance Testing: We periodically run the system on past real data and compare its outputs

to what human analysis at the time would suggest. For example, test the system on December 2024

data and see if it would have predicted the early 2025 crash with a bearish or neutral stance (since

we know some analysts did



). If it didn’t, investigate which feature or logic failed and improve

it. This kind of retrospective test, while anecdotal, helps fine-tune the system to capture real-world

important scenarios.


All tests are automated via GitHub Actions in the CI pipeline


. On each pull request, the suite of unit and

integration tests runs. Additionally, we schedule a weekly CI run of the backtest on the latest data to detect

any drift or issues (and possibly to update the performance metrics in documentation). 

Quality Metrics: We enforce coding standards with  black ,  isort ,  ruff  linters


 to maintain code

quality and consistency (this reduces chances of bugs). We also monitor the live system’s outputs: each

week after publishing the forecast, the system computes the last week’s forecast error once the actual price

is known. If errors exceed a certain threshold (say MAPE > 30% consistently or a big miss on direction), it

flags this for review, prompting us to investigate and adjust the model if needed. Essentially, the system has

a self-evaluation loop.

By   combining   thorough   pre-deployment   testing   with   ongoing   monitoring,   we   ensure   the   platform’s

forecasts remain reliable and improvements are continually integrated. This approach to testing mirrors

best practices in ML Ops, keeping the model’s performance in check and its predictions trustworthy.

12. DevOps & Maintenance

The development process follows a disciplined MLOps approach: - Version Control: All code (data pipeline,

features, models, API, dashboard) resides in a Git repository. Every change is code-reviewed and tested via

the   CI   pipeline   (unit   tests   +   backtest)   before   merging,   as   noted   in   §11


.   -  Reproducibility:  We

containerize the environment (Docker) to ensure consistency across dev, test, and prod. The environment

includes specific versions of libraries like scikit-learn, XGBoost, TensorFlow, etc., as noted in the architecture.
For example, the  Dockerfile  uses a Python base image, installs our requirements (including any needed

system libs for psycopg2, etc.), and sets up environment variables for database connection, etc. This means
anyone (or any automated system like Codex) can spin up the system with  docker compose up  and get

the same results locally


. -  MLflow Tracking:  We use MLflow to log experiments – each weekly model

update is logged with a tag (timestamp or week number). We save model artifacts (pickled XGBoost model,

sklearn model, and Keras model in HDF5) with versioning. This allows rollback to a prior model if a new

model is found to be problematic. The MLflow server can run locally (as a tracking URI with files) or on a

small  cloud  instance.  Over  time,  we  prune  old  experiments  to  save  space,  keeping  major  milestones.  -

Prometheus & Monitoring: A lightweight Prometheus exporter runs in the background to collect metrics
on pipeline execution – e.g., how long the   ingest_weekly   took (to monitor if an API is slowing down),

training duration, forecast computation time, etc.


. We also monitor system metrics like memory/CPU,

especially if running on a small VM or container. Alerts can be set if, say, a job exceeds expected time (which

could indicate an API hang or code issue). -  Secrets & Config:  All sensitive or environment-specific info
(though minimal here, since mostly public data) are stored in a   .env  file or environment variables (e.g.,

database passwords, API keys if any)


. The code reads config via a centralized config module (which can

also   manage   things   like   feature   toggle   flags   –   e.g.,   a   flag   to   enable/disable   the   LSTM   model   easily).   -

Maintaining Data: The TimescaleDB is the central data store. We set up continuous backup for the DB (if

on cloud, use automated snapshots). Since it’s mostly weekly data, size is small, but preserving history is

important for retraining. We also have a script to rebuild the DB from scratch by re-fetching all data (in case

of corruption or migration), as all sources are online and historical (CoinGecko provides entire history, FRED

and CSVs provide long history, etc.). This script is tested to ensure we can recover the whole pipeline on a

new  environment.  -  Regular  Maintenance:  We  schedule  a  monthly  review  of  feature  importance  and

model performance. Because the crypto market evolves, certain features may lose relevance or new ones

become   important   (for   instance,   if   a   new   on-chain   metric   becomes   available   or   macro   regime

fundamentally shifts). This review involves analyzing MLflow logs (did any features’ importance drop to near


zero?   Is   our   model   consistently   erring   in   certain   conditions?)   and   updating   the   feature   set   or   model

hyperparameters accordingly. The modular design (especially Feature Builder) allows adding new features

with minimal impact (just need to backfill and retrain). -  Extensibility:  The system is designed to allow

Codex   or   other   code   generation   tools  to   easily   plug   in   new   modules.   For   example,   if   we   decide   to

incorporate   a   sentiment   analysis   from   Twitter   feed,   we   could   create   a   new   data   source   module   and

corresponding feature function, and as long as we add it to the feature list, the rest of pipeline (training,

etc.) picks it up. The clear function definitions and interfaces make this possible without touching the core

logic. - Issue Tracking: Any errors or anomalies (e.g., a forecast that was wildly off) are logged and raised as

issues in our tracking system (could be GitHub Issues). We treat these as opportunities to improve – e.g.,

after   the   fact,   run   SHAP   on   the   instance   to   see   what   the   model   was   thinking,   and   whether   maybe   an

external factor not in our model played a role (this could inform adding a new feature, say something like

“ETF hype” maybe proxied by Google Trends next time).

Overall, the DevOps setup ensures the system runs reliably with minimal manual intervention, but also that

when   human   oversight   is   needed   (model   updates,   feature   engineering   tweaks),   it   can   be   done   in   a

controlled, versioned manner. Since the system is intended to be a long-lived platform, maintainability and
ease of updates are crucial – new methods or data can be integrated over time (e.g., regime detection could

evolve into a learned model, or new technical indicators can be added if they prove helpful).

13. Deployment

The system supports both local and cloud deployment:

- 
Local Development: Using Docker Compose, a developer can bring up all services on a single

machine. The compose file defines services: db (TimescaleDB), redis (for Celery queue), worker

. After
(which runs the scheduler and tasks A-F), api (FastAPI app), and dashboard (Streamlit UI)
configuring the  .env  file (with DB credentials, etc.),  docker compose up  will start everything.


The worker will automatically kick off the scheduled jobs (one can tweak the schedule for faster

testing) and the API/UI will be available at specified ports (e.g., FastAPI on localhost:8000, Streamlit

on 8501). This makes it easy to prototype and test changes.

- 
Cloud Deployment: We use a CI/CD pipeline (GitHub Actions) to build Docker images and push to

DockerHub on new releases


. The production deployment target is AWS ECS Fargate (serverless

containers) for scalability and ease of management


. Each service runs as a task in ECS:

- 
The worker service runs as a scheduled task (AWS EventBridge can trigger it weekly if we prefer not

to keep a service running 24/7; or we run it continuously with Celery Beat inside to manage

schedule). Given the weekly nature, a simple approach is to have a small EC2 or ECS Service always

on, running Celery Beat and worker.

- 
The API service runs as a long-lived ECS service, behind an Application Load Balancer, to serve

requests. We set auto-scaling if needed (though traffic is expected to be low to moderate).

- 
The Streamlit UI can be containerized similarly and served via an ALB or behind the FastAPI (some

integration possible by embedding it).

- 
The DB can be an AWS RDS PostgreSQL with Timescale extension enabled (or we use Timescale

Cloud). Redis can be AWS ElastiCache or simply an in-memory within the worker if simplified (for our

scale, even local Redis in container is fine, but production should use a managed Redis for reliability).

We ensure all secrets (DB password, etc.) are stored in AWS Secrets Manager and injected into containers as

env vars. 


Deployment steps are automated: pushing to main branch triggers CI which runs tests, then builds images.

Upon success, it deploys to ECS (or via a Terraform script or AWS CloudFormation that we maintain). 

Scalability & Cost: The design is lightweight – one could run it on a single t3.small instance with Docker if

needed. But splitting into services as above allows independent scaling. For instance, if heavy backtesting or

hyperparameter tuning is done, that might be run on a separate bigger instance rather than the weekly

forecasting instance. The free-tier data means no cost overhead for data. The main cost is compute for the

small models and hosting the DB. 

Monitoring in Production: We set up CloudWatch alarms for any task failures. The system logs (from each

container) are aggregated (e.g., to CloudWatch Logs or a logging service) so we can debug if something

goes wrong in the cloud. Also, after each weekly run, the system can send a summary Slack/email: “Forecast

published: Bullish with target X, key drivers: ...” – useful for both users and as a check that everything ran.

Updates: If we develop a new version of the model (say v4 with new features), we can deploy it in parallel

(by pointing it to the same data DB but logging forecasts to a separate table) to compare outputs, before

switching over the API to the new model’s results. This A/B testing is possible due to our modular setup and

versioned output.

In summary, the deployment strategy ensures the platform is accessible (via API/UI), reliable (with cloud-

managed infrastructure and monitoring), and easy to maintain (with CI/CD and containerization). A user or

stakeholder can trust that every week, just after week’s close, a new forecast will be live on the dashboard

and   API   without   manual   intervention.   And   if   any   part   fails,   the   alerting   and   modular   design   make   it

straightforward to pinpoint and fix the issue, keeping downtime minimal. 

With this robust design, we have a fully implementable system that marries data-driven modeling with

proven   domain   insights,   aimed   at   providing   a  state-of-the-art   Bitcoin   forecasting   tool  that   is   as

transparent as it is advanced. By integrating methods ranging from log-growth modeling to sentiment and

macro analysis, we expect improved foresight into market moves – much like those analysts who “predicted

the crash” by using a confluence of signals


, our platform strives to do the same systematically. 

Sources:  The  design  above  is  informed  by  the  previous  technical  specification



  and  incorporates

additional strategies validated by experts in the field






, thereby ensuring both continuity

with proven approaches and integration of new, effective methods. 












































Btc Pricing Recommendation System Spec_v1.pdf

file://file-5ywgEXX1qrE7j1SQDfnpK2











Medium_They Predicted Bitcoin’s Crash Before It Happened.pdf

file://file-NV9bBxcGuWMBt6NMfCPL6A


