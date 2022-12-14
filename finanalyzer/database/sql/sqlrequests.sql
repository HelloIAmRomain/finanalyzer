PRAGMA
foreign_keys = ON;

CREATE TABLE IF NOT EXISTS namesCompanies
(
    "id"
    INTEGER
    NOT
    NULL
    UNIQUE,
    "ticker"
    VARCHAR
(
    16
),
    "name" VARCHAR
(
    64
),
    "exchange" VARCHAR
(
    16
),
    PRIMARY KEY
(
    "id" AUTOINCREMENT
)
    );

CREATE TABLE IF NOT EXISTS valuesFinHistory
(
    "open"
    FLOAT,
    "high"
    FLOAT,
    "low"
    FLOAT,
    "close"
    FLOAT,
    "dateValue"
    DATE,
    "dateAdded"
    DATE,
    "namesId"
    INTEGER,
    FOREIGN
    KEY
(
    "namesId"
) REFERENCES "namesCompanies"
(
    "id"
)
    );

CREATE TABLE IF NOT EXISTS financialData
(
    "averageVolume"
    FLOAT,
    "beta"
    FLOAT,
    "bid"
    FLOAT,
    "bidSize"
    FLOAT,
    "bookValue"
    FLOAT,
    "currentPrice"
    FLOAT,
    "currentRatio"
    FLOAT,
    "dayHigh"
    FLOAT,
    "dayLow"
    FLOAT,
    "dividendRate"
    FLOAT,
    "dividendYield"
    FLOAT,
    "earningsGrowth"
    FLOAT,
    "earningsQuarterlyGrowth"
    FLOAT,
    "ebitda"
    FLOAT,
    "ebitdaMargins"
    FLOAT,
    "enterpriseValue"
    FLOAT,
    "fiftyDayAverage"
    FLOAT,
    "fiftyTwoWeekHigh"
    FLOAT,
    "fiftyTwoWeekLow"
    FLOAT,
    "forwardEps"
    FLOAT,
    "forwardPE"
    FLOAT,
    "freeCashflow"
    FLOAT,
    "grossProfits"
    FLOAT,
    "grossMargins"
    FLOAT,
    "marketCap"
    FLOAT,
    "numberOfAnalystOpinions"
    FLOAT,
    "operatingCashflow"
    FLOAT,
    "payoutRatio"
    FLOAT,
    "pegRatio"
    FLOAT,
    "priceToBook"
    FLOAT,
    "profitMargins"
    FLOAT,
    "quickRatio"
    FLOAT,
    "regularMarketPrice"
    FLOAT,
    "regularMarketVolume"
    FLOAT,
    "returnOnAssets"
    FLOAT,
    "returnOnEquity"
    FLOAT,
    "revenuePerShare"
    FLOAT,
    "sharesShort"
    FLOAT,
    "shortRatio"
    FLOAT,
    "totalCashPerShare"
    FLOAT,
    "totalDebt"
    FLOAT,
    "totalRevenue"
    FLOAT,
    "volume"
    FLOAT,
    "twoHundredDayAverage"
    FLOAT,
    "dateValue"
    DATE,
    "namesId"
    INTEGER
    NOT
    NULL,
    FOREIGN
    KEY
(
    "namesId"
) REFERENCES "namesCompanies"
(
    "id"
)
    );
