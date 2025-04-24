#!/usr/bin/env python
# coding: utf-8

# In[1]:


# app.py

import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder # Import LabelEncoder
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import BallTree
import numpy as np
import folium
from folium.plugins import MarkerCluster
from geopy.distance import geodesic
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit.components.v1 as components
import matplotlib.pyplot as plt


# —— CONFIG ——
st.set_page_config(page_title="Housing Explorer", layout="wide")

# —— LOAD DATA ——
@st.cache_data
def load_data(path):
    df = pd.read_csv(path, parse_dates=['SCRAPED_TIMESTAMP'])
    return df

# 2) Train models once, cache them so they don't re-train on every interaction
@st.cache_resource
def train_models(df):
    # --- XGBoost ---
    target = "RENT_PRICE"
    features = ['SQFT', 'BEDS', 'BATHS', 'GARAGE_COUNT', 'YEAR_BUILT',
        'LATITUDE', 'LONGITUDE', 'POOL', 'GYM', 'DOORMAN',
        'FURNISHED', 'LAUNDRY', 'GARAGE', 'CLUBHOUSE']
    
    train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)
    X_train = train_data[features].copy()
    y_train = train_data[target].copy()
    X_test = test_data[features].copy()
    y_test = test_data[target].copy()
    num_cols = X_train.select_dtypes(include=["float64", "int64"]).columns
    cat_cols = X_train.select_dtypes(include=["object"]).columns
    le = LabelEncoder()
    for col in cat_cols:
        # Fit LabelEncoder on combined unique values from both train and test sets
        all_values = pd.concat([X_train[col], X_test[col]], ignore_index=True).astype(str).unique()
        le.fit(all_values)

        X_train[col] = le.transform(X_train[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))
    
    bool_cols = ['POOL', 'GYM', 'DOORMAN', 'FURNISHED', 'LAUNDRY', 'GARAGE', 'CLUBHOUSE' ]
    for col in bool_cols:
        # Convert 'N' and other non-numeric values to 0 for boolean representation
        X_train[col] = pd.to_numeric(X_train[col], errors='coerce').fillna(0).astype(int)
        X_test[col] = pd.to_numeric(X_test[col], errors='coerce').fillna(0).astype(int)



    dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
    dtest = xgb.DMatrix(X_test, label=y_test, enable_categorical=True)


    params = {
        'objective': 'reg:squarederror',  
        'eval_metric': 'rmse', 
        'eta': 0.1, 
        'max_depth': 8, 
        'subsample': 0.8, 
        'colsample_bytree': 0.8, 
    }

    xgb_model = xgb.train(params, dtrain, num_boost_round=500, evals=[(dtest, "Test")], early_stopping_rounds=50, verbose_eval=50)
    
    dtest = xgb.DMatrix(X_test, enable_categorical=True)
    y_test_xgb = y_test
    y_pred_xgb = xgb_model.predict(dtest)
    
    # --- KNN ---
    df_filtered = df[features + ['RENT_PRICE']].dropna(subset=['RENT_PRICE'])
    X = df_filtered.drop(columns=['RENT_PRICE'])
    y = df_filtered['RENT_PRICE']
    
    numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    X_train, X_test, y_train, y_test_knn = train_test_split(
    X, y, test_size=0.2, random_state=42)
    
    # Preprocessor (reuse structure)
    preprocessor = ColumnTransformer(transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numerical_cols),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), categorical_cols)
    ])

    # KNN pipeline
    knn_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', KNeighborsRegressor())
    ])
    
    from sklearn.model_selection import GridSearchCV

    param_grid = {
        'regressor__n_neighbors': [5, 10, 15, 20],
        'regressor__weights': ['uniform', 'distance']
    }

    grid_search = GridSearchCV(
        estimator=knn_pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='r2',
        n_jobs=-1,
        verbose=2
    )

    grid_search.fit(X_train, y_train)
    knn_model = grid_search.best_estimator_

    y_pred_knn = knn_model.predict(X_test)
    
    test_df = train_test_split(df,test_size=0.2, random_state=42)

    return knn_model, xgb_model, y_test_knn, y_pred_knn, y_pred_xgb, y_test_xgb, test_df

# —— Recommendation ——
@st.cache_data
def load_and_preprocess(path="sampled_california_data_df.csv"):
    df = pd.read_csv(path).dropna(subset=['LATITUDE','LONGITUDE']).reset_index(drop=True)
    # bools
    bool_map = {'Y':1,'y':1,'Yes':1,'yes':1,'T':1,'TRUE':1,'1':1,
                'N':0,'n':0,'No':0,'no':0,'F':0,'FALSE':0,'0':0}
    for c in ['POOL','GYM','DOORMAN','FURNISHED','LAUNDRY','GARAGE','CLUBHOUSE']:
        df[c] = df[c].replace(bool_map).fillna(0).astype(int)
    # fill numeric
    for c in ['BEDS','BATHS','SQFT']:
        df[c] = df[c].fillna(df[c].median())
    df['BEDS'] = df['BEDS'].astype(int)
    # text TFIDF
    tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1,2), max_features=50)
    txt = tfidf.fit_transform(df['DESCRIPTION'].fillna(''))
    txt_df = pd.DataFrame(txt.toarray(),
                          columns=[f"TFIDF_{t}" for t in tfidf.get_feature_names_out()])
    df = pd.concat([df, txt_df], axis=1)
    # set globals
    FEATURES = ['BEDS','BATHS','SQFT'] + ['POOL','GYM','DOORMAN','FURNISHED','LAUNDRY','GARAGE','CLUBHOUSE'] + txt_df.columns.tolist()
    scaler = StandardScaler().fit(df[FEATURES])
    df[FEATURES] = scaler.transform(df[FEATURES])
    return df, FEATURES, scaler

class PropertyRecommender:
    def __init__(self, df, features, scaler):
        self.df = df
        self.features = features
        self.scaler = scaler
        self.global_meds = {
            'SQFT': df['SQFT'].median(),
            'RENT_PRICE': df['RENT_PRICE'].median()
        }
    def _safe_med(self, df, col):
        return df[col].median() if not df.empty else self.global_meds[col]
    def recommend(self, lat, lon, radius, beds, baths, top_n=5):
        tmp = self.df.copy()
        tmp['distance'] = tmp.apply(
            lambda r: geodesic((lat,lon),(r['LATITUDE'],r['LONGITUDE'])).miles, axis=1
        )
        filt = tmp[tmp['distance']<=radius]
        filt = filt[(filt['BEDS']>=beds)&(filt['BATHS']>=baths)]
        if filt.empty:
            return None
        # build user vector
        uv = np.zeros(len(self.features))
        for i, f in enumerate(self.features):
            if f=='BEDS': uv[i]=beds
            elif f=='BATHS': uv[i]=baths
            elif f=='SQFT': uv[i]=self._safe_med(filt,'SQFT')
            elif f=='RENT_PRICE': uv[i]=self._safe_med(filt,'RENT_PRICE')
            elif f in ['POOL','GYM','DOORMAN','FURNISHED','LAUNDRY','GARAGE','CLUBHOUSE']:
                uv[i]=1
            else:
                uv[i]=0
        uv_sc = self.scaler.transform([uv])
        knn = NearestNeighbors(n_neighbors=min(top_n,len(filt)), metric='cosine').fit(filt[self.features])
        dists, idxs = knn.kneighbors(uv_sc)
        res = filt.iloc[idxs[0]].copy()
        res['match_score'] = 1/(1+dists[0])
        return res.sort_values('match_score', ascending=False)

# Update this to your CSV path:
DATA_PATH = 'sampled_california_data_df.csv'
eda_df = load_data(DATA_PATH)
knn_model, xgb_model, y_test_knn, y_pred_knn, y_pred_xgb, y_test_xgb, test_df = train_models(eda_df)

df_rec, FEATURE_ORDER, scaler = load_and_preprocess("sampled_california_data_df.csv")

# —— SIDEBAR NAVIGATION ——
st.sidebar.title("Navigation")
tab1, tab2, tab3, tab4 = st.tabs(["🏠 Home", "🔍 EDA", "🤖 Prediction", "⭐ Recommendation"])


# —— HOME PAGE ——
with tab1:
    st.title("🏠 California Rental Market Analysis")
    st.header("Overview")
    st.write("""
    This app makes it easy to explore California rental listings over time and across regions.  
    It provides three main modules:
    1. **Exploratory Data Analysis (EDA):** Interactive charts and maps to uncover trends in rent prices, property sizes, and amenities.  
    2. **Rent Price Prediction:** Enter property features to forecast rental rates using trained KNN and XGBoost models.  
    3. **Recommendation System:** Get personalized suggestions of the top 5 rental listings matching your needs.
    """)

    st.subheader("Data Features")
    st.markdown("""
    - Dates: `SCRAPED_TIMESTAMP`  
    - Location: `CITY`, `LATITUDE`, `LONGITUDE`  
    - Specs: `BEDS`, `BATHS`, `SQFT`, `POOL`, `GYM`, `DESCRIPTION`  
    - Price: `RENT_PRICE`  
    """)
    st.markdown("Switch to the **EDA** page to filter data and dive into interactive visualizations.")
    st.markdown("Group 19: Chuyun Deng, Cindy Yang, Jiaqi Cheng, Jiayi Li, Keyou Wang, Qilian Wu")


# —— EDA PAGE ——
with tab2:
    st.title("🔍 Exploratory Data Analysis")

    # 1) Sidebar filters
    st.sidebar.header("EDA Filters")

    # — Date range picker —
    min_date, max_date = st.sidebar.date_input(
        "Date range",
        [eda_df.SCRAPED_TIMESTAMP.min().date(), eda_df.SCRAPED_TIMESTAMP.max().date()]
    )

    # — Building type multiselect —
    types = st.sidebar.multiselect(
        "Building type",
        options=eda_df['BUILDING_TYPE'].unique(),
        default=eda_df['BUILDING_TYPE'].unique()
    )

    # — City multiselect —
    # Checkbox to ignore the city filter
    show_all_cities = st.sidebar.checkbox("Show all cities on map", value=False)
    # compute top 20
    top_cities = (
        eda_df['CITY']
        .value_counts()
        .nlargest(20)
        .index
        .tolist()
    )

    # multiselect of just top 20
    selected_top = st.sidebar.multiselect(
        "City (top 20)",
        options=top_cities,
        default=top_cities
    )

    # text box for anything else
    other_city = st.sidebar.text_input(
        "Type another city to include"
    ).strip()

    # — Beds slider —
    bed_min, bed_max = st.sidebar.slider(
        "Beds",
        int(eda_df['BEDS'].min()), int(eda_df['BEDS'].max()),
        (int(eda_df['BEDS'].min()), int(eda_df['BEDS'].max()))
    )

    # — Baths slider —
    bath_min, bath_max = st.sidebar.slider(
        "Baths",
        int(eda_df['BATHS'].min()), int(eda_df['BATHS'].max()),
        (int(eda_df['BATHS'].min()), int(eda_df['BATHS'].max()))
    )

    # — Square Footage slider —
    sqft_min, sqft_max = st.sidebar.slider(
        "Square Footage",
        int(eda_df['SQFT'].min()), int(eda_df['SQFT'].max()),
        (int(eda_df['SQFT'].min()), int(eda_df['SQFT'].max()))
    )

    # — Pool checkbox —
    has_pool = st.sidebar.checkbox("Only show listings with a pool")
    
     # — Gym checkbox —
    has_gym = st.sidebar.checkbox("Only show listings with a gym")
    
    # 2) Apply filters
    date_only = eda_df['SCRAPED_TIMESTAMP'].dt.date
    # build your mask
    mask = (
        date_only.between(min_date, max_date) &
        eda_df['BUILDING_TYPE'].isin(types) &
        eda_df['BEDS'].between(bed_min, bed_max) &
        eda_df['BATHS'].between(bath_min, bath_max) &
        eda_df['SQFT'].between(sqft_min, sqft_max)
    )

        
    if has_pool:
        mask &= (eda_df['POOL'] == 'Y')
    if has_gym:
        mask &= (eda_df['GYM'] == 'Y')
        
    # THEN when you make df, override the city filter if the box is checked:
    if show_all_cities:
        df = eda_df[mask]         # ignore any cities
    else:
        # include any of the selected top cities
        # if they typed an “other” city, include substring‐matches too
        if other_city:
            mask |= eda_df['CITY'].str.contains(other_city, case=False, na=False)
        # mask &= eda_df['CITY'].isin(selected_top)
        df = eda_df[mask & eda_df['CITY'].isin(top_cities)]

    # 3) Rent over time (monthly bar chart, narrower bars)
    st.subheader("🏷️  Avg Rent Price by Month")
    
    # 3.1) Combined Avg Rent & Avg SQFT chart
    # — Compute monthly metrics —
    df['MONTH'] = df['SCRAPED_TIMESTAMP'].dt.to_period('M').dt.to_timestamp()
    monthly = (
        df
        .groupby('MONTH')
        .agg(
            avg_rent = ('RENT_PRICE','mean'),
            avg_sqft = ('SQFT','mean'),
            avg_beds = ('BEDS','mean'),
            listings = ('RENT_PRICE','size')
        )
        .reset_index()
    )

    # — Build a combined bar + line chart —
    import plotly.graph_objects as go

    fig = go.Figure()

    # Bar: Avg Rent
    fig.add_trace(go.Bar(
        x=monthly.MONTH,
        y=monthly.avg_rent,
        name="Avg Rent",
        marker_opacity=0.7
    ))

    # Line: Avg SQFT on secondary axis
    fig.add_trace(go.Scatter(
        x=monthly.MONTH,
        y=monthly.avg_sqft,
        name="Avg SQFT",
        mode="lines+markers",
        yaxis="y2"
    ))

    # Layout: two y-axes
    fig.update_layout(
        title="Monthly Rent vs. Square Footage",
        xaxis=dict(title="Month", tickformat="%b %Y", tickangle=45),
        yaxis=dict(title="Avg Rent"),
        yaxis2=dict(title="Avg SQFT", overlaying="y", side="right"),
        legend=dict(y=0.9, x=0.01),
        bargap=0.3,
        margin={"l":40,"r":40,"t":60,"b":40},
        height=450
    )

    st.plotly_chart(fig, use_container_width=True)

    # 3.2) Rent by Beds & Baths pivot table
    # 1) Build the pivot, filling NaNs with 0
    ct = df.pivot_table(
        index='BEDS',
        columns='BATHS',
        values='RENT_PRICE',
        aggfunc='mean'
    ).fillna(0)

    ct.index.name   = 'BEDS / BATHS'
    ct.columns.name = 'BATHS'

    # 2) Style it
    def clear_zero(v):
        # for zeros, wipe out any background so they stay blank
        return "background-color: transparent;" if v == 0 else ""

    styled = (
        ct.style
          .format(na_rep="0", precision=0)           # show zeros as “0”
          .background_gradient(cmap="Blues", axis=None)          # apply a blue ramp to all values…
          .applymap(clear_zero, subset=ct.columns)    # …but then clear the color for zeros
    )

    # 3) Display
    st.subheader("💡 Rent by Beds & Baths")
    st.write(styled)

    # 4) Scatter map by city
    st.subheader("📍  All Listings on the Map")

    fig_listings = px.scatter_mapbox(
        df,                                   # df is your filtered listing-level DataFrame
        lat="LATITUDE",
        lon="LONGITUDE",
        hover_name="CITY",                 # or "CITY" or "ID"—whatever you like as title
        hover_data={
            "ADDRESS": True,
            "RENT_PRICE": True,
            "BEDS": True,
            "BATHS": True,
            "SQFT": True,
            "POOL": True,
            "GYM": True,
            "LATITUDE": False,   # you can hide LAT/LON if too verbose
            "LONGITUDE": False
        },
        color="RENT_PRICE",                   # color scale for rent
        size_max=8,                           # max marker size
        zoom=5,
        height=700,                           # taller map
        mapbox_style="carto-positron"
    )

    # Optional: make markers semi‐transparent so overlaps show
    fig_listings.update_traces(marker=dict(opacity=0.6))
    
    # CA approximate center and zoom
    CA_CENTER = {"lat": 36.7783, "lon": -119.4179}
    CA_ZOOM = 5.0

    fig_listings.update_layout(
        mapbox=dict(
            center=CA_CENTER,
            zoom=CA_ZOOM,
            style="carto-positron"
        ),
        margin={"l":0,"r":0,"t":30,"b":0}
    )

    st.plotly_chart(fig_listings, use_container_width=True)

with tab3:
    st.title("🤖 Rent Price Prediction")
    st.write("Enter the property features below, then see the predicted range and location on the map:")

    # — Numeric inputs —
    sqft        = st.number_input("Square Footage (SQFT)", min_value=100,   max_value=20000, value=800)
    beds        = st.number_input("Bedrooms (BEDS)",          min_value=0,     max_value=10,    value=1)
    baths       = st.number_input("Bathrooms (BATHS)",        min_value=0.0,   max_value=10.0,  value=1.0, step=0.5)
    garage_cnt  = st.number_input("Garage Count",            min_value=0,     max_value=5,     value=0)
    year_built  = st.number_input("Year Built",              min_value=1900,  max_value=2025,  value=2000)
    latitude    = st.number_input("Latitude",                value=float(df['LATITUDE'].mean()))
    longitude   = st.number_input("Longitude",               value=float(df['LONGITUDE'].mean()))

    # — Boolean inputs —
    pool      = st.checkbox("Pool")
    gym       = st.checkbox("Gym")
    doorman   = st.checkbox("Doorman")
    furnished = st.checkbox("Furnished")
    laundry   = st.checkbox("Laundry")
    garage    = st.checkbox("Garage")
    clubhouse = st.checkbox("Clubhouse")

    if st.button("Predict Range"):
        # 1) Build the input row
        X_new = pd.DataFrame([{
            'SQFT': sqft,
            'BEDS': beds,
            'BATHS': baths,
            'GARAGE_COUNT': garage_cnt,
            'YEAR_BUILT': int(year_built),
            'LATITUDE': latitude,
            'LONGITUDE': longitude,
            'POOL': int(pool),
            'GYM': int(gym),
            'DOORMAN': int(doorman),
            'FURNISHED': int(furnished),
            'LAUNDRY': int(laundry),
            'GARAGE': int(garage),
            'CLUBHOUSE': int(clubhouse),
        }])

        # 2) Predict with both models
        knn_pred = knn_model.predict(X_new)[0]
        dmat     = xgb.DMatrix(X_new, enable_categorical=True)
        xgb_pred = xgb_model.predict(dmat)[0]

        low, high = min(knn_pred, xgb_pred), max(knn_pred, xgb_pred)
        st.metric("🔻 Low Estimate", f"${low:,.0f}")
        st.metric("🔺 High Estimate", f"${high:,.0f}")

        coords = np.radians(df[['LATITUDE','LONGITUDE']])
        tree   = BallTree(coords, metric='haversine')
        dist, idx = tree.query(np.radians([[latitude, longitude]]), k=21)
        neighbor_idx = idx[0][1:]                # drop the first (itself)
        nearby = df.iloc[neighbor_idx].copy()

        import plotly.graph_objects as go

        # Prediction trace
        pred_trace = go.Scattermapbox(
            lat=[latitude],
            lon=[longitude],
            mode="markers+text",
            marker=dict(size=15, color="red"),
            text=["Prediction"],
            textposition="top center",
            hovertemplate=(
                "<b>Prediction</b><br>"
                "Low: $%{customdata[0]:.0f}<br>"
                "High: $%{customdata[1]:.0f}<extra></extra>"
            ),
            customdata=[[low, high]]
        )

        # Nearby listings trace
        neigh_trace = go.Scattermapbox(
            lat=nearby["LATITUDE"],
            lon=nearby["LONGITUDE"],
            mode="markers",
            marker=dict(size=8, color="blue", opacity=0.6),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"  +  # ADDRESS or CITY
                "Rent: $%{customdata[1]:.0f}<br>" +
                "Beds: %{customdata[2]}  Baths: %{customdata[3]}<br>" +
                "SQFT: %{customdata[4]:.0f}<extra></extra>"
            ),
            customdata=nearby[["ADDRESS","RENT_PRICE","BEDS","BATHS","SQFT"]].values
        )

        # Build the figure
        fig = go.Figure([neigh_trace, pred_trace])
        fig.update_layout(
            mapbox=dict(
                style="carto-positron",
                center={"lat": latitude, "lon": longitude},
                zoom=10
            ),
            margin={"l":0,"r":0,"t":30,"b":0},
            height=600,
            showlegend=False
        )

        st.subheader("📍 Predicted Point + 20 Nearest Listings")
        st.plotly_chart(fig, use_container_width=True)

        
with tab4:
    st.title("⭐ Real Estate Recommendation")
    # — 2) Recommender class —

    recommender = PropertyRecommender(df_rec, FEATURE_ORDER, scaler)

    # — 3) Sidebar controls —
    st.sidebar.header("Recommendation Filters")
    latitude  = st.sidebar.number_input("Latitude",  value=34.0522, step=0.0001)
    longitude = st.sidebar.number_input("Longitude", value=-118.2437, step=0.0001)
    radius    = st.sidebar.slider("Radius (miles)", 0.5, 20.0, 2.0, 0.5)
    beds      = st.sidebar.slider("Min Bedrooms",  0, 5, 1)
    baths     = st.sidebar.slider("Min Bathrooms", 0.5, 4.0, 1.0, 0.5)
    top_n     = st.sidebar.slider("How Many to Recommend", 1, 20, 5)

    # — 4) Run recommendation —
    if st.sidebar.button("Show Recommendations"):
        results = recommender.recommend(latitude, longitude, radius, beds, baths, top_n)
        if results is None:
            st.warning("No matching properties found. Try widening your radius or lowering beds/baths.")
        else:
            st.subheader("🏘️  Top Recommendations")
            st.dataframe(
                results[['ADDRESS','CITY','RENT_PRICE','BEDS','BATHS','SQFT','distance','match_score']],
                use_container_width=True
            )

            # — 5) Build folium map —
            m = folium.Map(location=[latitude,longitude], zoom_start= int(max(6, radius*2)), tiles='cartodbpositron')
            folium.Marker([latitude,longitude], icon=folium.Icon(color='red',icon='star'), popup="You").add_to(m)
            mc = MarkerCluster().add_to(m)
            for _, r in results.iterrows():
                folium.Marker(
                    [r['LATITUDE'], r['LONGITUDE']],
                    icon=folium.Icon(color='green',icon='thumbs-up'),
                    popup=f"${r['RENT_PRICE']:.0f} | {r['BEDS']}BR/{r['BATHS']}BA"
                ).add_to(mc)

            # — 6) Display map in Streamlit —
            st.subheader("📍 Recommended Properties Around You")
            components.html(m._repr_html_(), height=600, scrolling=True)
