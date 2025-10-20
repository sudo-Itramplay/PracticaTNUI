# -*- coding: utf-8 -*-
"""
Script complet per a l'anàlisi exploratòria de dades (EDA) de les dades
dels taxis grocs de Nova York durant els anys 2019, 2020 i 2021.
"""

from datetime import datetime
import pandas as pd
import numpy as np
import urllib.request
import os
from tqdm import tqdm
import pyarrow.parquet as pq
import matplotlib.pyplot as plt

# -------------- Constants Globals --------------
YEARS = [2019, 2020, 2021]

# -------------- Descàrrega de Dades --------------

def download_data(years: list[int]):
    """
    Descarrega les dades dels trajectes dels taxis grocs de NY per als anys especificats.
    Crea una estructura de directoris 'data/any/' on desa els fitxers .parquet.
    
    Parameters:
        years (list[int]): Llista d'anys a descarregar.
    """
    print("Iniciant la descàrrega de dades...")
    for year in years:
        if not os.path.exists(f'data/{year}'):
            print(f"Descarregant dades per a l'any: {year}")
            os.makedirs(f'data/{year}', exist_ok=True)
            for month in tqdm(range(1, 13), desc=f"Mesos de {year}"):
                url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet'
                file_path = f'data/{year}/{month:02d}.parquet'
                try:
                    urllib.request.urlretrieve(url, file_path)
                except Exception as e:
                    print(f"No s'ha pogut descarregar {url}. Error: {e}")
    print("Descàrrega de dades finalitzada.")


# -------------- Carrega i Neteja de Dades --------------

def load_table(year: int, month: int, sampling: int = 100) -> pd.DataFrame:
    """
    Carrega les dades d'un fitxer Parquet, seleccionant columnes i aplicant mostreig.
    
    Parameters:
        year (int): Any de les dades.
        month (int): Mes de les dades.
        sampling (int): Interval de mostreig (1 de cada 'sampling' files).
    
    Returns:
        pd.DataFrame: DataFrame amb les dades carregades i mostrejades.
    """
    file_path = f'data/{year}/{str(month).zfill(2)}.parquet'
    if not os.path.exists(file_path):
        return pd.DataFrame() # Retorna un DataFrame buit si el fitxer no existeix

    data = pq.read_table(file_path).to_pandas()
    required_data = ['tpep_pickup_datetime', 'tpep_dropoff_datetime', 'passenger_count',
                     'trip_distance', 'PULocationID', 'DOLocationID', 'payment_type',
                     'fare_amount', 'total_amount']
    # Assegurem que totes les columnes requerides existeixen
    existing_cols = [col for col in required_data if col in data.columns]
    return data[existing_cols][::sampling]

def vmean(pickup: datetime, dropoff: datetime, distance: float) -> float:
    """
    Calcula la velocitat mitjana d'un viatge en milles per hora.
    
    Parameters:
        pickup (datetime): Data i hora de recollida.
        dropoff (datetime): Data i hora de deixada.
        distance (float): Distància del viatge en milles.
        
    Returns:
        float: Velocitat mitjana en mph. Retorna 66 si la durada és zero per evitar divisió per zero.
    """
    temps_hores = (dropoff - pickup).total_seconds() / 3600
    if temps_hores == 0:
        return 66  # Valor per identificar i filtrar viatges de durada zero
    return distance / temps_hores

def clean_data(data: pd.DataFrame, year: int, month: int) -> pd.DataFrame:
    """
    Neteja el DataFrame aplicant filtres per eliminar registres invàlids.
    
    Parameters:
        data (pd.DataFrame): DataFrame amb les dades crues.
        year (int): Any de les dades per a consistència temporal.
        month (int): Mes de les dades per a consistència temporal.

    Returns:
        pd.DataFrame: DataFrame netejat.
    """
    data = data.dropna()
    data = data[data["tpep_pickup_datetime"] < data["tpep_dropoff_datetime"]]
    data = data[(data["passenger_count"] > 0) & (data["passenger_count"] < 7)]
    data = data[data["trip_distance"] > 0]
    data = data[((data["PULocationID"] > 0) & (data["PULocationID"] < 264)) & 
                ((data["DOLocationID"] > 0) & (data["DOLocationID"] < 264))]
    data = data[data["payment_type"].isin([1, 2])]
    data = data[(data["fare_amount"] < data["total_amount"]) & 
                ((data["fare_amount"] > 0) & (data["total_amount"] > 0))]
    data = data[(data["tpep_pickup_datetime"].dt.year == year) & 
                (data["tpep_pickup_datetime"].dt.month == month) &
                (data["tpep_dropoff_datetime"].dt.year == year) &
                (data["tpep_dropoff_datetime"].dt.month == month)]
    
    # Filtrar per velocitat màxima permesa a NY (aprox. 65 mph)
    data = data[data.apply(lambda row: vmean(row["tpep_pickup_datetime"], 
                                                     row["tpep_dropoff_datetime"], 
                                                     row["trip_distance"]) < 66, axis=1)]
    
    return data.reset_index(drop=True)

def post_processing(data: pd.DataFrame) -> pd.DataFrame:
    """
    Realitza el post-processament de les dades, afegint noves columnes útils per a l'anàlisi.
    
    Parameters:
        data (pd.DataFrame): DataFrame netejat.

    Returns:
        pd.DataFrame: DataFrame processat i enriquit.
    """
    data["trip_duration"] = (data['tpep_dropoff_datetime'] - data['tpep_pickup_datetime']).dt.total_seconds() / 3600
    data["trip_distance"] *= 1.60934 # Milles a Km
    
    data.drop(columns=['fare_amount', 'total_amount', 'payment_type'], axis=1, inplace=True)

    data["month"] = data["tpep_pickup_datetime"].dt.month
    data["year"] = data["tpep_pickup_datetime"].dt.year
    data["pickup_hour"] = data["tpep_pickup_datetime"].dt.hour
    data["pickup_day"] = data["tpep_pickup_datetime"].dt.dayofweek # Dilluns=0, Diumenge=6
    data["pickup_week"] = data["tpep_pickup_datetime"].dt.isocalendar().week
    
    data["v_mean"] = data["trip_distance"] / data["trip_duration"]
    data.replace([np.inf, -np.inf], np.nan, inplace=True) # Gestionar possibles divisions per zero
    data.dropna(subset=['v_mean'], inplace=True)

    return data


# -------------- Funcions d'Anàlisi Quantitativa --------------

def calculate_total_percentage_change(df: pd.DataFrame, year1: int, year2: int) -> float:
    """
    Calcula el canvi percentual interanual del nombre total de viatges.
    """
    yearly_total = df['year'].value_counts()
    base = yearly_total.get(year1, 0)
    if base == 0:
        return float('inf')
    year_change = ((yearly_total.get(year2, 0) - base) / base) * 100
    return year_change

def calculate_percentage_change_per_passenger(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula el canvi percentual interanual desglossat per nombre de passatgers.
    """
    viatges = pd.crosstab(df['year'], df['passenger_count'])
    return viatges.pct_change() * 100

def calculate_mean_passenger_per_year(df: pd.DataFrame) -> pd.Series:
    """
    Calcula la mitjana de passatgers per viatge per a cada any.
    """
    return df.groupby('year')['passenger_count'].mean()

def analyze_pickup_dropoff_locations(df: pd.DataFrame) -> dict:
    """
    Analitza les localitzacions de recollida i deixada més freqüents.
    """
    top_pickup = df['PULocationID'].value_counts().head(5)
    top_index = top_pickup.index
    average_passengers = df[df['PULocationID'].isin(top_index)].groupby('PULocationID')['passenger_count'].mean()
    percentage_same_location = (df['PULocationID'] == df['DOLocationID']).mean() * 100
    
    return {
        'top_pickup_locations': top_pickup,
        'average_passengers': average_passengers,
        'percentage_same_location_trips': percentage_same_location
    }

# -------------- Visualització de Dades --------------

def bar_plot(df: pd.DataFrame, column: str, xlabel: str, ylabel: str, title: str):
    """
    Crea un gràfic de barres a partir d'una columna del DataFrame.
    """
    values = df[column].value_counts().sort_index()
    ax = plt.gca()
    ax.bar(values.index.astype(str), values.values, color='skyblue')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.xticks(rotation=45, ha='right')

def show_trips_by_month(df: pd.DataFrame, years: list[int]):
    """
    Mostra un gràfic de barres per any amb la quantitat de viatges per mes.
    """
    fig, axes = plt.subplots(1, len(years), figsize=(15, 5), sharey=True)
    fig.suptitle("Quantitat de viatges per mes i any", fontsize=16)
    
    for ax, year in zip(axes, years):
        plt.sca(ax)
        df_year = df[df['year'] == year]
        bar_plot(df_year, 'month', "Mes", "Quantitat de viatges", f"{year}")
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def passengers_per_year_plot(df: pd.DataFrame, norm: bool = False):
    """
    Crea un gràfic de barres de passatgers per any (absolut o percentual).
    """
    viatges = pd.crosstab(df['year'], df['passenger_count'])
    
    if norm:
        viatges = viatges.div(viatges.sum(axis=1), axis=0) * 100
        title = 'Recompte de passatgers per any (%)'
        ylabel = '% de viatges'
    else:
        title = 'Recompte de passatgers per any (Absolut)'
        ylabel = 'Recompte de viatges'
        
    ax = viatges.plot(kind='bar', figsize=(12, 7), stacked=False)
    
    plt.title(title, fontsize=16)
    plt.xlabel('Any')
    plt.ylabel(ylabel)
    plt.legend(title='Nº Passatgers', loc='upper right')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

def visualize_trips_agg(df: pd.DataFrame, column: str, title: str, xlabel: str, ylabel: str):
    """
    Visualitza el nombre de viatges per diferents agregacions temporals.
    """
    plt.figure(figsize=(12, 7))
    ax = plt.gca()
    
    for year, subset in df.groupby('year'):
        viatges = subset[column].value_counts().sort_index()
        ax.plot(viatges.index, viatges.values, marker='o', linestyle='--', label=str(year))
    
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(title='Any', loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

def visualize_histograms(df: pd.DataFrame, column: str, title: str, xlabel: str, xlim: tuple):
    """
    Crea un histograma superposat per a cada any per a una columna donada.
    """
    plt.figure(figsize=(10, 6))
    
    for year, subset in df.groupby('year'):
        plt.hist(subset[column], bins=50, alpha=0.5, label=str(year), density=True)
        
    plt.xlim(xlim)
    plt.xlabel(xlabel)
    plt.ylabel('Densitat')
    plt.title(title, fontsize=16)
    plt.legend(title='Any')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


