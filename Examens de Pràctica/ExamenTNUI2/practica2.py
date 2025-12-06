import math
import numpy as np
import pandas as pd
import datetime
import itertools
from tqdm.notebook import trange, tqdm
import matplotlib.pyplot as plt


# ---- Càrrega i exploració de dades ----


def read_table() -> pd.DataFrame:
    unames = ['user_id', 'gender', 'age', 'occupation', 'zip']
    users = pd.read_table('ml-1m/users.dat', sep='::', header=None, names=unames, engine='python')
    rnames = ['user_id', 'movie_id', 'rating', 'timestamp']
    ratings = pd.read_table('ml-1m/ratings.dat', sep='::', header=None, names=rnames, engine='python')
    mnames = ['movie_id', 'title', 'genres']
    movies = pd.read_table('ml-1m/movies.dat', sep='::', header=None, names=mnames, engine='python', encoding='latin-1')

    data = pd.merge(pd.merge(ratings, users), movies)

    # Per csv: movies = pd.read_table('ml-1m/movies.scv', sep=',', header=0, engine='python')
    return data


def mean_rating(df: pd.DataFrame) -> pd.DataFrame:
    """
    Retorna un dataframe amb la mitjana de valoracions per usuari.
    :param df: DataFrame original 
    :return: DataFrame amb la mitjana de valoracions per usuari
    """
    return df.pivot_table(index='user_id', values='rating', aggfunc='mean', dropna=True)


def best_movies_sorted(df: pd.DataFrame) -> pd.DataFrame:
    """
    Retorna un dataframe amb la mitjana de valoracions per peli, ordenat de millor a pitjor.
    :param df: DataFrame original
    :return: DataFrame amb la mitjana de valoracions per peli, ordenat de millor a pitjor
    """
    movies_mean_rating = df.pivot_table(index='movie_id', values='rating', aggfunc='mean', dropna=False)
    movies_mean_rating_df = movies_mean_rating.reset_index()
    
    movies_mean_rating_sorted = movies_mean_rating_df.sort_values(by=['rating', 'movie_id'], ascending=[False, True])

    return movies_mean_rating_sorted
    

def best_movie_id(df: pd.DataFrame) -> int:
    """
    Retorna l'ID de la peli amb la millor mitjana de valoracions.
    :param df: DataFrame original
    :return: ID de la peli amb la millor mitjana de valoracions
    """
    movies_mean_rating = df.pivot_table(index='movie_id', values='rating', aggfunc='mean', dropna=False)
    movies_mean_rating_df = movies_mean_rating.reset_index()
    
    movies_mean_rating_sorted = movies_mean_rating_df.sort_values(by=['rating', 'movie_id'], ascending=[False, True])
    
    best_movie_id = movies_mean_rating_sorted['movie_id'].iloc[0]

    return best_movie_id


def best_movie_rating(df: pd.DataFrame) -> float:
    """
    Retorna la millor mitjana de valoracions d'una peli.
    :param df: DataFrame original
    :return: Millor mitjana de valoracions d'una peli
    """
    movies_mean_rating = df.pivot_table(index='movie_id', values='rating', aggfunc='mean', dropna=False)
    movies_mean_rating_df = movies_mean_rating.reset_index()
    
    movies_mean_rating_sorted = movies_mean_rating_df.sort_values(by=['rating', 'movie_id'], ascending=[False, True])
    
    best_movie_rating = movies_mean_rating_sorted['rating'].iloc[0]

    return best_movie_rating


def best_movie_name(df: pd.DataFrame) -> str:
    """
    Retorna el nom de la peli amb la millor mitjana de valoracions.
    :param df: DataFrame original
    :return: Nom de la peli amb la millor mitjana de valoracions
    """
    movies_mean_rating = df.pivot_table(index='movie_id', values='rating', aggfunc='mean', dropna=False)
    movies_mean_rating_df = movies_mean_rating.reset_index()
    
    movies_mean_rating_sorted = movies_mean_rating_df.sort_values(by=['rating', 'movie_id'], ascending=[False, True])
    
    best_movie_id = movies_mean_rating_sorted['movie_id'].iloc[0]
    best_movie_name = df[df['movie_id'] == best_movie_id]['title'].iloc[0]

    return best_movie_name


def best_rating_maxviews_sorted(df: pd.DataFrame) -> pd.DataFrame:
    """
    Retorna un dataframe amb les pel·lícules que tenen la millor mitjana de valoracions (5 estrelles),
    ordenades per nombre de valoracions de manera descendent i per movie_id de manera ascendent.
    
    :param df: DataFrame original
    :return: DataFrame amb les pel·lícules amb millor mitjana de valoracions ordenades
    """

    # Calcular mitjana i nombre de valoracions
    movies_stats = df.groupby(['movie_id', 'title'])['rating'].agg(
        rating_mean='mean', 
        rating_count='count'
    )
    
    # Filtrar només les pel·lícules amb puntuació mitjana 5
    five_star_movies = movies_stats[movies_stats['rating_mean'] == 5]
    
    # Ordenar per nombre de valoracions descendent i per movie_id ascendent
    five_star_movies = five_star_movies.sort_values(by=['rating_count', 'movie_id'], ascending=[False, True])
    
    return five_star_movies
    

def top_movie_user(df: pd.DataFrame, usr: int) -> pd.Series:
    """
    Retorna la millor valoració feta per l'usuari `usr`. En cas d'empat,
    es retorna la peli amb menor `movie_id`.
    :param df: DataFrame original
    :param usr: ID de l'usuari
    :return: Serie amb la millor valoració feta per l'usuari `usr`
    """

    valoracions = df[df['user_id'] == usr]
    if valoracions.empty:
        return 0

    valoracions.head()
    valoracions = valoracions.sort_values(by=['rating','movie_id'], ascending = [False, True])
    return valoracions.iloc[0] 


def get_user_ratings(df: pd.DataFrame, user_id: int) -> pd.DataFrame:
    """
    Retorna totes les valoracions d'un usuari específic, ordenades per valoració.
    :param df: DataFrame original
    :param user_id: ID de l'usuari
    :return: DataFrame amb totes les valoracions de l'usuari ordenades per valoració
    """
    return df[df['user_id'] == user_id].sort_values('rating', ascending=False)


def get_movie_ratings(df: pd.DataFrame, movie_id: int) -> pd.DataFrame:
    """
    Retorna totes les valoracions d'una pel·lícula específica, amb dades demogràfiques.
    :param df: DataFrame original
    :param movie_id: ID de la pel·lícula
    :return: DataFrame amb totes les valoracions de la pel·lícula ordenades per valoració
    """
    return df[df['movie_id'] == movie_id].sort_values('rating', ascending=False)


def mean_rating_by_group(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """
    Retorna un dataframe amb la mitjana de valoracions agrupades per una columna específica.
    :param df: dataframe que conté totes les dades
    :param group_col: columna per la qual volem agrupar
    :return: dataframe amb la mitjana de valoracions per grup
    """
    return df.pivot_table(index=group_col, values='rating', aggfunc='mean')


def top_movies_by_group(df: pd.DataFrame, group_value: str, group_col: str) -> pd.DataFrame:
    """
    Retorna les pel·lícules millor valorades per un grup específic en ordre descendent.
    :param df: dataframe que conté totes les dades
    :param group_value: valor del grup pel qual volem filtrar
    :param group_col: columna per la qual volem agrupar
    :return: dataframe amb les n pel·lícules millor valorades pel grup
    Exemple: top_movies_by_group(data,"F","gender")
    """
    group_df = df[df[group_col] == group_value]
    movies_mean_rating = group_df.pivot_table(index='movie_id', values='rating', aggfunc='mean', dropna=False)
    movies_mean_rating_df = movies_mean_rating.reset_index()
    
    movies_mean_rating_sorted = movies_mean_rating_df.sort_values(by=['rating', 'movie_id'], ascending=[False, True])

    return movies_mean_rating_sorted


# ---- Preparació i Manipulació de Dades ----


def build_counts_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Retorna un dataframe on les columnes són els `movie_id`, les files `user_id` i els valors
    la valoració que un usuari ha donat a una peli d'un `movie_id`
    :param df: DataFrame original 
    :return: DataFrame descrit adalt
    """   
    return df.pivot_table(index='user_id', columns='movie_id' ,values='rating', dropna=True, sort=True )


def build_item_user_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Retorna un dataframe on les columnes són els `user_id`, les files `movie_id` i els valors
    la valoració que un usuari ha donat a una peli d'un `movie_id`
    :param df: DataFrame original 
    :return: DataFrame descrit adalt
    """
    return df.pivot_table(index='movie_id', columns='user_id', values='rating')


def get_count(df: pd.DataFrame, user_id: int, movie_id: int = None):
    """
    Retorna la valoració que l'usuari 'user_id' ha donat de 'movie_id'. Si no s'especifica 'movie_id',
    retorna totes les valoracions de l'usuari.
    :param df: DataFrame retornat per `build_counts_table`
    :param user_id: ID de l'usuari
    :param movie_id: ID de la peli
    :return: Enter amb la valoració de la peli
    """
    
    if movie_id==None:
        return df.loc[user_id]
    else:
        return df.loc[user_id, movie_id]
    

def get_max_movie_id(df: pd.DataFrame):
    """
    Retorna el nombre màxim de pel·lícules UNIQUES diferents que hi ha al DataFrame
    :param df: DataFrame original
    :return: Enter amb el nombre màxim de pel·lícules diferents
    """
    max_movie_id = pd.unique(df['movie_id'])
    return max_movie_id


def reset_id_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Retorna un DataFrame on els `user_id` i `movie_id` comencin a 0 i són consecutius.
    :param df: DataFrame original
    :return: DataFrame amb els `user_id` i `movie_id` reiniciats
    """
    # Transformar user_id perquè comenci a 0
    df['user_id'] = pd.Categorical(df['user_id']).codes
    
    # Transformar movie_id perquè comenci a 0
    df['movie_id'] = pd.Categorical(df['movie_id']).codes

    return df


def preprocess_genres(movies_df: pd.DataFrame) -> pd.DataFrame:
    """
    Afegeix columnes binàries per a cada gènere de pel·lícula
    :param movies_df: DataFrame amb les pel·lícules i els seus gèneres
    :return: DataFrame amb columnes binàries per a cada gènere
    """
    # Primer, aconseguir tots els gènere únics
    all_genres = set()
    for s in movies_df['genres']:
        all_genres.update(s.split('|'))

    # Crear una columna binària per a cada gènere
    for genre in all_genres:
        movies_df[f'genre_{genre}'] = movies_df['genres'].apply(lambda x: 1 if genre in x else 0)
    return movies_df


def filter_by_genre(df: pd.DataFrame, genre: str) -> pd.DataFrame:
    """
    Retorna un DataFrame que conté només valoracions de pel·lícules d'un gènere específic.
    :param df: DataFrame original
    :param genre: gènere pel qual filtrar (ex. 'Comedy')
    :return: DataFrame filtrat per gènere
    """
    return df[df['genres'].str.contains(genre)]


def filter_by_demographics(df: pd.DataFrame, gender: str = None, age_range: tuple = None, occupation: int = None) -> pd.DataFrame:
    """
    Selecciona els usuaris que compleixen certs criteris demogràfics (ex. age_range=(18, 24)).
    :param df: dataframe que conté totes les dades
    :param gender: gènere dels usuaris a seleccionar
    :param age_range: rang d'edat dels usuaris a seleccionar (tupla)
    :param occupation: ocupació dels usuaris a seleccionar
    :return: dataframe filtrat segons els criteris demogràfics
    """
    filtered_df = df.copy()
    if gender is not None:
        filtered_df = filtered_df[filtered_df['gender'] == gender]
    if age_range is not None:
        filtered_df = filtered_df[(filtered_df['age'] >= age_range[0]) & (filtered_df['age'] <= age_range[1])]
    if occupation is not None:
        filtered_df = filtered_df[filtered_df['occupation'] == occupation]
    return filtered_df


def filter_by_activity(df: pd.DataFrame, min_user_ratings: int = 5, min_movie_ratings: int = 5) -> pd.DataFrame:
    """
    Filtra usuaris i pel·lícules amb poques valoracions per reduir el soroll.
    Això s'anomena "K-core filtering".
    :param min_user_ratings: mínim nombre de valoracions que un usuari ha de tenir per ser inclòs
    :param min_movie_ratings: mínim nombre de valoracions que una pel·lícula ha de tenir per ser inclosa
    :return: DataFrame filtrat
    """
    # Filtra usuaris
    user_counts = df['user_id'].value_counts()
    valid_users = user_counts[user_counts >= min_user_ratings].index
    df_filtered = df[df['user_id'].isin(valid_users)]
    
    # Filtra pel·lícules
    movie_counts = df_filtered['movie_id'].value_counts()
    valid_movies = movie_counts[movie_counts >= min_movie_ratings].index
    df_filtered = df_filtered[df_filtered['movie_id'].isin(valid_movies)]
    
    return df_filtered

def mean_rating_by_demografics(data: pd.DataFrame, gender: bool = False, age: bool = False, occupation: bool = False) -> pd.DataFrame:
    """
    Retorna un dataframe amb la mitjana de valoracions agrupades per edat.
    :param df: dataframe que conté totes les dades
    :param gender: si es vol agrupar per gènere
    :param age: si es vol agrupar per edat
    :param occupation: si es vol agrupar per ocupació
    :return: dataframe amb la mitjana de valoracions per edat
    """
    if gender:
        mean_ratings = data.pivot_table(values= 'rating', index='title', columns='gender', aggfunc='mean')
    elif age:
        mean_ratings = data.pivot_table(values= 'rating', index='title', columns='age', aggfunc='mean')
    elif occupation:
        mean_ratings = data.pivot_table(values= 'rating', index='title', columns='occupation', aggfunc='mean')
    else:
        mean_ratings = data.pivot_table(values= 'rating', index='title', aggfunc='mean')
    
    return mean_ratings

def count_ratings_by_demografics(data: pd.DataFrame, gender: bool = False, age: bool = False, occupation: bool = False) -> pd.DataFrame:
    """
    Retorna un dataframe amb el nombre de valoracions agrupades per edat.
    :param df: dataframe que conté totes les dades
    :param gender: si es vol agrupar per gènere
    :param age: si es vol agrupar per edat
    :param occupation: si es vol agrupar per ocupació
    :return: dataframe amb el nombre de valoracions per edat
    """
    if gender:
        count_ratings = data.pivot_table(values= 'rating', index='title', columns='gender', aggfunc='count')
    elif age:
        count_ratings = data.pivot_table(values= 'rating', index='title', columns='age', aggfunc='count')
    elif occupation:
        count_ratings = data.pivot_table(values= 'rating', index='title', columns='occupation', aggfunc='count')
    else:
        count_ratings = data.pivot_table(values= 'rating', index='title', aggfunc='count')
    return count_ratings


# ---- Funcions de Similitud i Recomendació ----


def distEuclid(x: np.ndarray, y: np.ndarray) -> float:
    """
    Retorna la distancia euclidiana de dos vectors n-dimensionals.
    :param x: Primer vector
    :param y: Segon vector
    :return : Escalar (float) corresponent a la distancia euclidiana
    """
    
    return np.linalg.norm(x - y)


def simEuclid(Vec1: np.ndarray, Vec2: np.ndarray, norm: float) -> float:
    """
    Retorna la sembalça de dos vectors.
    :param Vec1: Primer vector
    :param Vec2: Segon vector
    :return : Escalar (float) corresponent a la semblança
    """
    # la vostra solució aquí
    similarity = np.pow((1+distEuclid(Vec1,Vec2)), (-1))
    similarity = similarity * (len(Vec1)/norm)

    return similarity


def simCosine(Vec1: np.ndarray, Vec2: np.ndarray) -> float:
    """
    Retorna la semblança de dos vectors basada en la similitud del cosinus.
    :param Vec1: Primer vector
    :param Vec2: Segon vector
    :return : Escalar (float) corresponent a la semblança
    """
    
    dot_product = np.dot(Vec1, Vec2)
    norm_vec1 = np.linalg.norm(Vec1)
    norm_vec2 = np.linalg.norm(Vec2)
    
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0
    
    cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
    
    return cosine_similarity


def simPearson(Vec1: np.ndarray, Vec2: np.ndarray) -> float:
    """
    Retorna la semblança de dos vectors basada en la correlació de Pearson.
    :param Vec1: Primer vector
    :param Vec2: Segon vector
    :return : Escalar (float) corresponent a la semblança
    """
    
    mean_vec1 = np.mean(Vec1)
    mean_vec2 = np.mean(Vec2)
    
    numerator = np.sum((Vec1 - mean_vec1) * (Vec2 - mean_vec2))
    denominator = np.sqrt(np.sum((Vec1 - mean_vec1) ** 2)) * np.sqrt(np.sum((Vec2 - mean_vec2) ** 2))
    
    if denominator == 0:
        return 0.0
    
    pearson_correlation = numerator / denominator
    
    return pearson_correlation


def simUsuaris(DataFrame: pd.DataFrame, User1: int, User2: int) -> float:
    """
    Retorna un score que representa la similitud entre user1 i user2 basada en la distancia euclidiana.
    :param DataFrame: dataframe que conté totes les dades
    :param User1: id user1
    :param User2: id user2
    :return : Escalar (float) corresponent al score
    """
    
    # la vostra solució aquí    
    ratings1 = get_count(DataFrame,User1) 
    ratings2 = get_count(DataFrame,User2) 


    # Agafem les pelicules valorades 
    mask1 = ~ratings1.isnull()
    mask2 = ~ratings2.isnull()
    
    # Agafem les pelicules valorades pels dos
    common_mask = mask1 & mask2
    
    # Apliquem mascara
    common_ratings1 = ratings1[common_mask]
    common_ratings2 = ratings2[common_mask]

    if len(common_ratings1) == 0:
        return 0.0 
    
    
    return simEuclid(common_ratings1, common_ratings2, DataFrame.shape[1])


def compute_similitude(fixed_arr: np.ndarray, var_arr: np.ndarray, num_movies: int) -> float:
    """
    Retorna un score que representa la similitud entre dos vectors basada en la distancia euclidiana.
    :param fixed_arr: Primer vector
    :param var_arr: Segon vector
    :param num_movies: nombre total de pel·lícules
    :return : Escalar (float) corresponent al score
    """
        
    # Creem una màscara per les valoracions que tenen els dos
    mask = ~np.isnan(fixed_arr) & ~np.isnan(var_arr)
    
    # Num de películes en comú
    pelis_comu = np.sum(mask)
    
    if pelis_comu == 0:
        return 0.0
    
    # Agafem només els valors comuns
    v1_comu = fixed_arr[mask]
    v2_comu = var_arr[mask]
       
    # Calculem la similitud
    sim_norm = simEuclid(v1_comu, v2_comu, num_movies)
    
    return sim_norm


def similarity_matrix_1(compute_distance: callable, df_counts: pd.DataFrame) -> np.ndarray:
    """
    Retorna una matriu de mida M x M on cada posició 
    indica la similitud entre usuaris (resp. ítems).
    :param df_counts: df amb els valor que cada usuari li ha donat a una peli.
    :return : Matriu numpy de mida M x M amb les similituds.
    """
    
    # Convertim el DataFrame a numpy
    matrix_data = df_counts.to_numpy()
    m = len(matrix_data)
    
    # Inicialitzem la matriu de similitud amb zeros
    sim_matrix = np.zeros((m,m))
    
    # Calculem la similitud només per la meitat superior i la copiem a la meitat inferior
    for i in range(m):
        for j in range(i+1, m):
            sim_matrix[i,j] = compute_distance(matrix_data[i], matrix_data[j]) 
            sim_matrix[j,i] = sim_matrix[i,j]

    return sim_matrix


def similarity_matrix_2(DataFrame: pd.DataFrame) -> np.ndarray:
    """
    Retorna una matriu de mida M x M on cada posició 
    indica la similitud entre usuaris (resp. ítems).
    Substitueix els nand per 0.
    :return : Matriu numpy de mida M x M amb les similituds.
    """

    # la vostra solució aquí

    # Substituïm NaN per 0
    x = DataFrame.fillna(0).values

    # Màscara, 1 si hi ha valoració, 0 si no
    B = (x != 0).astype(float)                  

    # Productes escalars i normes restringides
    x_y = np.dot(x, x.T)                       # Producte escalar entre usuaris
    x_sq = x ** 2                              # Quadrats de les valoracions
    masked_sq_norm = np.dot(x_sq, B.T)         # Norma de 'i' restringida a ítems de 'j'

    
    # Distància euclidiana entre usuaris (amb màscara)
    dist_sq = masked_sq_norm + masked_sq_norm.T - 2 * x_y
    dist_sq = np.maximum(dist_sq, 0)          # Correcció d'errors numèrics negatius
    dists = np.sqrt(dist_sq)                  # Distància euclidiana

    # Conversió a semblança i ponderació pels ítems comuns
    t1 = 1 / (1 + dists)                     # Passem de distància a semblança
    n_comu = np.dot(B, B.T)                  # Nombre d’ítems en comú per parella d’usuaris
    n_total_pelis = x.shape[1]               # Total d’ítems del sistema
    t2 = n_comu / n_total_pelis              # Ponderació segons % d’ítems comuns

    # Combinar semblança i ponderació
    sim_matrix = t1 * t2   

    # Evitar dividir per 0
    sim_matrix[n_comu == 0] = 0

    # Posem 0's a la diagonal per tal que un usuari no es recomani a si mateix
    np.fill_diagonal(sim_matrix, 0.0)           

    return sim_matrix


# ---- Funcions de Recomendació ----


def find_similar_users(DataFrame: pd.DataFrame, userID: int, m: int) -> dict:
    """
    Retorna un diccionari de usuaris similars amb les scores corresponents.
    :param DataFrame: dataframe que conté totes les dades
    :param userID: usuari respecte al qual fem la recomanació
    :param m: nombre d'usuaris que volem per fer la recomanació
    :param similarity: mesura de similitud
    :return : dictionary
    """
    similitud = DataFrame.index.to_series()\
        .apply(lambda usuari: simUsuaris(DataFrame, userID, usuari))\
        .drop(userID) # Eliminem la comparació amb ell mateix

    # Ens quedem amb els m millors
    top_m = similitud.sort_values(ascending = False).head(m)

    # Normalitzem
    normal_m = top_m / top_m.sum()
    dic_sim = normal_m.to_dict()
    
    return dic_sim


def find_similar_users_sim_matrix(DataFrame: pd.DataFrame, sim_mx: np.ndarray, userID: int, m: int) -> dict:
    """
    Retorna un diccionari de usuaris similars amb les scores corresponents.
    :param DataFrame: dataframe que conté totes les dades
    :param sim_mx: similarity_matrix
    :param userID: usuari respecte al qual fem la recomanació
    :param m: nombre d'usuaris que volem per fer la recomanació
    :return : dictionary
    """
    
    # Agafem la fila de la matriu de similitud
    serie = pd.Series(sim_mx[userID], index=DataFrame.index)
    
    # Treiem a l'usuari
    serie = serie.drop(userID)
    
    # Ordenem de més a menys semblant
    top_m = serie.sort_values(ascending=False).head(m)
    
    # Normalitzar
    top_m /= top_m.sum() if top_m.sum() > 0 else 1
    
    return top_m.to_dict()

    
def weighted_average(DataFrame: pd.DataFrame, user: int, sim_mx: np.ndarray, m: int) -> dict:
    """
    Retorna un diccionari amb les prediccions de valoració per a les pel·lícules que l'usuari no ha vist.
    :param DataFrame: dataframe que conté totes les dades
    :param user: usuari al qual fem la recomanació
    :param sim_mx: similarity_matrix
    :param m: nombre d'usuaris semblants a tenir en compte per les recomanacions
    :return: diccionari {peli_id: score predit}
    """

    df = build_counts_table(DataFrame)
    
    # Calculem els m usuaris més semblants al que volem recomanar
    dicc_top_users = find_similar_users_sim_matrix(df, sim_mx, user, m)
    if not dicc_top_users:
        return {}

    sim_users = pd.Series(dicc_top_users)

    # Matriu de valoracions dels veïns
    nei_ratings = df.loc[sim_users.index]

    # Mitjana de valoracions de l’usuari i dels veïns
    user_mean = df.loc[user].mean()
    nei_means = nei_ratings.mean(axis=1)

    # Desviacions de cada veí: r(v,i) - μ_v
    desviacions = nei_ratings.sub(nei_means, axis=0)

    # Suma ponderada de les desviacions segons les similituds
    numerador = desviacions.mul(sim_users, axis="index").sum(axis=0)
    denominador = sim_users.sum()

    # Predicció final: μ_u + (suma_ponderada / suma_de_similituds)
    pred = user_mean + numerador / denominador

    # Limitem les prediccions al rang de puntuació (1–5)
    pred = pred.clip(lower=1, upper=5)

    # Eliminem les pel·lícules que l’usuari ja ha vist
    seen = df.loc[user].dropna().index
    pred = pred.drop(seen, errors='ignore').dropna()

    return pred.sort_values(ascending=False).to_dict()


def getRecommendationsUser(DataFrame: pd.DataFrame, user: int, sim_mx: np.ndarray, n: int, m: int) -> pd.DataFrame:
    """
    Retorna un dataframe amb les n pel·lícules més recomanades per a l'usuari.
    :param DataFrame: dataframe que conté totes les dades
    :param user: usuari al qual fem la recomanació
    :param sim_mx: similarity_function
    :param n: nombre de pelis a recomanar
    :param m: nombre d'usuaris semblants a tenir en compte per les recomanacions
    :return : dataframe de pel·licules amb els scores.
    """
    
    perdiccions = weighted_average(DataFrame, user, sim_mx, m)
    
    if not perdiccions:
        return pd.DataFrame(columns=['movie_id', 'predicted_score'])
    
    # Convertim a DataFrame
    df = pd.DataFrame.from_dict(perdiccions, orient='index', columns=['predicted_score'])
    
    # Reiniciem l'índex i renombrem la columna
    df = df.reset_index().rename(columns={'index': 'movie_id'})
    df = df.sort_values(by='predicted_score', ascending=False).head(n)
    
    return df


# ---- Funcions d'Avaluació del Model ----


def get_sets(data: pd.DataFrame, frac: float = 0.1, random_state: int = 42) -> tuple:
    """
    Divideix les dades en un train set i un test set basat en usuaris.
    :param data: dataframe que conté totes les dades
    :param frac: fracció de dades que volem per al test set
    :param random_state: estat aleatori per a la reproduïbilitat
    :return : 
        - :param 1st: dataframe que conté les dades de train
        - :param 2nd: dataframe que conté les dades de test
    """
    users_unique = data['user_id'].unique()
    num_users = users_unique.shape[0]
    
    users_df = pd.DataFrame(users_unique, columns=['user_id'])
    
    test_users_df = users_df.sample(frac=frac, random_state=random_state)
    test_user_ids = test_users_df['user_id'].values
    
    train_users_df = users_df.drop(test_users_df.index)
    train_user_ids = train_users_df['user_id'].values
    
    train_set = data[data['user_id'].isin(train_user_ids)]
    test_set = data[data['user_id'].isin(test_user_ids)]

    return (train_set, test_set)


def add_testdata(traindf: pd.DataFrame, test_set: pd.DataFrame) -> tuple:
    """    
    Afegeix un 80% de les dades del test_set al traindf i deixa el 20% restant com a nou test_set.
    :param traindf: dataframe que conté les dades de train
    :param test_set: dataframe que conté les dades de test

    :return : 
        - :param 1st: dataframe que conté les dades de train juntament amb el 80% de test seleccionat
        - :param 2nd: dataframe que conté les dades de test que queden (20% restant)
    """
    # la vostra solució aquí
    
    train_extra_list = []
    test_final_list = []
    test_frac=0.2
    random_state=42

    for user_id, group in test_set.groupby('user_id'):

        n_test = max(1, int(len(group) * test_frac))  # sempre com a mínim 1

        test_user_part = group.sample(n=n_test, random_state=random_state)

        train_user_part = group.drop(test_user_part.index)

        test_final_list.append(test_user_part)
        train_extra_list.append(train_user_part)

    test_final = pd.concat(test_final_list, axis=0)
    train_extra = pd.concat(train_extra_list, axis=0)
    train_final = pd.concat([traindf, train_extra], axis=0)

    train_final = train_final.reset_index(drop=True)
    test_final = test_final.reset_index(drop=True)

    return train_final, test_final
    

def evaluateRecommendations(train: pd.DataFrame, test: pd.DataFrame, m: int, n: int, sim: np.ndarray) -> float:
    """
    Retorna l'error generat pel model
    :param DataFrame: dataframe que conté totes les dades
    :param userID: usuari respecte al qual fem la recomanació
    :param m: nombre d'usuaris que volem per fer la recomanació
    :param n: nombre de pelis a retornar (no)
    :param sim: matriu de similitud
    :return : Escalar (float) corresponent al MAE
    """
   
    # la vostra solució aquí
    
    errors = []

    # Iterem per cada usuari del test_set
    for user_id in test['user_id'].unique():

        # Obtenim les recomanacions per l'usuari
        recs = getRecommendationsUser(train, user_id, sim, n, m)  # dataframe amb scores

        # Agafem les pelis que realment l'usuari ha vist al test
        test_user = test[test['user_id'] == user_id]

        # Només considerem les pelis que han estat recomanades
        merged = pd.merge(test_user, recs, on='movie_id', how='inner', suffixes=('_true', '_pred'))

        # Si no hi ha intersecció, saltem aquest usuari
        if merged.empty:
            continue

        # Calcul de l'error absolut
        abs_error = np.abs(merged['rating'] - merged['predicted_score'])
        errors.extend(abs_error.tolist())

    # Mitjana de l'error absolut
    mae = np.mean(errors) if errors else np.nan
    return mae