import numpy as np
import pandas as pd
import re
from itertools import combinations
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# FUNCIONS INDEPENDENTS DE LA CLASSE MultinomialNB

def get_open_ngrams(word: str, n: int, include_boundaries: bool = True) -> set:
    """
    Genera un conjunt d’n-grams oberts per a una paraula donada, amb un tractament específic dels límits.

    Args:
        - word (str): La paraula d’entrada.
        - n (int): L’ordre de l’n-gram (per exemple, 2 per a bigrames, 3 per a trigrames).
        - include_boundaries (bool): Si és True, afegeix un guió baix '_' al principi i al
                                final de la paraula per incloure els límits inicial/final.

    Returns:
        - set: Un conjunt d’n-grams oberts únics.
    """
    
    open_ngrams = set()

    # 1. Generar N-grams sense boundaries (Core)
    # combinations retorna tuples, fem servir join per convertir-les a string
    if len(word) >= n:
        for ngram_tuple in combinations(word, n):
            open_ngrams.add("".join(ngram_tuple))

    # 2. Generar boundaries
    if include_boundaries:
        
        # Cas especial: n == 1
        # Simplement afegim el marcador de límit "_"
        if n == 1:
            open_ngrams.add("_")
        
        # Casos generals (n >= 2)
        else:
            # Regla: '_' + primer caràcter + (n-2 caràcters de la resta)
            # Això assegura que el límit només toca el primer caràcter real.
            first_char = word[0]
            rest_of_word = word[1:]
            
            # Necessitem triar (n-2) caràcters de la resta de la paraula
            # Si n=2, triem 0 caràcters (el bucle s'executa una vegada amb string buit)
            if len(rest_of_word) >= (n - 2):
                for combination in combinations(rest_of_word, n - 2):
                    ngram = "_" + first_char + "".join(combination)
                    open_ngrams.add(ngram)

            # Regla: (n-2 caràcters de l'inici fins al penúltim) + últim caràcter + '_'
            last_char = word[-1]
            start_of_word = word[:-1]
            
            if len(start_of_word) >= (n - 2):
                for combination in combinations(start_of_word, n - 2):
                    ngram = "".join(combination) + last_char + "_"
                    open_ngrams.add(ngram)

    return open_ngrams

# Colisions entre paraules

def load_dataset_info(path='dataset.csv'):
    dataset_file_path = path

    try:
        df = pd.read_csv(dataset_file_path)

        # Extracció de textes i etiquetes
        texts = df['Text'].tolist()
        labels = df['language'].tolist()

    except FileNotFoundError:
        print(f"Error: Fitxer '{dataset_file_path}' no trobat.")
        print("Assegura't que dataset.csv està accessible.")
    except Exception as e:
        print(f"Error al carregar el fitxer: {e}")

    return texts, labels

def analyze_word_collisions(texts):
    """
    Analitza col·lisions entre paraules basant-se en els seus bigrames oberts.
    Args:
        texts (list[str]): Llista de textos del dataset.
    Returns:
        unique_words (set): Conjunt de paraules úniques extretes.
        ngram_signature_map (dict): Mapa de signatures d'n-grams a llistes de paraules.
        collision_groups (list): Llista de grups de paraules en col·lisió.
    """
    # 1. Extreure totes les paraules úniques de 'texts'
    unique_words = set()

    for text in texts:
        if isinstance(text, str):
            # Utilitzem regex per trobar paraules (\w+)
            words = re.findall(r'\b\w+\b', text.lower())
            unique_words.update(words)

    # 2. Generar els bigrames oberts amb límits per a cada paraula
    # Utilitzem un diccionari on:
    # Clau: Una representació immutable del conjunt d'n-grams
    # Valor: Llista de paraules que generen aquest conjunt
    ngram_signature_map = defaultdict(list)

    for word in unique_words:
        # Generem el set
        ngrams_set = get_open_ngrams(word, n=2, include_boundaries=True)
        
        # Convertim el set a una tupla ordenada per poder usar-la com a clau de diccionari
        ngrams_signature = tuple(sorted(list(ngrams_set)))
        
        # Guardem la paraula
        ngram_signature_map[ngrams_signature].append(word)

    # 3. Identificar i comptar les col·lisions
    collision_groups = []
    words_in_collision_count = 0

    for signature, words_list in ngram_signature_map.items():
        if len(words_list) > 1:
            collision_groups.append(words_list)
            words_in_collision_count += len(words_list)

    # 4. Recomptes
    total_unique_words = len(unique_words)

    # 5. Càlcul del percentatge de col·lisió
    if total_unique_words > 0:
        percentatge = (words_in_collision_count / total_unique_words) * 100
    else:
        percentatge = 0.0

    return unique_words, ngram_signature_map, collision_groups, percentatge


def split_dataset(texts, labels, test_size=0.2, random_state=42):
    """
    Divideix el dataset en conjunts d'entrenament i prova.

    Args:
        texts (list[str]): Llista de textos.
        labels (list[str]): Llista d'etiquetes corresponents.
        test_size (float): Proporció del conjunt de prova.
        random_state (int): Estat aleatori per a la reproduïbilitat.

    Returns:
        X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


def extract_text_open_bigrams(text):
    """
    Extreu els bigrames oberts (open bigrams) d’un text.

    Args:
        - text (str): Text d’entrada del qual es volen extreure els bigrames.

    Returns:
        - list: Llista de bigrames oberts únics presents al text.
                Cada element correspon a un bigrama generat a partir de les paraules.
    """
    
    # 1. Normalització i extracció de paraules
    text_lower = text.lower()
    words_in_text = re.findall(r'\b\w+\b', text_lower)
    
    # 2. Unió dels conjunts de bigrames de totes les paraules
    text_bigrams_set = set()
    for word in words_in_text:
        # Cridem la funció feta anteriorment (n=2 per bigrames)
        ngrams = get_open_ngrams(word, n=2, include_boundaries=True)
        text_bigrams_set.update(ngrams)
        
    # 3. Retornem com a llista
    return list(text_bigrams_set)


def extract_ngrams_from_subset(X_train_test):
    """
    Extreu els bigrames oberts de cada text en un subconjunt.
    Args:
        - X_train_test (list[str]): Llista de textos del subconjunt d'entrenament o prova.
    Returns:
        - X_train_ngrams (list[list[str]]): Llista de llistes de bigrames per cada text.
    """
    
    X_ngrams = []

    for text in X_train_test:
        bigrams_list = extract_text_open_bigrams(text)
        X_ngrams.append(bigrams_list)

    return X_ngrams
    # print(f"Exemple del primer text (primers 10 bigrames): {X_ngrams[0][:10]}")


class MultinomialNB:
    """
    Implementació d'un classificador ingenu de Bayes multimonial
    amb n-grames oberts i TF-IDF.
    """

    def __init__(self, alpha: float = 1.0):
        """
        Inicialitza el model.

        Args:
            alpha (float): Paràmetre de suavitzat de Laplace.
        """
        self.alpha = alpha
        self.languages = None           # classes
        self.Pprior = None             # log P(c)
        self.Pconditional = None       # log P(w_i | c)
        self.vectorizer = None         # TfidfVectorizer
        self._trained = False          # flag intern

    
    # N-GRAMES OBERTS I PREPROCESSAT

    def get_open_ngrams(word: str, n: int, include_boundaries: bool = True) -> set:
        """
        Genera un conjunt d’n-grams oberts per a una paraula.

        Args:
            word (str): Paraula d’entrada.
            n (int): Ordre de l’n-gram.
            include_boundaries (bool): Si és True, afegeix '_' al principi i al final.

        Returns:
            set[str]: Conjunt d’n-grams oberts.
        """
        open_ngrams = set()

        # N-grams interns
        if len(word) >= n:
            for ngram_tuple in combinations(word, n):
                open_ngrams.add("".join(ngram_tuple))

        if include_boundaries:
            if n == 1:
                open_ngrams.add("_")
            else:
                first_char = word[0]
                rest = word[1:]
                if len(rest) >= (n - 2):
                    for sub_combo in combinations(rest, n - 2):
                        open_ngrams.add("_" + first_char + "".join(sub_combo))

                last_char = word[-1]
                start = word[:-1]
                if len(start) >= (n - 2):
                    for sub_combo in combinations(start, n - 2):
                        open_ngrams.add("".join(sub_combo) + last_char + "_")

        return open_ngrams

    def get_all_ngrams_from_text(self, text_list, n: int = 2,
                                 include_boundaries: bool = True):
        """
        Converteix una llista de textos en una llista de llistes d’n-grams oberts.

        Args:
            text_list (list[str]): Textos d’entrada.
            n (int): Ordre de l’n-gram.
            include_boundaries (bool): Si s’han d’incloure límits.

        Returns:
            list[list[str]]: Llista de llistes d’n-grams per text.
        """
        processed_texts = []

        for text in text_list:
            if not isinstance(text, str):
                processed_texts.append([])
                continue

            words = re.findall(r'\b\w+\b', text.lower())
            ngrams_set = set()
            for word in words:
                ngrams_set.update(self.get_open_ngrams(word, n, include_boundaries))

            processed_texts.append(list(ngrams_set))

        return processed_texts


    # CARREGA DATASET

    
    def get_text_label_from_df(path: str = 'dataset.csv'):
        """
        Llegeix dataset.csv i retorna textos i etiquetes.

        Args:
            path (str): Ruta al CSV.

        Returns:
            (list[str], list[str]): texts, labels
        """
        df = pd.read_csv(path)
        texts = df['Text'].tolist()
        labels = df['language'].tolist()
        return texts, labels


    # ENTRENAMENT COMPLET (PIPELINE)

    def do_model(self, path: str = 'dataset.csv', n: int = 2,
                       test_size: float = 0.2, random_state: int = 42):
        """
        Llegeix el dataset, genera n-grams, vectoritza amb TF-IDF i entrena el model.

        Guarda:
            - self.vectorizer
            - self.languages, self.Pprior, self.Pconditional

        Retorna:
            X_train_tfidf, X_test_tfidf, y_train, y_test
        """
        texts, labels = self.get_text_label_from_df(path)

        # Split
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            texts, labels, test_size=test_size, random_state=random_state
        )

        # Extraure n-grams
        X_train_ngrams = self.get_all_ngrams_from_text(X_train_raw, n=n,
                                                       include_boundaries=True)
        X_test_ngrams = self.get_all_ngrams_from_text(X_test_raw, n=n,
                                                      include_boundaries=True)

        # Vectorització TF-IDF
        if self.vectorizer is None:
            self.vectorizer = TfidfVectorizer(analyzer=lambda x: x)

        X_train_tfidf = self.vectorizer.fit_transform(X_train_ngrams)
        X_test_tfidf = self.vectorizer.transform(X_test_ngrams)

        # Entrenament MultinomialNB
        self.MultinomialNBfit(X_train_tfidf, y_train)
        self._trained = True

        return X_train_tfidf, X_test_tfidf, np.array(y_train), np.array(y_test)


    # FIT I PREDICT DEL MULTINOMIAL

    def MultinomialNBfit(self, X, y):
        """
        Entrena el model Multinomial Naive Bayes amb suavitzat de Laplace.

        Args:
            X (sparse matrix): Característiques TF-IDF.
            y (array-like): Etiquetes.

        Returns:
            self
        """
        y = np.array(y)

        if self.languages is None:
            self.languages = np.unique(y)

        num_lang = len(self.languages)
        num_ngram = X.shape[1]

        if self.Pprior is None:
            self.Pprior = np.zeros(num_lang, dtype=np.float32)
        if self.Pconditional is None:
            self.Pconditional = np.zeros((num_lang, num_ngram), dtype=np.float32)

        for idx, lang in enumerate(self.languages):
            # Màscara per seleccionar documents d'aquest idioma
            mask = (y == lang)

            # Càlcul de P(c)
            n_docs_class = np.sum(mask)
            self.Pprior[idx] = np.log(n_docs_class / len(y))

            # Càlcul de P(w_i | c)
            n_ngram_class = np.array(X[mask, :].sum(axis=0)).flatten()
            total_ngram_class = n_ngram_class.sum()

            numerador = np.log(n_ngram_class + self.alpha)
            denominador = np.log(total_ngram_class + num_ngram * self.alpha)

            self.Pconditional[idx, :] = numerador - denominador

        return self

    def MultinomialNBpredict(self, X):
        """
        Prediu etiquetes per a una matriu de mostres X.

        Args:
            X (sparse matrix): Característiques TF-IDF.

        Returns:
            np.ndarray: Etiquetes predites.
        """
        # log P(c) + sum_i x_i log P(w_i | c)
        scores = X @ self.Pconditional.T
        scores += self.Pprior

        best_idx = np.argmax(scores, axis=1)
        return self.languages[best_idx]


    # PREDICCIÓ D’UN SOL TEXT

    def predict_language(self, text: str, path: str = 'dataset.csv',
                         n: int = 2) -> str:
        """
        Prediu l'idioma d'un sol text.
        Si el model no està entrenat, entrena primer amb el dataset.

        Args:
            text (str): Text a classificar.

        Returns:
            str: Idioma predit.
        """
        if not self._trained or self.vectorizer is None:
            self.do_model(path=path, n=n)

        ngrams = self.get_all_ngrams_from_text([text], n=n,
                                               include_boundaries=True)
        X = self.vectorizer.transform(ngrams)
        pred = self.MultinomialNBpredict(X)
        return pred[0]


    # INFORMACIÓ SOBRE BIGRAMES / N-GRAMES
    
    def get_top_ngrams(self, X, top_k: int = 5):
        """
        Retorna els top_k n-grams globals més freqüents al conjunt X.

        Args:
            X (sparse matrix): Característiques (TF-IDF o comptatges).
            top_k (int): Nombre de n-grams.

        Returns:
            list[tuple[str, float]]: (ngram, suma_valor) ordenats descendentment.
        """
        if self.vectorizer is None:
            raise ValueError("Cal tenir un vectorizer entrenat per mapejar n-grams.")

        vocab = np.array(self.vectorizer.get_feature_names_out())
        sums = np.array(X.sum(axis=0)).flatten()
        idx_sorted = np.argsort(-sums)[:top_k]

        return [(vocab[i], float(sums[i])) for i in idx_sorted]

    def get_ngrams_percentages(self, X):
        """
        Calcula el percentatge d’aparició de cada n-gram al conjunt X.

        Args:
            X (sparse matrix)

        Returns:
            list[tuple[str, float]]: (ngram, percentatge) ordenats descendentment.
        """
        if self.vectorizer is None:
            raise ValueError("Cal tenir un vectorizer entrenat per mapejar n-grams.")

        vocab = np.array(self.vectorizer.get_feature_names_out())
        sums = np.array(X.sum(axis=0)).flatten()
        total = sums.sum()
        if total == 0:
            return []

        percentages = sums / total * 100.0
        idx_sorted = np.argsort(-percentages)

        return [(vocab[i], float(percentages[i])) for i in idx_sorted]

    def find_colliding_words(self, words, n: int = 2,
                             include_boundaries: bool = True,
                             max_groups: int = 5):
        """
        Troba grups de paraules que tenen EXACTAMENT el mateix conjunt d’n-grams.

        Args:
            words (iterable[str]): Paraules úniques.
            n (int): Ordre de l’n-gram.
            include_boundaries (bool): Si s'inclouen límits.
            max_groups (int): Màxim de grups a retornar.

        Returns:
            list[list[str]]: Cada element és la llista de paraules en col·lisió.
        """
        sig_map = defaultdict(list)

        for w in words:
            ngrams_set = self.get_open_ngrams(w, n, include_boundaries)
            sig = tuple(sorted(list(ngrams_set)))
            sig_map[sig].append(w)

        collision_groups = [group for group in sig_map.values() if len(group) > 1]

        return collision_groups[:max_groups]


    # ACCURACY AMB DIFERENTS MÈTODES
    def accuracy_manual(y_true, y_pred) -> float:
        """
        Calcula accuracy manualment (percentatge d'encerts).
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        return float(np.mean(y_true == y_pred))

    def accuracy_sklearn(y_true, y_pred) -> float:
        """
        Calcula accuracy amb sklearn.metrics.accuracy_score.
        """
        return float(accuracy_score(y_true, y_pred))

    def evaluate(self, X_test, y_test):
        """
        Calcula l'accuracy de test amb dos mètodes i la retorna.

        Returns:
            dict: {'manual': acc1, 'sklearn': acc2}
        """
        y_pred = self.MultinomialNBpredict(X_test)
        acc_manual = self.accuracy_manual(y_test, y_pred)
        acc_skl = self.accuracy_sklearn(y_test, y_pred)
        return {
            'manual': acc_manual,
            'sklearn': acc_skl
        }


    # EXPERIMENT: DROP D’UN IDIOMA

    def experiment_drop_language(self, drop_lang: str,
                                 path: str = 'dataset.csv',
                                 n: int = 2,
                                 test_size: float = 0.2,
                                 random_state: int = 42):
        """
        Entrena i avalua el model amb i sense un idioma determinat.

        Retorna:
            dict amb accuracy abans i després de treure l’idioma.
        """
        # 1) Entenament normal
        X_train_full, X_test_full, y_train_full, y_test_full = self.do_model(
            path=path, n=n, test_size=test_size, random_state=random_state
        )
        res_full = self.evaluate(X_test_full, y_test_full)

        # 2) Elimina l’idioma drop_lang del dataset i torna a entrenar
        texts, labels = self.get_text_label_from_df(path)
        texts_drop = []
        labels_drop = []
        for t, l in zip(texts, labels):
            if l != drop_lang:
                texts_drop.append(t)
                labels_drop.append(l)

        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            texts_drop, labels_drop, test_size=test_size, random_state=random_state
        )

        X_train_ngrams = self.get_all_ngrams_from_text(X_train_raw, n=n,
                                                       include_boundaries=True)
        X_test_ngrams = self.get_all_ngrams_from_text(X_test_raw, n=n,
                                                      include_boundaries=True)

        # Nou vectorizer per aquest experiment
        vec = TfidfVectorizer(analyzer=lambda x: x)
        X_train_tfidf = vec.fit_transform(X_train_ngrams)
        X_test_tfidf = vec.transform(X_test_ngrams)

        # Guardem estat antic
        old_vec = self.vectorizer
        old_Pprior = self.Pprior
        old_Pcond = self.Pconditional
        old_langs = self.languages

        # Reentrenem sobre el conjunt reduït
        self.vectorizer = vec
        self.languages = None
        self.Pprior = None
        self.Pconditional = None
        self.MultinomialNBfit(X_train_tfidf, y_train)
        res_drop = self.evaluate(X_test_tfidf, y_test)

        # Restaurem estat del model “principal”
        self.vectorizer = old_vec
        self.Pprior = old_Pprior
        self.Pconditional = old_Pcond
        self.languages = old_langs

        return {
            'with_lang': res_full,
            'without_lang': res_drop
        }
    

    # EXPERIMENT: MATRIU DE CONFUSIÓ
    
    def get_confusion_matrix(self, y_true, y_pred):
        """
        Genera una matriu de confusió com a DataFrame de pandas.
        Dius quins idiomes es confonen més entre ells.
        Args:
            y_true (array-like): Etiquetes reals.
            y_pred (array-like): Etiquetes predites.
        Returns:
            pd.DataFrame: Matriu de confusió amb índex i columnes etiquetades.
        """
        cm = confusion_matrix(y_true, y_pred, labels=self.languages)
        df_cm = pd.DataFrame(cm, index=self.languages, columns=self.languages)
        return df_cm

    # EXPERIMENT: Recall i Precision per IDIOMA
    
    def get_classification_report(self, y_true, y_pred):
        """
        Genera un informe de clasificación amb Precision, Recall i F1-Score per idioma.
        Per mirar Recall i Precision de cada idioma.
        Args:
            y_true (array-like): Etiquetes reals.
            y_pred (array-like): Etiquetes predites.
        Returns:
            str: Informe de classificació en format de text.
        """
        return classification_report(y_true, y_pred, target_names=self.languages)


    # EXPERIMENT: EXPLICAR PREDICCIÓ
    
    def explain_prediction(self, text, n=2, top_k=5):
        """
        Explica per què el model ha pres una decisió.
        Mostra quins n-grames han aportat més punts a la classe guanyadora.
        Args:
            text (str): Text a explicar.
            n (int): Ordre dels n-grams.
            top_k (int): Nombre d’n-grams més determinants a mostrar.
        Returns:
            str: Missatge si el model no està entrenat.
        """
        if not self._trained: return "Model no entrenat"

        # 1. Obtenir features i scores
        ngrams_list = self.get_all_ngrams_from_text([text], n=n)[0]
        # Mapa n-grama -> index vectorizer
        vocab = self.vectorizer.vocabulary_
        
        # Filtrem només els n-grames que existeixen al vocabulari
        known_ngrams = [ng for ng in ngrams_list if ng in vocab]
        indices = [vocab[ng] for ng in known_ngrams]
        
        # Predicció
        X = self.vectorizer.transform([ngrams_list])
        scores = (X @ self.Pconditional.T + self.Pprior).flatten()
        winner_idx = np.argmax(scores)
        winner_lang = self.languages[winner_idx]
        
        # Contribució de cada paraula a la classe guanyadora
        # Pes = P(w|Winner)
        contributions = []
        for ng, idx in zip(known_ngrams, indices):
            score = self.Pconditional[winner_idx, idx]
            contributions.append((ng, score))
            
        # Ordenar per score (més alt = menys negatiu = més probabilitat)
        contributions.sort(key=lambda x: x[1], reverse=True)
        
        return {
            'predicted_language': winner_lang,
            'top_ngrams': contributions[:top_k]
        }
    

    # EXPERIMENT: TOP N-GRAMS PER IDIOMA
    
    def get_top_features_per_language(self, top_k=5):
        """
        Mostra els n-grams més representatius per a cada idioma segons P(w_i | c).
        Args:
            top_k (int): Nombre d’n-grams a mostrar per idioma.
        Returns:
            pd.DataFrame: DataFrame amb els top n-grams per idioma.
        """
        feature_names = np.array(self.vectorizer.get_feature_names_out())
        
        results = {}
        for i, lang in enumerate(self.languages):
            # Ordenem els índexs segons la probabilitat condicional (descendent)
            top_indices = np.argsort(self.Pconditional[i])[-top_k:][::-1]
            top_features = feature_names[top_indices]
            results[lang] = top_features
            
        return pd.DataFrame(results)

def man_regex():
    text = """
            Regular expressions (regex) are patterns to match, search, and manipulate text. Below is a compact “what it does + example” summary of the core pieces you’ll use 95% of the time.  
                ## Literal characters
                
                - `abc` – Matches the exact substring “abc”.  
                  - Example: Pattern `cat` matches “cat” in “my cat is here”.[1][2]
                
                ## Character classes
                
                - `[abc]` – Any one of `a`, `b`, or `c`.  
                  - Example: `[cb]at` matches “cat” and “bat”.[3]
                - `[^abc]` – Any character except `a`, `b`, or `c`.  
                  - Example: `[^0-9]` matches any non-digit.[3]
                - `[a-z]` – Any lowercase letter from a to z (ranges).  
                  - Example: `[A-Z][a-z]+` matches “Hello”.[4][3]
                - Shorthands:  
                  - `\d` digit, `\D` non-digit.  
                  - `\w` word char (letters, digits, `_`), `\W` non-word.  
                  - `\s` whitespace, `\S` non-whitespace.  
                  - Example: `\d\d/\d\d/\d{4}` matches dates like “12/03/2025”.[5][4][3]
                
                ## Quantifiers (how many times)
                
                - `a*` – 0 or more `a`.  
                  - Example: `ba*` matches “b”, “ba”, “baaa”.[6][7][3]
                - `a+` – 1 or more `a`.  
                  - Example: `ba+` matches “ba”, “baaa” but not “b”.[7][6][3]
                - `a?` – 0 or 1 `a` (optional).  
                  - Example: `colou?r` matches “color” and “colour”.[6][7]
                - `a{3}` – Exactly 3 `a`.  
                  - Example: `a{3}` matches “aaa” in “baaaac”.[6][3]
                - `a{2,4}` – Between 2 and 4 `a`.  
                  - Example: `a{2,4}` matches “aa”, “aaa” or “aaaa”.[3][6]
                
                ## Anchors (positions)
                
                - `^` – Start of string/line.  
                  - Example: `^Hello` matches “Hello world” but not “He said Hello”.[7][3]
                - `$` – End of string/line.  
                  - Example: `world$` matches “Hello world” but not “world says hi”.[7][3]
                - `\b` – Word boundary.  
                  - Example: `\bcat\b` matches “cat” but not “scatter”.[3]
                
                ## Groups and alternation
                
                - `(abc)` – Capturing group, treats `abc` as a unit and stores the match.  
                  - Example: `(ab)+` matches “ab”, “abab”, “ababab”.[4][3]
                - `(?:abc)` – Non‑capturing group (groups for logic, not for capture).  
                  - Example: `(?:ab)+c` matches “abc”, “ababc”.[3]
                - `a|b` – Alternation, matches `a` or `b`.  
                  - Example: `cat|dog` matches “cat” or “dog”. [8][5]  
                
                ## Dot and escaping
                
                - `.` – Any single character except newline (engine‑dependent).  
                  - Example: `a.c` matches “abc”, “a-c”, “a c”.[8][5]
                - `\` – Escape to treat a special char as literal.  
                  - Example: `\.` matches a literal dot in “example.com”.[9][3]
                
                ## Lookaround (advanced but useful)
                
                - `(?=...)` – Positive lookahead: must be followed by `...`, but does not consume it.  
                  - Example: `\d+(?=px)` matches “12” in “12px”.[3]
                - `(?!...)` – Negative lookahead: must not be followed by `...`.  
                  - Example: `\d+(?!px)` matches “12” in “12kg” but not in “12px”.[3]
                - `(?<=...)` / `(?<!...)` – Positive/negative lookbehind (similar but “before” the match).  
                  - Example: `(?<=\$)\d+` matches “100” in “$100”.[3]
                
                ## Tiny end‑to‑end examples
                
                - Email-ish: `\w+@\w+\.\w+` matches basic emails like “user@mail.com”.[10]
                - Simple IPv4: `\d{1,3}(\.\d{1,3}){3}` matches “192.168.0.1”.[11][5]
            
            """
    return textwrap.dedent(text).strip()

# Exemple d’ús bàsic (només si s’executa com a script)
def class_run(path='dataset.csv'):
    model = MultinomialNB()

    # Entrenar pipeline complet
    X_train, X_test, y_train, y_test = model.do_model(
        path=path,
        n=2,
        test_size=0.2,
        random_state=42
    )

    # Accuracy amb diferents mètodes
    results = model.evaluate(X_test, y_test)
    print(f"Accuracy (manual):  {results['manual'] * 100:.2f}%")
    print(f"Accuracy (sklearn): {results['sklearn'] * 100:.2f}%")

    # Top 5 bigrames globals al test
    top_bi = model.get_top_ngrams(X_test, top_k=5)
    print("\nTop 5 bigrames (test):")
    for bg, val in top_bi:
        print(f"{bg}: {val:.4f}")

    # Percentatges d’ús de bigrames
    perc = model.get_ngrams_percentages(X_test)[:5]
    print("\nTop 5 bigrames per percentatge (test):")
    for bg, p in perc:
        print(f"{bg}: {p:.4f}%")

    # Exemple de col·lisions de paraules en bigrames
    texts_all, labels_all = model.get_text_label_from_df('dataset.csv')
    unique_words = set()
    for txt in texts_all:
        if isinstance(txt, str):
            unique_words.update(re.findall(r'\b\w+\b', txt.lower()))

    collisions = model.find_colliding_words(unique_words, n=2,
                                            include_boundaries=True,
                                            max_groups=5)
    print("\nExemples de paraules en col·lisió (mateixos bigrames):")
    for i, group in enumerate(collisions, 1):
        print(f"{i}. {group}")

    # Experiment dropejant l'anglès (o qualsevol altre idioma)
    drop_res = model.experiment_drop_language('English', path='dataset.csv')
    print("\nExperiment drop English:")
    print(f"  Amb English  -> manual:  {drop_res['with_lang']['manual']*100:.2f}%")
    print(f"  Sense English -> manual: {drop_res['without_lang']['manual']*100:.2f}%")

    # Predicció d’un sol text
    print("\nPrediccions de textos individuals:")
    lang = model.predict_language("This is a test sentence.")
    print("This is a test sentence.")
    print(lang)

    lang2 = model.predict_language("Saya suka kacang-kacangan dengan chorizo dan saus ikan kod")
    print("Saya suka kacang-kacangan dengan chorizo dan saus ikan kod")
    print(lang2)

    # Experiment matriu de confusió
    y_pred = model.MultinomialNBpredict(X_test)
    cm_df = model.get_confusion_matrix(y_test, y_pred)
    # Mostrar top 5 idiomes amb més confusions
    print("\nMatriu de confusió (top 5 idiomes):")
    print(cm_df.head())

    # Experiment classification report
    report = model.get_classification_report(y_test, y_pred)
    print("\nInforme de classificació:")
    print(report)

    # Experiment explicar predicció
    explanation = model.explain_prediction("This is a test sentence.", n=2, top_k=5)
    print("\nExplicació de la predicció:")
    print(f"Idioma predit: {explanation['predicted_language']}")
    print("Top n-grams que van influir en la decisió:")
    for ng, score in explanation['top_ngrams']:
        print(f"{ng}: {score:.4f}")
    
    # Experiment top n-grams per idioma
    top_features_df = model.get_top_features_per_language(top_k=5)
    print("\nTop n-grams per idioma:")
    print(top_features_df.head())

"""
Electric bugaloo
"""

"""
    def get_confusion_matrix(self, y_true= , y_pred):
        
        Genera una matriu de confusió com a DataFrame de pandas.
        Dius quins idiomes es confonen més entre ells.
        Args:
            y_true (array-like): Etiquetes reals.
            y_pred (array-like): Etiquetes predites.
        Returns:
            pd.DataFrame: Matriu de confusió amb índex i columnes etiquetades.

        
        cm = confusion_matrix(y_true, y_pred, labels=self.languages)
        df_cm = pd.DataFrame(cm, index=self.languages, columns=self.languages)
        return df_cm

model = do_model()
cm_df = model.get_confusion_matrix(y_test, y_pred)

"""

def most_confused_language(df_cm):
    
    confusion_per_language = df_cm.sum(axis=1) - df_cm.values.diagonal()
    return confusion_per_language.sort_values(ascending=False)

"""
Pot ser més confos per les poques dades que tenim, si volem algo més 
constant, creible o simplement no dependent de la població hem de normalitzar
"""
def confusion_rate(df_cm):
    normalized = df_cm.div(df_cm.sum(axis=1), axis=0)
    confusion_rate = normalized.sum(axis=1) - normalized.values.diagonal()
    return confusion_rate.sort_values(ascending=False)


def compare_alphas(alphas=[0.1, 1.0, 5.0], path='dataset.csv'):
    """
    Entrena i avalua 3 models MultinomialNB amb diferents alphas.
    
    Args:
        alphas (list): Llista de 3 valors float per al paràmetre alpha.
        path (str): Ruta al fitxer de dades.
    """
    results = {}

    for alpha in alphas:
        model = MultinomialNB(alpha=alpha)
        
        # Fixem random_state=42 per assegurar que tots els models
        # veuen exactament el mateix conjunt de Train i Test.
        _, X_test, _, y_test = model.do_model(path=path, test_size=0.2, random_state=42)
        
        metrics = model.evaluate(X_test, y_test)
        acc = metrics['sklearn']
        results[alpha] = acc
        
        print(f"🔹 Model Alpha = {alpha:<4} | Accuracy: {acc*100:.2f}%")

    best_alpha = max(results, key=results.get)
    print(f"\nMillor Alpha: {best_alpha} amb {results[best_alpha]*100:.2f}% d'encert.")
