# practica3.py

import numpy as np
import pandas as pd
import re
from itertools import combinations
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


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

    # ----------------------------------------------------------------------
    # N-GRAMES OBERTS I PREPROCESSAT
    # ----------------------------------------------------------------------
    @staticmethod
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

    # ----------------------------------------------------------------------
    # CARREGA DATASET
    # ----------------------------------------------------------------------
    @staticmethod
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

    # ----------------------------------------------------------------------
    # ENTRENAMENT COMPLET (PIPELINE)
    # ----------------------------------------------------------------------
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

        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            texts, labels, test_size=test_size, random_state=random_state
        )

        X_train_ngrams = self.get_all_ngrams_from_text(X_train_raw, n=n,
                                                       include_boundaries=True)
        X_test_ngrams = self.get_all_ngrams_from_text(X_test_raw, n=n,
                                                      include_boundaries=True)

        if self.vectorizer is None:
            self.vectorizer = TfidfVectorizer(analyzer=lambda x: x)

        X_train_tfidf = self.vectorizer.fit_transform(X_train_ngrams)
        X_test_tfidf = self.vectorizer.transform(X_test_ngrams)

        self.MultinomialNBfit(X_train_tfidf, y_train)
        self._trained = True

        return X_train_tfidf, X_test_tfidf, np.array(y_train), np.array(y_test)

    # ----------------------------------------------------------------------
    # FIT I PREDICT DEL MULTINOMIAL
    # ----------------------------------------------------------------------
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
            mask = (y == lang)

            n_docs_class = np.sum(mask)
            self.Pprior[idx] = np.log(n_docs_class / len(y))

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

    # ----------------------------------------------------------------------
    # PREDICCIÓ D’UN SOL TEXT
    # ----------------------------------------------------------------------
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

    # ----------------------------------------------------------------------
    # INFORMACIÓ SOBRE BIGRAMES / N-GRAMES
    # ----------------------------------------------------------------------
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

    # ----------------------------------------------------------------------
    # ACCURACY AMB DIFERENTS MÈTODES
    # ----------------------------------------------------------------------
    @staticmethod
    def accuracy_manual(y_true, y_pred) -> float:
        """
        Calcula accuracy manualment (percentatge d'encerts).
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        return float(np.mean(y_true == y_pred))

    @staticmethod
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

    # ----------------------------------------------------------------------
    # EXPERIMENT: DROP D’UN IDIOMA
    # ----------------------------------------------------------------------
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
        X_train_full, X_test_full, y_train_full, y_test_full = self.build_pipeline(
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


# ----------------------------------------------------------------------
# Exemple d’ús bàsic (només si s’executa com a script)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    model = MultinomialNB(alpha=1.0)

    # Entrenar pipeline complet
    X_train, X_test, y_train, y_test = model.build_pipeline(
        path='dataset.csv',
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
