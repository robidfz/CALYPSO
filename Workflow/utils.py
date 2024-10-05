from string import ascii_letters, digits
from random import choice
import matplotlib.pyplot as plt


def random_string(n):
    lettersdigits = ascii_letters + digits
    my_list = [choice(lettersdigits) for _ in range(n)]
    my_str = ''.join(my_list)
    return my_str


import re


def file_to_dict(file_path):
    result_dict = {}
    with open(file_path, 'r') as file:
        for line in file:
            match = re.search(r"Esperimento \('(\d+)', '(\d+)'\) eseguito con il file: .*\\(TEST1_[\w]+)", line)
            if match:
                # Estrai i due numeri e il nome del file
                num1, num2, file_name = match.groups()
                # Aggiungi al dizionario il nome del file come chiave e la tupla dei numeri come valore
                result_dict[file_name] = (num1, num2)

    return result_dict


def generate_colors(n):
    cmap = plt.get_cmap('tab10')  # Puoi cambiare la palette (ad es. 'viridis', 'tab20', ecc.)
    return [cmap(i % cmap.N) for i in range(n)]


