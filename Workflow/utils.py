from string import ascii_letters, digits
from random import choice


def random_string(n):
    lettersdigits = ascii_letters + digits
    my_list = [choice(lettersdigits) for _ in range(n)]
    my_str = ''.join(my_list)
    return my_str
