import sys
from Workflow.execution import execute
from itertools import product

from Workflow.utils import random_string

if __name__ == "__main__":
    if len(sys.argv) == 7:
        register = dict()
        parameters = list()
        # definition of the first
        number_of_CI = sys.argv[5].split(',')
        noise_inside_CI = sys.argv[6].split(',')
        experiments = list(product(number_of_CI, noise_inside_CI))
        for experiment in experiments:
            # change the name of the file
            new_name = sys.argv[4] + "_" + random_string(4)
            parameters = list(sys.argv[1:4])
            parameters.append(new_name)
            parameters.append(experiment[0])
            parameters.append(experiment[1])
            t = tuple(parameters)
            register[experiment] = new_name
            execute(*t)
            print('ciao')
        dictionary_handler = open('final_register.txt','w') #todo: trasformare in una lista di tuple e esportare in CSV
        dictionary_handler.write(register)
        dictionary_handler.close()


