from Workflow.execution import execute
from Workflow.utils import random_string
import concurrent.futures
from itertools import product
import sys
import random




def execute_in_parallel(register, experiment):
    new_name = sys.argv[4] + "_" + random_string(4)
    parameters = list(sys.argv[1:4])
    parameters.append(new_name)
    parameters.append(experiment[0])
    parameters.append(experiment[1])
    t = tuple(parameters)
    register[experiment] = new_name
    execute(*t)
    return (experiment, new_name)


def main():
    register = dict()
    number_of_processes=int(sys.argv[5])
    number_of_experiments=int(sys.argv[6])
    number_of_CI = sys.argv[7].split(',')
    noise_inside_CI = sys.argv[8].split(',')

    experiments = list(product(number_of_CI, noise_inside_CI)) * number_of_experiments


    with open('final_register.txt', 'a+') as dictionary_handler:
        with concurrent.futures.ProcessPoolExecutor(max_workers=number_of_processes) as executor:
            future_results = {executor.submit(execute_in_parallel, register, exp): exp for exp in experiments}

            for future in concurrent.futures.as_completed(future_results):
                experiment, new_name = future.result()
                s = f"Esperimento {experiment} eseguito con il file: {new_name}\n"
                print(s)
                dictionary_handler.write(s)
                dictionary_handler.flush()




if __name__ == "__main__":
    main()


