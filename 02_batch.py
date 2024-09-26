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
    number_of_CI = sys.argv[5].split(',')
    noise_inside_CI = sys.argv[6].split(',')
    experiments = list(product(number_of_CI, noise_inside_CI))
    experiments=experiments*5
    with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
        future_results = {executor.submit(execute_in_parallel, register, exp): exp for exp in experiments}

        for future in concurrent.futures.as_completed(future_results):
            experiment, new_name = future.result()
            print(f"Esperimento {experiment} eseguito con il file: {new_name}")
    with open('final_register.txt', 'w+') as dictionary_handler:
        dictionary_handler.write(str(register))


if __name__ == "__main__":
    main()


