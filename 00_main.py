import sys
from Workflow.execution import execute

if __name__ == "__main__":
    if len(sys.argv) == 7:
        t = tuple(sys.argv[1:])
        execute(*t)

