import argparse
import importlib
import timeit
import sys

import numpy as np

from problems.api import Problem
from agent.llm_optimize import LLMOptimizer

def run_agent(problem: Problem, ref_out, agent: LLMOptimizer):
    llvmir = problem.cfn_src

    print('Agent input:')
    print('\n================================')
    print(llvmir)
    print('================================\n')

    # run your agent here!
    # TODO: for now just return a copy of the original IR
    # optimized = str(llvmir)
    optimized = agent.optimize(llvmir)
    
    print('\n================================')
    print(optimized)
    print('================================\n')    
    
    # try to compile the agent-generated IR

    problem.optimize(optimized)
        # problem.ai_cfn(*ref_out)
    # after calling .optimize(), you can use "problem.ai_cfn(*ref_out)" to run your function
    # and perhaps compare it with the reference output
    # if you want to recompile, please call "problem.reset()" before calling "problem.optimize()"
    # again
    

def benchmark(fn, data):
    # return in milliseconds
    return timeit.timeit('fn(*data)', globals={ 'fn': fn, 'data': data }, number=100) * 1000


def check_the_same(a, b):
    assert a is not b
    if isinstance(a, np.ndarray):
        if a.dtype == np.float32:
            return np.allclose(a, b)
        else:
            return (a == b).all()
    return (a == b)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--problem', required=True, type=int, help='Problem number to run')
    args = parser.parse_args()

    pidx = args.problem
    try:
        pmod = importlib.import_module(f'problems.problem{pidx}')
    except:
        import traceback
        print(f'Failed to import the problem with the provided id={pidx}')
        traceback.print_exc()
        return 1

    p = pmod.problem

    ref = p.fn(*p.get_test_data())
    cref = p.cfn(*p.get_test_data())
    check_the_same(ref, cref)

    agent = LLMOptimizer()
    run_agent(p, ref, agent=agent)

    ai = p.ai_cfn(*p.get_test_data())
    if not check_the_same(cref, ai):
        raise ValueError('Output mismatch!')

    print('All outputs match. Benchmarking...')
    print('Base:', benchmark(p.fn, p.get_test_data()))
    print('Compiled:', benchmark(p.cfn, p.get_test_data()))
    print('AI-Opt:', benchmark(p.ai_cfn, p.get_test_data()))


if __name__ == '__main__':
    sys.exit(main() or 0)
