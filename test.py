from generate_values import generateConfig
from simple_client import SimpleClient
from parcs_client import ParcsClient
from solver import Output
from test_runner import TestInfo, show_plot

import matplotlib.pyplot as plt

DEFAULT_JOB_NAME = 'triangulation'
DEFAULT_TEMP_FILENAME = 'points-temp.json'

class TestRunner:
    def __init__(self, job_name = DEFAULT_JOB_NAME):
        self.job_name = job_name

    def run_test(self, num_points, world_size, solution_file):
        print(f'Running test: points={num_points}')
        config = generateConfig(num_points, world_size, 1)
        config.writeTo(DEFAULT_TEMP_FILENAME)

        result = []
        
        job_info, _ = SimpleClient().run_job(self.job_name, solution_file, DEFAULT_TEMP_FILENAME)
        result.append(TestInfo('Simple',num_points,job_info['duration']))
        
        parcs_client = ParcsClient('localhost','8080')
        
        for i in range(2, len(parcs_client.get_workers()) + 1, 2):
            print(f'Running test: points={num_points} workers={i}')
            config.workers_num = i
            config.writeTo(DEFAULT_TEMP_FILENAME)
            job_info, job_output = parcs_client.run_job(self.job_name, solution_file, DEFAULT_TEMP_FILENAME)
            
            world_size, points, edges, workers_num = Output.readFromString(job_output).desctruct()
            result.append(TestInfo('Parcs' + str(config.workers_num), num_points, job_info['duration']))

        return result

def run_tests():
    test_runner = TestRunner()

    results = []
    for i in range(1000, 3000, 1000):
        results.append(test_runner.run_test(i, 2*i, './solver.py'))
    return results

test_result = run_tests()
show_plot(test_result)
