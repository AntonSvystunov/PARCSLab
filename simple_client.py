import time
from solver import Input, Output, World, lexigraphic_sort, triangulate


class SimpleClient:
    def run_job(self, job_name, solution_file, data_file):
        start = time.time()
        world_size, points, workers_num = self.read_input(data_file)
        world = World(world_size)

        # start = time.time()
        job_result = triangulate(points)
        elapsed = time.time() - start

        job_info = {
            'name': job_name,
            'duration': elapsed
        }
        
        return job_info, job_result
        

    def read_input(self, data_file):
        world_size, points, workers_num = Input.readFrom(filename=data_file).desctruct()
        return world_size, lexigraphic_sort(points), workers_num