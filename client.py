import sys
from drawresults import draw
from solver import Output
from generate_values import generateConfig
from parcs_client import ParcsClient

SERVICE_URL = 'http://localhost:8080'
DEFAULT_TEMP_FILENAME = 'points-temp.json'

def main():  
    if len(sys.argv) < 4:
        raise Exception('Not enough arguments!\n\tUSAGE:\t./client.py <num_points> <world_size> <workers_num>')

    num_points = int(sys.argv[1])
    world_size = int(sys.argv[2])
    workers_num = int(sys.argv[3])
    temp_filename = DEFAULT_TEMP_FILENAME

    input_config = generateConfig(num_points, world_size, workers_num)
    input_config.writeTo(temp_filename)

    parcs_client = ParcsClient('localhost','8080')
    jobInfo, outputFile = parcs_client.run_job('triangulation', './solver.py', temp_filename)

    world_size, points, edges, workers_num = Output.readFromString(outputFile).desctruct()
    
    #print(f'Job info: {jobInfo} Workers: {workers_num}')
    draw(world_size, points, edges)


if __name__ == '__main__':
    main()
    


