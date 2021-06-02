from parcs_client import ParcsClient

def main():
    client = ParcsClient(master_ip='localhost',master_port='8080',scheme='http')
    info, result = client.run_job(
        job_name='triangulation', 
        solution_file='./solver.py', 
        data_file='./points-temp.json'
    )
    print(info)
    print(result)
    

if __name__ == '__main__':
    main()