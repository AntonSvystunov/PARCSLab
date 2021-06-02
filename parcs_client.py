import time
import requests
from requests_toolbelt.multipart.encoder import MultipartEncoder

class ParcsClient:
    def __init__(self, master_ip, master_port, scheme='http'):
        self.service_url = scheme + '://' + master_ip + ':' + master_port

    def get_workers(self):
        response = requests.get(self.service_url + '/api/worker')
        if response.status_code != 200:
            raise Exception('Unable to fetch workers')
        return response.json()['workers']

    def get_worker(self, worker_id):
        response = requests.get(self.service_url + '/api/worker/' + str(worker_id))
        if response.status_code != 200:
            raise Exception('Unable to fetch worker: ' + str(worker_id))
        return response.json()

    def get_jobs(self):
        response = requests.get(self.service_url + '/api/job/')
        if response.status_code != 200:
            raise Exception('Unable to fetch jobs')
        return response.json()

    def get_job(self, job_id):
        response = requests.get(self.service_url + '/api/job/' + str(job_id))
        if response.status_code != 200:
            raise Exception('Unable to fetch job: ' + str(job_id))
        return response.json()

    def send_job(self, job_name, solution_file_name, input_file_name):
        payload = MultipartEncoder(
            fields={
                'job_name': job_name,
                'solution_file': ('solution.py', open(solution_file_name, 'rb'), 'text/plain'),
                'input_file': ('input.json', open(input_file_name, 'rb'), 'text/plain')
            }
        )

        response = requests.post(self.service_url + '/api/job', data=payload, headers={'Content-Type': payload.content_type})
        return response.json()

    def download_file(self, job_id, file_name):
        url = self.service_url + '/api/job/' + str(job_id) + '/' + str(file_name)
        response = requests.get(url)
        if response.status_code != 200:
            raise Exception('Unable to fetch job file: ' + str(job_id))
        return response.text

    def run_job(self, job_name, solution_file, data_file):
        job_info = self.send_job(job_name, solution_file, data_file)
        
        while True:
            try:
                job_result = self.download_file(job_info['id'], 'output')
                break
            except:
                time.sleep(1)

        job_info = self.get_job(job_info['id'])

        return job_info, job_result