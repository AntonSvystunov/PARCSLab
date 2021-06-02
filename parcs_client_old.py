import requests
import time

from bs4 import BeautifulSoup
from requests_toolbelt.multipart.encoder import MultipartEncoder

class ParcsClient:
    def __init__(self, service_url):
        self.service_url = service_url

    def send_file(self, job_name, solution_file, data_file):
        payload = MultipartEncoder(
            fields={
                'job_name': job_name,
                'input_file': ('input.json', open(data_file, 'rb'), 'text/plain'),
                'solution_file': ('solution.py', open(solution_file, 'rb'), 'text/plain')
            }
        )

        response = requests.post(self.service_url + '/api/job', data=payload, headers={'Content-Type': payload.content_type})
        return response.status_code

    def get_active_jobs(self):
        while True:
            try:
                response = requests.get(self.service_url + '/jobs')
                soup = BeautifulSoup(response.text, features="html.parser")
                return [ 
                    { 
                        'id': e.select('.i-id')[0].text.replace('#',''),
                        'name': e.select('.job-name')[0].text.strip(),
                        'duration': e.select('.i-attr-value')[1].text.strip() 
                    } for e in soup.select('[data-job-id]') 
                ]
            except:
                time.sleep(1)

    def download_output(self, job_id):
        while True:
            response = requests.get(self.service_url + '/api/job/' + f'{job_id}' + '/output')
            if response.status_code == 200:
                return response.text
            time.sleep(1)

    def get_workers(self):
        while True:
            try:
                response = requests.get(self.service_url + '/workers')
                soup = BeautifulSoup(response.text, features="html.parser")
                return [ 
                    { 
                        'id': e.select('.i-id')[0].text.replace('#',''),
                        'ip': e.select('.i-attr-value')[0].text.strip(),
                        'port': e.select('.i-attr-value')[1].text.strip(),
                        'cpu': e.select('.i-attr-value')[2].text.strip(),
                        'ram': e.select('.i-attr-value')[3].text.strip()
                    } for e in soup.select('.enabled-worker') 
                ]
            except:
                time.sleep(1)

    def run_job(self, job_name, solution_file, data_file):
        is_sent = self.send_file(job_name, solution_file, data_file)
        if (is_sent != 200):
            raise Exception('Unable to send file to server. Check params and try again.')

        job_info = self.get_active_jobs()[-1]
        if len(job_info) == 0:
            raise Exception('Unable to load jobs.')

        job_result = self.download_output(job_info['id'])
        return job_info, job_result