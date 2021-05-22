import json
import requests
import os
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse
import argparse

def submit(user_key='', file_path = '', desc=""):
    if not user_key:
        raise Exception("No UserKey" )

    f = open(file_path, 'rb')

    url = urlparse('http://ec2-13-124-161-225.ap-northeast-2.compute.amazonaws.com:8000/api/v1/competition/35/presigned_url/?description=&hyperparameters={%22training%22:{},%22inference%22:{}}')
    qs = dict(parse_qsl(url.query))
    qs['description'] = desc
    parts = url._replace(query=urlencode(qs))
    url = urlunparse(parts)

    print(url)
    headers = {
        'Authorization': user_key
    }
    res = requests.get(url, headers=headers)
    print(res.text)
    data = json.loads(res.text)
    
    submit_url = data['url']
    body = {
        'key':'app/Competitions/000035/Users/{}/Submissions/{}/output.csv'.format(str(data['submission']['user']).zfill(8),str(data['submission']['local_id']).zfill(4)),
        'x-amz-algorithm':data['fields']['x-amz-algorithm'],
        'x-amz-credential':data['fields']['x-amz-credential'],
        'x-amz-date':data['fields']['x-amz-date'],
        'policy':data['fields']['policy'],
        'x-amz-signature':data['fields']['x-amz-signature']
    }
    requests.post(url=submit_url, data=body, files={'file': f})


####################################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--folder', '-f', type=str, default='')
args = parser.parse_args()

user_key = "Bearer 841fe768880cc574158c7367d6512bf5ed115c7e" # 수정 필요 : Authorization (http://boostcamp.stages.ai/competitions/1/discussion/post/29 참고)
csv_file = f'work_dirs/{args.folder}/submission.csv'
submit(user_key, csv_file, args.folder) # 폴더 이름을 description으로 사용

# 파일 위치: code/mmdetection_trash
