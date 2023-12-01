"""Example Python client for vllm.entrypoints.api_server"""
import sys
import json
import os
import glob
import argparse
import json
from typing import Iterable, List
import requests

def post_http_request(frames: json,
                      api_url: str) -> requests.Response:
    headers = {"User-Agent": "Test Client"}
    response = requests.post(api_url, headers=headers, json=frames, stream=True)
    return response


def get_response(response: requests.Response) -> List[str]:
    return response.content


if __name__ == "__main__":
    api_url = f"http://127.0.0.1:8876/predict"

    import time
    files = glob.glob("/home/deepspeed/dislyte/test_data/*.json")
    bs = 200
    frames = []
    for fpath in files:
        fid='_'.join(fpath.split("/")[-2:])
        with open(fpath, 'r') as f:
            tmp_frames = json.load(f)
            frames.extend(tmp_frames)
            if len(frames) > bs:
                print(f"process frame num {len(frames)}")
                response = post_http_request(frames, api_url)
                act_json = get_response(response)
                print(act_json)
                frames = []
