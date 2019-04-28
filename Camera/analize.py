import boto3
import json
import time
import requests
import cv2
import os
import numpy as np
import uuid
from faced import FaceDetector
from utils import *
from faced.utils import annotate_image
from threading import Thread

