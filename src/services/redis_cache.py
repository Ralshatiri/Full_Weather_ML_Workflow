import json # used to conert to json strings so redis can store 
import hashlib # used to hash input data
import redis 
from src.config import REDIS_HOST,REDIS_DB,REDIS_PORT,REDIS_TTL_SECONDS,JOB_TTL_SECONDS

redis_client = None



# This function tries to connect to redis 

def connect_redis():

    global redis_client

    if not REDIS_HOST:
        print("REDIS_HOST is not set. Redis cache disabled.")
        redis_client = None
        return None
    
    try:

# create redis connection
        redis_client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            decode_responses=True 
        )

# sends small test redis to make sure connection works
        redis_client.ping()
        print("Connected to Redis successfully.")
        return redis_client
    except Exception as e:
        print(f"Redis unavailable. Continuing without cache. Error: {str(e)}")
        redis_client = None
        return None

# creates a unique redis key for a prediction input
def make_cache_key(prefix: str, data: dict) -> str:

    # this coverts the dictionary to json string
    normalized_data = json.dumps(data, sort_keys=True, default=str)

    # creates hash from input
    hashed_data = hashlib.sha256(normalized_data.encode()).hexdigest()
    return f"{prefix}:{hashed_data}"

# this checks redis to see if a saved prediction exists
def get_cached_prediction(cache_key: str):

    if redis_client is None:
        return None
    
    try:

        # checks if there is a value saved under this key
        cached_value = redis_client.get(cache_key)

        # if there is a cached value the return it as py dict
        if cached_value:
            print(f"cached hit: {cache_key}")
            return json.loads(cached_value)
        
        print(f"Cache miss: {cache_key}")
        return None
    
    except Exception as e:
        print(f"Failed to read from Redis cache: {str(e)}")
        return None

# Here we save model prediction to redis 
def save_prediction_to_cache(cache_key: str, response: dict):
    
    if redis_client is None:
        return
    
    try:

        # this saves prediction to redis
        redis_client.setex(
            cache_key,
            REDIS_TTL_SECONDS,
            json.dumps(response)
        )

        print(f"Saved prediction to Redis cache: {cache_key}")

    except Exception as e:
        print(f"Failed to save prediction to Redis cache: {str(e)}")


def save_job_status(job_id: str, status_data: dict):
    if redis_client is None:
        raise RuntimeError("Redis is required for queued jobs but is not connected.")

    job_key = f"job:{job_id}"

    redis_client.setex(
        job_key,
        JOB_TTL_SECONDS,
        json.dumps(status_data)
    )


def get_job_status(job_id: str):
    if redis_client is None:
        raise RuntimeError("Redis is required for queued jobs but is not connected.")

    job_key = f"job:{job_id}"
    value = redis_client.get(job_key)

    if not value:
        return None

    return json.loads(value)