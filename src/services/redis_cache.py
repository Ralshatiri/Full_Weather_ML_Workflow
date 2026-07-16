import json 
import hashlib 
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
        if redis_client is None:
            redis_client = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                db=REDIS_DB,
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2 
            )

# sends small test redis to make sure connection works
        redis_client.ping()
        print("Connected to Redis successfully.")
        return redis_client
    except Exception as e:
        print(f"Redis unavailable. Continuing without cache. Error: {str(e)}")
        redis_client = None
        return None
    
def get_redis_client():
    global redis_client
    if redis_client is None:
        return connect_redis();

    return redis_client

# creates a unique redis key for a prediction input
def make_cache_key(prefix: str, data: dict):

    # this coverts the dictionary to json string
    normalized_data = json.dumps(data, sort_keys=True, default=str)

    # creates hash from input
    hashed_data = hashlib.sha256(normalized_data.encode()).hexdigest()
    return f"{prefix}:{hashed_data}"

# this checks redis to see if a saved prediction exists
def get_cached_prediction(cache_key: str):

    client = get_redis_client()

    if client is None:
        return None
    
    try:

        # checks if there is a value saved under this key
        cached_value = client.get(cache_key)

        # if there is a cached value the return it as py dict
        if not cached_value:
            return None
        
        return json.loads(cached_value)
    
    except Exception as e:
        print(
             f"Redis forecast cache read failed.Continuing as a cache miss: {e}"
        )
        return None

# Here we save model prediction to redis 
def save_prediction_to_cache(cache_key: str, response: dict):
    
    client = get_redis_client()
    
    if client is None:
        return False
    
    try:
        # this saves prediction to redis
        redis_client.setex(
            cache_key,
            REDIS_TTL_SECONDS,
            json.dumps(response)
        )
        return True

    except Exception as e:
        print(
            f"Redis forecast cache write failed.Continuing without cache: {e}"
        )
        return False


def save_job_status(job_id: str, status_data: dict):

    client = get_redis_client()

    if client is None:
        return False
    
    try:

        job_key = f"job:{job_id}"

        client.setex(
            job_key,
            JOB_TTL_SECONDS,
            json.dumps(status_data)
    )
        return True
    except Exception as e:
        print(
          f"Redis job-status write failed. Continuing without Redis status: {e}"  
        )
        return False


def get_job_status(job_id: str):

    client = get_redis_client()
    if client is None:
        return None
    try:
        job_key = f"job:{job_id}"
        status_value = client.get(job_key)

        if not status_value:
            return None
        
        return json.loads(status_value)

    except Exception as e:
        print(
            f"Redis job-status read failed. Using database fallback: {e}"
        )
        return None
    
    
