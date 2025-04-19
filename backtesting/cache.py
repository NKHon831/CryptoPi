import time

class L1Cache:
    def __init__(self, ttl_seconds=3600):
        self.cache = {}
        self.ttl_seconds = ttl_seconds
        self.hits = 0
        self.misses = 0

    def is_expired(self, key):
        """Check if the cache for the key is expired."""
        if key in self.cache:
            cached_time = self.cache[key]["timestamp"]
            current_time = time.time()
            if current_time - cached_time > self.ttl_seconds:
                print(f"🕒 Cache expired for key: {key}")
                return True
        return False

    def get(self, key):
        entry = self.cache.get(key)
        print(f"Attempting to get data for key: {key}")
        if entry:
            data, timestamp = entry
            if time.time() - timestamp < self.ttl_seconds:
                self.hits += 1
                print(f"✅ Cache hit for key: {key}")
                return data
            else:
                print(f"🕒 Cache expired for key: {key}")
                del self.cache[key]
        self.misses += 1
        print(f"❌ Cache miss for key: {key}")
        return None

    def set(self, key, value):
        print(f"Setting cache for key {key}")
        self.cache[key] = (value, time.time())
        print(f"💾 Cache updated for key: {key}")

    def print_cache(self):
        """Prints all the data in the cache."""
        if not self.cache:
            print("Cache is empty.")
        else:
            for key, value in self.cache.items():
                data, timestamp = value
                print(f"Key: {key}, Data: {data.head()}, Timestamp: {timestamp}")

    def get_cache_metrics(self):
        return {"hits": self.hits, "misses": self.misses}