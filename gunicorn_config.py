import multiprocessing

bind = "0.0.0.0:8081"

worker_class = "gthread"
workers = (multiprocessing.cpu_count() * 2) + 1
threads = 4

# accesslog = "-"  # Log to stdout
# errorlog = "-"   # Log to stderr
loglevel = "info"

max_requests = 1000
max_requests_jitter = 50
