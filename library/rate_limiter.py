import time
from collections import deque
from datetime import datetime, timedelta

class ANSIColors:
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    RED = '\033[31m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    INVERSE = '\033[7m'
    END = '\033[0m'


class RateLimiter:
    def __init__(self, requests_per_minute):
        self.rate_limit = requests_per_minute
        self.window_size = 60  # seconds
        self.requests = deque()  # will store timestamps of requests

    def wait_if_needed(self):
        current_time = datetime.now()

        # Remove timestamps older than our window
        while self.requests and (current_time - self.requests[0]).total_seconds() > self.window_size:
            self.requests.popleft()

        # If we're at or over the limit
        if len(self.requests) >= self.rate_limit:
            # Calculate when we can make the next request
            oldest_request = self.requests[0]
            wait_until = oldest_request + timedelta(seconds=self.window_size)
            wait_seconds = (wait_until - current_time).total_seconds()

            if wait_seconds > 0:
                # Print countdown
                while wait_seconds > 0:
                    print(f"\r{ANSIColors.RED}Rate limit reached. Waiting {wait_seconds:.1f} seconds...{ANSIColors.END}",
                          end='', flush=True)
                    time.sleep(0.1)  # Update every 0.1 seconds
                    wait_seconds -= 0.1
                print(f"\r{ANSIColors.GREEN}Resuming operations...                      {ANSIColors.END}")  # Clear the line

                # After waiting, clean up old timestamps again
                current_time = datetime.now()
                while self.requests and (current_time - self.requests[0]).total_seconds() > self.window_size:
                    self.requests.popleft()

        # Add current request
        self.requests.append(current_time)