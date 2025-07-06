class QuotaExceededException(Exception):
    def __init__(self, retry_after: int, message: str):
        self.retry_after = retry_after
        self.message = message
        super().__init__(self.message)