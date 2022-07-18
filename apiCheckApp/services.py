import logging

logger = logging.getLogger(__name__)  # the __name__ resolve to "uicheckapp.services"


# This will load the apiCheckApp logger

class EchoService:
    @staticmethod
    def echo(msg):
        logger.info("echoing from apiCheckApp logger:")
        print(msg)
