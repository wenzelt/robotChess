
import logging

logger = logging.getLogger(__name__)  # the __name__ resolve to "uicheckapp.services"
                                      # This will load the uicheckapp logger

class EchoService:
  def echo(self, msg):
    logger.info("echoing something from the uicheckapp logger")
    print(msg)
