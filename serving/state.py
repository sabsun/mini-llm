from serving.generator import TextGenerator
from serving.loader import ModelLoader


class AppState:
    """
    Shared application state.

    Everything loaded once during server startup.
    """

    def __init__(self, loader: ModelLoader):
        self.loader = loader
        self.generator = TextGenerator(loader)