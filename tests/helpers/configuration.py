class MockConfigStorage:
    def __init__(self):
        self.configs = {}

    def get_available_configs(self):
        return []

    def get_config(self, camera_name):
        return None

    def save_config(self, camera_name, camera_config):
        pass
