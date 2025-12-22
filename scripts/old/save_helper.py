import json
import os


class Result:
    def __init__(self, data, metadata):
        self.metadata = metadata
        new_data = {}
        for key in data:
            new_data[key] = list(data[key])
        self.data = new_data

    def save(self, folder, base_filename):
        filename = base_filename
        count = 1
        while os.path.exists(os.path.join(folder, filename + '.json')):
            filename = f"{base_filename}_{count}"
            count += 1

        with open(os.path.join(folder, filename + '.json'), 'w') as file:
            json.dump({'metadata': self.metadata, 'data': self.data}, file, indent=4, sort_keys=True)

    @classmethod
    def load(self, filename):
        with open(filename, 'r') as file:
            result_dict = json.load(file)
            self.data = result_dict['data']
            self.metadata = result_dict['metadata']
        return self
