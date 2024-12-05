import yaml
import importlib

with open('config.yaml') as f:
    config = yaml.safe_load(f)

scheduler_config = config['training']['scheduler']
module_path, class_name = scheduler_config['type'].rsplit('.', 1)

scheduler_module = importlib.import_module(module_path)
scheduler_class = getattr(scheduler_module, class_name)
scheduler = scheduler_class(**scheduler_config['params'])
print(scheduler)