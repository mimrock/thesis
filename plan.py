import yaml

class Plan:
    def __init__(self, plan_file):
        # path to the plan file
        self.plan_file = plan_file

        with open(plan_file, 'r') as f:
            try:
                plan = yaml.safe_load(f)
            except yaml.YAMLError as exc:
                print(exc)
                exit(1)

        self.data_path = plan['data_file']
        self.mapping = plan['mapping']
        self.test_ratio = plan['test_ratio']

        self.default = {
            'use': 'ignore',
            'preprocess': 'original'
        }

    #@todo fix, user_input gets overwritten.
    def apply(self, fields):
        plan = {}
        for field in fields:
            plan[field] = self.default.copy()
            if field in self.mapping:
                for key in self.mapping[field]:
                    plan[field][key] = self.mapping[field][key]

        return plan